import os
import json
import hashlib
import secrets
from datetime import datetime, timezone, timedelta
from flask import (
    Flask, render_template, request, jsonify, send_file, make_response,
    redirect, url_for, g, session
)
from werkzeug.security import generate_password_hash, check_password_hash
from dotenv import load_dotenv
from io import BytesIO
from sqlalchemy import text, or_

from models import db, AppSetting, User, InviteCode, DeviceSession, Calibration, Measurement, CalibrationModel
from calibration_engine import (
    extract_basic_features,
    estimate_blood_pressure,
    train_simple_linear_model,
    train_polynomial_global_model,
    MIN_PERSONAL,
    MIN_GLOBAL,
)
from bias_analysis import analyze_demographic_bias
from validation_outliers import (
    detect_signal_outliers,
    detect_calibration_outliers,
    filter_calibrations_remove_outliers,
    validate_global_model,
    residual_outliers_with_model,
)
from hrv import build_variability_report, assess_bp_confidence
from neural_ecg import (
    detect_pulse_irregularity,
    train_neural_ppg_to_ecg_model,
    predict_ecg_from_ppg,
)
from pdf_report import build_measurement_pdf
from notify import send_email_with_pdf, send_email_plain, send_whatsapp_notification

load_dotenv()

app = Flask(__name__)
app.config["SECRET_KEY"] = os.getenv("SECRET_KEY", "dev-secret-change-me")
app.config["SQLALCHEMY_DATABASE_URI"] = os.getenv(
    "DATABASE_URL", "sqlite:///cardiasync_dev.db"
)
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
app.config["PERMANENT_SESSION_LIFETIME"] = timedelta(days=30)
app.config["SESSION_COOKIE_SAMESITE"] = "Lax"
app.config["SESSION_COOKIE_HTTPONLY"] = True

uri = app.config["SQLALCHEMY_DATABASE_URI"]
if uri and uri.startswith("postgres://"):
    app.config["SQLALCHEMY_DATABASE_URI"] = uri.replace("postgres://", "postgresql://", 1)

db.init_app(app)


def _ensure_schema():
    """
    create_all só cria tabelas novas — não adiciona colunas.
    No Neon/Render, deploys anteriores deixam o schema desatualizado
    e /historico quebra ao fazer SELECT em colunas inexistentes.
    """
    db.create_all()

    alterations = [
        ("users", "password_hash", "VARCHAR(256)"),
        ("users", "is_blocked", "BOOLEAN"),
        ("invite_codes", "recipient_phone", "VARCHAR(40)"),
        ("users", "recovery_code_hash", "VARCHAR(256)"),
        ("users", "reset_code_hash", "VARCHAR(256)"),
        ("users", "reset_code_expires", "TIMESTAMP"),
        ("users", "updated_at", "TIMESTAMP"),
        ("users", "age", "INTEGER"),
        ("users", "sex", "VARCHAR(20)"),
        ("users", "skin_tone", "VARCHAR(30)"),
        ("calibrations", "age", "INTEGER"),
        ("calibrations", "sex", "VARCHAR(20)"),
        ("calibrations", "skin_tone", "VARCHAR(30)"),
        ("calibrations", "notes", "TEXT"),
        ("calibrations", "features_json", "TEXT"),
        ("calibrations", "peak_amplitude", "FLOAT"),
        ("calibrations", "pulse_width", "FLOAT"),
        ("calibrations", "mean_red", "FLOAT"),
        ("calibrations", "std_red", "FLOAT"),
        ("calibrations", "signal_quality", "FLOAT"),
        ("calibrations", "heart_rate", "FLOAT"),
        ("calibrations", "condition", "VARCHAR(30)"),
        ("measurements", "sbp_std", "FLOAT"),
        ("measurements", "dbp_std", "FLOAT"),
        ("measurements", "features_json", "TEXT"),
        ("measurements", "mean_red", "FLOAT"),
        ("measurements", "std_red", "FLOAT"),
        ("measurements", "feeding", "VARCHAR(20)"),
        ("measurements", "position", "VARCHAR(20)"),
        ("measurements", "arm", "VARCHAR(20)"),
        ("measurements", "pulse_height", "VARCHAR(20)"),
        ("measurements", "period", "VARCHAR(20)"),
        ("measurements", "observation", "TEXT"),
        ("measurements", "condition", "VARCHAR(30)"),
        ("measurements", "calibration_id", "INTEGER"),
        ("calibration_models", "scope", "VARCHAR(20)"),
        ("calibration_models", "degree", "INTEGER"),
        ("calibration_models", "residual_std_sbp", "FLOAT"),
        ("calibration_models", "residual_std_dbp", "FLOAT"),
        ("calibration_models", "bias_report_json", "TEXT"),
        ("calibration_models", "coefficients_json", "TEXT"),
        ("calibration_models", "intercept_systolic", "FLOAT"),
        ("calibration_models", "intercept_diastolic", "FLOAT"),
        ("calibration_models", "r2_score", "FLOAT"),
        ("calibration_models", "n_samples", "INTEGER"),
        ("calibration_models", "is_active", "BOOLEAN"),
        ("calibration_models", "model_type", "VARCHAR(50)"),
        ("calibration_models", "updated_at", "TIMESTAMP"),
    ]

    uri = app.config.get("SQLALCHEMY_DATABASE_URI") or ""
    is_pg = uri.startswith("postgresql") or uri.startswith("postgres")

    with db.engine.begin() as conn:
        for table, column, coltype in alterations:
            try:
                if is_pg:
                    conn.execute(
                        text(
                            f'ALTER TABLE "{table}" ADD COLUMN IF NOT EXISTS "{column}" {coltype}'
                        )
                    )
                else:
                    rows = conn.execute(text(f"PRAGMA table_info({table})")).fetchall()
                    existing = {r[1] for r in rows}
                    if column not in existing:
                        conn.execute(
                            text(f"ALTER TABLE {table} ADD COLUMN {column} {coltype}")
                        )
            except Exception:
                pass


DEVICE_COOKIE_NAME = "cs_device_token"
DEVICE_COOKIE_MAX_AGE = 10 * 365 * 24 * 3600  # 10 anos (permanece ativo até apagar o app)


def _hash_token(raw_token: str) -> str:
    return hashlib.sha256((raw_token or "").encode("utf-8")).hexdigest()


def _generate_rescue_code() -> str:
    alphabet = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"
    p1 = "".join(secrets.choice(alphabet) for _ in range(4))
    p2 = "".join(secrets.choice(alphabet) for _ in range(4))
    return f"CS-{p1}-{p2}"


def _issue_device_session(user: User, user_agent: str = "") -> str:
    raw_token = secrets.token_urlsafe(48)
    sess = DeviceSession(
        user_id=user.id,
        token_hash=_hash_token(raw_token),
        device_label=(user_agent or "Celular")[:110],
    )
    db.session.add(sess)
    db.session.commit()
    return raw_token


def _attach_device_cookie(response, raw_token: str):
    response.set_cookie(
        DEVICE_COOKIE_NAME,
        raw_token,
        max_age=DEVICE_COOKIE_MAX_AGE,
        httponly=True,
        samesite="Lax",
        secure=request.is_secure,
        path="/",
    )
    return response


def _find_user_by_identifier(identifier: str):
    ident = (identifier or "").strip()
    if not ident:
        return None
    if "@" in ident:
        return User.query.filter(db.func.lower(User.email) == ident.lower()).first()
    return User.query.filter(db.func.lower(User.name) == ident.lower()).first()


@app.context_processor
def inject_current_user():
    return {"current_user": getattr(g, "current_user", None)}


@app.before_request
def enforce_device_lock():
    if not getattr(app, "_schema_ready", False):
        with app.app_context():
            try:
                _ensure_schema()
            except Exception as e:
                app.logger.exception("Schema ensure failed: %s", e)
        app._schema_ready = True

    g.current_user = None
    g.device_session = None
    g.raw_device_token = None

    raw_token = (
        request.cookies.get(DEVICE_COOKIE_NAME)
        or request.headers.get("X-Device-Token")
        or ""
    ).strip()

    if raw_token:
        try:
            sess = DeviceSession.query.filter_by(
                token_hash=_hash_token(raw_token),
                revoked=False,
            ).first()
            if sess and sess.user and not getattr(sess.user, "is_blocked", False):
                g.current_user = sess.user
                g.device_session = sess
                g.raw_device_token = raw_token
                now = datetime.now(timezone.utc)
                last_seen = sess.last_seen_at
                if last_seen and last_seen.tzinfo is None:
                    last_seen = last_seen.replace(tzinfo=timezone.utc)
                if not last_seen or (now - last_seen).total_seconds() > 3600:
                    sess.last_seen_at = now
                    db.session.commit()
        except Exception:
            try:
                db.session.rollback()
            except Exception:
                pass

    public_endpoints = {
        "static",
        "health",
        "service_worker",
        "acesso_page",
        "admin_page",
        "api_admin_setup",
        "api_admin_login",
        "api_admin_logout",
        "api_admin_create_invite",
        "api_admin_revoke_invite",
        "api_admin_toggle_user",
        "api_admin_link_this_device",
        "api_auth_register",
        "api_auth_login",
        "api_auth_restore_device",
        "api_auth_request_reset",
        "api_auth_verify_reset",
        "api_auth_recover_rescue",
    }

    if (
        request.endpoint in public_endpoints
        or request.endpoint is None
        or (request.endpoint and request.endpoint.startswith("api_admin_"))
    ):
        return None

    if g.current_user is None:
        if request.path.startswith("/api/"):
            return jsonify({
                "ok": False,
                "auth_required": True,
                "error": "Aparelho não vinculado. Crie ou informe sua senha de acesso.",
            }), 401
        return redirect(url_for("acesso_page"))

    return None


@app.after_request
def refresh_device_cookie(response):
    raw_token = getattr(g, "raw_device_token", None)
    if raw_token and request.endpoint != "api_auth_unlink_device":
        _attach_device_cookie(response, raw_token)
    return response


def _parse_features_safe(features_json_str):
    if not features_json_str:
        return {}
    try:
        data = json.loads(features_json_str)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _attach_ecg_metadata(record):
    feats = _parse_features_safe(getattr(record, "features_json", None))
    ecg_val = feats.get("ecg_validation") or {}
    bp_conf = ecg_val.get("bp_confidence") or {}
    provided = bool(ecg_val.get("ecg_provided"))
    record.ecg_provided = provided
    if provided:
        if bp_conf.get("level") == "moderate_mismatch":
            record.ecg_status_label = "⚠️ Divergência ECG"
            record.ecg_badge_class = "medium"
        else:
            record.ecg_status_label = "⚡ Validado ECG"
            record.ecg_badge_class = "good"
    else:
        record.ecg_status_label = "Sem ECG"
        record.ecg_badge_class = "medium"

    irreg = feats.get("pulse_irregularity") or {}
    record.irregularity_label = irreg.get("label") or "Ritmo Regular"
    record.irregularity_badge_class = irreg.get("badge_class") or "good"
    record.has_neural_ecg = bool((feats.get("neural_ecg_prediction") or {}).get("available"))


def get_or_create_user(
    name: str,
    email: str = None,
    phone: str = None,
    age: int = None,
    sex: str = None,
    skin_tone: str = None,
):
    user = User.query.filter_by(name=name).first()
    if not user:
        user = User(
            name=name, email=email, phone=phone,
            age=age, sex=sex, skin_tone=skin_tone,
        )
        db.session.add(user)
        db.session.flush()
    else:
        if email and not user.email:
            user.email = email
        if phone and not user.phone:
            user.phone = phone
        if age is not None:
            user.age = age
        if sex:
            user.sex = sex
        if skin_tone:
            user.skin_tone = skin_tone
    return user


def _load_coeffs(model: CalibrationModel):
    if not model or not model.coefficients_json:
        return None
    try:
        return json.loads(model.coefficients_json)
    except Exception:
        return None


def get_active_coeffs(user_id=None):
    """Prioriza modelo pessoal ativo; senão modelo global polinomial."""
    q = CalibrationModel.query.filter_by(is_active=True).filter(
        CalibrationModel.scope.in_(["personal", "global"])
    )
    if user_id:
        m = (
            q.filter_by(user_id=user_id, scope="personal")
            .order_by(CalibrationModel.created_at.desc())
            .first()
        )
        coeffs = _load_coeffs(m)
        if coeffs:
            return coeffs

    m = (
        q.filter(CalibrationModel.user_id.is_(None), CalibrationModel.scope == "global")
        .order_by(CalibrationModel.created_at.desc())
        .first()
    )
    coeffs = _load_coeffs(m)
    if coeffs:
        return coeffs
    return None


def get_active_neural_ecg_weights():
    """Carrega pesos ativos da rede neural PPG-to-ECG, se já treinada."""
    m = (
        CalibrationModel.query.filter_by(is_active=True, scope="neural_ecg")
        .order_by(CalibrationModel.created_at.desc())
        .first()
    )
    return _load_coeffs(m)


def maybe_train_neural_ecg_model(min_pairs: int = 3):
    """
    Coleta todas as calibrações pareadas (PPG + ECG) no banco de dados e
    treina/atualiza a rede neural MLP PPG-to-ECG.
    """
    cals = (
        Calibration.query.filter(Calibration.signal_quality >= 0.3)
        .order_by(Calibration.created_at.desc())
        .limit(400)
        .all()
    )
    paired_records = []
    for c in cals:
        feats = _parse_features_safe(c.features_json)
        ecg_val = feats.get("ecg_validation") or {}
        if not ecg_val.get("ecg_provided"):
            continue
        ppg_ibi = feats.get("ppg_ibi_ms") or []
        ecg_series = ecg_val.get("ecg_series") or {}
        ecg_rr = ecg_series.get("clean_rr_ms") or []
        if len(ppg_ibi) >= 3 and len(ecg_rr) >= 3:
            std_r = float(c.std_red or 1.0)
            peak_a = float(c.peak_amplitude or 0.35)
            paired_records.append({
                "ppg_ibi_ms": ppg_ibi,
                "ecg_rr_ms": ecg_rr,
                "sbp": c.ref_systolic,
                "dbp": c.ref_diastolic,
                "norm_amp": peak_a / max(1.0, std_r),
                "pulse_width": float(c.pulse_width or 0.28),
            })

    if len(paired_records) < min_pairs:
        return None, len(paired_records)

    weights = train_neural_ppg_to_ecg_model(paired_records, min_pairs=min_pairs)
    if not weights:
        return None, len(paired_records)

    CalibrationModel.query.filter_by(scope="neural_ecg", is_active=True).update(
        {"is_active": False}
    )
    model = CalibrationModel(
        user_id=None,
        model_type="mlp_ppg_to_ecg",
        scope="neural_ecg",
        degree=1,
        coefficients_json=json.dumps(weights),
        n_samples=weights.get("n_paired_recordings", len(paired_records)),
        is_active=True,
    )
    db.session.add(model)
    db.session.commit()
    return weights, len(paired_records)


def resolve_calibration_state(user_name: str):
    """
    Verifica se o sistema possui modelo calibrado (pessoal ou global)
    para liberar medições. Se já houver amostras suficientes no banco mas
    o modelo ainda não foi compilado, dispara o treino automaticamente.
    """
    clean_name = (user_name or "").strip() or "Anônimo"
    user = User.query.filter_by(name=clean_name).first()
    user_id = user.id if user else None

    personal_count = (
        Calibration.query.filter_by(user_id=user_id)
        .filter(Calibration.signal_quality >= 0.3)
        .count()
        if user_id
        else 0
    )
    global_count = Calibration.query.filter(Calibration.signal_quality >= 0.3).count()

    coeffs = get_active_coeffs(user_id)

    # Auto-cura: se já atingiu o mínimo de calibrações mas ainda não há modelo ativo
    if coeffs is None and user_id and personal_count >= MIN_PERSONAL:
        coeffs, _ = maybe_train_model(user_id)
    if coeffs is None and global_count >= MIN_GLOBAL:
        g_coeffs, _, _, _ = train_global_polynomial(degree=2)
        if g_coeffs:
            coeffs = g_coeffs

    is_calibrated = coeffs is not None
    scope = (coeffs or {}).get("scope") if coeffs else None
    model_type = (coeffs or {}).get("model_type") if coeffs else None

    if is_calibrated:
        if scope == "personal":
            msg = f"Modelo pessoal calibrado ({personal_count} calibrações). Pronto para medir automaticamente."
        else:
            msg = f"Modelo global calibrado ({global_count} amostras). Pronto para medir automaticamente."
    else:
        faltam_p = max(0, MIN_PERSONAL - personal_count)
        msg = (
            f"Calibração obrigatória antes de medir. "
            f"Você tem {personal_count}/{MIN_PERSONAL} calibrações pessoais "
            f"(faltam {faltam_p})."
        )

    return {
        "is_calibrated": is_calibrated,
        "coeffs": coeffs,
        "user_id": user_id,
        "user_name": clean_name,
        "model_scope": scope,
        "model_type": model_type,
        "personal_calibrations": personal_count,
        "min_personal": MIN_PERSONAL,
        "global_calibrations": global_count,
        "min_global": MIN_GLOBAL,
        "message": msg,
    }


def maybe_train_model(user_id: int):
    """Treina Ridge pessoal se ≥ 5 calibrações boas do usuário."""
    cals = (
        Calibration.query.filter_by(user_id=user_id)
        .filter(Calibration.signal_quality >= 0.3)
        .order_by(Calibration.created_at.desc())
        .limit(40)
        .all()
    )
    if len(cals) < MIN_PERSONAL:
        return None, len(cals)

    coeffs = train_simple_linear_model(cals)
    if not coeffs:
        return None, len(cals)

    CalibrationModel.query.filter_by(user_id=user_id, is_active=True).update(
        {"is_active": False}
    )
    model = CalibrationModel(
        user_id=user_id,
        model_type=coeffs.get("model_type", "ridge"),
        scope="personal",
        degree=1,
        coefficients_json=json.dumps(coeffs),
        intercept_systolic=coeffs["systolic"]["intercept"],
        intercept_diastolic=coeffs["diastolic"]["intercept"],
        r2_score=(coeffs.get("r2_systolic", 0) + coeffs.get("r2_diastolic", 0)) / 2,
        residual_std_sbp=coeffs.get("residual_std_sbp"),
        residual_std_dbp=coeffs.get("residual_std_dbp"),
        n_samples=coeffs.get("n_samples", len(cals)),
        is_active=True,
    )
    db.session.add(model)
    db.session.commit()
    return coeffs, len(cals)


def train_global_polynomial(degree: int = 2, remove_outliers: bool = True):
    """
    Treina modelo polinomial populacional com calibrações boas,
    opcionalmente removendo outliers de features/PA.
    """
    cals = (
        Calibration.query.filter(Calibration.signal_quality >= 0.3)
        .order_by(Calibration.created_at.desc())
        .limit(500)
        .all()
    )
    outlier_info = None
    train_set = cals
    if remove_outliers and len(cals) >= MIN_GLOBAL:
        train_set, outlier_info = filter_calibrations_remove_outliers(cals)
        if len(train_set) < MIN_GLOBAL:
            train_set = cals
            if outlier_info is not None:
                outlier_info["fallback_used_all"] = True

    if len(train_set) < MIN_GLOBAL:
        return None, len(cals), None, outlier_info

    coeffs = train_polynomial_global_model(
        train_set, degree=degree, min_samples=MIN_GLOBAL
    )
    if not coeffs:
        return None, len(cals), None, outlier_info

    bias = analyze_demographic_bias(train_set, coeffs)
    validation = validate_global_model(
        cals, degree=degree, remove_outliers=remove_outliers
    )

    CalibrationModel.query.filter(
        CalibrationModel.user_id.is_(None),
        CalibrationModel.is_active.is_(True),
    ).update({"is_active": False})

    extra = {
        "bias": bias,
        "validation": validation,
        "outliers": outlier_info,
    }

    model = CalibrationModel(
        user_id=None,
        model_type="polynomial",
        scope="global",
        degree=degree,
        coefficients_json=json.dumps(coeffs),
        intercept_systolic=coeffs["systolic"]["intercept"],
        intercept_diastolic=coeffs["diastolic"]["intercept"],
        r2_score=(coeffs.get("r2_systolic", 0) + coeffs.get("r2_diastolic", 0)) / 2,
        residual_std_sbp=coeffs.get("residual_std_sbp"),
        residual_std_dbp=coeffs.get("residual_std_dbp"),
        n_samples=coeffs.get("n_samples", len(train_set)),
        is_active=True,
        bias_report_json=json.dumps(extra, ensure_ascii=False),
    )
    db.session.add(model)
    db.session.commit()
    return coeffs, len(train_set), extra, outlier_info


# ──────────────────────────────────────────────
# Páginas
# ──────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/medir")
def measure_page():
    return render_template("measure.html")


@app.route("/calibrar")
def calibrate_page():
    return render_template("calibrate.html")


@app.route("/historico")
def historico():
    measurements = []
    calibrations = []
    error_msg = None
    try:
        if getattr(g, "current_user", None):
            uid = g.current_user.id
            measurements = (
                Measurement.query.filter_by(user_id=uid)
                .order_by(Measurement.created_at.desc())
                .limit(100)
                .all()
            )
            calibrations = (
                Calibration.query.filter_by(user_id=uid)
                .order_by(Calibration.created_at.desc())
                .limit(50)
                .all()
            )
        else:
            measurements = (
                Measurement.query.order_by(Measurement.created_at.desc()).limit(100).all()
            )
            calibrations = (
                Calibration.query.order_by(Calibration.created_at.desc()).limit(50).all()
            )
        for m in measurements:
            _attach_ecg_metadata(m)
        for c in calibrations:
            _attach_ecg_metadata(c)
    except Exception as e:
        app.logger.exception("Erro ao carregar histórico: %s", e)
        error_msg = (
            "Não foi possível ler o banco ainda. "
            "Se acabou de fazer deploy, recarregue em alguns segundos. "
            "Se persistir, confira DATABASE_URL no Render."
        )
        try:
            db.session.rollback()
        except Exception:
            pass
    return render_template(
        "historico.html",
        measurements=measurements,
        calibrations=calibrations,
        error_msg=error_msg,
    )


# ──────────────────────────────────────────────
# API – Status de Calibração (Pré-verificação)
# ──────────────────────────────────────────────

@app.route("/api/calibration-status", methods=["GET"])
def api_calibration_status():
    user_name = (request.args.get("user_name") or "").strip() or "Anônimo"
    state = resolve_calibration_state(user_name)
    return jsonify({
        "ok": True,
        "is_calibrated": state["is_calibrated"],
        "user_name": state["user_name"],
        "model_scope": state["model_scope"],
        "model_type": state["model_type"],
        "personal_calibrations": state["personal_calibrations"],
        "min_personal": state["min_personal"],
        "global_calibrations": state["global_calibrations"],
        "min_global": state["min_global"],
        "message": state["message"],
    })


# ──────────────────────────────────────────────
# API – Medição (Somente quando calibrado)
# ──────────────────────────────────────────────

@app.route("/api/measure", methods=["POST"])
def api_measure():
    data = request.get_json(force=True) or {}

    red_signal = data.get("red_signal", [])
    fps = float(data.get("fps", 30))
    condition = data.get("condition", "Repouso")
    user_name = (data.get("user_name") or "").strip() or "Anônimo"
    email = (data.get("email") or "").strip() or None
    phone = (data.get("phone") or "").strip() or None

    feeding = data.get("feeding")
    position = data.get("position")
    arm = data.get("arm")
    pulse_height = data.get("pulse_height")
    observation = data.get("observation")
    send_email = bool(data.get("send_email"))
    send_whatsapp = bool(data.get("send_whatsapp"))

    # 1. Bloqueia medição se não houver calibração ativa
    cal_state = resolve_calibration_state(user_name)
    if not cal_state["is_calibrated"] or cal_state["coeffs"] is None:
        return jsonify({
            "ok": False,
            "requires_calibration": True,
            "personal_calibrations": cal_state["personal_calibrations"],
            "min_personal": cal_state["min_personal"],
            "global_calibrations": cal_state["global_calibrations"],
            "min_global": cal_state["min_global"],
            "error": (
                f"Calibração obrigatória antes de medir. Você possui "
                f"{cal_state['personal_calibrations']}/{cal_state['min_personal']} calibrações pessoais. "
                f"Acesse a aba Calibrar primeiro."
            ),
        }), 400

    if not red_signal or len(red_signal) < 30:
        return jsonify({"ok": False, "error": "Sinal insuficiente. Tente novamente."}), 400

    features = extract_basic_features(red_signal, fps=fps)

    outlier = detect_signal_outliers(features, red_signal)
    if features.get("signal_quality", 0) < 0.25 or outlier.get("is_outlier"):
        reasons = ", ".join(outlier.get("reasons") or ["qualidade insuficiente"])
        return jsonify({
            "ok": False,
            "error": f"Sinal rejeitado. Motivos: {reasons}. Reposicione o dedo e tente de novo.",
            "outlier": outlier,
        }), 400

    # HRV/PRV a partir dos picos PPG; ECG opcional para validação profissional
    peaks = features.pop("peak_indices", []) or []
    ppg_wave_snippet = features.pop("ppg_wave_snippet", []) or []
    variability = build_variability_report(
        peaks=peaks,
        fps=fps,
        ecg_rr_ms=data.get("ecg_rr_ms"),
        ecg_peaks_sec=data.get("ecg_peaks_sec"),
    )

    user = get_or_create_user(user_name, email=email, phone=phone)
    coeffs = cal_state["coeffs"]
    sbp, dbp, status, sbp_std, dbp_std = estimate_blood_pressure(features, coeffs)

    if status != "calibrado" or sbp is None or dbp is None:
        return jsonify({
            "ok": False,
            "requires_calibration": True,
            "error": "Não foi possível estimar a pressão arterial com o modelo atual. Refaça a calibração na aba Calibrar.",
        }), 400

    bp_confidence = assess_bp_confidence(
        ecg_status=variability.get("ecg_status", "missing"),
        comparison=variability.get("comparison"),
        base_sbp_std=sbp_std,
        base_dbp_std=dbp_std,
    )
    final_sbp_std = (
        bp_confidence["adjusted_sbp_std"]
        if bp_confidence["adjusted_sbp_std"] is not None
        else sbp_std
    )
    final_dbp_std = (
        bp_confidence["adjusted_dbp_std"]
        if bp_confidence["adjusted_dbp_std"] is not None
        else dbp_std
    )

    # Triagem automática de irregularidade de pulso + Tradução Neural PPG-to-ECG
    ecg_clean_rr = ((variability.get("ecg_series") or {}).get("clean_rr_ms")) if variability.get("ecg_provided") else None
    pulse_irregularity = detect_pulse_irregularity(
        ppg_ibi_ms=variability.get("ppg_ibi_ms") or [],
        ecg_rr_ms=ecg_clean_rr,
    )
    neural_weights = get_active_neural_ecg_weights()
    if neural_weights is None:
        neural_weights, _ = maybe_train_neural_ecg_model()

    neural_ecg_prediction = predict_ecg_from_ppg(
        ppg_ibi_ms=variability.get("ppg_ibi_ms") or [],
        features=features,
        estimated_sbp=sbp,
        estimated_dbp=dbp,
        neural_weights=neural_weights,
        irregularity_report=pulse_irregularity,
        ppg_wave_snippet=ppg_wave_snippet,
        fps=fps,
    )

    features["hrv"] = variability.get("prv")
    features["ppg_ibi_ms"] = variability.get("ppg_ibi_ms") or []
    features["pulse_irregularity"] = pulse_irregularity
    features["neural_ecg_prediction"] = neural_ecg_prediction
    features["ecg_validation"] = {
        "ecg_provided": variability.get("ecg_provided"),
        "ecg_status": variability.get("ecg_status"),
        "hrv_ecg": variability.get("hrv_ecg"),
        "comparison": variability.get("comparison"),
        "ecg_series": variability.get("ecg_series"),
        "bp_confidence": bp_confidence,
    }

    measurement = Measurement(
        user_id=user.id,
        heart_rate=features.get("heart_rate"),
        estimated_systolic=sbp,
        estimated_diastolic=dbp,
        signal_quality=features.get("signal_quality"),
        mean_red=features.get("mean_red"),
        std_red=features.get("std_red"),
        features_json=json.dumps(features),
        condition=condition,
        feeding=feeding,
        position=position,
        arm=arm,
        pulse_height=pulse_height,
        period=condition,
        observation=observation,
        sbp_std=final_sbp_std,
        dbp_std=final_dbp_std,
    )
    db.session.add(measurement)
    db.session.commit()

    msg = (
        "Estimativa calibrada e validada por ECG (alta confiança)."
        if bp_confidence["level"] == "high"
        else "Estimativa calibrada apenas por PPG (sem ECG — menor segurança clínica)."
    )

    # PDF + envios opcionais
    pdf_info = None
    email_result = None
    wa_result = None

    pdf_data = {
        "user_name": user_name,
        "created_at": measurement.created_at.strftime("%d/%m/%Y %H:%M:%S") if measurement.created_at else None,
        "heart_rate": features.get("heart_rate"),
        "estimated_systolic": sbp,
        "estimated_diastolic": dbp,
        "sbp_std": final_sbp_std,
        "dbp_std": final_dbp_std,
        "signal_quality": features.get("signal_quality"),
        "condition": condition,
        "feeding": feeding,
        "position": position,
        "arm": arm,
        "pulse_height": pulse_height,
        "observation": observation,
        "status": status,
        "ecg_validation": features["ecg_validation"],
        "bp_confidence": bp_confidence,
        "pulse_irregularity": pulse_irregularity,
        "neural_ecg_prediction": neural_ecg_prediction,
    }

    try:
        pdf_bytes = build_measurement_pdf(pdf_data)
        pdf_info = {"size": len(pdf_bytes)}

        if send_email and email:
            ok, emsg = send_email_with_pdf(
                to_email=email,
                subject=f"CardiaSync — Relatório {user_name}",
                body_text=(
                    f"Olá {user_name},\n\n"
                    f"Segue o relatório da sua medição PPG.\n"
                    f"FC: {features.get('heart_rate')} bpm\n"
                    f"PA estimada: {sbp or 'N/A'}/{dbp or 'N/A'} mmHg\n"
                    f"Confiança: {bp_confidence.get('label')}\n\n"
                    "Este é um sistema experimental. Não use para diagnóstico clínico."
                ),
                pdf_bytes=pdf_bytes,
            )
            email_result = {"ok": ok, "message": emsg}

        if send_whatsapp and phone:
            ok, wmsg = send_whatsapp_notification(
                to_phone=phone,
                message=(
                    f"CardiaSync — Medição de {user_name}\n"
                    f"FC: {features.get('heart_rate')} bpm\n"
                    f"PA: {sbp or 'N/A'}/{dbp or 'N/A'} mmHg\n"
                    f"Segurança: {bp_confidence.get('label')}\n"
                    f"Qualidade: {int((features.get('signal_quality') or 0)*100)}%\n"
                    "⚠️ Sistema experimental — não use para diagnóstico."
                ),
            )
            wa_result = {"ok": ok, "message": wmsg}
    except Exception as e:
        pdf_info = {"error": str(e)}

    return jsonify({
        "ok": True,
        "measurement_id": measurement.id,
        "heart_rate": features.get("heart_rate"),
        "estimated_systolic": sbp,
        "estimated_diastolic": dbp,
        "sbp_std": final_sbp_std,
        "dbp_std": final_dbp_std,
        "signal_quality": features.get("signal_quality"),
        "variability": variability,
        "bp_confidence": bp_confidence,
        "pulse_irregularity": pulse_irregularity,
        "neural_ecg_prediction": neural_ecg_prediction,
        "status": status,
        "model_scope": (coeffs or {}).get("scope") if coeffs else None,
        "model_type": (coeffs or {}).get("model_type") if coeffs else None,
        "message": msg,
        "features": features,
        "pdf": pdf_info,
        "email": email_result,
        "whatsapp": wa_result,
    })


# ──────────────────────────────────────────────
# API – Calibração + treino automático
# ──────────────────────────────────────────────

@app.route("/api/calibrate", methods=["POST"])
def api_calibrate():
    data = request.get_json(force=True) or {}

    try:
        ref_systolic = float(data.get("ref_systolic"))
        ref_diastolic = float(data.get("ref_diastolic"))
    except (TypeError, ValueError):
        return jsonify({"ok": False, "error": "Informe pressão sistólica e diastólica válidas."}), 400

    if not (70 <= ref_systolic <= 220 and 40 <= ref_diastolic <= 130):
        return jsonify({"ok": False, "error": "Valores de pressão fora da faixa fisiológica."}), 400
    if ref_diastolic >= ref_systolic:
        return jsonify({"ok": False, "error": "Diastólica deve ser menor que a sistólica."}), 400

    red_signal = data.get("red_signal", [])
    fps = float(data.get("fps", 30))
    condition = data.get("condition", "Repouso")
    user_name = (data.get("user_name") or "").strip() or "Anônimo"
    notes = data.get("notes", "")

    age = data.get("age")
    try:
        age = int(age) if age not in (None, "") else None
    except (TypeError, ValueError):
        age = None
    sex = (data.get("sex") or "").strip() or None
    skin_tone = (data.get("skin_tone") or "").strip() or None

    if not red_signal or len(red_signal) < 30:
        return jsonify({"ok": False, "error": "Sinal da câmera insuficiente."}), 400

    features = extract_basic_features(red_signal, fps=fps)
    outlier = detect_signal_outliers(features, red_signal)

    if features.get("signal_quality", 0) < 0.35 or outlier.get("is_outlier"):
        reasons = ", ".join(outlier.get("reasons") or ["qualidade insuficiente"])
        return jsonify({
            "ok": False,
            "error": f"Sinal rejeitado (outlier/qualidade). Motivos: {reasons}. Reposicione o dedo e tente novamente.",
            "outlier": outlier,
            "features": {k: v for k, v in features.items() if k != "peak_indices"},
        }), 400

    peaks = features.pop("peak_indices", []) or []
    ppg_wave_snippet = features.pop("ppg_wave_snippet", []) or []
    variability = build_variability_report(
        peaks=peaks,
        fps=fps,
        ecg_rr_ms=data.get("ecg_rr_ms"),
        ecg_peaks_sec=data.get("ecg_peaks_sec"),
    )
    bp_confidence = assess_bp_confidence(
        ecg_status=variability.get("ecg_status", "missing"),
        comparison=variability.get("comparison"),
    )
    ecg_clean_rr = ((variability.get("ecg_series") or {}).get("clean_rr_ms")) if variability.get("ecg_provided") else None
    pulse_irregularity = detect_pulse_irregularity(
        ppg_ibi_ms=variability.get("ppg_ibi_ms") or [],
        ecg_rr_ms=ecg_clean_rr,
    )
    neural_ecg_prediction = predict_ecg_from_ppg(
        ppg_ibi_ms=variability.get("ppg_ibi_ms") or [],
        features=features,
        estimated_sbp=ref_systolic,
        estimated_dbp=ref_diastolic,
        neural_weights=get_active_neural_ecg_weights(),
        irregularity_report=pulse_irregularity,
        ppg_wave_snippet=ppg_wave_snippet,
        fps=fps,
    )

    features["hrv"] = variability.get("prv")
    features["ppg_ibi_ms"] = variability.get("ppg_ibi_ms") or []
    features["pulse_irregularity"] = pulse_irregularity
    features["neural_ecg_prediction"] = neural_ecg_prediction
    features["ecg_validation"] = {
        "ecg_provided": variability.get("ecg_provided"),
        "ecg_status": variability.get("ecg_status"),
        "hrv_ecg": variability.get("hrv_ecg"),
        "comparison": variability.get("comparison"),
        "ecg_series": variability.get("ecg_series"),
        "bp_confidence": bp_confidence,
    }

    user = get_or_create_user(
        user_name, age=age, sex=sex, skin_tone=skin_tone,
    )

    calibration = Calibration(
        user_id=user.id,
        ref_systolic=ref_systolic,
        ref_diastolic=ref_diastolic,
        heart_rate=features.get("heart_rate"),
        signal_quality=features.get("signal_quality"),
        mean_red=features.get("mean_red"),
        std_red=features.get("std_red"),
        peak_amplitude=features.get("peak_amplitude"),
        pulse_width=features.get("pulse_width"),
        features_json=json.dumps(features),
        condition=condition,
        notes=notes,
        age=age or user.age,
        sex=sex or user.sex,
        skin_tone=skin_tone or user.skin_tone,
    )
    db.session.add(calibration)
    db.session.commit()

    coeffs, total = maybe_train_model(user.id)
    trained = coeffs is not None

    neural_info = None
    if variability.get("ecg_provided"):
        n_weights, n_paired = maybe_train_neural_ecg_model()
        if n_weights:
            neural_info = {
                "trained": True,
                "n_paired_recordings": n_paired,
                "n_beats_learned": n_weights.get("n_beats_learned"),
                "mae_rr_ms": n_weights.get("mae_rr_ms"),
            }

    # Tenta atualizar modelo global se já houver volume
    global_info = None
    all_n = Calibration.query.filter(Calibration.signal_quality >= 0.3).count()
    if all_n >= MIN_GLOBAL:
        g_coeffs, g_n, g_extra, _ = train_global_polynomial(degree=2)
        if g_coeffs:
            val = (g_extra or {}).get("validation") or {}
            global_info = {
                "trained": True,
                "n_samples": g_n,
                "r2_systolic": g_coeffs.get("r2_systolic"),
                "r2_diastolic": g_coeffs.get("r2_diastolic"),
                "residual_std_sbp": g_coeffs.get("residual_std_sbp"),
                "residual_std_dbp": g_coeffs.get("residual_std_dbp"),
                "validation_mae_sbp": (val.get("systolic") or {}).get("mae"),
                "validation_mae_dbp": (val.get("diastolic") or {}).get("mae"),
                "bhs_grade_sbp": val.get("bhs_grade_systolic"),
                "bhs_grade_dbp": val.get("bhs_grade_diastolic"),
            }

    msg = f"Calibração salva. Total pessoal: {total}."
    if variability.get("ecg_provided"):
        msg += " Validação por ECG registrada."
    if neural_info:
        msg += f" Rede neural PPG→ECG atualizada com {neural_info['n_paired_recordings']} pares (MAE {neural_info['mae_rr_ms']} ms)."
    if trained:
        msg += (
            f" Modelo pessoal atualizado "
            f"(R² sistólica ≈ {coeffs.get('r2_systolic', 0):.2f}, "
            f"diastólica ≈ {coeffs.get('r2_diastolic', 0):.2f})."
        )
    elif total < MIN_PERSONAL:
        msg += f" Faltam {MIN_PERSONAL - total} calibração(ões) pessoais para o modelo individual."
    if global_info:
        msg += f" Modelo global polinomial ativo (n={global_info['n_samples']})."
    elif all_n < MIN_GLOBAL:
        msg += f" Modelo global: faltam {MIN_GLOBAL - all_n} calibrações populacionais."

    return jsonify({
        "ok": True,
        "calibration_id": calibration.id,
        "total_calibrations": total,
        "model_trained": trained,
        "ecg_validated": bool(variability.get("ecg_provided")),
        "variability": variability,
        "bp_confidence": bp_confidence,
        "model_info": {
            "r2_systolic": coeffs.get("r2_systolic") if coeffs else None,
            "r2_diastolic": coeffs.get("r2_diastolic") if coeffs else None,
            "n_samples": coeffs.get("n_samples") if coeffs else None,
        } if trained else None,
        "global_model": global_info,
        "message": msg,
        "features": features,
    })


# ──────────────────────────────────────────────
# Detalhes visuais (ECG + PPG + Segurança) para Histórico
# ──────────────────────────────────────────────

@app.route("/api/measurement/<int:mid>/detail")
def api_measurement_detail(mid):
    m = Measurement.query.get_or_404(mid)
    feats = _parse_features_safe(m.features_json)
    ecg_val = feats.get("ecg_validation") or {}
    bp_conf = ecg_val.get("bp_confidence") or {}
    return jsonify({
        "ok": True,
        "kind": "measurement",
        "id": m.id,
        "user_name": m.user.name if m.user else "Anônimo",
        "created_at": m.created_at.strftime("%d/%m/%Y %H:%M:%S") if m.created_at else "—",
        "heart_rate": m.heart_rate,
        "estimated_systolic": m.estimated_systolic,
        "estimated_diastolic": m.estimated_diastolic,
        "sbp_std": m.sbp_std,
        "dbp_std": m.dbp_std,
        "signal_quality": m.signal_quality,
        "condition": m.condition,
        "observation": m.observation,
        "variability": {
            "prv": feats.get("hrv") or {},
            "hrv_ecg": ecg_val.get("hrv_ecg"),
            "comparison": ecg_val.get("comparison") or {},
            "ecg_provided": bool(ecg_val.get("ecg_provided")),
            "ecg_status": ecg_val.get("ecg_status") or "missing",
            "ppg_ibi_ms": feats.get("ppg_ibi_ms") or [],
            "ecg_series": ecg_val.get("ecg_series") or {},
        },
        "bp_confidence": bp_conf,
        "pulse_irregularity": feats.get("pulse_irregularity") or {},
        "neural_ecg_prediction": feats.get("neural_ecg_prediction") or {},
    })


@app.route("/api/calibration/<int:cid>/detail")
def api_calibration_detail(cid):
    c = Calibration.query.get_or_404(cid)
    feats = _parse_features_safe(c.features_json)
    ecg_val = feats.get("ecg_validation") or {}
    bp_conf = ecg_val.get("bp_confidence") or {}
    return jsonify({
        "ok": True,
        "kind": "calibration",
        "id": c.id,
        "user_name": c.user.name if c.user else "Anônimo",
        "created_at": c.created_at.strftime("%d/%m/%Y %H:%M:%S") if c.created_at else "—",
        "heart_rate": c.heart_rate,
        "ref_systolic": c.ref_systolic,
        "ref_diastolic": c.ref_diastolic,
        "signal_quality": c.signal_quality,
        "condition": c.condition,
        "observation": c.notes,
        "variability": {
            "prv": feats.get("hrv") or {},
            "hrv_ecg": ecg_val.get("hrv_ecg"),
            "comparison": ecg_val.get("comparison") or {},
            "ecg_provided": bool(ecg_val.get("ecg_provided")),
            "ecg_status": ecg_val.get("ecg_status") or "missing",
            "ppg_ibi_ms": feats.get("ppg_ibi_ms") or [],
            "ecg_series": ecg_val.get("ecg_series") or {},
        },
        "bp_confidence": bp_conf,
        "pulse_irregularity": feats.get("pulse_irregularity") or {},
        "neural_ecg_prediction": feats.get("neural_ecg_prediction") or {},
    })


# ──────────────────────────────────────────────
# PDF download
# ──────────────────────────────────────────────

@app.route("/api/measurement/<int:mid>/pdf")
def measurement_pdf(mid):
    m = Measurement.query.get_or_404(mid)
    feats = _parse_features_safe(m.features_json)
    ecg_val = feats.get("ecg_validation") or {}
    bp_conf = ecg_val.get("bp_confidence") or {}

    data = {
        "user_name": m.user.name if m.user else "—",
        "created_at": m.created_at.strftime("%d/%m/%Y %H:%M:%S") if m.created_at else None,
        "heart_rate": m.heart_rate,
        "estimated_systolic": m.estimated_systolic,
        "estimated_diastolic": m.estimated_diastolic,
        "sbp_std": m.sbp_std,
        "dbp_std": m.dbp_std,
        "signal_quality": m.signal_quality,
        "condition": m.condition,
        "feeding": m.feeding,
        "position": m.position,
        "arm": m.arm,
        "pulse_height": m.pulse_height,
        "observation": m.observation,
        "status": "calibrado" if m.estimated_systolic else "sem_calibracao",
        "ecg_validation": ecg_val,
        "bp_confidence": bp_conf,
        "pulse_irregularity": feats.get("pulse_irregularity") or {},
        "neural_ecg_prediction": feats.get("neural_ecg_prediction") or {},
    }
    pdf_bytes = build_measurement_pdf(data)
    return send_file(
        BytesIO(pdf_bytes),
        mimetype="application/pdf",
        as_attachment=True,
        download_name=f"cardiasync_medicao_{mid}.pdf",
    )


@app.route("/api/send-report", methods=["POST"])
def api_send_report():
    """Reenvia PDF de uma medição por e-mail e/ou WhatsApp."""
    data = request.get_json(force=True) or {}
    mid = data.get("measurement_id")
    email = (data.get("email") or "").strip() or None
    phone = (data.get("phone") or "").strip() or None

    if not mid:
        return jsonify({"ok": False, "error": "measurement_id obrigatório"}), 400

    m = Measurement.query.get(mid)
    if not m:
        return jsonify({"ok": False, "error": "Medição não encontrada"}), 404

    feats = _parse_features_safe(m.features_json)
    ecg_val = feats.get("ecg_validation") or {}
    bp_conf = ecg_val.get("bp_confidence") or {}

    pdf_data = {
        "user_name": m.user.name if m.user else "—",
        "created_at": m.created_at.strftime("%d/%m/%Y %H:%M:%S") if m.created_at else None,
        "heart_rate": m.heart_rate,
        "estimated_systolic": m.estimated_systolic,
        "estimated_diastolic": m.estimated_diastolic,
        "sbp_std": m.sbp_std,
        "dbp_std": m.dbp_std,
        "signal_quality": m.signal_quality,
        "condition": m.condition,
        "feeding": m.feeding,
        "position": m.position,
        "arm": m.arm,
        "pulse_height": m.pulse_height,
        "observation": m.observation,
        "status": "calibrado" if m.estimated_systolic else "sem_calibracao",
        "ecg_validation": ecg_val,
        "bp_confidence": bp_conf,
    }
    pdf_bytes = build_measurement_pdf(pdf_data)

    results = {}
    if email:
        ok, msg = send_email_with_pdf(
            to_email=email,
            subject=f"CardiaSync — Relatório #{mid}",
            body_text="Segue o relatório da medição PPG (sistema experimental).",
            pdf_bytes=pdf_bytes,
            filename=f"cardiasync_{mid}.pdf",
        )
        results["email"] = {"ok": ok, "message": msg}

    if phone:
        ok, msg = send_whatsapp_notification(
            to_phone=phone,
            message=(
                f"CardiaSync — Medição #{mid}\n"
                f"FC: {m.heart_rate} bpm | PA: {m.estimated_systolic or 'N/A'}/{m.estimated_diastolic or 'N/A'}\n"
                "⚠️ Experimental — não use para diagnóstico."
            ),
        )
        results["whatsapp"] = {"ok": ok, "message": msg}

    if not results:
        return jsonify({"ok": False, "error": "Informe email e/ou phone"}), 400

    return jsonify({"ok": True, "results": results})


# ──────────────────────────────────────────────
# Modelo global polinomial + análise de viés
# ──────────────────────────────────────────────

@app.route("/api/train-neural-ecg", methods=["POST"])
def api_train_neural_ecg():
    """Força treino da rede neural MLP PPG-to-ECG com os pares PPG+ECG do banco."""
    weights, n_paired = maybe_train_neural_ecg_model(min_pairs=2)
    if not weights:
        return jsonify({
            "ok": False,
            "n_paired_recordings": n_paired,
            "error": f"São necessárias pelo menos 2 calibrações pareadas (PPG + ECG) para treinar a rede neural. Atual: {n_paired}.",
        }), 400
    return jsonify({
        "ok": True,
        "n_paired_recordings": n_paired,
        "n_beats_learned": weights.get("n_beats_learned"),
        "mae_rr_ms": weights.get("mae_rr_ms"),
        "architecture": weights.get("architecture"),
        "message": f"Rede neural PPG→ECG treinada com {n_paired} gravações pareadas ({weights.get('n_beats_learned')} batimentos).",
    })


@app.route("/api/train-global", methods=["POST"])
def api_train_global():
    """Força retreino do modelo polinomial populacional."""
    data = request.get_json(silent=True) or {}
    degree = int(data.get("degree", 2))
    degree = max(1, min(3, degree))
    remove_outliers = data.get("remove_outliers", True)

    coeffs, n, extra, outlier_info = train_global_polynomial(
        degree=degree, remove_outliers=bool(remove_outliers)
    )
    if not coeffs:
        return jsonify({
            "ok": False,
            "error": f"Dados insuficientes (mínimo {MIN_GLOBAL} calibrações com qualidade ≥ 0,3). Atual: {n}.",
            "n_samples": n,
            "outliers": outlier_info,
        }), 400

    return jsonify({
        "ok": True,
        "n_samples": n,
        "degree": degree,
        "r2_systolic": coeffs.get("r2_systolic"),
        "r2_diastolic": coeffs.get("r2_diastolic"),
        "residual_std_sbp": coeffs.get("residual_std_sbp"),
        "residual_std_dbp": coeffs.get("residual_std_dbp"),
        "extra": extra,
        "outliers": outlier_info,
        "message": f"Modelo polinomial global (grau {degree}) treinado com {n} amostras (outliers filtrados).",
    })


@app.route("/api/validate-global")
def api_validate_global():
    """Validação cruzada da precisão do modelo global."""
    degree = int(request.args.get("degree", 2))
    cals = (
        Calibration.query.filter(Calibration.signal_quality >= 0.3)
        .order_by(Calibration.created_at.desc())
        .limit(500)
        .all()
    )
    report = validate_global_model(cals, degree=degree, remove_outliers=True)
    return jsonify(report)


@app.route("/api/outliers")
def api_outliers():
    """Outliers de calibração (features) + resíduos do modelo ativo."""
    cals = (
        Calibration.query.filter(Calibration.signal_quality >= 0.3)
        .order_by(Calibration.created_at.desc())
        .limit(500)
        .all()
    )
    feature_outliers = detect_calibration_outliers(cals)
    coeffs = get_active_coeffs()
    residual = residual_outliers_with_model(cals, coeffs)
    return jsonify({
        "ok": True,
        "feature_outliers": feature_outliers,
        "residual_outliers": residual,
    })


@app.route("/api/bias-report")
def api_bias_report():
    """Relatório de viés demográfico do modelo global ativo (ou retreina análise)."""
    m = (
        CalibrationModel.query.filter_by(is_active=True, scope="global")
        .order_by(CalibrationModel.created_at.desc())
        .first()
    )
    if not m:
        m = (
            CalibrationModel.query.filter(
                CalibrationModel.user_id.is_(None),
                CalibrationModel.is_active.is_(True),
            )
            .order_by(CalibrationModel.created_at.desc())
            .first()
        )

    coeffs = _load_coeffs(m)
    cals = (
        Calibration.query.filter(Calibration.signal_quality >= 0.3)
        .order_by(Calibration.created_at.desc())
        .limit(500)
        .all()
    )

    if m and m.bias_report_json:
        try:
            cached = json.loads(m.bias_report_json)
            bias_part = cached.get("bias") if isinstance(cached, dict) and "bias" in cached else cached
            n_used = (bias_part or {}).get("total_used", 0)
            if n_used >= len(cals) * 0.9 and bias_part:
                return jsonify({
                    "ok": True,
                    "cached": True,
                    "report": bias_part,
                    "validation": cached.get("validation") if isinstance(cached, dict) else None,
                    "outliers": cached.get("outliers") if isinstance(cached, dict) else None,
                    "model_id": m.id,
                })
        except Exception:
            pass

    report = analyze_demographic_bias(cals, coeffs)
    validation = validate_global_model(cals, degree=m.degree if m and m.degree else 2)
    if m:
        m.bias_report_json = json.dumps({
            "bias": report,
            "validation": validation,
        }, ensure_ascii=False)
        db.session.commit()

    return jsonify({
        "ok": True,
        "cached": False,
        "report": report,
        "validation": validation,
        "model_id": m.id if m else None,
        "model_type": m.model_type if m else None,
        "n_samples_model": m.n_samples if m else None,
    })


# ──────────────────────────────────────────────
# Painel do Administrador & Convites de Uso Único
# ──────────────────────────────────────────────

def _get_admin_secret_fingerprint():
    env_pw = (os.getenv("ADMIN_PASSWORD") or "").strip()
    if env_pw:
        return "env", hashlib.sha256(env_pw.encode("utf-8")).hexdigest()[:20]
    row = AppSetting.query.get("admin_password_hash")
    if row and row.value:
        return "db", hashlib.sha256(row.value.encode("utf-8")).hexdigest()[:20]
    return None, None


def _is_admin_authenticated() -> bool:
    if not session.get("is_cardiasync_admin"):
        return False
    _, fp = _get_admin_secret_fingerprint()
    if not fp:
        return False
    return session.get("admin_pw_fp") == fp


def _generate_single_use_invite_code() -> str:
    for _ in range(30):
        digits = f"{secrets.randbelow(9000) + 1000}"
        candidate = f"CS-{digits}"
        if not InviteCode.query.filter(db.func.upper(InviteCode.code) == candidate).first():
            return candidate
    return f"CS-{secrets.token_hex(3).upper()}"


@app.route("/admin")
def admin_page():
    mode, _ = _get_admin_secret_fingerprint()
    needs_setup = mode is None
    is_admin = _is_admin_authenticated()
    invites = []
    users = []
    if is_admin:
        invites = InviteCode.query.order_by(InviteCode.created_at.desc()).limit(100).all()
        users = User.query.order_by(User.created_at.desc()).limit(100).all()
    return render_template(
        "admin.html",
        is_admin=is_admin,
        needs_setup=needs_setup,
        invites=invites,
        users=users,
    )


@app.route("/api/admin/setup", methods=["POST"])
def api_admin_setup():
    mode, _ = _get_admin_secret_fingerprint()
    if mode is not None:
        return jsonify({"ok": False, "error": "A senha mestra já está configurada."}), 400
    data = request.get_json(silent=True) or {}
    password = data.get("password") or ""
    if len(password) < 6:
        return jsonify({"ok": False, "error": "A senha mestra deve ter pelo menos 6 caracteres."}), 400

    pw_hash = generate_password_hash(password)
    row = AppSetting(key="admin_password_hash", value=pw_hash)
    db.session.add(row)
    db.session.commit()

    _, fp = _get_admin_secret_fingerprint()
    session["is_cardiasync_admin"] = True
    session["admin_pw_fp"] = fp
    session.permanent = True
    return jsonify({"ok": True})


@app.route("/api/admin/login", methods=["POST"])
def api_admin_login():
    data = request.get_json(silent=True) or {}
    password = data.get("password") or ""
    env_pw = (os.getenv("ADMIN_PASSWORD") or "").strip()

    valid = False
    if env_pw:
        valid = secrets.compare_digest(password, env_pw)
    else:
        row = AppSetting.query.get("admin_password_hash")
        if row and row.value:
            valid = check_password_hash(row.value, password)

    if not valid:
        return jsonify({"ok": False, "error": "Senha mestra do administrador incorreta."}), 401

    _, fp = _get_admin_secret_fingerprint()
    session["is_cardiasync_admin"] = True
    session["admin_pw_fp"] = fp
    session.permanent = True
    return jsonify({"ok": True})


@app.route("/api/admin/logout", methods=["POST"])
def api_admin_logout():
    session.pop("is_cardiasync_admin", None)
    session.pop("admin_pw_fp", None)
    return jsonify({"ok": True})


@app.route("/api/admin/invites", methods=["POST"])
def api_admin_create_invite():
    if not _is_admin_authenticated():
        return jsonify({"ok": False, "error": "Sessão do administrador expirada. Entre novamente com a Senha Mestra."}), 401

    data = request.get_json(silent=True) or {}
    recipient = (data.get("recipient") or "").strip() or None
    recipient_phone = (data.get("recipient_phone") or "").strip() or None
    raw_custom = (data.get("custom_code") or "").strip().upper()
    expires_hours = int(data.get("expires_hours") or 0)

    if raw_custom:
        cleaned = "".join(ch for ch in raw_custom if ch.isalnum() or ch in ("-", "_", "$", "#"))[:28]
        if len(cleaned) < 3:
            return jsonify({"ok": False, "error": "O código personalizado deve ter ao menos 3 caracteres."}), 400
        if InviteCode.query.filter(db.func.upper(InviteCode.code) == cleaned).first():
            return jsonify({"ok": False, "error": f"O código '{cleaned}' já existe. Escolha outro ou deixe em branco."}), 409
        code = cleaned
    else:
        code = _generate_single_use_invite_code()

    expires_at = None
    if expires_hours > 0:
        expires_at = datetime.now(timezone.utc) + timedelta(hours=expires_hours)

    inv = InviteCode(
        code=code,
        recipient_label=recipient,
        recipient_phone=recipient_phone,
        status="active",
        expires_at=expires_at,
    )
    db.session.add(inv)
    db.session.commit()

    return jsonify({
        "ok": True,
        "invite": {
            "id": inv.id,
            "code": inv.code,
            "recipient_label": inv.recipient_label,
            "recipient_phone": inv.recipient_phone,
            "status": inv.status,
        },
    })


@app.route("/api/admin/link-this-device", methods=["POST"])
def api_admin_link_this_device():
    """Permite ao próprio administrador vincular o celular atual sem precisar digitar código."""
    if not _is_admin_authenticated():
        return jsonify({"ok": False, "error": "Acesso restrito ao administrador."}), 401

    data = request.get_json(silent=True) or {}
    name = (data.get("name") or "Administrador").strip()
    phone = (data.get("phone") or "").strip() or None
    password = (data.get("password") or "").strip()

    user = User.query.filter(db.func.lower(User.name) == name.lower()).first()
    if not user:
        user = User(name=name, phone=phone)
        db.session.add(user)
    elif phone:
        user.phone = phone

    if password and len(password) >= 4:
        user.password_hash = generate_password_hash(password)
    elif not user.password_hash:
        user.password_hash = generate_password_hash(secrets.token_urlsafe(16))

    if not user.recovery_code_hash:
        user.recovery_code_hash = generate_password_hash(_generate_rescue_code())
    user.is_blocked = False
    db.session.commit()

    raw_token = _issue_device_session(user, request.headers.get("User-Agent", ""))
    resp = make_response(jsonify({
        "ok": True,
        "device_token": raw_token,
        "user": {"id": user.id, "name": user.name},
    }))
    return _attach_device_cookie(resp, raw_token)


@app.route("/api/admin/invites/<int:iid>/revoke", methods=["POST"])
def api_admin_revoke_invite(iid):
    if not _is_admin_authenticated():
        return jsonify({"ok": False, "error": "Acesso restrito ao administrador."}), 401
    inv = InviteCode.query.get_or_404(iid)
    if inv.status == "used":
        return jsonify({"ok": False, "error": "Este convite já foi utilizado e não pode ser cancelado."}), 400
    inv.status = "revoked"
    db.session.commit()
    return jsonify({"ok": True})


@app.route("/api/admin/users/<int:uid>/<action>", methods=["POST"])
def api_admin_toggle_user(uid, action):
    if not _is_admin_authenticated():
        return jsonify({"ok": False, "error": "Acesso restrito ao administrador."}), 401
    user = User.query.get_or_404(uid)
    if action == "block":
        user.is_blocked = True
        for s in user.device_sessions:
            s.revoked = True
    elif action == "unblock":
        user.is_blocked = False
    else:
        return jsonify({"ok": False, "error": "Ação inválida."}), 400
    db.session.commit()
    return jsonify({"ok": True})


# ──────────────────────────────────────────────
# Autenticação Persistente por Aparelho & Recuperação de Senha
# ──────────────────────────────────────────────

@app.route("/acesso")
def acesso_page():
    if getattr(g, "current_user", None):
        return redirect(url_for("index"))
    return render_template("acesso.html")


@app.route("/seguranca")
def seguranca_page():
    return render_template("seguranca.html")


@app.route("/api/auth/register", methods=["POST"])
def api_auth_register():
    data = request.get_json(silent=True) or {}
    invite_code = (data.get("invite_code") or "").strip().upper()
    name = (data.get("name") or "").strip()
    phone = (data.get("phone") or "").strip() or None
    email = (data.get("email") or "").strip().lower() or None
    password = data.get("password") or ""

    if not invite_code:
        return jsonify({
            "ok": False,
            "error": "Código de convite obrigatório. Solicite um convite de uso único ao administrador.",
        }), 403

    invite = InviteCode.query.filter(db.func.upper(InviteCode.code) == invite_code).first()
    if not invite:
        return jsonify({
            "ok": False,
            "error": "Código de convite inválido. Verifique com o administrador do CardiaSync.",
        }), 403
    if invite.status == "used":
        return jsonify({
            "ok": False,
            "error": "Este código de convite já foi utilizado em outro aparelho. Cada convite é de uso único.",
        }), 403
    if invite.status == "revoked":
        return jsonify({
            "ok": False,
            "error": "Este código de convite foi cancelado pelo administrador.",
        }), 403
    if invite.expires_at:
        exp = invite.expires_at
        if exp.tzinfo is None:
            exp = exp.replace(tzinfo=timezone.utc)
        if datetime.now(timezone.utc) > exp:
            return jsonify({
                "ok": False,
                "error": "Este código de convite expirou. Solicite um novo código ao administrador.",
            }), 403

    if len(name) < 2:
        return jsonify({"ok": False, "error": "Informe seu nome (mínimo 2 caracteres)."}), 400
    if len(password) < 4:
        return jsonify({"ok": False, "error": "A senha deve ter pelo menos 4 caracteres."}), 400

    # Verifica se já existe usuário protegido com este e-mail ou nome (sem queimar o convite)
    existing = None
    if email:
        existing = User.query.filter(db.func.lower(User.email) == email).first()
    if not existing:
        existing = User.query.filter(db.func.lower(User.name) == name.lower()).first()

    if existing and existing.password_hash:
        return jsonify({
            "ok": False,
            "error": "Já existe uma senha cadastrada para este nome/e-mail. Use a aba 'Já Tenho Senha' ou 'Recuperar'.",
        }), 409

    rescue_code = _generate_rescue_code()
    effective_phone = phone or getattr(invite, "recipient_phone", None)
    if existing:
        user = existing
        user.name = name
        if email:
            user.email = email
        if effective_phone:
            user.phone = effective_phone
    else:
        user = User(name=name, phone=effective_phone, email=email)
        db.session.add(user)

    user.password_hash = generate_password_hash(password)
    user.recovery_code_hash = generate_password_hash(rescue_code)
    user.is_blocked = False
    db.session.flush()

    # Consome o convite de uso único definitivamente
    invite.status = "used"
    invite.used_at = datetime.now(timezone.utc)
    invite.used_by_user_id = user.id
    invite.used_device_label = (request.headers.get("User-Agent") or "Celular")[:110]
    db.session.commit()

    raw_token = _issue_device_session(user, request.headers.get("User-Agent", ""))
    resp = make_response(jsonify({
        "ok": True,
        "user": {"id": user.id, "name": user.name, "email": user.email},
        "device_token": raw_token,
        "recovery_code": rescue_code,
    }))
    return _attach_device_cookie(resp, raw_token)


@app.route("/api/auth/login", methods=["POST"])
def api_auth_login():
    data = request.get_json(silent=True) or {}
    identifier = (data.get("identifier") or "").strip()
    password = data.get("password") or ""

    user = _find_user_by_identifier(identifier)
    if not user or not user.password_hash or not check_password_hash(user.password_hash, password):
        return jsonify({"ok": False, "error": "Nome/e-mail ou senha incorretos."}), 401
    if getattr(user, "is_blocked", False):
        return jsonify({"ok": False, "error": "Acesso suspenso pelo administrador."}), 403

    raw_token = _issue_device_session(user, request.headers.get("User-Agent", ""))
    resp = make_response(jsonify({
        "ok": True,
        "user": {"id": user.id, "name": user.name, "email": user.email},
        "device_token": raw_token,
    }))
    return _attach_device_cookie(resp, raw_token)


@app.route("/api/auth/restore-device", methods=["POST"])
def api_auth_restore_device():
    data = request.get_json(silent=True) or {}
    raw_token = (data.get("device_token") or "").strip()
    if not raw_token:
        return jsonify({"ok": False, "error": "Token ausente."}), 400

    sess = DeviceSession.query.filter_by(
        token_hash=_hash_token(raw_token),
        revoked=False,
    ).first()
    if not sess or not sess.user or getattr(sess.user, "is_blocked", False):
        return jsonify({"ok": False, "error": "Vínculo inválido ou revogado."}), 401

    sess.last_seen_at = datetime.now(timezone.utc)
    db.session.commit()

    resp = make_response(jsonify({
        "ok": True,
        "user": {"id": sess.user.id, "name": sess.user.name},
    }))
    return _attach_device_cookie(resp, raw_token)


@app.route("/api/auth/request-reset", methods=["POST"])
def api_auth_request_reset():
    data = request.get_json(silent=True) or {}
    email = (data.get("email") or "").strip().lower()
    if not email or "@" not in email:
        return jsonify({"ok": False, "error": "Informe um e-mail válido."}), 400

    user = User.query.filter(db.func.lower(User.email) == email).first()
    if not user or not user.password_hash:
        return jsonify({
            "ok": False,
            "error": "Nenhuma conta com senha encontrada para este e-mail.",
        }), 404

    code = f"{secrets.randbelow(900000) + 100000}"
    user.reset_code_hash = generate_password_hash(code)
    user.reset_code_expires = datetime.now(timezone.utc) + timedelta(minutes=15)
    db.session.commit()

    body = (
        f"Olá, {user.name}!\n\n"
        f"Seu código de recuperação de senha do CardiaSync é: {code}\n"
        f"Este código expira em 15 minutos.\n\n"
        f"Se você não solicitou esta alteração, ignore este e-mail."
    )
    ok, msg = send_email_plain(email, "Código de Recuperação — CardiaSync", body)
    if not ok:
        return jsonify({
            "ok": False,
            "error": f"{msg} Use a Opção 2 (Código de Resgate Offline) abaixo.",
        }), 503

    return jsonify({"ok": True, "message": "Código de 6 dígitos enviado para seu e-mail!"})


@app.route("/api/auth/verify-reset", methods=["POST"])
def api_auth_verify_reset():
    data = request.get_json(silent=True) or {}
    email = (data.get("email") or "").strip().lower()
    code = (data.get("code") or "").strip()
    new_password = data.get("new_password") or ""

    if len(new_password) < 4:
        return jsonify({"ok": False, "error": "A nova senha deve ter pelo menos 4 caracteres."}), 400

    user = User.query.filter(db.func.lower(User.email) == email).first()
    if not user or not user.reset_code_hash or not user.reset_code_expires:
        return jsonify({"ok": False, "error": "Solicite primeiro o envio do código por e-mail."}), 400

    expires = user.reset_code_expires
    if expires.tzinfo is None:
        expires = expires.replace(tzinfo=timezone.utc)
    if datetime.now(timezone.utc) > expires:
        return jsonify({"ok": False, "error": "Código expirado. Solicite um novo código."}), 400

    if not check_password_hash(user.reset_code_hash, code):
        return jsonify({"ok": False, "error": "Código de 6 dígitos incorreto."}), 401

    user.password_hash = generate_password_hash(new_password)
    user.reset_code_hash = None
    user.reset_code_expires = None
    db.session.commit()

    raw_token = _issue_device_session(user, request.headers.get("User-Agent", ""))
    resp = make_response(jsonify({
        "ok": True,
        "device_token": raw_token,
    }))
    return _attach_device_cookie(resp, raw_token)


@app.route("/api/auth/recover-rescue", methods=["POST"])
def api_auth_recover_rescue():
    data = request.get_json(silent=True) or {}
    identifier = (data.get("identifier") or "").strip()
    rescue_code = (data.get("recovery_code") or "").strip().upper()
    new_password = data.get("new_password") or ""

    if len(new_password) < 4:
        return jsonify({"ok": False, "error": "A nova senha deve ter pelo menos 4 caracteres."}), 400

    user = _find_user_by_identifier(identifier)
    if not user or not user.recovery_code_hash:
        return jsonify({"ok": False, "error": "Usuário não encontrado ou sem código de resgate."}), 404

    if not check_password_hash(user.recovery_code_hash, rescue_code):
        return jsonify({"ok": False, "error": "Código de resgate inválido."}), 401

    new_rescue = _generate_rescue_code()
    user.password_hash = generate_password_hash(new_password)
    user.recovery_code_hash = generate_password_hash(new_rescue)
    db.session.commit()

    raw_token = _issue_device_session(user, request.headers.get("User-Agent", ""))
    resp = make_response(jsonify({
        "ok": True,
        "device_token": raw_token,
        "new_recovery_code": new_rescue,
    }))
    return _attach_device_cookie(resp, raw_token)


@app.route("/api/auth/update-security", methods=["POST"])
def api_auth_update_security():
    user = getattr(g, "current_user", None)
    if not user:
        return jsonify({"ok": False, "error": "Não autenticado."}), 401

    data = request.get_json(silent=True) or {}
    current_password = data.get("current_password") or ""
    new_email = (data.get("email") or "").strip().lower() or None
    new_password = data.get("new_password") or ""

    if not user.password_hash or not check_password_hash(user.password_hash, current_password):
        return jsonify({"ok": False, "error": "Senha atual incorreta."}), 401

    if new_email:
        user.email = new_email
    if new_password:
        if len(new_password) < 4:
            return jsonify({"ok": False, "error": "A nova senha deve ter pelo menos 4 caracteres."}), 400
        user.password_hash = generate_password_hash(new_password)

    db.session.commit()
    return jsonify({"ok": True, "message": "Configurações de segurança atualizadas!"})


@app.route("/api/auth/regenerate-rescue", methods=["POST"])
def api_auth_regenerate_rescue():
    user = getattr(g, "current_user", None)
    if not user:
        return jsonify({"ok": False, "error": "Não autenticado."}), 401

    new_rescue = _generate_rescue_code()
    user.recovery_code_hash = generate_password_hash(new_rescue)
    db.session.commit()
    return jsonify({"ok": True, "recovery_code": new_rescue})


@app.route("/api/auth/unlink-device", methods=["POST"])
def api_auth_unlink_device():
    sess = getattr(g, "device_session", None)
    if sess:
        sess.revoked = True
        db.session.commit()
    resp = make_response(jsonify({"ok": True}))
    resp.delete_cookie(DEVICE_COOKIE_NAME, path="/")
    return resp


@app.route("/vies")
def vies_page():
    return render_template("vies.html")


@app.route("/sw.js")
def service_worker():
    """Service worker na raiz — exigido para escopo PWA / instalação."""
    response = app.send_static_file("sw.js")
    response.headers["Content-Type"] = "application/javascript; charset=utf-8"
    response.headers["Service-Worker-Allowed"] = "/"
    response.headers["Cache-Control"] = "no-cache"
    return response


@app.route("/health")
def health():
    return jsonify({"status": "ok", "time": datetime.now(timezone.utc).isoformat()})


if __name__ == "__main__":
    with app.app_context():
        db.create_all()
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 5000)), debug=True)
