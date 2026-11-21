import os
import json
from datetime import datetime, timezone
from flask import (
    Flask, render_template, request, jsonify, send_file, make_response
)
from dotenv import load_dotenv
from io import BytesIO
from sqlalchemy import text

from models import db, User, Calibration, Measurement, CalibrationModel
from calibration_engine import (
    extract_basic_features,
    estimate_blood_pressure,
    train_simple_linear_model,
    train_polynomial_global_model,
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
from hrv import build_variability_report
from pdf_report import build_measurement_pdf
from notify import send_email_with_pdf, send_whatsapp_notification

load_dotenv()

app = Flask(__name__)
app.config["SECRET_KEY"] = os.getenv("SECRET_KEY", "dev-secret-change-me")
app.config["SQLALCHEMY_DATABASE_URI"] = os.getenv(
    "DATABASE_URL", "sqlite:///cardiasync_dev.db"
)
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

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

    # colunas que podem faltar em bancos antigos
    alterations = [
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
                    # SQLite: verifica colunas existentes
                    rows = conn.execute(text(f"PRAGMA table_info({table})")).fetchall()
                    existing = {r[1] for r in rows}
                    if column not in existing:
                        conn.execute(
                            text(f"ALTER TABLE {table} ADD COLUMN {column} {coltype}")
                        )
            except Exception:
                # tabela ainda não existe ou coluna já presente — ignore
                pass


@app.before_request
def create_tables():
    if not getattr(app, "_schema_ready", False):
        with app.app_context():
            try:
                _ensure_schema()
            except Exception as e:
                # não derruba o app inteiro; loga e segue
                app.logger.exception("Schema ensure failed: %s", e)
        app._schema_ready = True


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
    q = CalibrationModel.query.filter_by(is_active=True)
    if user_id:
        m = (
            q.filter_by(user_id=user_id)
            .order_by(CalibrationModel.created_at.desc())
            .first()
        )
        coeffs = _load_coeffs(m)
        if coeffs:
            return coeffs

    m = (
        q.filter(CalibrationModel.user_id.is_(None))
        .order_by(CalibrationModel.created_at.desc())
        .first()
    )
    coeffs = _load_coeffs(m)
    if coeffs:
        return coeffs
    return None


def maybe_train_model(user_id: int):
    """Treina Ridge pessoal se ≥ 5 calibrações boas do usuário."""
    cals = (
        Calibration.query.filter_by(user_id=user_id)
        .filter(Calibration.signal_quality >= 0.3)
        .order_by(Calibration.created_at.desc())
        .limit(40)
        .all()
    )
    if len(cals) < 5:
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
            # se removeu demais, usa conjunto original
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
        measurements = (
            Measurement.query.order_by(Measurement.created_at.desc()).limit(100).all()
        )
        calibrations = (
            Calibration.query.order_by(Calibration.created_at.desc()).limit(50).all()
        )
    except Exception as e:
        # Schema desatualizado / conexão — mostra página vazia em vez de 500
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
# API – Medição
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

    # HRV/PRV a partir dos picos PPG; ECG opcional para validação
    peaks = features.pop("peak_indices", []) or []
    variability = build_variability_report(
        peaks=peaks,
        fps=fps,
        ecg_rr_ms=data.get("ecg_rr_ms"),
        ecg_peaks_sec=data.get("ecg_peaks_sec"),
    )
    features["hrv"] = variability.get("prv")
    features["ecg_validation"] = {
        "ecg_provided": variability.get("ecg_provided"),
        "hrv_ecg": variability.get("hrv_ecg"),
        "comparison": variability.get("comparison"),
    }

    user = get_or_create_user(user_name, email=email, phone=phone)
    coeffs = get_active_coeffs(user.id)
    sbp, dbp, status, sbp_std, dbp_std = estimate_blood_pressure(features, coeffs)

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
        sbp_std=sbp_std,
        dbp_std=dbp_std,
    )
    db.session.add(measurement)
    db.session.commit()

    msg = (
        "Estimativa calibrada com modelo treinado."
        if status == "calibrado"
        else "Frequência cardíaca ok. Pressão ainda sem calibração suficiente (faça ≥ 5 calibrações)."
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
        "signal_quality": features.get("signal_quality"),
        "condition": condition,
        "feeding": feeding,
        "position": position,
        "arm": arm,
        "pulse_height": pulse_height,
        "observation": observation,
        "status": status,
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
                    f"PA estimada: {sbp or 'N/A'}/{dbp or 'N/A'} mmHg\n\n"
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
        "sbp_std": sbp_std,
        "dbp_std": dbp_std,
        "signal_quality": features.get("signal_quality"),
        "variability": variability,
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
    variability = build_variability_report(
        peaks=peaks,
        fps=fps,
        ecg_rr_ms=data.get("ecg_rr_ms"),
        ecg_peaks_sec=data.get("ecg_peaks_sec"),
    )
    features["hrv"] = variability.get("prv")
    features["ecg_validation"] = {
        "ecg_provided": variability.get("ecg_provided"),
        "comparison": variability.get("comparison"),
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
    if trained:
        msg += (
            f" Modelo pessoal atualizado "
            f"(R² sistólica ≈ {coeffs.get('r2_systolic', 0):.2f}, "
            f"diastólica ≈ {coeffs.get('r2_diastolic', 0):.2f})."
        )
    elif total < 5:
        msg += f" Faltam {5 - total} calibração(ões) pessoais para o modelo individual."
    if global_info:
        msg += f" Modelo global polinomial ativo (n={global_info['n_samples']})."
    elif all_n < MIN_GLOBAL:
        msg += f" Modelo global: faltam {MIN_GLOBAL - all_n} calibrações populacionais."

    return jsonify({
        "ok": True,
        "calibration_id": calibration.id,
        "total_calibrations": total,
        "model_trained": trained,
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
# PDF download
# ──────────────────────────────────────────────

@app.route("/api/measurement/<int:mid>/pdf")
def measurement_pdf(mid):
    m = Measurement.query.get_or_404(mid)
    data = {
        "user_name": m.user.name if m.user else "—",
        "created_at": m.created_at.strftime("%d/%m/%Y %H:%M:%S") if m.created_at else None,
        "heart_rate": m.heart_rate,
        "estimated_systolic": m.estimated_systolic,
        "estimated_diastolic": m.estimated_diastolic,
        "signal_quality": m.signal_quality,
        "condition": m.condition,
        "feeding": m.feeding,
        "position": m.position,
        "arm": m.arm,
        "pulse_height": m.pulse_height,
        "observation": m.observation,
        "status": "calibrado" if m.estimated_systolic else "sem_calibracao",
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

    pdf_data = {
        "user_name": m.user.name if m.user else "—",
        "created_at": m.created_at.strftime("%d/%m/%Y %H:%M:%S") if m.created_at else None,
        "heart_rate": m.heart_rate,
        "estimated_systolic": m.estimated_systolic,
        "estimated_diastolic": m.estimated_diastolic,
        "signal_quality": m.signal_quality,
        "condition": m.condition,
        "feeding": m.feeding,
        "position": m.position,
        "arm": m.arm,
        "pulse_height": m.pulse_height,
        "observation": m.observation,
        "status": "calibrado" if m.estimated_systolic else "sem_calibracao",
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


@app.route("/vies")
def vies_page():
    return render_template("vies.html")


@app.route("/health")
def health():
    return jsonify({"status": "ok", "time": datetime.now(timezone.utc).isoformat()})


if __name__ == "__main__":
    with app.app_context():
        db.create_all()
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 5000)), debug=True)
