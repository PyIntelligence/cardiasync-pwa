import os
import json
from datetime import datetime, timezone
from flask import (
    Flask, render_template, request, jsonify, send_file, make_response
)
from dotenv import load_dotenv
from io import BytesIO

from models import db, User, Calibration, Measurement, CalibrationModel
from calibration_engine import (
    extract_basic_features,
    estimate_blood_pressure,
    train_simple_linear_model,
)
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


@app.before_request
def create_tables():
    if not hasattr(app, "_tables_created"):
        with app.app_context():
            db.create_all()
        app._tables_created = True


def get_or_create_user(name: str, email: str = None, phone: str = None):
    user = User.query.filter_by(name=name).first()
    if not user:
        user = User(name=name, email=email, phone=phone)
        db.session.add(user)
        db.session.flush()
    else:
        if email and not user.email:
            user.email = email
        if phone and not user.phone:
            user.phone = phone
    return user


def get_active_coeffs(user_id=None):
    """Busca modelo ativo (prioriza do usuário, senão global)."""
    q = CalibrationModel.query.filter_by(is_active=True)
    if user_id:
        m = q.filter_by(user_id=user_id).order_by(CalibrationModel.created_at.desc()).first()
        if m and m.coefficients_json:
            try:
                return json.loads(m.coefficients_json)
            except Exception:
                pass
    m = q.filter(CalibrationModel.user_id.is_(None)).order_by(
        CalibrationModel.created_at.desc()
    ).first()
    if m and m.coefficients_json:
        try:
            return json.loads(m.coefficients_json)
        except Exception:
            pass
    # fallback: qualquer ativo
    m = q.order_by(CalibrationModel.created_at.desc()).first()
    if m and m.coefficients_json:
        try:
            return json.loads(m.coefficients_json)
        except Exception:
            pass
    return None


def maybe_train_model(user_id: int):
    """
    Se o usuário tiver ≥ 5 calibrações com qualidade razoável,
    treina regressão linear e salva em CalibrationModel.
    """
    cals = (
        Calibration.query.filter_by(user_id=user_id)
        .filter(Calibration.signal_quality >= 0.3)
        .order_by(Calibration.created_at.desc())
        .limit(30)
        .all()
    )
    if len(cals) < 5:
        return None, len(cals)

    coeffs = train_simple_linear_model(cals)
    if not coeffs:
        return None, len(cals)

    # Desativa modelos anteriores do usuário
    CalibrationModel.query.filter_by(user_id=user_id, is_active=True).update(
        {"is_active": False}
    )

    model = CalibrationModel(
        user_id=user_id,
        model_type=coeffs.get("model_type", "ridge"),
        coefficients_json=json.dumps(coeffs),
        intercept_systolic=coeffs["systolic"]["intercept"],
        intercept_diastolic=coeffs["diastolic"]["intercept"],
        r2_score=(coeffs.get("r2_systolic", 0) + coeffs.get("r2_diastolic", 0)) / 2,
        n_samples=coeffs.get("n_samples", len(cals)),
        is_active=True,
    )
    db.session.add(model)
    db.session.commit()
    return coeffs, len(cals)


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
    measurements = (
        Measurement.query.order_by(Measurement.created_at.desc()).limit(100).all()
    )
    calibrations = (
        Calibration.query.order_by(Calibration.created_at.desc()).limit(50).all()
    )
    return render_template(
        "historico.html", measurements=measurements, calibrations=calibrations
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

    if features.get("signal_quality", 0) < 0.25:
        return jsonify({
            "ok": False,
            "error": "Qualidade do sinal muito baixa. Reposicione o dedo cobrindo bem a câmera e o flash, fique parado e tente de novo.",
        }), 400

    user = get_or_create_user(user_name, email=email, phone=phone)
    coeffs = get_active_coeffs(user.id)
    sbp, dbp, status = estimate_blood_pressure(features, coeffs)

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
        "signal_quality": features.get("signal_quality"),
        "status": status,
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

    if not red_signal or len(red_signal) < 30:
        return jsonify({"ok": False, "error": "Sinal da câmera insuficiente."}), 400

    features = extract_basic_features(red_signal, fps=fps)

    if features.get("signal_quality", 0) < 0.35:
        return jsonify({
            "ok": False,
            "error": "Qualidade do sinal baixa. Reposicione o dedo e tente novamente.",
        }), 400

    user = get_or_create_user(user_name)

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
    )
    db.session.add(calibration)
    db.session.commit()

    # Treina modelo se ≥ 5 calibrações
    coeffs, total = maybe_train_model(user.id)
    trained = coeffs is not None

    msg = f"Calibração salva. Total: {total}."
    if trained:
        msg += (
            f" Modelo treinado com sucesso "
            f"(R² sistólica ≈ {coeffs.get('r2_systolic', 0):.2f}, "
            f"diastólica ≈ {coeffs.get('r2_diastolic', 0):.2f})."
        )
    elif total < 5:
        msg += f" Faltam {5 - total} calibração(ões) de boa qualidade para treinar o modelo."
    else:
        msg += " Ainda não foi possível treinar o modelo (verifique qualidade dos sinais)."

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


@app.route("/health")
def health():
    return jsonify({"status": "ok", "time": datetime.now(timezone.utc).isoformat()})


if __name__ == "__main__":
    with app.app_context():
        db.create_all()
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 5000)), debug=True)
