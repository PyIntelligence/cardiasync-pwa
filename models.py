from datetime import datetime, timezone
from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()


class User(db.Model):
    """Usuário / paciente (dados demográficos opcionais para análise de viés)."""
    __tablename__ = "users"

    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(120), nullable=False)
    phone = db.Column(db.String(30))
    email = db.Column(db.String(120))

    # Demografia (opcional — LGPD: coletar só com consentimento)
    age = db.Column(db.Integer)                    # anos
    sex = db.Column(db.String(20))                 # Feminino, Masculino, Outro, Prefiro não dizer
    skin_tone = db.Column(db.String(30))           # Clara, Média, Escura (proxy grosseiro)

    created_at = db.Column(db.DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

    measurements = db.relationship("Measurement", backref="user", lazy=True)
    calibrations = db.relationship("Calibration", backref="user", lazy=True)


class Calibration(db.Model):
    """
    Par (referência do esfigmomanômetro + features PPG).
    Usado no treino pessoal e no modelo polinomial global.
    """
    __tablename__ = "calibrations"

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=True)

    ref_systolic = db.Column(db.Float, nullable=False)
    ref_diastolic = db.Column(db.Float, nullable=False)

    heart_rate = db.Column(db.Float)
    signal_quality = db.Column(db.Float)
    mean_red = db.Column(db.Float)
    std_red = db.Column(db.Float)
    peak_amplitude = db.Column(db.Float)
    pulse_width = db.Column(db.Float)
    features_json = db.Column(db.Text)

    condition = db.Column(db.String(30))
    notes = db.Column(db.Text)

    # Snapshot demográfico no momento da calibração (para viés, mesmo se user mudar depois)
    age = db.Column(db.Integer)
    sex = db.Column(db.String(20))
    skin_tone = db.Column(db.String(30))

    created_at = db.Column(db.DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

    def __repr__(self):
        return f"<Calibration {self.id} SBP={self.ref_systolic}/{self.ref_diastolic}>"


class Measurement(db.Model):
    __tablename__ = "measurements"

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=True)

    heart_rate = db.Column(db.Float)
    estimated_systolic = db.Column(db.Float)
    estimated_diastolic = db.Column(db.Float)
    signal_quality = db.Column(db.Float)

    mean_red = db.Column(db.Float)
    std_red = db.Column(db.Float)
    features_json = db.Column(db.Text)

    condition = db.Column(db.String(30))
    feeding = db.Column(db.String(20))
    position = db.Column(db.String(20))
    arm = db.Column(db.String(20))
    pulse_height = db.Column(db.String(20))
    period = db.Column(db.String(20))
    observation = db.Column(db.Text)

    calibration_id = db.Column(db.Integer, db.ForeignKey("calibrations.id"), nullable=True)

    # Intervalo de confiança da previsão (se modelo global polinomial)
    sbp_std = db.Column(db.Float)
    dbp_std = db.Column(db.Float)

    created_at = db.Column(db.DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

    def __repr__(self):
        return f"<Measurement {self.id} HR={self.heart_rate}>"


class CalibrationModel(db.Model):
    """
    Modelo treinado (pessoal ou global).
    user_id NULL = modelo populacional global.
    """
    __tablename__ = "calibration_models"

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=True)

    model_type = db.Column(db.String(50), default="ridge")  # ridge, polynomial
    scope = db.Column(db.String(20), default="personal")    # personal | global
    degree = db.Column(db.Integer, default=1)               # grau polinomial
    coefficients_json = db.Column(db.Text)
    intercept_systolic = db.Column(db.Float)
    intercept_diastolic = db.Column(db.Float)
    r2_score = db.Column(db.Float)
    residual_std_sbp = db.Column(db.Float)                  # para intervalo de predição
    residual_std_dbp = db.Column(db.Float)
    n_samples = db.Column(db.Integer)
    is_active = db.Column(db.Boolean, default=True)
    bias_report_json = db.Column(db.Text)                   # último relatório de viés (só global)

    created_at = db.Column(db.DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    updated_at = db.Column(db.DateTime(timezone=True), onupdate=lambda: datetime.now(timezone.utc))
