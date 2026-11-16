from datetime import datetime, timezone
from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()


class User(db.Model):
    """Usuário simples (nome do dono do celular / paciente)."""
    __tablename__ = "users"

    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(120), nullable=False)
    phone = db.Column(db.String(30))
    email = db.Column(db.String(120))
    created_at = db.Column(db.DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

    measurements = db.relationship("Measurement", backref="user", lazy=True)
    calibrations = db.relationship("Calibration", backref="user", lazy=True)


class Calibration(db.Model):
    """
    Tabela de calibrações.
    Guarda o valor de referência do esfigmomanômetro + features do sinal PPG
    para treinar / ajustar o modelo de estimativa.
    """
    __tablename__ = "calibrations"

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=True)

    # Valores de referência do aparelho de pressão
    ref_systolic = db.Column(db.Float, nullable=False)
    ref_diastolic = db.Column(db.Float, nullable=False)

    # Features extraídas do sinal PPG no momento da calibração
    heart_rate = db.Column(db.Float)               # BPM médio
    signal_quality = db.Column(db.Float)           # 0-1
    mean_red = db.Column(db.Float)                 # intensidade média canal vermelho
    std_red = db.Column(db.Float)                  # desvio padrão
    peak_amplitude = db.Column(db.Float)           # amplitude média dos picos
    pulse_width = db.Column(db.Float)              # largura média do pulso (s)
    features_json = db.Column(db.Text)             # JSON com features extras (futuro)

    # Contexto da calibração
    condition = db.Column(db.String(30))           # Repouso, Acordado, etc.
    notes = db.Column(db.Text)

    created_at = db.Column(db.DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

    def __repr__(self):
        return f"<Calibration {self.id} SBP={self.ref_systolic}/{self.ref_diastolic}>"


class Measurement(db.Model):
    """Medições realizadas pela câmera (após ou sem calibração)."""
    __tablename__ = "measurements"

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=True)

    # Resultados
    heart_rate = db.Column(db.Float)
    estimated_systolic = db.Column(db.Float)
    estimated_diastolic = db.Column(db.Float)
    signal_quality = db.Column(db.Float)

    # Features brutas (para reprocessamento futuro)
    mean_red = db.Column(db.Float)
    std_red = db.Column(db.Float)
    features_json = db.Column(db.Text)

    # Metadados clínicos / contexto
    condition = db.Column(db.String(30))           # Repouso, Acordado
    feeding = db.Column(db.String(20))             # Jejum, Normal
    position = db.Column(db.String(20))            # Sentado, Deitado
    arm = db.Column(db.String(20))                 # Direito, Esquerdo
    pulse_height = db.Column(db.String(20))        # Coxa, Peito, etc.
    period = db.Column(db.String(20))              # Repouso, Acordado
    observation = db.Column(db.Text)

    # Calibração usada (se houver)
    calibration_id = db.Column(db.Integer, db.ForeignKey("calibrations.id"), nullable=True)

    created_at = db.Column(db.DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

    def __repr__(self):
        return f"<Measurement {self.id} HR={self.heart_rate}>"


class CalibrationModel(db.Model):
    """
    Placeholder para o modelo de calibração treinado.
    Quando houver dados suficientes, um job pode treinar uma regressão
    e salvar os coeficientes aqui.
    """
    __tablename__ = "calibration_models"

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=True)  # null = modelo global

    model_type = db.Column(db.String(50), default="linear")  # linear, ridge, etc.
    coefficients_json = db.Column(db.Text)                   # coeficientes serializados
    intercept_systolic = db.Column(db.Float)
    intercept_diastolic = db.Column(db.Float)
    r2_score = db.Column(db.Float)
    n_samples = db.Column(db.Integer)
    is_active = db.Column(db.Boolean, default=True)

    created_at = db.Column(db.DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    updated_at = db.Column(db.DateTime(timezone=True), onupdate=lambda: datetime.now(timezone.utc))