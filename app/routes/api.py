from flask import Blueprint, request, jsonify
from app.models.calibration import CalibrationModel
from app.models.measurement import MeasurementModel
from app.services.model_placeholder import placeholder_calibration_model

api_bp = Blueprint('api', __name__, url_prefix='/api')

@api_bp.route('/calibrate', methods=['POST'])
def save_calibration():
    data = request.json
    user_id = data.get('user_id', 'usuario_demo')
    systolic_ref = data.get('systolic_ref')
    diastolic_ref = data.get('diastolic_ref')
    signal_features = data.get('signal_features')

    if not systolic_ref or not diastolic_ref or not signal_features:
        return jsonify({"error": "Dados incompletos"}), 400

    cal_id = CalibrationModel.save(user_id, systolic_ref, diastolic_ref, signal_features)
    return jsonify({"success": True, "calibration_id": cal_id, "message": "Calibração salva com sucesso!"})

@api_bp.route('/measure', methods=['POST'])
def measure_pressure():
    data = request.json
    user_id = data.get('user_id', 'usuario_demo')
    raw_features = data.get('signal_features', {})
    condition = data.get('condition', 'Repouso')

    calibration = CalibrationModel.get_latest(user_id)
    systolic, diastolic, is_calibrated = placeholder_calibration_model(raw_features, calibration)

    MeasurementModel.save(user_id, systolic, diastolic, condition, calibration.get('id') if calibration else None)

    return jsonify({
        "systolic": systolic,
        "diastolic": diastolic,
        "calibrated": is_calibrated,
        "warning": None if is_calibrated else "Aviso: Calibração ausente. Faça uma calibração prévia."
    })