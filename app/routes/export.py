from flask import Blueprint, send_file, request
from app.models.calibration import CalibrationModel
from app.models.measurement import MeasurementModel
from app.services.pdf_generator import generate_patient_pdf

export_bp = Blueprint('export', __name__, url_prefix='/export')

@export_bp.route('/pdf', methods=['GET'])
def download_pdf():
    user_id = request.args.get('user_id', 'usuario_demo')
    
    calibrations = CalibrationModel.get_by_user(user_id)
    measurements = MeasurementModel.get_by_user(user_id)
    
    pdf_file_path = generate_patient_pdf(user_id, calibrations, measurements)
    
    return send_file(
        pdf_file_path,
        mimetype='application/pdf',
        as_attachment=True,
        download_name=f"Relatorio_Pressao_PPG_{user_id}.pdf"
    )