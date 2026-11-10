import os
from flask import render_template
from weasyprint import HTML

def generate_patient_pdf(user_id, calibrations, measurements):
    rendered_html = render_template(
        'pdf_template.html',
        user_id=user_id,
        calibrations=calibrations,
        measurements=measurements
    )
    
    output_filename = f"relatorio_{user_id}.pdf"
    output_path = os.path.join('/tmp', output_filename)
    
    HTML(string=rendered_html).write_pdf(output_path)
    return output_path