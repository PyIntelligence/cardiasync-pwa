from flask import Blueprint, render_template

main_bp = Blueprint('main', __name__)

@main_bp.route('/')
def index():
    return render_template('index.html')

@main_bp.route('/calibrate')
def calibrate_page():
    return render_template('calibrate.html')

@main_bp.route('/measure')
def measure_page():
    return render_template('measure.html')