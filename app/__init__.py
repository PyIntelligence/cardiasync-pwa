from flask import Flask
from app.config import Config

def create_app():
    app = Flask(__name__)
    app.config.from_object(Config)

    # Registro de Blueprints
    from app.routes.main import main_bp
    from app.routes.api import api_bp
    from app.routes.export import export_bp

    app.register_blueprint(main_bp)
    app.register_blueprint(api_bp)
    app.register_blueprint(export_bp)

    return app