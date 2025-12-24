from flask import Flask
from app.extentions import settings


def create_app():
    """Factory function to create and configure Flask app."""
    app = Flask(__name__)
    
    # Configuration
    app.config['JSON_SORT_KEYS'] = False
    app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max upload
    
    # Register blueprints
    from app.api.id_card_scan.routes import api_v1_bp
    app.register_blueprint(api_v1_bp, url_prefix='/id_card_scan/')
    
    return app
