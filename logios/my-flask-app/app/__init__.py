from flask import Flask
from config import config
from wtforms.validators import DataRequired, Length
from flask_wtf.file import FileAllowed, FileRequired
from flask_sqlalchemy import SQLAlchemy
import os

db = SQLAlchemy()


def create_app(config_name=None):
    app = Flask(__name__)
    
    # Use environment variable or default to development
    if config_name is None:
        config_name = os.environ.get('FLASK_CONFIG', 'development')
    
    app.config.from_object(config[config_name])
    
    # Override with Docker-specific settings
    app.config["UPLOAD_FOLDER"] = "/app/uploads"  # Use absolute path for Docker
    app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///site.db"

    # OCR service URL - adjust based on your Docker Compose setup
    app.config["OCR_SERVICE_URL"] = os.environ.get("OCR_SERVICE_URL", "http://ocr:8000")

    config[config_name].init_app(app)
    db.init_app(app)

    from app.routes import app as main_blueprint
    from app.auth import auth as auth_blueprint

    app.register_blueprint(main_blueprint)
    app.register_blueprint(auth_blueprint)

    return app
