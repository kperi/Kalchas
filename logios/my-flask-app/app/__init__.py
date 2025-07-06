from flask import Flask
from flask_wtf.csrf import CSRFProtect
from config import config
from wtforms.validators import DataRequired, Length
from flask_wtf.file import FileAllowed, FileRequired
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager
import os

db = SQLAlchemy()
login_manager = LoginManager()
csrf = CSRFProtect()


def create_app(config_name=None):
    app = Flask(__name__)

    # Use environment variable or default to development
    if config_name is None:
        config_name = os.environ.get("FLASK_CONFIG", "development")

    app.config.from_object(config[config_name])

    # Override with Docker-specific settings
    app.config["UPLOAD_FOLDER"] = "/app/uploads"  # Use absolute path for Docker
    app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///site.db"

    # OCR service URL - adjust based on your Docker Compose setup
    app.config["OCR_SERVICE_URL"] = os.environ.get("OCR_SERVICE_URL", "http://ocr:8000")

    config[config_name].init_app(app)
    db.init_app(app)

    # Initialize CSRF protection
    csrf.init_app(app)

    # Initialize Flask-Login
    login_manager.init_app(app)
    login_manager.login_view = "auth.login"
    login_manager.login_message = "Please log in to access this page."
    login_manager.login_message_category = "info"

    # User loader callback
    @login_manager.user_loader
    def load_user(user_id):
        from app.models import User

        return User.query.get(int(user_id))

    # Add security headers to all responses
    @app.after_request
    def add_security_headers(response):
        # Prevent XSS attacks
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"

        # Content Security Policy
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; "
            "script-src 'self' 'unsafe-inline' 'unsafe-eval' https://cdnjs.cloudflare.com; "
            "style-src 'self' 'unsafe-inline' https://cdnjs.cloudflare.com https://fonts.googleapis.com; "
            "img-src 'self' data:; "
            "font-src 'self' https://cdnjs.cloudflare.com https://fonts.gstatic.com; "
            "connect-src 'self'; "
            "frame-ancestors 'none';"
        )

        # Prevent information disclosure
        response.headers["Server"] = "Logios"

        # Force HTTPS in production
        if app.config.get("FLASK_ENV") == "production":
            response.headers["Strict-Transport-Security"] = (
                "max-age=31536000; includeSubDomains"
            )

        return response

    from app.routes import app as main_blueprint
    from app.auth import auth as auth_blueprint

    app.register_blueprint(main_blueprint)
    app.register_blueprint(auth_blueprint)

    # Initialize database tables
    with app.app_context():
        db.create_all()
        # Create default admin user
        from app.models import create_admin_user

        create_admin_user()

    return app
