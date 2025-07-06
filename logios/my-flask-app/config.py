import os
import secrets


class Config:
    # Generate a cryptographically secure secret key if not provided
    SECRET_KEY = os.environ.get("SECRET_KEY") or secrets.token_hex(32)
    MAX_CONTENT_LENGTH = 100 * 1024 * 1024  # 100 MB limit for uploads
    UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), "uploads")
    ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "gif", "pdf"}

    @staticmethod
    def init_app(app):
        pass


class DevelopmentConfig(Config):
    DEBUG = True
    FLASK_ENV = "development"

    @staticmethod
    def init_app(app):
        Config.init_app(app)


class ProductionConfig(Config):
    DEBUG = False
    FLASK_ENV = "production"


config = {
    "development": DevelopmentConfig,
    "production": ProductionConfig,
    "default": DevelopmentConfig,
}
