import os

class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'a_default_secret_key'
    MAX_CONTENT_LENGTH = 100 * 1024 * 1024  # 100 MB limit for uploads
    UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), 'uploads')
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'pdf'}

    @staticmethod
    def init_app(app):
        pass