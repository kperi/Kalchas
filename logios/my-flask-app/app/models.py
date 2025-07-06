from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin
from datetime import datetime
import os
from app import db


class User(UserMixin, db.Model):
    """User model for authentication and session management"""

    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False, index=True)
    email = db.Column(db.String(120), unique=True, nullable=False, index=True)
    password_hash = db.Column(db.String(255), nullable=False)

    # User profile information
    first_name = db.Column(db.String(50), nullable=True)
    last_name = db.Column(db.String(50), nullable=True)

    # Account status and timestamps
    is_active = db.Column(db.Boolean, default=True, nullable=False)
    is_admin = db.Column(db.Boolean, default=False, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)
    updated_at = db.Column(
        db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False
    )
    last_login = db.Column(db.DateTime, nullable=True)

    def __repr__(self):
        return f"<User {self.username}>"

    def get_id(self):
        """Return the user ID as a string (required by Flask-Login)"""
        return str(self.id)

    @property
    def full_name(self):
        """Return the user's full name"""
        if self.first_name and self.last_name:
            return f"{self.first_name} {self.last_name}"
        elif self.first_name:
            return self.first_name
        elif self.last_name:
            return self.last_name
        else:
            return self.username

    @property
    def display_name(self):
        """Return the best available display name"""
        return self.full_name if (self.first_name or self.last_name) else self.username

    def is_authenticated(self):
        """Return True if the user is authenticated"""
        return True

    def is_anonymous(self):
        """Return False as this is not an anonymous user"""
        return False

    def get_user_folder_name(self):
        """Return the folder name used for this user's uploads"""
        # Use the username as the folder name for consistency
        # This maintains compatibility with existing file structure
        return self.username


class UserSession(db.Model):
    """Track user sessions for security and analytics"""

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=False)
    session_token = db.Column(db.String(255), unique=True, nullable=False, index=True)
    ip_address = db.Column(db.String(45), nullable=True)  # IPv6 compatible
    user_agent = db.Column(db.Text, nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)
    expires_at = db.Column(db.DateTime, nullable=False)
    is_active = db.Column(db.Boolean, default=True, nullable=False)

    # Relationship
    user = db.relationship("User", backref=db.backref("sessions", lazy=True))

    def __repr__(self):
        return f"<UserSession {self.user.username}:{self.session_token[:8]}>"


def init_db():
    """Initialize the database with tables"""
    db.create_all()


import secrets


def create_admin_user(username=None, email=None, password=None):
    """Create a default admin user if it doesn't exist"""
    from werkzeug.security import generate_password_hash

    # Use environment variables or secure defaults
    username = username or os.environ.get("ADMIN_USERNAME", "admin")
    email = email or os.environ.get("ADMIN_EMAIL", "admin@logios.phil.uoa.gr")
    password = password or os.environ.get("ADMIN_PASSWORD")

    # If no password is set in environment, generate a secure one and warn
    if not password:
        password = secrets.token_urlsafe(16)
        print(
            f"WARNING: No ADMIN_PASSWORD environment variable set. Generated secure password: {password}"
        )
        print(
            f"Please save this password and set ADMIN_PASSWORD environment variable for future deployments."
        )

    existing_user = User.query.filter_by(username=username).first()
    if not existing_user:
        admin_user = User(
            username=username,
            email=email,
            password_hash=generate_password_hash(password),
            first_name="System",
            last_name="Administrator",
            is_admin=True,
            is_active=True,
        )
        db.session.add(admin_user)
        db.session.commit()
        print(f"Created admin user: {username}")
        return admin_user
    else:
        print(f"Admin user {username} already exists")
        return existing_user


class UploadedFile(db.Model):
    """Model to track uploaded files for each user"""

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=False)

    # Original filename as uploaded by user (no sanitization)
    original_filename = db.Column(db.String(255), nullable=False)

    # Sanitized filename used in filesystem (for compatibility)
    stored_filename = db.Column(db.String(255), nullable=False)

    # File information
    file_type = db.Column(db.String(10), nullable=False)  # 'pdf', 'image'
    file_size = db.Column(db.Integer, nullable=False)  # Size in bytes
    mime_type = db.Column(db.String(100), nullable=True)

    # Processing status
    processing_status = db.Column(db.String(20), default="in_progress", nullable=False)
    # Values: 'in_progress' (after upload), 'under_ocr' (after Move to OCR), 'completed' (after manual completion), 'failed'

    # Document folder name (for organizing related files)
    document_folder = db.Column(db.String(255), nullable=False)

    # Timestamps
    uploaded_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)
    updated_at = db.Column(
        db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False
    )

    # File paths (relative to upload root)
    file_path = db.Column(db.String(500), nullable=False)

    # OCR and processing metadata
    page_count = db.Column(db.Integer, nullable=True)  # For PDFs
    ocr_completed = db.Column(db.Boolean, default=False, nullable=False)

    # Relationships
    user = db.relationship(
        "User",
        backref=db.backref("uploaded_files", lazy=True, cascade="all, delete-orphan"),
    )

    def __repr__(self):
        return f"<UploadedFile {self.original_filename} by {self.user.username}>"

    @property
    def file_size_mb(self):
        """Return file size in MB"""
        return round(self.file_size / (1024 * 1024), 2)

    @property
    def is_pdf(self):
        """Check if file is PDF"""
        return self.file_type.lower() == "pdf"

    @property
    def is_image(self):
        """Check if file is image"""
        return self.file_type.lower() == "image"

    def get_full_path(self, config):
        """Get full filesystem path to the file"""
        upload_root = config.get("UPLOAD_FOLDER", "/app/uploads")
        return os.path.join(upload_root, self.file_path)

    def update_processing_status(self, status):
        """Update processing status with timestamp"""
        self.processing_status = status
        self.updated_at = datetime.utcnow()
        db.session.commit()
