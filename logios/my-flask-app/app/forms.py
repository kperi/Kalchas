from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField, FileField, PasswordField
from wtforms.validators import (
    DataRequired,
    Length,
    Email,
    EqualTo,
    ValidationError,
    Regexp,
)
from flask_wtf.file import FileAllowed, FileRequired
from app.models import User
import re


def validate_strong_password(form, field):
    """Custom validator for strong password requirements"""
    password = field.data
    if len(password) < 8:
        raise ValidationError("Password must be at least 8 characters long.")
    if not re.search(r"[A-Z]", password):
        raise ValidationError("Password must contain at least one uppercase letter.")
    if not re.search(r"[a-z]", password):
        raise ValidationError("Password must contain at least one lowercase letter.")
    if not re.search(r"\d", password):
        raise ValidationError("Password must contain at least one digit.")
    if not re.search(r'[!@#$%^&*()_+\-=\[\]{};\':"\\|,.<>\?]', password):
        raise ValidationError("Password must contain at least one special character.")
    if re.search(r"\s", password):
        raise ValidationError("Password cannot contain spaces.")


class LoginForm(FlaskForm):
    username = StringField(
        "Username",
        validators=[
            DataRequired(),
            Length(min=3, max=25),
            Regexp(
                r"^[a-zA-Z0-9_]+$",
                message="Username can only contain letters, numbers, and underscores.",
            ),
        ],
    )
    password = PasswordField(
        "Password", validators=[DataRequired(), Length(min=6, max=128)]
    )
    submit = SubmitField("Login")


class RegistrationForm(FlaskForm):
    username = StringField(
        "Username",
        validators=[
            DataRequired(),
            Length(min=3, max=25),
            Regexp(
                r"^[a-zA-Z0-9_]+$",
                message="Username can only contain letters, numbers, and underscores.",
            ),
        ],
    )
    email = StringField("Email", validators=[DataRequired(), Email(), Length(max=120)])
    first_name = StringField(
        "First Name",
        validators=[
            Length(max=50),
            Regexp(
                r"^[a-zA-Z\s\-\'\.]*$",
                message="First name can only contain letters, spaces, hyphens, apostrophes, and periods.",
            ),
        ],
    )
    last_name = StringField(
        "Last Name",
        validators=[
            Length(max=50),
            Regexp(
                r"^[a-zA-Z\s\-\'\.]*$",
                message="Last name can only contain letters, spaces, hyphens, apostrophes, and periods.",
            ),
        ],
    )
    password = PasswordField(
        "Password",
        validators=[DataRequired(), Length(min=8, max=128), validate_strong_password],
    )
    password_confirm = PasswordField(
        "Confirm Password",
        validators=[
            DataRequired(),
            EqualTo("password", message="Passwords must match"),
        ],
    )
    submit = SubmitField("Register")

    def validate_username(self, username):
        """Check if username already exists"""
        user = User.query.filter_by(username=username.data).first()
        if user:
            raise ValidationError(
                "Username already exists. Please choose a different one."
            )

    def validate_email(self, email):
        """Check if email already exists"""
        user = User.query.filter_by(email=email.data).first()
        if user:
            raise ValidationError(
                "Email already registered. Please choose a different one."
            )


class UploadForm(FlaskForm):
    file = FileField(
        "Upload Image or PDF",
        validators=[
            FileRequired(),
            FileAllowed(["jpg", "jpeg", "png", "pdf"], "Images and PDFs only!"),
        ],
    )
    submit = SubmitField("Upload")
