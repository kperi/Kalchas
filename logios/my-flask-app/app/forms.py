from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField, FileField, PasswordField
from wtforms.validators import DataRequired, Length, Email, EqualTo, ValidationError
from flask_wtf.file import FileAllowed, FileRequired
from app.models import User


class LoginForm(FlaskForm):
    username = StringField(
        "Username", validators=[DataRequired(), Length(min=3, max=25)]
    )
    password = PasswordField(
        "Password", validators=[DataRequired(), Length(min=6, max=35)]
    )
    submit = SubmitField("Login")


class RegistrationForm(FlaskForm):
    username = StringField(
        "Username", validators=[DataRequired(), Length(min=3, max=25)]
    )
    email = StringField(
        "Email", validators=[DataRequired(), Email(), Length(max=120)]
    )
    first_name = StringField(
        "First Name", validators=[Length(max=50)]
    )
    last_name = StringField(
        "Last Name", validators=[Length(max=50)]
    )
    password = PasswordField(
        "Password", validators=[DataRequired(), Length(min=6, max=35)]
    )
    password_confirm = PasswordField(
        "Confirm Password", 
        validators=[DataRequired(), EqualTo('password', message='Passwords must match')]
    )
    submit = SubmitField("Register")
    
    def validate_username(self, username):
        """Check if username already exists"""
        user = User.query.filter_by(username=username.data).first()
        if user:
            raise ValidationError('Username already exists. Please choose a different one.')
    
    def validate_email(self, email):
        """Check if email already exists"""
        user = User.query.filter_by(email=email.data).first()
        if user:
            raise ValidationError('Email already registered. Please choose a different one.')


class UploadForm(FlaskForm):
    file = FileField(
        "Upload Image or PDF",
        validators=[
            FileRequired(),
            FileAllowed(["jpg", "jpeg", "png", "pdf"], "Images and PDFs only!"),
        ],
    )
    submit = SubmitField("Upload")
