from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField, FileField, PasswordField
from wtforms.validators import DataRequired, Length
from flask_wtf.file import FileAllowed, FileRequired


class LoginForm(FlaskForm):
    username = StringField(
        "Username", validators=[DataRequired(), Length(min=4, max=25)]
    )
    password = PasswordField(
        "Password", validators=[DataRequired(), Length(min=6, max=35)]
    )
    submit = SubmitField("Login")


class UploadForm(FlaskForm):
    file = FileField(
        "Upload Image or PDF",
        validators=[
            FileRequired(),
            FileAllowed(["jpg", "jpeg", "png", "pdf"], "Images and PDFs only!"),
        ],
    )
    submit = SubmitField("Upload")
