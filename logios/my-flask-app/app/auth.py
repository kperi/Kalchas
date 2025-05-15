from flask import Blueprint, render_template, redirect, url_for, flash, request, session
from werkzeug.security import generate_password_hash, check_password_hash
from app.forms import LoginForm
from app import db

auth = Blueprint("auth", __name__)


@auth.route("/login", methods=["GET", "POST"])
def login():
    form = LoginForm()
    if form.validate_on_submit():
        user = db.get_user_by_username(form.username.data)
        if user and check_password_hash(user.password, form.password.data):
            session["user_id"] = user.id
            flash("Login successful!", "success")
            return redirect(url_for("main.index"))
        else:
            flash("Invalid username or password", "danger")
    return render_template("login.html", form=form)


@auth.route("/logout")
def logout():
    session.pop("user_id", None)
    flash("You have been logged out.", "success")
    return redirect(url_for("auth.login"))


@auth.route("/register", methods=["GET", "POST"])
def register():
    form = RegistrationForm()
    if form.validate_on_submit():
        hashed_password = generate_password_hash(form.password.data, method="sha256")
        db.add_user(form.username.data, hashed_password)
        flash("Registration successful! You can now log in.", "success")
        return redirect(url_for("auth.login"))
    return render_template("register.html", form=form)
