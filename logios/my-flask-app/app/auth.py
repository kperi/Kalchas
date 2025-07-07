from flask import Blueprint, render_template, redirect, url_for, flash, request, session
from flask_login import login_user, logout_user, login_required, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from app.forms import LoginForm, RegistrationForm
from app.models import User
from app import db
from app.rate_limiting import rate_limit_auth
from datetime import datetime

auth = Blueprint("auth", __name__)


@auth.route("/login", methods=["GET", "POST"])
@rate_limit_auth(max_requests=5, window_seconds=900)  # 5 attempts per 15 minutes
def login():
    # Redirect if user is already logged in
    if current_user.is_authenticated:
        return redirect(url_for("app.index"))

    form = LoginForm()
    if form.validate_on_submit():
        user = User.query.filter_by(username=form.username.data).first()

        if user and check_password_hash(user.password_hash, form.password.data):
            if user.is_active:
                # Update last login time
                user.last_login = datetime.utcnow()
                db.session.commit()

                # Log the user in
                login_user(user, remember=True)

                # Set user_id in session for backward compatibility
                session["user_id"] = (
                    user.username
                )  # Use username as user_id for file system compatibility

                flash(f"Welcome back, {user.display_name}!", "success")

                # Redirect to next page if available
                next_page = request.args.get("next")
                if next_page:
                    return redirect(next_page)
                return redirect(url_for("app.index"))
            else:
                flash(
                    "Your account has been deactivated. Please contact an administrator.",
                    "danger",
                )
        else:
            flash("Invalid username or password", "danger")

    return render_template("login.html", form=form)


@auth.route("/register", methods=["GET", "POST"])
def register():
    # Registration is disabled
    flash(
        "Registration is currently disabled. Please contact an administrator for account creation.",
        "warning",
    )
    return redirect(url_for("auth.login"))


@auth.route("/logout")
@login_required
def logout():
    user_name = current_user.display_name
    logout_user()
    session.pop("user_id", None)
    flash(f"Goodbye, {user_name}! You have been logged out.", "success")
    return redirect(url_for("auth.login"))


@auth.route("/profile")
@login_required
def profile():
    """User profile page"""
    return render_template("profile.html", user=current_user)
