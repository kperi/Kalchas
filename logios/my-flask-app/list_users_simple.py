#!/usr/bin/env python3
"""
Simple script to list all users in the database (basic info only).
This script is designed to be run through Docker Compose exec.

Usage:
docker compose exec flaskui python list_users_simple.py
"""

import sys
from datetime import datetime

# Add the app directory to Python path
sys.path.insert(0, "/app")

# Import Flask app and models
from app import create_app, db
from app.models import User


def format_date(dt):
    """Format datetime for display"""
    if dt is None:
        return "Never"
    return dt.strftime("%Y-%m-%d %H:%M:%S")


def list_users_simple():
    """List all users in the database with basic information"""

    # Create Flask app context
    app = create_app("development")

    with app.app_context():
        try:
            # Query all users
            users = User.query.all()

            if not users:
                print("No users found in the database.")
                return

            print(f"\n{'='*100}")
            print(f"{'DATABASE USERS - SIMPLE LIST':^100}")
            print(f"{'='*100}")
            print(f"Total Users: {len(users)}")
            print(f"Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"{'='*100}\n")

            # Table header
            print(
                f"{'ID':<4} {'Username':<20} {'Email':<35} {'Full Name':<25} {'Admin':<6} {'Active':<7} {'Created':<20}"
            )
            print("-" * 117)

            for user in users:
                # Format display values
                admin_status = "Yes" if user.is_admin else "No"
                active_status = "Yes" if user.is_active else "No"

                # Get full name
                full_name = ""
                if user.first_name and user.last_name:
                    full_name = f"{user.first_name} {user.last_name}"
                elif user.first_name:
                    full_name = user.first_name
                elif user.last_name:
                    full_name = user.last_name
                else:
                    full_name = "-"

                # Print user row
                print(
                    f"{user.id:<4} {user.username:<20} {user.email:<35} {full_name:<25} {admin_status:<6} {active_status:<7} {format_date(user.created_at):<20}"
                )

            # Summary
            print("-" * 117)
            print(f"\nSUMMARY:")
            print(f"Total Users: {len(users)}")
            print(f"Active Users: {sum(1 for u in users if u.is_active)}")
            print(f"Inactive Users: {sum(1 for u in users if not u.is_active)}")
            print(f"Admin Users: {sum(1 for u in users if u.is_admin)}")

            print(f"\n{'='*100}")

        except Exception as e:
            print(f"Error accessing database: {e}")
            sys.exit(1)


if __name__ == "__main__":
    list_users_simple()
