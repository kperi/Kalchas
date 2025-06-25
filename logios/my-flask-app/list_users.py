#!/usr/bin/env python3
"""
Script to list all users in the database.
This script is designed to be run through Docker Compose exec.

Usage:
docker compose exec flaskui python list_users.py
"""

import sys
import os
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


def format_size(size_bytes):
    """Format file size in human readable format"""
    if size_bytes == 0:
        return "0 B"

    size_names = ["B", "KB", "MB", "GB", "TB"]
    i = 0
    while size_bytes >= 1024 and i < len(size_names) - 1:
        size_bytes /= 1024.0
        i += 1
    return f"{size_bytes:.2f} {size_names[i]}"


def get_user_storage_size(user_id):
    """Calculate total storage used by a user"""
    user_upload_dir = f"/app/uploads/{user_id}"
    total_size = 0

    if os.path.exists(user_upload_dir):
        for dirpath, dirnames, filenames in os.walk(user_upload_dir):
            for filename in filenames:
                filepath = os.path.join(dirpath, filename)
                try:
                    total_size += os.path.getsize(filepath)
                except (OSError, IOError):
                    # Skip files that can't be accessed
                    continue

    return total_size


def count_user_documents(user_id):
    """Count number of document folders for a user"""
    user_upload_dir = f"/app/uploads/{user_id}"
    doc_count = 0

    if os.path.exists(user_upload_dir):
        try:
            # Count subdirectories (each represents a document)
            for item in os.listdir(user_upload_dir):
                item_path = os.path.join(user_upload_dir, item)
                if os.path.isdir(item_path):
                    doc_count += 1
        except (OSError, IOError):
            pass

    return doc_count


def list_users():
    """List all users in the database with their details"""

    # Create Flask app context
    app = create_app("development")

    with app.app_context():
        try:
            # Query all users
            users = User.query.all()

            if not users:
                print("No users found in the database.")
                return

            print(f"\n{'='*80}")
            print(f"{'DATABASE USERS REPORT':^80}")
            print(f"{'='*80}")
            print(f"Total Users: {len(users)}")
            print(f"Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"{'='*80}\n")

            # Table header
            print(
                f"{'ID':<4} {'Username':<20} {'Email':<30} {'Name':<25} {'Admin':<6} {'Active':<7} {'Documents':<10} {'Storage':<12} {'Created':<20} {'Last Login':<20}"
            )
            print("-" * 155)

            total_storage = 0
            total_documents = 0

            for user in users:
                # Get user statistics
                storage_size = get_user_storage_size(user.id)
                doc_count = count_user_documents(user.id)

                total_storage += storage_size
                total_documents += doc_count

                # Format display values
                admin_status = "Yes" if user.is_admin else "No"
                active_status = "Yes" if user.is_active else "No"
                full_name = (
                    user.full_name
                    if hasattr(user, "full_name")
                    else f"{user.first_name or ''} {user.last_name or ''}".strip()
                )
                if not full_name:
                    full_name = "-"

                # Print user row
                print(
                    f"{user.id:<4} {user.username:<20} {user.email:<30} {full_name:<25} {admin_status:<6} {active_status:<7} {doc_count:<10} {format_size(storage_size):<12} {format_date(user.created_at):<20} {format_date(user.last_login):<20}"
                )

            # Summary
            print("-" * 155)
            print(f"\nSUMMARY:")
            print(f"Total Users: {len(users)}")
            print(f"Active Users: {sum(1 for u in users if u.is_active)}")
            print(f"Admin Users: {sum(1 for u in users if u.is_admin)}")
            print(f"Total Documents: {total_documents}")
            print(f"Total Storage Used: {format_size(total_storage)}")

            # User status breakdown
            print(f"\nUSER STATUS BREAKDOWN:")
            active_users = [u for u in users if u.is_active]
            inactive_users = [u for u in users if not u.is_active]
            admin_users = [u for u in users if u.is_admin]

            print(f"Active Users ({len(active_users)}):")
            for user in active_users:
                print(f"  - {user.username} ({user.email})")

            if inactive_users:
                print(f"\nInactive Users ({len(inactive_users)}):")
                for user in inactive_users:
                    print(f"  - {user.username} ({user.email})")

            if admin_users:
                print(f"\nAdmin Users ({len(admin_users)}):")
                for user in admin_users:
                    print(f"  - {user.username} ({user.email})")

            print(f"\n{'='*80}")

        except Exception as e:
            print(f"Error accessing database: {e}")
            sys.exit(1)


if __name__ == "__main__":
    list_users()
