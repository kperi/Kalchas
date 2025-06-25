#!/usr/bin/env python3
"""
Comprehensive user management script for the Flask OCR application.
This script provides various user listing and management functions.

Usage:
docker compose exec flaskui python user_manager.py [command] [options]

Commands:
  list          - List all users (default)
  list-simple   - List users with basic info only
  list-admins   - List only admin users
  list-active   - List only active users
  list-inactive - List only inactive users
  stats         - Show user statistics
  help          - Show this help message

Examples:
  docker compose exec flaskui python user_manager.py
  docker compose exec flaskui python user_manager.py list
  docker compose exec flaskui python user_manager.py list-admins
  docker compose exec flaskui python user_manager.py stats
"""

import sys
import os
import argparse
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
                    continue

    return total_size


def count_user_documents(user_id):
    """Count number of document folders for a user"""
    user_upload_dir = f"/app/uploads/{user_id}"
    doc_count = 0

    if os.path.exists(user_upload_dir):
        try:
            for item in os.listdir(user_upload_dir):
                item_path = os.path.join(user_upload_dir, item)
                if os.path.isdir(item_path):
                    doc_count += 1
        except (OSError, IOError):
            pass

    return doc_count


def print_header(title):
    """Print formatted header"""
    print(f"\n{'='*100}")
    print(f"{title:^100}")
    print(f"{'='*100}")
    print(f"Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*100}\n")


def print_user_table(users, include_storage=False):
    """Print users in table format"""
    if not users:
        print("No users found.")
        return

    if include_storage:
        print(
            f"{'ID':<4} {'Username':<20} {'Email':<30} {'Name':<20} {'Admin':<6} {'Active':<7} {'Docs':<6} {'Storage':<12} {'Created':<20}"
        )
        print("-" * 125)

        total_storage = 0
        total_docs = 0

        for user in users:
            storage_size = get_user_storage_size(user.id)
            doc_count = count_user_documents(user.id)
            total_storage += storage_size
            total_docs += doc_count

            admin_status = "Yes" if user.is_admin else "No"
            active_status = "Yes" if user.is_active else "No"
            full_name = f"{user.first_name or ''} {user.last_name or ''}".strip() or "-"

            print(
                f"{user.id:<4} {user.username:<20} {user.email:<30} {full_name:<20} {admin_status:<6} {active_status:<7} {doc_count:<6} {format_size(storage_size):<12} {format_date(user.created_at):<20}"
            )

        print("-" * 125)
        print(
            f"TOTALS: {len(users)} users, {total_docs} documents, {format_size(total_storage)} storage"
        )

    else:
        print(
            f"{'ID':<4} {'Username':<20} {'Email':<35} {'Name':<25} {'Admin':<6} {'Active':<7} {'Created':<20}"
        )
        print("-" * 117)

        for user in users:
            admin_status = "Yes" if user.is_admin else "No"
            active_status = "Yes" if user.is_active else "No"
            full_name = f"{user.first_name or ''} {user.last_name or ''}".strip() or "-"

            print(
                f"{user.id:<4} {user.username:<20} {user.email:<35} {full_name:<25} {admin_status:<6} {active_status:<7} {format_date(user.created_at):<20}"
            )

        print("-" * 117)
        print(f"Total: {len(users)} users")


def list_all_users():
    """List all users with full details"""
    app = create_app("development")
    with app.app_context():
        users = User.query.all()
        print_header("ALL USERS - FULL DETAILS")
        print_user_table(users, include_storage=True)


def list_users_simple():
    """List all users with basic info"""
    app = create_app("development")
    with app.app_context():
        users = User.query.all()
        print_header("ALL USERS - BASIC INFO")
        print_user_table(users, include_storage=False)


def list_admin_users():
    """List only admin users"""
    app = create_app("development")
    with app.app_context():
        users = User.query.filter_by(is_admin=True).all()
        print_header("ADMIN USERS")
        print_user_table(users, include_storage=False)


def list_active_users():
    """List only active users"""
    app = create_app("development")
    with app.app_context():
        users = User.query.filter_by(is_active=True).all()
        print_header("ACTIVE USERS")
        print_user_table(users, include_storage=False)


def list_inactive_users():
    """List only inactive users"""
    app = create_app("development")
    with app.app_context():
        users = User.query.filter_by(is_active=False).all()
        print_header("INACTIVE USERS")
        print_user_table(users, include_storage=False)


def show_user_stats():
    """Show comprehensive user statistics"""
    app = create_app("development")
    with app.app_context():
        users = User.query.all()

        print_header("USER STATISTICS")

        total_users = len(users)
        active_users = len([u for u in users if u.is_active])
        inactive_users = len([u for u in users if not u.is_active])
        admin_users = len([u for u in users if u.is_admin])

        # Recent activity
        from datetime import datetime, timedelta

        thirty_days_ago = datetime.utcnow() - timedelta(days=30)
        recent_logins = len(
            [u for u in users if u.last_login and u.last_login > thirty_days_ago]
        )

        print(f"Total Users:           {total_users}")
        print(f"Active Users:          {active_users}")
        print(f"Inactive Users:        {inactive_users}")
        print(f"Admin Users:           {admin_users}")
        print(f"Recent Logins (30d):   {recent_logins}")

        if users:
            # Newest and oldest users
            newest_user = max(users, key=lambda u: u.created_at)
            oldest_user = min(users, key=lambda u: u.created_at)

            print(
                f"\nNewest User:           {newest_user.username} ({format_date(newest_user.created_at)})"
            )
            print(
                f"Oldest User:           {oldest_user.username} ({format_date(oldest_user.created_at)})"
            )

            # Last login stats
            users_with_login = [u for u in users if u.last_login]
            if users_with_login:
                most_recent_login = max(users_with_login, key=lambda u: u.last_login)
                print(
                    f"Most Recent Login:     {most_recent_login.username} ({format_date(most_recent_login.last_login)})"
                )

        print(f"\n{'='*100}")


def show_help():
    """Show help message"""
    print(__doc__)


def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description="User management for Flask OCR app", add_help=False
    )
    parser.add_argument(
        "command",
        nargs="?",
        default="list",
        choices=[
            "list",
            "list-simple",
            "list-admins",
            "list-active",
            "list-inactive",
            "stats",
            "help",
        ],
        help="Command to execute",
    )

    args = parser.parse_args()

    try:
        if args.command == "list":
            list_all_users()
        elif args.command == "list-simple":
            list_users_simple()
        elif args.command == "list-admins":
            list_admin_users()
        elif args.command == "list-active":
            list_active_users()
        elif args.command == "list-inactive":
            list_inactive_users()
        elif args.command == "stats":
            show_user_stats()
        elif args.command == "help":
            show_help()
        else:
            show_help()

    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
