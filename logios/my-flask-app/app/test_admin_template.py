import unittest
from datetime import datetime
from types import SimpleNamespace

from flask import Flask, render_template


def create_test_app():
    app = Flask(__name__)
    app.add_url_rule("/", endpoint="app.index", view_func=lambda: "")
    app.add_url_rule("/login", endpoint="auth.login", view_func=lambda: "")
    app.add_url_rule("/register", endpoint="auth.register", view_func=lambda: "")
    app.jinja_env.globals["csrf_token"] = lambda: "test-token"
    app.jinja_env.globals["current_user"] = SimpleNamespace(is_authenticated=False)
    return app


def render_admin(app, users):
    with app.test_request_context("/admin"):
        return render_template(
            "admin.html",
            users_data=users,
            total_users=len(users),
            total_storage_mb=12.5,
        )


class AdminTemplateTestCase(unittest.TestCase):
    def setUp(self):
        self.app = create_test_app()
        self.user = {
            "user_id": "alice",
            "db_user_id": 1,
            "username": "alice",
            "email": "alice@example.test",
            "display_name": "Alice Example",
            "is_admin": True,
            "is_active": False,
            "created_at": datetime(2025, 1, 1),
            "last_login": None,
            "has_folder": True,
            "total_documents": 2,
            "completed_documents": 1,
            "in_progress_documents": 1,
            "storage_size_mb": 12.5,
        }

    def test_styles_modal_and_user_actions_are_rendered(self):
        html = render_admin(self.app, [self.user])

        self.assertIn(".user-item {", html.split("</head>")[0])
        self.assertEqual(html.count('id="confirmation-modal"'), 1)
        self.assertIn("viewUserDocuments('alice')", html)
        self.assertIn("editUser('1')", html)
        self.assertIn("confirmDeleteUser('alice')", html)
        self.assertIn('class="user-badge">Admin', html)
        self.assertIn('class="user-badge user-badge-inactive">Inactive', html)
        self.assertIn("Never logged in", html)

    def test_admin_post_requests_send_csrf_token(self):
        html = render_admin(self.app, [self.user])

        self.assertEqual(html.count("'X-CSRFToken':"), 4)

    def test_empty_list_and_user_without_uploads(self):
        empty_html = render_admin(self.app, [])
        self.assertIn("No users found.", empty_html)

        self.user["has_folder"] = False
        html = render_admin(self.app, [self.user])
        self.assertIn("No uploads folder", html)
        self.assertNotIn("viewUserDocuments('alice')", html)
        self.assertIn("editUser('1')", html)


if __name__ == "__main__":
    unittest.main()
