import os
import sqlite3
import tempfile
import unittest

from flask import Flask
from werkzeug.security import generate_password_hash

from app import db, load_user, login_manager
from app import routes
from app.auth import auth
from app.models import User, ensure_user_schema
from app.rate_limiting import rate_limiter


class SoftDeleteUserTestCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.app = Flask("app")
        self.app.config.update(
            SECRET_KEY="test-secret",
            TESTING=True,
            WTF_CSRF_ENABLED=False,
            SQLALCHEMY_DATABASE_URI=f"sqlite:///{self.tmp.name}/test.db",
            SQLALCHEMY_TRACK_MODIFICATIONS=False,
            UPLOAD_FOLDER=self.tmp.name,
        )
        db.init_app(self.app)
        login_manager.init_app(self.app)
        login_manager.user_loader(load_user)
        self.app.register_blueprint(routes.app)
        self.app.register_blueprint(auth)
        self.client = self.app.test_client()
        rate_limiter.requests.clear()

        with self.app.app_context():
            db.create_all()
            db.session.add_all(
                [
                    User(
                        username="admin",
                        email="admin@example.test",
                        password_hash=generate_password_hash("Secret123!"),
                        is_admin=True,
                    ),
                    User(
                        username="alice",
                        email="alice@example.test",
                        password_hash=generate_password_hash("Secret123!"),
                    ),
                ]
            )
            db.session.commit()
            self.admin_id = User.query.filter_by(username="admin").one().id
            self.alice_id = User.query.filter_by(username="alice").one().id

        self.alice_folder = os.path.join(self.tmp.name, "alice")
        os.makedirs(self.alice_folder)

    def tearDown(self):
        with self.app.app_context():
            db.session.remove()
            db.engine.dispose()
        rate_limiter.requests.clear()
        self.tmp.cleanup()

    def login_as(self, client, user_id):
        with client.session_transaction() as session:
            session["_user_id"] = str(user_id)
            session["_fresh"] = True

    def delete_alice(self):
        self.login_as(self.client, self.admin_id)
        return self.client.post("/api/admin/delete_user", data={"user_id": "alice"})

    def test_delete_flags_user_and_keeps_row_and_documents(self):
        response = self.delete_alice()

        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.get_json()["success"])
        with self.app.app_context():
            alice = db.session.get(User, self.alice_id)
            self.assertIsNotNone(alice)
            self.assertIsNotNone(alice.deleted_at)
            self.assertFalse(alice.is_active)
        self.assertTrue(os.path.isdir(self.alice_folder))

        self.assertEqual(
            self.client.get(f"/api/admin/get_user/{self.alice_id}").status_code, 404
        )
        repeat = self.client.post("/api/admin/delete_user", data={"user_id": "alice"})
        self.assertEqual(repeat.status_code, 404)

    def test_deleted_user_cannot_log_in(self):
        self.delete_alice()

        client = self.app.test_client()
        response = client.post(
            "/login", data={"username": "alice", "password": "Secret123!"}
        )

        self.assertEqual(response.status_code, 200)
        self.assertIn("Invalid username or password", response.get_data(as_text=True))

    def test_existing_session_of_deleted_user_is_rejected(self):
        alice_client = self.app.test_client()
        self.login_as(alice_client, self.alice_id)
        self.assertEqual(alice_client.get("/api/upload_progress/x").status_code, 404)

        self.delete_alice()

        response = alice_client.get("/api/upload_progress/x")
        self.assertEqual(response.status_code, 401)


class EnsureUserSchemaTestCase(unittest.TestCase):
    def test_adds_deleted_at_to_existing_user_table(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "old.db")
            with sqlite3.connect(path) as connection:
                connection.execute('CREATE TABLE "user" (id INTEGER PRIMARY KEY)')

            app = Flask(__name__)
            app.config.update(
                SQLALCHEMY_DATABASE_URI=f"sqlite:///{path}",
                SQLALCHEMY_TRACK_MODIFICATIONS=False,
            )
            db.init_app(app)
            with app.app_context():
                ensure_user_schema()
                ensure_user_schema()
                db.engine.dispose()

            with sqlite3.connect(path) as connection:
                columns = [row[1] for row in connection.execute('PRAGMA table_info("user")')]
            self.assertIn("deleted_at", columns)


if __name__ == "__main__":
    unittest.main()
