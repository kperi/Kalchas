import io
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from flask import Flask
from flask_login import LoginManager
from flask_wtf.csrf import CSRFProtect

from app import routes
from app.auth import auth


class UploadApiTestCase(unittest.TestCase):
    def setUp(self):
        self.app = Flask(__name__)
        self.app.config.update(
            LOGIN_DISABLED=True,
            SECRET_KEY="test-secret",
            TESTING=True,
            WTF_CSRF_ENABLED=False,
        )
        login_manager = LoginManager(self.app)
        login_manager.user_loader(lambda _user_id: None)
        CSRFProtect(self.app)
        self.app.register_blueprint(routes.app)
        self.app.register_blueprint(auth)
        self.client = self.app.test_client()
        self.user = SimpleNamespace(id=7, get_user_folder_name=lambda: "alice")
        routes.upload_progress_store.clear()

    def tearDown(self):
        routes.upload_progress_store.clear()

    def test_upload_requires_a_file_and_returns_json(self):
        with patch.object(routes, "current_user", self.user):
            response = self.client.post("/api/upload_with_progress")

        self.assertEqual(response.status_code, 400)
        self.assertEqual(
            response.get_json(),
            {"success": False, "message": "No file part in the request"},
        )

    def test_upload_returns_server_generated_id(self):
        with (
            patch.object(routes, "current_user", self.user),
            patch.object(
                routes.file_operations,
                "handle_pdf_upload_with_auto_folder_no_sanitize",
                return_value=(
                    "/uploads/alice/document",
                    "/uploads/alice/document/page.png",
                    "document",
                ),
            ) as save_upload,
            patch.object(routes.threading, "Thread") as thread_class,
        ):
            response = self.client.post(
                "/api/upload_with_progress",
                data={"file": (io.BytesIO(b"image data"), "page.png")},
                content_type="multipart/form-data",
            )

        payload = response.get_json()
        self.assertEqual(response.status_code, 200)
        self.assertTrue(payload["success"])
        self.assertIn(payload["upload_id"], routes.upload_progress_store)
        self.assertEqual(
            routes.upload_progress_store[payload["upload_id"]]["user_id"], "alice"
        )
        save_upload.assert_called_once()
        thread_class.assert_called_once()
        worker_args = thread_class.call_args.kwargs["args"]
        self.assertNotIn(self.user, worker_args)
        self.assertFalse(any(hasattr(argument, "stream") for argument in worker_args))
        thread_class.return_value.start.assert_called_once_with()

    def test_progress_is_scoped_to_the_current_user(self):
        routes.upload_progress_store["job-id"] = {
            "status": "processing",
            "user_id": "alice",
        }

        with patch.object(routes, "current_user", self.user):
            own_response = self.client.get("/api/upload_progress/job-id")

        other_user = SimpleNamespace(id=8, get_user_folder_name=lambda: "bob")
        with patch.object(routes, "current_user", other_user):
            other_response = self.client.get("/api/upload_progress/job-id")

        self.assertEqual(own_response.status_code, 200)
        self.assertEqual(own_response.get_json()["progress"]["status"], "processing")
        self.assertEqual(other_response.status_code, 403)

    def test_upload_page_uses_json_api_contract(self):
        response = self.client.get("/show_upload_page")
        html = response.get_data(as_text=True)

        self.assertEqual(response.status_code, 200)
        self.assertIn('action="/api/upload_with_progress"', html)
        self.assertIn("/api/upload_progress/", html)
        self.assertNotIn("new DOMParser", html)
        self.assertNotIn("fetch(`/progress/", html)

    def test_upload_page_does_not_accept_legacy_form_posts(self):
        response = self.client.post(
            "/show_upload_page",
            data={"file": (io.BytesIO(b"image data"), "page.png")},
            content_type="multipart/form-data",
        )

        self.assertEqual(response.status_code, 405)


if __name__ == "__main__":
    unittest.main()
