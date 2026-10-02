import unittest
from types import SimpleNamespace
from unittest.mock import patch

from flask import Flask
from flask_login import LoginManager
from flask_wtf.csrf import CSRFProtect

from app import routes
from app.auth import auth


class ImagePreviewStateTestCase(unittest.TestCase):
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
        self.user = SimpleNamespace(
            id=7,
            is_admin=False,
            get_user_folder_name=lambda: "alice",
        )

    def render_preview(self, query_string=None):
        with (
            patch.object(routes, "current_user", self.user),
            patch.object(
                routes.file_operations,
                "get_user_documents_list",
                return_value={"completed": ["document"]},
            ),
            patch.object(
                routes.file_operations,
                "get_document_info",
                return_value={"is_completed": True},
            ),
            patch.object(
                routes.file_operations,
                "get_document_completed_dir",
                return_value="/completed/document/TOOCR",
            ),
            patch.object(
                routes.file_operations,
                "fetch_page_segments_data",
                return_value=[{"id": "012", "text": "Selected line"}],
            ) as fetch_segments,
            patch.object(routes.os.path, "exists", return_value=True),
            patch.object(routes.os, "listdir", return_value=["page_021.png"]),
        ):
            response = self.client.get("/image_preview", query_string=query_string)

        return response, fetch_segments

    def test_reload_bootstraps_controls_before_restoring_selected_segment(self):
        response, fetch_segments = self.render_preview(
            {
                "selected_folder": "document",
                "selected_png": "page_021.png",
                "selected_segment": "12",
            }
        )
        html = response.get_data(as_text=True)

        self.assertEqual(response.status_code, 200)
        self.assertIn('folder: "document"', html)
        self.assertIn('page: "page_021.png"', html)
        self.assertIn('segmentIndex: "12"', html)
        self.assertNotIn("{ { selected_png", html)
        self.assertNotIn("{ { selected_segment", html)
        self.assertLess(
            html.index("showOCRControls(selectedFolder, selectedPng);"),
            html.index("checkForExistingSegments(selectedFolder, selectedPng);"),
        )
        self.assertIn(
            "const segmentIndexFromUrl = urlParams.get('selected_segment');", html
        )
        self.assertIn("displaySegmentByIndex(initialIndex, false);", html)
        fetch_segments.assert_called_once_with(
            self.app.config,
            "alice",
            "document",
            "page_021",
            is_document_completed=True,
        )

    def test_empty_reload_state_is_serialized_as_javascript_null(self):
        response, fetch_segments = self.render_preview()
        html = response.get_data(as_text=True)

        self.assertEqual(response.status_code, 200)
        self.assertIn("folder: null", html)
        self.assertIn("page: null", html)
        self.assertIn("segmentIndex: null", html)
        self.assertNotIn('const selectedPng = "None"', html)
        fetch_segments.assert_not_called()


if __name__ == "__main__":
    unittest.main()
