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

    def render_preview(self, query_string=None, png_files=None):
        if png_files is None:
            png_files = ["page_021.png"]
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
            patch.object(routes.os, "listdir", return_value=png_files),
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
        self.assertIn("const initialPageParams = new URLSearchParams", html)
        self.assertIn("folder: initialPageParams.get('selected_folder')", html)
        self.assertIn("page: initialPageParams.get('selected_png')", html)
        self.assertIn("segmentIndex: initialPageParams.get('selected_segment')", html)
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
        self.assertNotIn("{ { selected_png", html)
        self.assertNotIn("{ { selected_segment", html)
        self.assertNotIn('const selectedPng = "None"', html)
        fetch_segments.assert_not_called()

    def test_workspace_language_selector_defaults_to_greek_polytonic(self):
        response, _ = self.render_preview()
        html = response.get_data(as_text=True)

        self.assertEqual(response.status_code, 200)
        self.assertIn(
            '<option value="greek-polytonic" selected>Greek Polytonic</option>',
            html,
        )
        self.assertIn('<option value="coptic">Coptic</option>', html)
        for value, label in (
            ("old-english", "Old English"),
            ("latin", "Latin"),
            ("catalan", "Catalan"),
        ):
            self.assertIn(
                f'<option value="{value}" disabled>{label}</option>',
                html,
            )

    def test_single_page_numeric_filename_renders_without_underscore(self):
        response, _ = self.render_preview(
            {"selected_folder": "document"},
            png_files=["001.png"],
        )
        html = response.get_data(as_text=True)

        self.assertEqual(response.status_code, 200)
        self.assertIn('<option value="001.png"', html)
        self.assertIn("Page 1", html)

    def test_polytonic_greek_uses_specialized_ocr_service(self):
        config = {
            "OCR_SERVICE_URL": "http://ocr:8000",
            "VLM_OCR_SERVICE_URL": "http://vlm_ocr:8000",
        }

        self.assertEqual(
            routes._get_ocr_service_url(config, "greek-polytonic"),
            "http://ocr:8000",
        )

    def test_non_greek_languages_use_vlm_ocr_service(self):
        config = {
            "OCR_SERVICE_URL": "http://ocr:8000",
            "VLM_OCR_SERVICE_URL": "http://vlm_ocr:8000",
        }

        for language in ("coptic", "old-english", "latin", "catalan"):
            with self.subTest(language=language):
                self.assertEqual(
                    routes._get_ocr_service_url(config, language),
                    "http://vlm_ocr:8000",
                )

    def test_unsupported_language_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Unsupported OCR language"):
            routes._get_ocr_service_url({}, "unknown")


if __name__ == "__main__":
    unittest.main()
