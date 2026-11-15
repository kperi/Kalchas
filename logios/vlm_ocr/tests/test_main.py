import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from fastapi.testclient import TestClient
from PIL import Image, ImageDraw

from app.main import app
from app.response_parser import NoRecognizedText


class ProcessImageTestCase(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.image_path = self.root / "page.png"
        image = Image.new("RGB", (200, 100), "white")
        draw = ImageDraw.Draw(image)
        for top in (20, 60):
            for left in range(20, 180, 12):
                draw.rectangle((left, top, left + 6, top + 8), fill="black")
        image.save(self.image_path)
        self.model = Mock()
        self.model.recognize_image.return_value = [
            {"text": "line text", "bbox": [0, 0, 10, 10]}
        ]
        self.model.process.return_value = [
            {"text": "whole page", "bbox": [0, 0, 200, 100]}
        ]
        app.state.ocr_model = self.model
        self.addCleanup(lambda: delattr(app.state, "ocr_model"))
        self.env = patch.dict(os.environ, {"OCR_IMAGE_ROOT": str(self.root)})
        self.env.start()
        self.addCleanup(self.env.stop)
        self.client = TestClient(app)
        self.addCleanup(self.client.close)

    def test_manuscript_request_returns_original_envelope_and_numbered_crops(self):
        response = self.client.post("/process_image/", json={
            "value": str(self.image_path), "segmentation": "manuscript",
        })
        self.assertEqual(response.status_code, 200)
        result = response.json()
        self.assertEqual(set(result), {"received_value", "ret"})
        self.assertEqual(result["received_value"], str(self.image_path))
        self.assertEqual(set(result["ret"]), {
            "status", "file", "message", "recognized_text", "segments",
        })
        self.assertEqual(result["ret"]["status"], "ok")
        self.assertEqual(result["ret"]["recognized_text"], "line text\nline text")
        self.assertEqual(len(result["ret"]["segments"]), 2)
        self.assertEqual(self.model.recognize_image.call_count, 2)
        self.model.process.assert_not_called()
        self.assertTrue((self.root / "page" / "000.png").is_file())
        self.assertTrue((self.root / "page" / "001.png").is_file())

    def test_existing_default_keeps_whole_page_recognition(self):
        response = self.client.post("/process_image/", json={"value": str(self.image_path)})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["ret"]["segments"], self.model.process.return_value)
        self.model.process.assert_called_once_with(self.image_path)
        self.model.recognize_image.assert_not_called()

    def test_invalid_segmentation_is_rejected(self):
        response = self.client.post("/process_image/", json={
            "value": str(self.image_path), "segmentation": "unknown",
        })
        self.assertEqual(response.status_code, 422)
        self.model.process.assert_not_called()

    def test_segmentation_failure_is_reported(self):
        Image.new("RGB", (200, 100), "white").save(self.image_path)
        response = self.client.post("/process_image/", json={
            "value": str(self.image_path), "segmentation": "manuscript",
        })
        self.assertEqual(response.status_code, 500)
        self.assertIn("No text lines detected", response.json()["detail"])
        self.model.recognize_image.assert_not_called()

    def test_all_empty_line_responses_are_reported_without_saving_crops(self):
        self.model.recognize_image.side_effect = NoRecognizedText("no line text")
        response = self.client.post("/process_image/", json={
            "value": str(self.image_path), "segmentation": "manuscript",
        })
        self.assertEqual(response.status_code, 502)
        self.assertIn("no recognized text on any line", response.json()["detail"])
        self.assertFalse((self.root / "page").exists())

    def test_crop_write_failure_is_not_returned_as_success(self):
        with patch.object(Image.Image, "save", side_effect=OSError("cannot write crop")):
            response = self.client.post("/process_image/", json={
                "value": str(self.image_path), "segmentation": "manuscript",
            })
        self.assertEqual(response.status_code, 500)
        self.assertIn("cannot write crop", response.json()["detail"])


if __name__ == "__main__":
    unittest.main()
