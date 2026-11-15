import hashlib
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

import numpy as np
from PIL import Image, ImageDraw

from app.coptic import process_coptic
from app.response_parser import NoRecognizedText
from app.segmentation import segment_lines


def manuscript():
    image = Image.new("RGB", (240, 180), (240, 225, 200))
    draw = ImageDraw.Draw(image)
    for top, color in ((25, (20, 20, 20)), (75, (190, 20, 20)), (125, (20, 20, 20))):
        for left in range(30, 200, 12):
            draw.rectangle((left, top, left + 6, top + 10), fill=color)
    return image


class SegmentationTestCase(unittest.TestCase):
    def test_detects_black_and_red_lines_in_order_with_rgb_source_boxes(self):
        image = manuscript()
        lines = segment_lines(image)
        self.assertEqual(len(lines), 3)
        for line, top, color in zip(
            lines, (25, 75, 125), ((20, 20, 20), (190, 20, 20), (20, 20, 20))
        ):
            self.assertEqual(line.bbox, [25, top - 5, 210, top + 16])
            self.assertEqual(line.image.mode, "RGB")
            self.assertEqual(line.image.size, (195, 31))
            self.assertTrue(np.any(np.all(np.asarray(line.image) == color, axis=2)))
        self.assertEqual(image.tobytes(), manuscript().tobytes())

    def test_blank_and_tiny_pages_do_not_produce_spurious_lines(self):
        for size in ((240, 180), (1, 1), (2, 2)):
            with self.subTest(size=size):
                self.assertEqual(segment_lines(Image.new("RGB", size, "white")), [])

    def test_rejects_oversized_image_before_detection(self):
        image = Mock(width=6000, height=5000)
        with self.assertRaisesRegex(ValueError, "24 megapixels"):
            segment_lines(image)
        image.convert.assert_not_called()

    def test_coptic_contract_crops_and_source_are_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "page_001.png"
            manuscript().save(path)
            output_dir = path.parent / path.stem
            output_dir.mkdir()
            (output_dir / "099.png").write_bytes(b"stale")
            (output_dir / "preview.png").write_bytes(b"keep")
            original_hash = hashlib.sha256(path.read_bytes()).hexdigest()
            recognize = Mock(side_effect=[
                [{"text": "first", "bbox": [0, 0, 195, 31]}],
                [{"text": "second", "bbox": [0, 0, 195, 31]}],
                [{"text": "third", "bbox": [0, 0, 195, 31]}],
            ])
            segments = process_coptic(path, recognize)
            self.assertEqual([s["text"] for s in segments], ["first", "second", "third"])
            self.assertEqual(recognize.call_count, 3)
            for index, segment in enumerate(segments):
                self.assertEqual(set(segment), {"text", "bbox", "coords"})
                self.assertEqual(segment["coords"], segment["bbox"])
                self.assertEqual(segment["bbox"], [25, 20 + 50 * index, 210, 41 + 50 * index])
                with Image.open(path.parent / path.stem / f"{index:03d}.png") as crop:
                    self.assertEqual(crop.mode, "RGB")
                    self.assertEqual(crop.size, (195, 31))
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), original_hash)
            self.assertFalse((output_dir / "099.png").exists())
            self.assertTrue((output_dir / "preview.png").exists())

    def test_blank_page_is_an_explicit_error_without_whole_page_ocr(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "blank.png"
            Image.new("RGB", (240, 180), "white").save(path)
            recognize = Mock()
            with self.assertRaisesRegex(ValueError, "No text lines detected"):
                process_coptic(path, recognize)
            recognize.assert_not_called()
            self.assertFalse((path.parent / path.stem).exists())

    def test_empty_line_recognition_is_preserved_when_other_lines_have_text(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "page.png"
            manuscript().save(path)
            recognize = Mock(side_effect=[
                NoRecognizedText("no text"),
                [{"text": "second", "bbox": [0, 0, 10, 10]}],
                NoRecognizedText("no text"),
            ])
            segments = process_coptic(path, recognize)
            self.assertEqual([segment["text"] for segment in segments], ["", "second", ""])
            self.assertTrue((path.parent / path.stem / "000.png").is_file())
            self.assertTrue((path.parent / path.stem / "002.png").is_file())

    def test_all_empty_line_recognition_is_not_returned_as_success(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "page.png"
            manuscript().save(path)
            with self.assertRaisesRegex(NoRecognizedText, "on any line"):
                process_coptic(path, Mock(side_effect=NoRecognizedText("no text")))
            self.assertFalse((path.parent / path.stem).exists())


if __name__ == "__main__":
    unittest.main()
