import unittest

from app.response_parser import ImageSize, InvalidModelResponse, parse_segments


class ParseSegmentsTestCase(unittest.TestCase):
    def test_parses_segments_object(self):
        result = parse_segments(
            '{"segments":[{"text":"χαῖρε","bbox":[1,2,101,22]}]}',
            ImageSize(width=200, height=100),
        )

        self.assertEqual(
            result,
            [{"text": "χαῖρε", "bbox": [1, 2, 101, 22]}],
        )

    def test_parses_json_code_fence(self):
        result = parse_segments(
            '```json\n[{"text":"Latin","bbox":[0,0,80,20]}]\n```',
            ImageSize(width=100, height=50),
        )

        self.assertEqual(result[0]["text"], "Latin")

    def test_rejects_out_of_bounds_bbox(self):
        with self.assertRaisesRegex(
            InvalidModelResponse,
            "invalid horizontal coordinates",
        ):
            parse_segments(
                '[{"text":"text","bbox":[0,0,101,20]}]',
                ImageSize(width=100, height=50),
            )

    def test_rejects_non_json_response(self):
        with self.assertRaisesRegex(InvalidModelResponse, "not valid JSON"):
            parse_segments("recognized text", ImageSize(width=100, height=50))


if __name__ == "__main__":
    unittest.main()

