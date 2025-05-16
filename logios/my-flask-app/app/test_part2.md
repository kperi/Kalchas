````

**Explanation and How to Run:**

1.  **Save the Code:** Save the test code above as `test_file_operations.py` in a location where Python can find it and the `app.file_operations` module (e.g., in a `tests` subdirectory, and ensure your `PYTHONPATH` is set up, or place it alongside `file_operations.py` if your project structure is flat).
2.  **`MOCK_APP_CONFIG`**: This dictionary simulates the `current_app.config` used in your routes.
3.  **`@patch`**: This decorator from `unittest.mock` is used extensively to replace objects (like `os.path.exists`, `os.listdir`, `open`, `json.load`, `logger`, etc.) with `MagicMock` objects. This allows you to:
    *   Control their return values.
    *   Assert that they were called with the correct arguments.
    *   Prevent actual file system operations or external calls.
4.  **`mock_open`**: Used to mock the built-in `open()` function, allowing you to simulate reading from and writing to files without touching the disk.
5.  **`setUp` and `tearDown`**: Used here to patch the `logger` for all test methods in the class, ensuring it's reset for each test.
6.  **Test Cases**:
    *   **Path Helpers**: Verify that the path construction functions return the expected string paths.
    *   **`get_segment_image_file_details`**: Tests scenarios like file/directory existence and different ID formats.
    *   **`fetch_page_segments_data`**: Tests successful data loading, directory not found, and JSON errors.
    *   **`resolve_segment_image_filename_from_index`**: Tests index logic, fallbacks, and error conditions.
    *   **`save_uploaded_cropped_image`**: Tests filename generation for new and subsequent crops, and error handling.
    *   **`save_ocr_processed_data`**: Tests different OCR payload types and verifies that file creation/writing is attempted correctly. This is a complex function, so these tests cover key paths.
    *   **`copy_files_to_ocr_folder`**: Tests the logic for copying originals and prioritizing cropped versions.
7.  **Assertions**: Methods like `assertEqual`, `assertTrue`, `assertIsNone`, `assert_called_once_with`, `assert_any_call`, `assert_has_calls` are used to verify the behavior of the functions.
8.  **Running the Tests**:
    You can run these tests from your terminal. Navigate to the directory containing `test_file_operations.py` (or your main project directory if your test runner can discover them) and run:
    ```bash
    python -m unittest test_file_operations.py
    ```
    Or, if you have a test runner like `pytest` or use your IDE's test runner, you can use that.

This suite provides a good starting point for testing your `file_operations.py` module. You can expand it further by adding more specific scenarios or edge cases as needed.# filepath: /Users/konstantinosperifanos/code/projects/logios/Kalchas/logios/my-flask-app/app/test_file_operations.py
import unittest
from unittest.mock import patch, mock_open, MagicMock, call
import os
import json
import base64
import tempfile # For actual temporary directory in some tests if needed

# Assuming file_operations.py is in a module 'app'
# Adjust the import according to your project structure
from app import file_operations

# Mock app_config for tests
MOCK_APP_CONFIG = {"UPLOAD_FOLDER": "/tmp/test_uploads"}
MOCK_USER_ID = "test_user"
MOCK_FOLDER_NAME = "test_doc"
MOCK_PAGE_ID_BASE = "page_001"

class TestFileOperations(unittest.TestCase):

    def setUp(self):
        # Patch the logger for all tests in this class
        self.patcher_logger = patch('app.file_operations.logger', MagicMock())
        self.mock_logger = self.patcher_logger.start()

    def tearDown(self):
        self.patcher_logger.stop()

    # --- Test Path Helper Functions ---
    def test_get_upload_root(self):
        self.assertEqual(file_operations.get_upload_root(MOCK_APP_CONFIG), "/tmp/test_uploads")

    def test_get_user_dir(self):
        expected = os.path.join("/tmp/test_uploads", MOCK_USER_ID)
        self.assertEqual(file_operations.get_user_dir(MOCK_APP_CONFIG, MOCK_USER_ID), expected)

    def test_get_document_dir(self):
        expected = os.path.join("/tmp/test_uploads", MOCK_USER_ID, MOCK_FOLDER_NAME)
        self.assertEqual(file_operations.get_document_dir(MOCK_APP_CONFIG, MOCK_USER_ID, MOCK_FOLDER_NAME), expected)

    def test_get_toocr_dir(self):
        expected = os.path.join("/tmp/test_uploads", MOCK_USER_ID, MOCK_FOLDER_NAME, "TOOCR")
        self.assertEqual(file_operations.get_toocr_dir(MOCK_APP_CONFIG, MOCK_USER_ID, MOCK_FOLDER_NAME), expected)

    def test_get_page_segments_dir(self):
        expected = os.path.join("/tmp/test_uploads", MOCK_USER_ID, MOCK_FOLDER_NAME, "TOOCR", MOCK_PAGE_ID_BASE)
        self.assertEqual(file_operations.get_page_segments_dir(MOCK_APP_CONFIG, MOCK_USER_ID, MOCK_FOLDER_NAME, MOCK_PAGE_ID_BASE), expected)

    def test_get_combined_segments_parent_dir(self):
        expected = os.path.join("/tmp/test_uploads", MOCK_USER_ID, MOCK_FOLDER_NAME, "TOOCR", "segments")
        self.assertEqual(file_operations.get_combined_segments_parent_dir(MOCK_APP_CONFIG, MOCK_USER_ID, MOCK_FOLDER_NAME), expected)

    def test_get_cropped_images_dir(self):
        expected = os.path.join("/tmp/test_uploads", MOCK_USER_ID, MOCK_FOLDER_NAME, "cropped")
        self.assertEqual(file_operations.get_cropped_images_dir(MOCK_APP_CONFIG, MOCK_USER_ID, MOCK_FOLDER_NAME), expected)

    # --- Test get_segment_image_file_details ---
    @patch('app.file_operations.os.path.isdir')
    @patch('app.file_operations.os.path.exists')
    def test_get_segment_image_file_details_success_numeric_id(self, mock_exists, mock_isdir):
        mock_isdir.return_value = True
        mock_exists.return_value = True
        segment_id_str = "1"
        expected_dir = file_operations.get_page_segments_dir(MOCK_APP_CONFIG, MOCK_USER_ID, MOCK_FOLDER_NAME, MOCK_PAGE_ID_BASE)
        expected_filename = "001.png"

        result_dir, result_filename = file_operations.get_segment_image_file_details(
            MOCK_APP_CONFIG, MOCK_USER_ID, MOCK_FOLDER_NAME, MOCK_PAGE_ID_BASE, segment_id_str
        )
        self.assertEqual(result_dir, expected_dir)
        self.assertEqual(result_filename, expected_filename)
        mock_isdir.assert_called_once_with(expected_dir)
        mock_exists.assert_called_once_with(os.path.join(expected_dir, expected_filename))

    @patch('app.file_operations.os.path.isdir')
    @patch('app.file_operations.os.path.exists')
    def test_get_segment_image_file_details_success_direct_filename(self, mock_exists, mock_isdir):
        mock_isdir.return_value = True
        mock_exists.return_value = True # Mock that direct filename.png exists
        segment_id_str = "my_image.png" # This will fail int() but then be used directly
        expected_dir = file_operations.get_page_segments_dir(MOCK_APP_CONFIG, MOCK_USER_ID, MOCK_FOLDER_NAME, MOCK_PAGE_ID_BASE)
        expected_filename = "my_image.png"

        result_dir, result_filename = file_operations.get_segment_image_file_details(
            MOCK_APP_CONFIG, MOCK_USER_ID, MOCK_FOLDER_NAME, MOCK_PAGE_ID_BASE, segment_id_str
        )
        self.assertEqual(result_dir, expected_dir)
        self.assertEqual(result_filename, expected_filename)
        # os.path.exists is called twice in this path: once for the int-cast name, once for direct name
        mock_exists.assert_any_call(os.path.join(expected_dir, segment_id_str))


    @patch('app.file_operations.os.path.isdir')
    def test_get_segment_image_file_details_dir_not_found(self, mock_isdir):
        mock_isdir.return_value = False
        result_dir, result_filename = file_operations.get_segment_image_file_details(
            MOCK_APP_CONFIG, MOCK_USER_ID, MOCK_FOLDER_NAME, MOCK_PAGE_ID_BASE, "1"
        )
        self.assertIsNone(result_dir)
        self.assertIsNone(result_filename)

    @patch('app.file_operations.os.path.isdir')
    @patch('app.file_operations.os.path.exists')
    def test_get_segment_image_file_details_file_not_found(self, mock_exists, mock_isdir):
        mock_isdir.return_value = True
        mock_exists.return_value = False # Image file does not exist
        result_dir, result_filename = file_operations.get_segment_image_file_details(
            MOCK_APP_CONFIG, MOCK_USER_ID, MOCK_FOLDER_NAME, MOCK_PAGE_ID_BASE, "1"
        )
        self.assertIsNone(result_dir)
        self.assertIsNone(result_filename)

    @patch('app.file_operations.os.path.isdir')
    @patch('app.file_operations.os.path.exists')
    def test_get_segment_image_file_details_invalid_id_format(self, mock_exists, mock_isdir):
        mock_isdir.return_value = True
        mock_exists.return_value = False # For the direct check after ValueError
        result_dir, result_filename = file_operations.get_segment_image_file_details(
            MOCK_APP_CONFIG, MOCK_USER_ID, MOCK_FOLDER_NAME, MOCK_PAGE_ID_BASE, "invalid"
        )
        self.assertIsNone(result_dir)
        self.assertIsNone(result_filename)

    # --- Test fetch_page_segments_data ---
    @patch('app.file_operations.os.path.isdir')
    @patch('app.file_operations.os.listdir')
    @patch('app.file_operations.open', new_callable=mock_open)
    @patch('app.file_operations.json.load')
    def test_fetch_page_segments_data_success(self, mock_json_load, mock_file_open, mock_listdir, mock_isdir):
        mock_isdir.return_value = True
        mock_listdir.return_value = ["001.json", "002.json", "other.txt"]
        mock_json_load.side_effect = [{"id": "001", "text": "text1"}, {"id": "002", "text": "text2"}]

        expected_data = [{"id": "001", "text": "text1"}, {"id": "002", "text": "text2"}]
        result = file_operations.fetch_page_segments_data(MOCK_APP_CONFIG, MOCK_USER_ID, MOCK_FOLDER_NAME, MOCK_PAGE_ID_BASE)
        self.assertEqual(result, expected_data)
        self.assertEqual(mock_json_load.call_count, 2)

    @patch('app.file_operations.os.path.isdir')
    def test_fetch_page_segments_data_dir_not_found(self, mock_isdir):
        mock_isdir.return_value = False
        result = file_operations.fetch_page_segments_data(MOCK_APP_CONFIG, MOCK_USER_ID, MOCK_FOLDER_NAME, MOCK_PAGE_ID_BASE)
        self.assertEqual(result, [])

    @patch('app.file_operations.os.path.isdir')
    @patch('app.file_operations.os.listdir')
    @patch('app.file_operations.open', new_callable=mock_open)
    @patch('app.file_operations.json.load')
    def test_fetch_page_segments_data_json_load_error(self, mock_json_load, mock_file_open, mock_listdir, mock_isdir):
        mock_isdir.return_value = True
        mock_listdir.return_value = ["error.json"]
        mock_json_load.side_effect = json.JSONDecodeError("Error", "doc", 0)
        result = file_operations.fetch_page_segments_data(MOCK_APP_CONFIG, MOCK_USER_ID, MOCK_FOLDER_NAME, MOCK_PAGE_ID_BASE)
        self.assertEqual(result, []) # Should skip the erroneous file
        self.mock_logger.error.assert_called()


    # --- Test resolve_segment_image_filename_from_index ---
    @patch('app.file_operations.os.path.isdir')
    @patch('app.file_operations.os.listdir')
    def test_resolve_segment_image_filename_from_index_success(self, mock_listdir, mock_isdir):
        mock_isdir.return_value = True
        mock_listdir.return_value = ["000.png", "001.png", "002.png"]
        result = file_operations.resolve_segment_image_filename_from_index(
            MOCK_APP_CONFIG, MOCK_USER_ID, MOCK_FOLDER_NAME, MOCK_PAGE_ID_BASE, "1"
        )
        self.assertEqual(result, "001.png")

    @patch('app.file_operations.os.path.isdir')
    def test_resolve_segment_image_filename_from_index_dir_not_found(self, mock_isdir):
        mock_isdir.return_value = False
        result = file_operations.resolve_segment_image_filename_from_index(
            MOCK_APP_CONFIG, MOCK_USER_ID, MOCK_FOLDER_NAME, MOCK_PAGE_ID_BASE, "0"
        )
        self.assertEqual(result, "error_directory_not_found.png")

    @patch('app.file_operations.os.path.isdir')
    @patch('app.file_operations.os.listdir')
    @patch('app.file_operations.os.path.exists') # For fallback checks
    def test_resolve_segment_image_filename_from_index_out_of_bounds_fallback_direct(self, mock_exists, mock_listdir, mock_isdir):
        mock_isdir.return_value = True
        mock_listdir.return_value = ["000.png", "001.png"]
        mock_exists.return_value = True # Fallback "002.png" exists
        result = file_operations.resolve_segment_image_filename_from_index(
            MOCK_APP_CONFIG, MOCK_USER_ID, MOCK_FOLDER_NAME, MOCK_PAGE_ID_BASE, "2" # Index 2 is out of bounds
        )
        self.assertEqual(result, "2.png") # Assumes fallback logic constructs this

    @patch('app.file_operations.os.path.isdir')
    @patch('app.file_operations.os.listdir')
    @patch('app.file_operations.os.path.exists')
    def test_resolve_segment_image_filename_from_index_value_error_fallback_direct(self, mock_exists, mock_listdir, mock_isdir):
        mock_isdir.return_value = True
        mock_listdir.return_value = ["segment_abc.png"]
        mock_exists.return_value = True # Fallback "segment_abc.png" exists
        result = file_operations.resolve_segment_image_filename_from_index(
            MOCK_APP_CONFIG, MOCK_USER_ID, MOCK_FOLDER_NAME, MOCK_PAGE_ID_BASE, "segment_abc"
        )
        self.assertEqual(result, "segment_abc.png")


    # --- Test save_uploaded_cropped_image ---
    @patch('app.file_operations.os.makedirs')
    @patch('app.file_operations.os.listdir')
    @patch('app.file_operations.base64.b64decode')
    @patch('app.file_operations.open', new_callable=mock_open)
    def test_save_uploaded_cropped_image_success_first_crop(self, mock_file, mock_b64decode, mock_listdir, mock_makedirs):
        mock_listdir.return_value = [] # No existing crops
        mock_b64decode.return_value = b"imagedata"
        image_data_url = "data:image/png;base64,SOMETHING"

        success, filename = file_operations.save_uploaded_cropped_image(
            MOCK_APP_CONFIG, MOCK_USER_ID, MOCK_FOLDER_NAME, "original.png", image_data_url
        )
        self.assertTrue(success)
        self.assertEqual(filename, "original_cropped_001.png")
        mock_makedirs.assert_called_once()
        mock_file.assert_called_once_with(os.path.join(file_operations.get_cropped_images_dir(MOCK_APP_CONFIG, MOCK_USER_ID, MOCK_FOLDER_NAME), "original_cropped_001.png"), "wb")
        mock_file().write.assert_called_once_with(b"imagedata")

    @patch('app.file_operations.os.makedirs')
    @patch('app.file_operations.os.listdir')
    @patch('app.file_operations.base64.b64decode')
    @patch('app.file_operations.open', new_callable=mock_open)
    def test_save_uploaded_cropped_image_success_next_crop(self, mock_file, mock_b64decode, mock_listdir, mock_makedirs):
        mock_listdir.return_value = ["original_cropped_001.png"] # Existing crop
        mock_b64decode.return_value = b"imagedata"
        image_data_url = "data:image/png;base64,SOMETHING"

        success, filename = file_operations.save_uploaded_cropped_image(
            MOCK_APP_CONFIG, MOCK_USER_ID, MOCK_FOLDER_NAME, "original.png", image_data_url
        )
        self.assertTrue(success)
        self.assertEqual(filename, "original_cropped_002.png")

    @patch('app.file_operations.os.makedirs')
    @patch('app.file_operations.base64.b64decode')
    def test_save_uploaded_cropped_image_b64_error(self, mock_b64decode, mock_makedirs):
        mock_b64decode.side_effect = base64.binascii.Error("B64 error")
        image_data_url = "data:image/png;base64,INVALID"
        success, result_msg = file_operations.save_uploaded_cropped_image(
            MOCK_APP_CONFIG, MOCK_USER_ID, MOCK_FOLDER_NAME, "original.png", image_data_url
        )
        self.assertFalse(success)
        self.assertIn("B64 error", result_msg)


    # --- Test save_ocr_processed_data ---
    # This is a complex function, testing a few key scenarios
    @patch('app.file_operations.os.makedirs')
    @patch('app.file_operations.open', new_callable=mock_open)
    @patch('app.file_operations.json.dump')
    @patch('app.file_operations.base64.b64decode')
    def test_save_ocr_processed_data_list_payload_with_image(self, mock_b64decode, mock_json_dump, mock_file, mock_makedirs):
        ocr_payload = [{
            "text": "Segment 1 text",
            "coords": [1,2,3,4],
            "image_data": "data:image/png;base64,testimgdata"
        }]
        mock_b64decode.return_value = b"decoded_image"

        summary, segments = file_operations.save_ocr_processed_data(
            MOCK_APP_CONFIG, MOCK_USER_ID, MOCK_FOLDER_NAME, MOCK_PAGE_ID_BASE, ocr_payload
        )
        self.assertEqual(summary, "Segment 1 text")
        self.assertEqual(len(segments), 1)
        self.assertEqual(segments[0]["id"], "000")
        self.assertTrue(segments[0]["has_image"])

        mock_makedirs.assert_any_call(file_operations.get_combined_segments_parent_dir(MOCK_APP_CONFIG, MOCK_USER_ID, MOCK_FOLDER_NAME), exist_ok=True)
        mock_makedirs.assert_any_call(file_operations.get_page_segments_dir(MOCK_APP_CONFIG, MOCK_USER_ID, MOCK_FOLDER_NAME, MOCK_PAGE_ID_BASE), exist_ok=True)

        # Check individual segment JSON save
        individual_json_path = os.path.join(file_operations.get_page_segments_dir(MOCK_APP_CONFIG, MOCK_USER_ID, MOCK_FOLDER_NAME, MOCK_PAGE_ID_BASE), "000.json")
        # Check image save
        image_path = os.path.join(file_operations.get_page_segments_dir(MOCK_APP_CONFIG, MOCK_USER_ID, MOCK_FOLDER_NAME, MOCK_PAGE_ID_BASE), "000.png")
        # Check combined JSON save
        combined_json_path = os.path.join(file_operations.get_combined_segments_parent_dir(MOCK_APP_CONFIG, MOCK_USER_ID, MOCK_FOLDER_NAME), f"{MOCK_PAGE_ID_BASE}.json")

        # Verify open calls for writing files
        mock_file.assert_any_call(individual_json_path, "w")
        mock_file.assert_any_call(image_path, "wb")
        mock_file.assert_any_call(combined_json_path, "w")

        mock_json_dump.assert_any_call(segments[0], mock_file()) # For individual
        mock_json_dump.assert_any_call(segments, mock_file())    # For combined
        mock_b64decode.assert_called_once_with("testimgdata")
        mock_file().write.assert_any_call(b"decoded_image")


    @patch('app.file_operations.os.makedirs')
    @patch('app.file_operations.open', new_callable=mock_open)
    @patch('app.file_operations.json.dump')
    def test_save_ocr_processed_data_status_dict_payload(self, mock_json_dump, mock_file, mock_makedirs):
        ocr_payload = {"status": "Processing", "file": "/path/to/file.png"}
        summary, segments = file_operations.save_ocr_processed_data(
            MOCK_APP_CONFIG, MOCK_USER_ID, MOCK_FOLDER_NAME, MOCK_PAGE_ID_BASE, ocr_payload
        )
        self.assertEqual(summary, "OCR status: Processing")
        self.assertEqual(len(segments), 1)
        self.assertEqual(segments[0]["id"], "000")
        self.assertFalse(segments[0]["has_image"])
        self.assertEqual(segments[0]["file"], "/path/to/file.png")
        # Combined JSON should still be saved
        combined_json_path = os.path.join(file_operations.get_combined_segments_parent_dir(MOCK_APP_CONFIG, MOCK_USER_ID, MOCK_FOLDER_NAME), f"{MOCK_PAGE_ID_BASE}.json")
        mock_file.assert_any_call(combined_json_path, "w")
        mock_json_dump.assert_any_call(segments, mock_file())


    # --- Test copy_files_to_ocr_folder ---
    @patch('app.file_operations.os.makedirs')
    @patch('app.file_operations.os.path.exists')
    @patch('app.file_operations.os.listdir')
    @patch('app.file_operations.shutil.copy2')
    def test_copy_files_to_ocr_folder_with_crops(self, mock_copy, mock_listdir, mock_path_exists, mock_makedirs):
        source_doc_dir = file_operations.get_document_dir(MOCK_APP_CONFIG, MOCK_USER_ID, MOCK_FOLDER_NAME)
        source_cropped_dir = file_operations.get_cropped_images_dir(MOCK_APP_CONFIG, MOCK_USER_ID, MOCK_FOLDER_NAME)
        dest_toocr_dir = file_operations.get_toocr_dir(MOCK_APP_CONFIG, MOCK_USER_ID, MOCK_FOLDER_NAME)

        mock_path_exists.return_value = True # Both source_cropped_dir and source_doc_dir exist

        def listdir_side_effect(path):
            if path == source_doc_dir:
                return ["001.png", "002.png"] # Original page names
            if path == source_cropped_dir:
                return ["001_cropped_001.png", "001_cropped_002.png"] # Cropped files for page 001
            return []
        mock_listdir.side_effect = listdir_side_effect

        success, count = file_operations.copy_files_to_ocr_folder(MOCK_APP_CONFIG, MOCK_USER_ID, MOCK_FOLDER_NAME)

        self.assertTrue(success)
        self.assertEqual(count, 3) # 2 crops for page 001, 1 original for page 002
        mock_makedirs.assert_called_once_with(dest_toocr_dir, exist_ok=True)

        expected_calls = [
            call(os.path.join(source_cropped_dir, "001_cropped_001.png"), os.path.join(dest_toocr_dir, "page_001_crop_001.png")),
            call(os.path.join(source_cropped_dir, "001_cropped_002.png"), os.path.join(dest_toocr_dir, "page_001_crop_002.png")),
            call(os.path.join(source_doc_dir, "002.png"), os.path.join(dest_toocr_dir, "page_002.png")),
        ]
        mock_copy.assert_has_calls(expected_calls, any_order=True)


if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)

````

**Explanation and How to Run:**

1.  **Save the Code:** Save the test code above as `test_file_operations.py` in a location where Python can find it and the `app.file_operations` module (e.g., in a `tests` subdirectory, and ensure your `PYTHONPATH` is set up, or place it alongside `file_operations.py` if your project structure is flat).
2.  **`MOCK_APP_CONFIG`**: This dictionary simulates the `current_app.config` used in your routes.
3.  **`@patch`**: This decorator from `unittest.mock` is used extensively to replace objects (like `os.path.exists`, `os.listdir`, `open`, `json.load`, `logger`, etc.) with `MagicMock` objects. This allows you to:
    - Control their return values.
    - Assert that they were called with the correct arguments.
    - Prevent actual file system operations or external calls.
4.  **`mock_open`**: Used to mock the built-in `open()` function, allowing you to simulate reading from and writing to files without touching the disk.
5.  **`setUp` and `tearDown`**: Used here to patch the `logger` for all test methods in the class, ensuring it's reset for each test.
6.  **Test Cases**:
    - **Path Helpers**: Verify that the path construction functions return the expected string paths.
    - **`get_segment_image_file_details`**: Tests scenarios like file/directory existence and different ID formats.
    - **`fetch_page_segments_data`**: Tests successful data loading, directory not found, and JSON errors.
    - **`resolve_segment_image_filename_from_index`**: Tests index logic, fallbacks, and error conditions.
    - **`save_uploaded_cropped_image`**: Tests filename generation for new and subsequent crops, and error handling.
    - **`save_ocr_processed_data`**: Tests different OCR payload types and verifies that file creation/writing is attempted correctly. This is a complex function, so these tests cover key paths.
    - **`copy_files_to_ocr_folder`**: Tests the logic for copying originals and prioritizing cropped versions.
7.  **Assertions**: Methods like `assertEqual`, `assertTrue`, `assertIsNone`, `assert_called_once_with`, `assert_any_call`, `assert_has_calls` are used to verify the behavior of the functions.
8.  **Running the Tests**:
    You can run these tests from your terminal. Navigate to the directory containing `test_file_operations.py` (or your main project directory if your test runner can discover them) and run:
    ```bash
    python -m unittest test_file_operations.py
    ```
    Or, if you have a test runner like `pytest` or use your IDE's test runner, you can use that.

This suite provides a good starting point for testing your `file_operations.py` module. You can expand it further by adding more specific scenarios or edge cases as needed.
