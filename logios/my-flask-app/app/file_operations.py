import os
import json
import base64
import re
import shutil
from collections import defaultdict
from loguru import logger
from datetime import datetime
from werkzeug.utils import secure_filename

# Attempt to import pdf2image, but allow it to fail gracefully if not installed
# The actual conversion function will need to handle this.
try:
    from pdf2image import convert_from_path
except ImportError:
    convert_from_path = None

# --- Path Helper Functions (Revised and New) ---


def get_upload_root(app_config):
    """Root directory for all user uploads."""
    return app_config["UPLOAD_FOLDER"]


def get_user_upload_dir(app_config, user_id):
    """Directory for a specific user's uploads."""
    return os.path.join(get_upload_root(app_config), str(user_id))


def get_document_dir(app_config, user_id, document_folder_name: str):
    """
    Base directory for a specific document.
    e.g., /UPLOAD_FOLDER/<user_id>/<pdf_file_basename_X>
    """
    return os.path.join(get_user_upload_dir(app_config, user_id), document_folder_name)


def get_document_png_dir(app_config, user_id, document_folder_name):
    """
    Directory for PNGs (originals and crops) within a document folder.
    e.g., /UPLOAD_FOLDER/<user_id>/<pdf_file_basename_X>/PNG
    This is used before marking as completed.
    """
    return os.path.join(
        get_document_dir(app_config, user_id, document_folder_name), "PNG"
    )


def get_document_completed_dir(app_config, user_id, document_folder_name):
    """
    The 'TOOCR' subfolder indicating a document's editing is completed.
    Files are moved here.
    e.g., /UPLOAD_FOLDER/<user_id>/<pdf_file_basename_X>/TOOCR
    """
    return os.path.join(
        get_document_dir(app_config, user_id, document_folder_name), "TOOCR"
    )


def get_ocr_source_png_dir(
    app_config, user_id, document_folder_name, is_completed=True
):
    """
    Directory containing PNG files to be used for OCR processing.
    If is_completed=True, looks in TOOCR/PNG directory.
    If is_completed=False, looks in PNG directory.

    Args:
        app_config: Flask app configuration
        user_id: The ID of the user
        document_folder_name: Name of the document folder
        is_completed: Whether to look in completed (TOOCR) or in-progress directory

    Returns:
        str: Path to the PNG directory containing OCR source images
    """
    if is_completed:
        return os.path.join(
            get_document_completed_dir(app_config, user_id, document_folder_name), "PNG"
        )
    else:
        return get_document_png_dir(app_config, user_id, document_folder_name)


# --- Task 1: PDF File Upload and Folder Creation ---


def handle_pdf_upload(app_config, user_id, uploaded_pdf_file_storage):
    """
    Handles PDF upload, creates a unique document folder, and saves the PDF.
    Args:
        app_config: Flask app configuration.
        user_id: The ID of the user.
        uploaded_pdf_file_storage: The FileStorage object from Flask request.
    Returns:
        tuple: (document_folder_path, saved_pdf_path, document_folder_name) or (None, None, None) on error.
    """

    user_dir = get_user_upload_dir(app_config, user_id)
    os.makedirs(user_dir, exist_ok=True)

    logger.info(f"User Dir = {user_dir}")

    original_filename = (
        uploaded_pdf_file_storage.filename
    )  # secure_filename(uploaded_pdf_file_storage.filename)
    pdf_basename = os.path.splitext(original_filename)[0]
    logger.info(
        f"PDF BASE NAME : {pdf_basename}, ORIGINAL FILENAME = {uploaded_pdf_file_storage.filename}"
    )

    document_folder_name = pdf_basename
    document_folder_path = get_document_dir(app_config, user_id, document_folder_name)
    logger.info(f"document folder path : {document_folder_path}")

    counter = 1
    while os.path.exists(document_folder_path):
        counter += 1
        document_folder_name = f"{pdf_basename}_{counter}"
        document_folder_path = get_document_dir(
            app_config, user_id, document_folder_name
        )

    try:
        os.makedirs(document_folder_path, exist_ok=True)
        saved_pdf_path = os.path.join(
            document_folder_path, original_filename
        )  # Use original filename for the PDF itself
        uploaded_pdf_file_storage.save(saved_pdf_path)
        logger.info(f"Created document folder: {document_folder_path}")
        logger.info(f"Saved PDF to: {saved_pdf_path}")
        return document_folder_path, saved_pdf_path, document_folder_name
    except Exception as e:
        logger.error(
            f"Error creating folder or saving PDF for {original_filename}: {e}"
        )
        return None, None, None


def sanitize_folder_name(folder_name):
    """
    Sanitize folder name to ensure it's safe for file system use.
    Args:
        folder_name (str): The raw folder name to sanitize.
    Returns:
        str: Sanitized folder name safe for file system use.
    """
    if not folder_name:
        return "untitled"

    # Remove/replace problematic characters
    sanitized = re.sub(
        r'[<>:"/\\|?*]', "", folder_name
    )  # Remove Windows-forbidden chars
    sanitized = re.sub(
        r"[^\w\s\-_.]", "", sanitized
    )  # Keep only alphanumeric, spaces, hyphens, underscores, dots
    sanitized = re.sub(r"\s+", "_", sanitized)  # Replace spaces with underscores
    sanitized = re.sub(
        r"_{2,}", "_", sanitized
    )  # Replace multiple underscores with single
    sanitized = sanitized.strip("._")  # Remove leading/trailing dots and underscores

    # Ensure it's not empty and not too long
    if not sanitized:
        sanitized = "untitled"
    if len(sanitized) > 50:  # Reasonable limit for folder names
        sanitized = sanitized[:50]

    return sanitized.lower()


def create_folder_name_from_filename(filename):
    """
    Create a folder name from the uploaded filename according to requirements:
    - Remove file extension
    - Limit to 20 characters maximum
    - Replace spaces with underscores
    - Sanitize for filesystem safety

    Args:
        filename (str): The original filename

    Returns:
        str: Sanitized folder name (max 20 chars, spaces as underscores)
    """
    if not filename:
        return "untitled"

    # Remove file extension
    base_name = os.path.splitext(filename)[0]

    # Limit to 20 characters
    if len(base_name) > 20:
        base_name = base_name[:20]

    # Replace spaces with underscores
    base_name = base_name.replace(" ", "_")

    # Apply general sanitization
    sanitized = sanitize_folder_name(base_name)

    # Ensure it's not longer than 20 characters after sanitization
    if len(sanitized) > 20:
        sanitized = sanitized[:20]

    return sanitized


def handle_pdf_upload_with_custom_name(
    app_config, user_id, uploaded_pdf_file_storage, custom_folder_name
):
    """
    Handles PDF upload with a custom folder name, creates a unique document folder, and saves the PDF.

    This follows the instructions for Task 1: PDF File Upload and Folder Creation.
    Creates folder structure: /<user_id>/<custom_folder_name> or /<user_id>/<custom_folder_name_X>

    Args:
        app_config: Flask app configuration.
        user_id: The ID of the user.
        uploaded_pdf_file_storage: The FileStorage object from Flask request.
        custom_folder_name: Custom name for the document folder (will be sanitized).

    Returns:
        tuple: (document_folder_path, saved_pdf_path, document_folder_name) or (None, None, None) on error.
    """
    try:
        # Ensure user directory exists
        user_dir = get_user_upload_dir(app_config, user_id)
        os.makedirs(user_dir, exist_ok=True)
        logger.info(f"User directory: {user_dir}")

        # Get and secure the original filename
        original_filename = secure_filename(uploaded_pdf_file_storage.filename)
        if not original_filename:
            logger.error("Invalid or empty filename")
            return None, None, None

        # Sanitize the custom folder name
        base_folder_name = sanitize_folder_name(custom_folder_name)
        logger.info(
            f"Original custom folder name: '{custom_folder_name}' -> Sanitized: '{base_folder_name}'"
        )

        # Determine unique folder name following Task 1 instructions
        # Start with base name, then try base_name_2, base_name_3, etc.
        document_folder_name = base_folder_name
        document_folder_path = get_document_dir(
            app_config, user_id, document_folder_name
        )

        counter = 1
        while os.path.exists(document_folder_path):
            counter += 1
            document_folder_name = f"{base_folder_name}_{counter}"
            document_folder_path = get_document_dir(
                app_config, user_id, document_folder_name
            )
            logger.info(
                f"Folder '{base_folder_name}' exists, trying: {document_folder_name}"
            )

        # Create the unique document folder
        os.makedirs(document_folder_path, exist_ok=True)
        logger.info(f"Created document folder: {document_folder_path}")

        # Save the PDF file in the document folder with original filename
        saved_pdf_path = os.path.join(document_folder_path, original_filename)
        uploaded_pdf_file_storage.save(saved_pdf_path)
        logger.info(f"Saved PDF to: {saved_pdf_path}")

        # Verify the file was saved successfully
        if not os.path.exists(saved_pdf_path):
            logger.error(f"PDF file was not saved successfully: {saved_pdf_path}")
            return None, None, None

        file_size = os.path.getsize(saved_pdf_path)
        logger.info(f"PDF file saved successfully. Size: {file_size} bytes")

        return document_folder_path, saved_pdf_path, document_folder_name

    except Exception as e:
        logger.error(
            f"Error in handle_pdf_upload_with_custom_name for file '{uploaded_pdf_file_storage.filename}': {str(e)}"
        )

        # Cleanup on error - remove any partially created folders
        try:
            if "document_folder_path" in locals() and os.path.exists(
                document_folder_path
            ):
                import shutil

                shutil.rmtree(document_folder_path)
                logger.info(
                    f"Cleaned up partially created folder: {document_folder_path}"
                )
        except Exception as cleanup_error:
            logger.error(f"Error during cleanup: {cleanup_error}")

        return None, None, None


def handle_pdf_upload_with_auto_folder(app_config, user_id, uploaded_pdf_file_storage):
    """
    Handles PDF upload with automatic folder name creation based on filename.
    Creates folder structure with PNG and TOOCR subfolders.

    Requirements:
    - Folder name based on PDF filename (without extension)
    - Max 20 characters, spaces replaced with underscores
    - Creates PNG subfolder for converted pages
    - Creates TOOCR subfolder for processed images

    Args:
        app_config: Flask app configuration.
        user_id: The ID of the user.
        uploaded_pdf_file_storage: The FileStorage object from Flask request.

    Returns:
        tuple: (document_folder_path, saved_pdf_path, document_folder_name) or (None, None, None) on error.
    """
    try:
        # Ensure user directory exists
        user_dir = get_user_upload_dir(app_config, user_id)
        os.makedirs(user_dir, exist_ok=True)
        logger.info(f"User directory: {user_dir}")

        # Get and secure the original filename
        original_filename = secure_filename(uploaded_pdf_file_storage.filename)
        if not original_filename:
            logger.error("Invalid or empty filename")
            return None, None, None

        # Create folder name from filename according to requirements
        base_folder_name = create_folder_name_from_filename(original_filename)
        logger.info(
            f"Original filename: '{original_filename}' -> Folder name: '{base_folder_name}'"
        )

        # Determine unique folder name
        document_folder_name = base_folder_name
        document_folder_path = get_document_dir(
            app_config, user_id, document_folder_name
        )

        counter = 1
        while os.path.exists(document_folder_path):
            counter += 1
            document_folder_name = f"{base_folder_name}_{counter}"
            document_folder_path = get_document_dir(
                app_config, user_id, document_folder_name
            )
            logger.info(
                f"Folder '{base_folder_name}' exists, trying: {document_folder_name}"
            )

        # Create the unique document folder
        os.makedirs(document_folder_path, exist_ok=True)
        logger.info(f"Created document folder: {document_folder_path}")

        # Create PNG subfolder for converted pages
        png_dir = get_document_png_dir(app_config, user_id, document_folder_name)
        os.makedirs(png_dir, exist_ok=True)
        logger.info(f"Created PNG subfolder: {png_dir}")

        # Create TOOCR subfolder for processed images
        toocr_dir = get_document_completed_dir(
            app_config, user_id, document_folder_name
        )
        os.makedirs(toocr_dir, exist_ok=True)
        logger.info(f"Created TOOCR subfolder: {toocr_dir}")

        # Save the PDF file in the document folder with original filename
        saved_pdf_path = os.path.join(document_folder_path, original_filename)
        uploaded_pdf_file_storage.save(saved_pdf_path)
        logger.info(f"Saved PDF to: {saved_pdf_path}")

        # Verify the file was saved successfully
        if not os.path.exists(saved_pdf_path):
            logger.error(f"PDF file was not saved successfully: {saved_pdf_path}")
            return None, None, None

        file_size = os.path.getsize(saved_pdf_path)
        logger.info(f"PDF file saved successfully. Size: {file_size} bytes")

        return document_folder_path, saved_pdf_path, document_folder_name

    except Exception as e:
        logger.error(
            f"Error in handle_pdf_upload_with_auto_folder for file '{uploaded_pdf_file_storage.filename}': {str(e)}"
        )

        # Cleanup on error - remove any partially created folders
        try:
            if "document_folder_path" in locals() and os.path.exists(
                document_folder_path
            ):
                shutil.rmtree(document_folder_path)
                logger.info(
                    f"Cleaned up partially created folder: {document_folder_path}"
                )
        except Exception as cleanup_error:
            logger.error(f"Error during cleanup: {cleanup_error}")

        return None, None, None


def handle_pdf_upload_with_auto_folder_no_sanitize(
    app_config, user_id, uploaded_pdf_file_storage
):
    """
    Handles PDF upload with automatic folder name creation based on filename.
    DOES NOT sanitize the filename - stores original filename as provided.
    Creates folder structure with PNG and TOOCR subfolders.

    Requirements:
    - Folder name based on PDF filename (without extension)
    - Max 20 characters, spaces replaced with underscores
    - Creates PNG subfolder for converted pages
    - Creates TOOCR subfolder for processed images
    - Original filename is preserved without sanitization

    Args:
        app_config: Flask app configuration.
        user_id: The ID of the user.
        uploaded_pdf_file_storage: The FileStorage object from Flask request.

    Returns:
        tuple: (document_folder_path, saved_pdf_path, document_folder_name) or (None, None, None) on error.
    """
    try:
        # Ensure user directory exists
        user_dir = get_user_upload_dir(app_config, user_id)
        os.makedirs(user_dir, exist_ok=True)
        logger.info(f"User directory: {user_dir}")

        # Get the original filename WITHOUT sanitization
        original_filename = uploaded_pdf_file_storage.filename
        if not original_filename:
            logger.error("Invalid or empty filename")
            return None, None, None

        # Create folder name from filename (this still needs sanitization for folder names)
        base_folder_name = create_folder_name_from_filename(original_filename)
        logger.info(
            f"Original filename: '{original_filename}' -> Folder name: '{base_folder_name}'"
        )

        # Determine unique folder name
        document_folder_name = base_folder_name
        document_folder_path = get_document_dir(
            app_config, user_id, document_folder_name
        )

        counter = 1
        while os.path.exists(document_folder_path):
            counter += 1
            document_folder_name = f"{base_folder_name}_{counter}"
            document_folder_path = get_document_dir(
                app_config, user_id, document_folder_name
            )
            logger.info(
                f"Folder '{base_folder_name}' exists, trying: {document_folder_name}"
            )

        # Create the unique document folder
        os.makedirs(document_folder_path, exist_ok=True)
        logger.info(f"Created document folder: {document_folder_path}")

        # Create PNG subfolder for converted pages
        png_dir = get_document_png_dir(app_config, user_id, document_folder_name)
        os.makedirs(png_dir, exist_ok=True)
        logger.info(f"Created PNG subfolder: {png_dir}")

        # Create TOOCR subfolder for processed images
        toocr_dir = get_document_completed_dir(
            app_config, user_id, document_folder_name
        )
        os.makedirs(toocr_dir, exist_ok=True)
        logger.info(f"Created TOOCR subfolder: {toocr_dir}")

        # Save the PDF file in the document folder with ORIGINAL filename (no sanitization)
        saved_pdf_path = os.path.join(document_folder_path, original_filename)
        uploaded_pdf_file_storage.save(saved_pdf_path)
        logger.info(f"Saved PDF to: {saved_pdf_path}")

        # Verify the file was saved successfully
        if not os.path.exists(saved_pdf_path):
            logger.error(f"PDF file was not saved successfully: {saved_pdf_path}")
            return None, None, None

        file_size = os.path.getsize(saved_pdf_path)
        logger.info(f"PDF file saved successfully. Size: {file_size} bytes")

        return document_folder_path, saved_pdf_path, document_folder_name

    except Exception as e:
        logger.error(
            f"Error in handle_pdf_upload_with_auto_folder_no_sanitize for file '{uploaded_pdf_file_storage.filename}': {str(e)}"
        )

        # Cleanup on error - remove any partially created folders
        try:
            if "document_folder_path" in locals() and os.path.exists(
                document_folder_path
            ):
                shutil.rmtree(document_folder_path)
                logger.info(
                    f"Cleaned up partially created folder: {document_folder_path}"
                )
        except Exception as cleanup_error:
            logger.error(f"Error during cleanup: {cleanup_error}")

        return None, None, None


# --- Task 2: PDF to PNG Conversion ---


def convert_pdf_to_png_pages(app_config, user_id, document_folder_name, pdf_file_path):
    """
    Converts a PDF to PNG images, saving them in a 'PNG' subfolder.
    Args:
        app_config: Flask app configuration.
        user_id: The ID of the user.
        document_folder_name: The name of the document's base folder (e.g., mydoc or mydoc_2).
        pdf_file_path: Full path to the PDF file to convert.
    Returns:
        tuple: (success_boolean, message_or_list_of_png_paths)
    """
    if not convert_from_path:
        logger.error("pdf2image library is not installed. Cannot convert PDF to PNG.")
        return False, "PDF conversion library not available."

    png_dir = get_document_png_dir(app_config, user_id, document_folder_name)
    os.makedirs(png_dir, exist_ok=True)

    try:
        images = convert_from_path(pdf_file_path, dpi=200)  # Adjust DPI as needed
        saved_png_paths = []
        for i, image in enumerate(images):
            png_filename = f"page_{(i + 1):03d}.png"
            png_path = os.path.join(png_dir, png_filename)
            image.save(png_path, "PNG")
            saved_png_paths.append(png_path)
        logger.info(
            f"Converted PDF {pdf_file_path} to {len(saved_png_paths)} PNGs in {png_dir}"
        )
        return True, saved_png_paths
    except Exception as e:
        logger.error(f"Error converting PDF {pdf_file_path} to PNGs: {e}")
        return False, str(e)


# --- Task 3: Image Cropping and Storage ---


def save_cropped_page_image(
    app_config,
    user_id,
    document_folder_name,
    original_page_filename_base,
    cropped_image_data_url,
):
    """
    Saves a base64 encoded cropped image to the document's PNG folder.
    Args:
        app_config: Flask app configuration.
        user_id: The ID of the user.
        document_folder_name: The name of the document's base folder.
        original_page_filename_base: Base name of the original page (e.g., "page_001").
        cropped_image_data_url: Base64 encoded image data.
    Returns:
        tuple: (success_boolean, filename_or_error_message)
    """
    png_dir = get_document_png_dir(app_config, user_id, document_folder_name)
    if not os.path.exists(png_dir):  # Should have been created by PDF conversion
        os.makedirs(png_dir, exist_ok=True)
        logger.warning(
            f"PNG directory {png_dir} did not exist, created it. This might indicate an issue if PDF conversion didn't run."
        )

    # Determine next crop number
    existing_crops = [
        f
        for f in os.listdir(png_dir)
        if f.startswith(f"{original_page_filename_base}_crop_")
        and f.lower().endswith(".png")
    ]
    crop_numbers = []
    for cf in existing_crops:
        match = re.search(r"_crop_(\d{3})\.png$", cf, re.IGNORECASE)
        if match:
            crop_numbers.append(int(match.group(1)))

    next_crop_num = max(crop_numbers) + 1 if crop_numbers else 1
    cropped_filename = f"{original_page_filename_base}_crop_{next_crop_num:03d}.png"
    cropped_file_path = os.path.join(png_dir, cropped_filename)

    try:
        image_data_b64 = re.sub(r"^data:image/\w+;base64,", "", cropped_image_data_url)
        binary_data = base64.b64decode(image_data_b64)
        with open(cropped_file_path, "wb") as f:
            f.write(binary_data)
        logger.info(f"Saved cropped image to {cropped_file_path}")
        return True, cropped_filename
    except Exception as e:
        logger.error(f"Error saving cropped image {cropped_filename}: {str(e)}")
        return False, str(e)


# --- Task 4: Marking Editing as Completed ---


def mark_document_editing_completed(app_config, user_id, document_folder_name):
    """
    Marks document editing as completed by moving its contents into a 'TOOCR' subfolder.
    Args:
        app_config: Flask app configuration.
        user_id: The ID of the user.
        document_folder_name: The name of the document's base folder.
    Returns:
        tuple: (success_boolean, message)
    """
    doc_base_dir = get_document_dir(app_config, user_id, document_folder_name)
    completed_dir = get_document_completed_dir(
        app_config, user_id, document_folder_name
    )

    if not os.path.isdir(doc_base_dir):
        logger.error(f"Document base directory not found: {doc_base_dir}")
        return False, "Document directory not found."

    if os.path.exists(completed_dir):
        logger.warning(
            f"Document {document_folder_name} seems to be already marked as completed (TOOCR folder exists). Archiving existing TOOCR folder."
        )
        # Simple archive: rename existing TOOCR to TOOCR_old_timestamp
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        try:
            shutil.move(completed_dir, f"{completed_dir}_old_{timestamp}")
        except Exception as e_archive:
            logger.error(
                f"Could not archive existing TOOCR folder {completed_dir}: {e_archive}"
            )
            # Decide if this is a critical failure or if we can proceed
            # For now, let's proceed with creating a new TOOCR folder if archiving failed but it's gone
            if os.path.exists(completed_dir):  # Check again if it's still there
                return False, f"Could not archive existing TOOCR folder: {e_archive}"

    os.makedirs(completed_dir, exist_ok=True)

    items_to_move = os.listdir(doc_base_dir)
    moved_count = 0
    errors = []

    for item_name in items_to_move:
        source_item_path = os.path.join(doc_base_dir, item_name)
        destination_item_path = os.path.join(completed_dir, item_name)

        if item_name == "TOOCR":  # Don't try to move the TOOCR folder into itself
            continue

        try:
            shutil.move(source_item_path, destination_item_path)
            logger.info(f"Moved {source_item_path} to {destination_item_path}")
            moved_count += 1
        except Exception as e:
            logger.error(f"Error moving {source_item_path} to {completed_dir}: {e}")
            errors.append(f"Could not move {item_name}: {e}")

    if errors:
        return (
            False,
            f"Completed moving {moved_count} items to {completed_dir} with errors: {'; '.join(errors)}",
        )

    logger.info(
        f"Successfully moved {moved_count} items into {completed_dir} for document {document_folder_name}."
    )
    return (
        True,
        f"Document {document_folder_name} marked as completed. Contents moved to TOOCR subfolder.",
    )


def get_user_documents_list(
    app_config, user_id, include_completed=True, include_in_progress=True
):
    """
    Get a list of all documents for a user, categorized by completion status.

    Args:
        app_config: Flask app configuration.
        user_id: The ID of the user.
        include_completed: Include documents that have been marked as completed (TOOCR folder exists).
        include_in_progress: Include documents that are still being edited (no TOOCR folder).

    Returns:
        dict: {
            'completed': [list of completed document folder names],
            'in_progress': [list of in-progress document folder names],
            'all': [list of all document folder names]
        }
    """
    user_dir = get_user_upload_dir(app_config, user_id)
    documents = {"completed": [], "in_progress": [], "all": []}

    if not os.path.exists(user_dir):
        logger.info(f"User directory does not exist: {user_dir}")
        return documents

    try:
        # Get all directories in user folder
        all_items = os.listdir(user_dir)
        document_folders = [
            item for item in all_items if os.path.isdir(os.path.join(user_dir, item))
        ]

        logger.info(
            f"Found {len(document_folders)} document folders for user {user_id}"
        )

        for doc_folder in document_folders:
            doc_path = get_document_dir(app_config, user_id, doc_folder)
            toocr_path = get_document_completed_dir(app_config, user_id, doc_folder)

            # Check if TOOCR folder exists to determine completion status
            # According to instructions: completed documents have files moved to TOOCR subfolder
            is_completed = os.path.exists(toocr_path) and os.path.isdir(toocr_path)

            if is_completed:
                if include_completed:
                    documents["completed"].append(doc_folder)
                    logger.debug(
                        f"Document {doc_folder} is completed (TOOCR folder exists)"
                    )
            else:
                if include_in_progress:
                    documents["in_progress"].append(doc_folder)
                    logger.debug(
                        f"Document {doc_folder} is in progress (no TOOCR folder)"
                    )

            # Always add to 'all' list regardless of filters
            documents["all"].append(doc_folder)

        # Sort all lists alphabetically for consistent ordering
        documents["completed"].sort()
        documents["in_progress"].sort()
        documents["all"].sort()

        logger.info(
            f"User {user_id} documents: {len(documents['completed'])} completed, {len(documents['in_progress'])} in progress, {len(documents['all'])} total"
        )

    except Exception as e:
        logger.error(f"Error listing documents for user {user_id}: {e}")
        # Return empty structure on error
        documents = {"completed": [], "in_progress": [], "all": []}

    return documents


def get_user_workspace_info(app_config, user_id):
    """
    Get comprehensive workspace information for a user.

    Args:
        app_config: Flask app configuration.
        user_id: The ID of the user.

    Returns:
        dict: Workspace information including document counts and storage usage.
    """
    user_dir = get_user_upload_dir(app_config, user_id)
    documents_data = get_user_documents_list(app_config, user_id)

    workspace_info = {
        "user_id": user_id,
        "user_directory": user_dir,
        "directory_exists": os.path.exists(user_dir),
        "total_documents": len(documents_data["all"]),
        "completed_documents": len(documents_data["completed"]),
        "in_progress_documents": len(documents_data["in_progress"]),
        "documents": documents_data,
        "storage_usage": 0,
        "last_activity": None,
    }

    if workspace_info["directory_exists"]:
        try:
            # Calculate storage usage
            total_size = 0
            latest_modification = 0

            for root, dirs, files in os.walk(user_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    if os.path.exists(file_path):
                        file_size = os.path.getsize(file_path)
                        total_size += file_size

                        file_mtime = os.path.getmtime(file_path)
                        if file_mtime > latest_modification:
                            latest_modification = file_mtime

            workspace_info["storage_usage"] = total_size
            if latest_modification > 0:
                workspace_info["last_activity"] = datetime.fromtimestamp(
                    latest_modification
                )

        except Exception as e:
            logger.error(f"Error calculating workspace info for user {user_id}: {e}")

    logger.info(
        f"Workspace info for user {user_id}: {workspace_info['total_documents']} documents, {workspace_info['storage_usage']} bytes"
    )

    return workspace_info


def get_document_info(app_config, user_id, document_folder_name):
    """
    Get detailed information about a specific document.

    This function analyzes the document's current state according to the file storage structure:
    - In-progress documents: files in /<user_id>/<document_folder_name>/
    - Completed documents: files moved to /<user_id>/<document_folder_name>/TOOCR/

    Args:
        app_config: Flask app configuration.
        user_id: The ID of the user.
        document_folder_name: The name of the document folder (e.g., 'mydoc' or 'mydoc_2').

    Returns:
        dict: Document information including status, file counts, dates, etc.
    """
    doc_dir = get_document_dir(app_config, user_id, document_folder_name)
    toocr_dir = get_document_completed_dir(app_config, user_id, document_folder_name)

    info = {
        "name": document_folder_name,
        "is_completed": False,
        "pdf_files": [],
        "png_files": [],
        "original_png_count": 0,  # Original pages (page_001.png, page_002.png, etc.)
        "crop_count": 0,  # Cropped images (page_001_crop_001.png, etc.)
        "total_png_count": 0,  # Total PNG files
        "creation_date": None,
        "completion_date": None,
        "document_path": doc_dir,
        "toocr_path": toocr_dir,
        "status": "unknown",
    }

    if not os.path.exists(doc_dir):
        logger.warning(f"Document directory does not exist: {doc_dir}")
        info["status"] = "missing"
        return info

    try:
        # Get creation date from folder
        info["creation_date"] = datetime.fromtimestamp(os.path.getctime(doc_dir))

        # Determine if document is completed by checking for TOOCR folder
        info["is_completed"] = os.path.exists(toocr_dir) and os.path.isdir(toocr_dir)

        if info["is_completed"]:
            # Document is completed - files are in TOOCR folder (Task 4 completed)
            info["status"] = "completed"
            info["completion_date"] = datetime.fromtimestamp(
                os.path.getctime(toocr_dir)
            )

            # Look for PDF files in TOOCR folder
            try:
                toocr_files = os.listdir(toocr_dir)
                info["pdf_files"] = [
                    f for f in toocr_files if f.lower().endswith(".pdf")
                ]
                logger.debug(
                    f"Found {len(info['pdf_files'])} PDF files in TOOCR folder"
                )
            except Exception as e:
                logger.error(f"Error listing TOOCR folder contents: {e}")

            # Look for PNG files in TOOCR/PNG folder
            png_dir = os.path.join(toocr_dir, "PNG")
            if os.path.exists(png_dir):
                try:
                    png_files = [
                        f for f in os.listdir(png_dir) if f.lower().endswith(".png")
                    ]
                    info["png_files"] = png_files

                    # Count original pages vs crops
                    # Original pages: page_001.png, page_002.png, etc.
                    # Crops: page_001_crop_001.png, page_002_crop_005.png, etc.
                    original_pages = [
                        f
                        for f in png_files
                        if re.match(r"^page_\d{3}\.png$", f, re.IGNORECASE)
                    ]
                    crop_images = [f for f in png_files if "_crop_" in f.lower()]

                    info["original_png_count"] = len(original_pages)
                    info["crop_count"] = len(crop_images)
                    info["total_png_count"] = len(png_files)

                    logger.debug(
                        f"Completed document {document_folder_name}: {info['original_png_count']} original pages, {info['crop_count']} crops"
                    )
                except Exception as e:
                    logger.error(
                        f"Error listing PNG files from completed document: {e}"
                    )
            else:
                logger.warning(f"PNG folder not found in completed document: {png_dir}")

        else:
            # Document is in progress - files are in main folder and PNG subfolder
            info["status"] = "in_progress"

            # Look for PDF files in main document folder
            try:
                doc_files = os.listdir(doc_dir)
                # Exclude the PNG and TOOCR subfolders from file listing
                doc_files = [
                    f for f in doc_files if not os.path.isdir(os.path.join(doc_dir, f))
                ]
                info["pdf_files"] = [f for f in doc_files if f.lower().endswith(".pdf")]
                logger.debug(
                    f"Found {len(info['pdf_files'])} PDF files in main document folder"
                )
            except Exception as e:
                logger.error(f"Error listing main document folder contents: {e}")

            # Look for PNG files in PNG subfolder (Task 2 and Task 3 outputs)
            png_dir = get_document_png_dir(app_config, user_id, document_folder_name)
            if os.path.exists(png_dir):
                try:
                    png_files = [
                        f for f in os.listdir(png_dir) if f.lower().endswith(".png")
                    ]
                    info["png_files"] = png_files

                    # Count original pages vs crops
                    original_pages = [
                        f
                        for f in png_files
                        if re.match(r"^page_\d{3}\.png$", f, re.IGNORECASE)
                    ]
                    crop_images = [f for f in png_files if "_crop_" in f.lower()]

                    info["original_png_count"] = len(original_pages)
                    info["crop_count"] = len(crop_images)
                    info["total_png_count"] = len(png_files)

                    logger.debug(
                        f"In-progress document {document_folder_name}: {info['original_png_count']} original pages, {info['crop_count']} crops"
                    )
                except Exception as e:
                    logger.error(
                        f"Error listing PNG files from in-progress document: {e}"
                    )
            else:
                logger.info(f"PNG folder not yet created for document: {png_dir}")

        # Additional document analysis
        if info["original_png_count"] == 0 and len(info["pdf_files"]) > 0:
            info["status"] = (
                "uploaded" if not info["is_completed"] else "completed_no_conversion"
            )
        elif info["original_png_count"] > 0 and info["crop_count"] == 0:
            info["status"] = "converted" if not info["is_completed"] else "completed"
        elif info["crop_count"] > 0:
            info["status"] = "edited" if not info["is_completed"] else "completed"

        logger.info(
            f"Document info for {document_folder_name}: Status={info['status']}, PDFs={len(info['pdf_files'])}, PNGs={info['total_png_count']} ({info['original_png_count']} originals, {info['crop_count']} crops)"
        )

    except Exception as e:
        logger.error(f"Error getting document info for {document_folder_name}: {e}")
        info["status"] = "error"

    return info


def get_document_summary_stats(app_config, user_id):
    """
    Get summary statistics for all documents of a user.

    Args:
        app_config: Flask app configuration.
        user_id: The ID of the user.

    Returns:
        dict: Summary statistics including counts by status, total files, etc.
    """
    documents_data = get_user_documents_list(app_config, user_id)

    stats = {
        "total_documents": len(documents_data["all"]),
        "completed_documents": len(documents_data["completed"]),
        "in_progress_documents": len(documents_data["in_progress"]),
        "total_pdfs": 0,
        "total_pngs": 0,
        "total_crops": 0,
        "status_breakdown": {
            "uploaded": 0,  # PDF uploaded, not converted
            "converted": 0,  # PDF converted to PNGs
            "edited": 0,  # Has cropped images
            "completed": 0,  # Marked as completed
            "error": 0,  # Error state
            "missing": 0,  # Missing files
        },
    }

    try:
        for doc_name in documents_data["all"]:
            doc_info = get_document_info(app_config, user_id, doc_name)

            stats["total_pdfs"] += len(doc_info["pdf_files"])
            stats["total_pngs"] += doc_info["total_png_count"]
            stats["total_crops"] += doc_info["crop_count"]

            status = doc_info["status"]
            if status in stats["status_breakdown"]:
                stats["status_breakdown"][status] += 1

        logger.info(f"Document summary for user {user_id}: {stats}")

    except Exception as e:
        logger.error(f"Error generating document summary stats for user {user_id}: {e}")

    return stats


# --- OCR Data Processing Functions ---


def save_ocr_processed_data(
    app_config,
    user_id,
    document_folder_name,
    page_image_filename_base,
    ocr_result_data,
    is_document_completed=True,
):
    """
    Save OCR processed segments data for a specific page.

    Args:
        app_config: Flask app configuration
        user_id: The ID of the user
        document_folder_name: Name of the document folder
        page_image_filename_base: Base filename of the page (e.g., "page_001" or "page_001_crop_001")
        ocr_result_data: OCR results data from OCR service
        is_document_completed: Whether document is in TOOCR folder

    Returns:
        tuple: (text_summary, segments_data_list)
    """
    try:
        if is_document_completed:
            base_dir = get_document_completed_dir(
                app_config, user_id, document_folder_name
            )
        else:
            base_dir = get_document_png_dir(app_config, user_id, document_folder_name)

        # Create segments directory for this page under PNG subdirectory
        # Pattern: /uploads/kostas/<document_name>/TOOCR/PNG/page_004_crop_002/
        png_subdir = os.path.join(base_dir, "PNG")
        segments_dir = os.path.join(png_subdir, page_image_filename_base)
        os.makedirs(segments_dir, exist_ok=True)

        segments_data_list = []
        text_summary = ""

        # Process each segment from OCR results
        if isinstance(ocr_result_data, dict) and "segments" in ocr_result_data:
            segments = ocr_result_data["segments"]
        elif isinstance(ocr_result_data, list):
            segments = ocr_result_data
        else:
            logger.warning(
                f"Unexpected OCR result data format: {type(ocr_result_data)}"
            )
            segments = []

        for i, segment in enumerate(segments):
            segment_id = f"{i:03d}"
            segment_file = os.path.join(segments_dir, f"{segment_id}.json")

            # Save individual segment data
            with open(segment_file, "w", encoding="utf-8") as f:
                json.dump(segment, f, ensure_ascii=False, indent=2)

            # Extract text for summary and client data
            segment_text = segment.get("text", "")
            text_summary += segment_text + "\n"

            # Prepare segment data for client
            segment_data = {
                "id": segment_id,
                "text": segment_text,
                "confidence": segment.get("confidence", 0.0),
                "bbox": segment.get("bbox", []),
                "file_path": segment_file,
            }
            segments_data_list.append(segment_data)

        # Save combined summary in the same directory as the segments
        summary_file = os.path.join(segments_dir, "summary.txt")

        with open(summary_file, "w", encoding="utf-8") as f:
            f.write(text_summary.strip())

        logger.info(
            f"Saved OCR data for {page_image_filename_base}: {len(segments_data_list)} segments"
        )
        return text_summary.strip(), segments_data_list

    except Exception as e:
        logger.error(
            f"Error saving OCR processed data for {page_image_filename_base}: {str(e)}"
        )
        return "", []


def fetch_page_segments_data(
    app_config,
    user_id,
    document_folder_name,
    page_image_filename_base,
    is_document_completed=True,
):
    """
    Fetch OCR segments data for a specific page that has been processed.

    Args:
        app_config: Flask app configuration
        user_id: The ID of the user
        document_folder_name: Name of the document folder
        page_image_filename_base: Base filename of the page (e.g., "page_001" or "page_001_crop_001")
        is_document_completed: Whether document is in TOOCR folder

    Returns:
        list: List of segment data dictionaries, or None if not found
    """
    try:
        if is_document_completed:
            base_dir = get_document_completed_dir(
                app_config, user_id, document_folder_name
            )
        else:
            base_dir = get_document_png_dir(app_config, user_id, document_folder_name)

        # Look for segments directory for this page under PNG subdirectory
        # Pattern: /uploads/kostas/<document_name>/TOOCR/PNG/page_004_crop_002/
        png_subdir = os.path.join(base_dir, "PNG")
        segments_dir = os.path.join(png_subdir, page_image_filename_base)

        if not os.path.exists(segments_dir):
            logger.warning(f"Segments directory not found: {segments_dir}")
            return None

        segments_data_list = []

        # Read all segment JSON files
        try:
            segment_files = [f for f in os.listdir(segments_dir) if f.endswith(".json")]
            segment_files.sort()  # Ensure proper order (000.json, 001.json, etc.)

            for segment_file in segment_files:
                segment_path = os.path.join(segments_dir, segment_file)
                segment_id = os.path.splitext(segment_file)[0]

                with open(segment_path, "r", encoding="utf-8") as f:
                    segment_data = json.load(f)

                # Extract text from the data structure
                text_content = ""
                if "text" in segment_data:
                    if isinstance(segment_data["text"], list):
                        text_content = " ".join(segment_data["text"])
                    else:
                        text_content = str(segment_data["text"])

                # Extract coordinates (handle both 'coords' and 'bbox' formats)
                coordinates = segment_data.get("coords", segment_data.get("bbox", []))

                # Prepare segment data for client
                client_segment_data = {
                    "id": segment_id,
                    "text": text_content,
                    "confidence": segment_data.get("confidence", 0.0),
                    "bbox": coordinates,
                    "file_path": segment_path,
                }
                segments_data_list.append(client_segment_data)

        except Exception as e:
            logger.error(f"Error reading segment files from {segments_dir}: {str(e)}")
            return None

        logger.info(
            f"Fetched {len(segments_data_list)} segments for {page_image_filename_base}"
        )
        return segments_data_list

    except Exception as e:
        logger.error(
            f"Error fetching page segments data for {page_image_filename_base}: {str(e)}"
        )
        return None


def get_segment_image_file_details(
    app_config,
    user_id,
    document_folder_name,
    page_image_filename_base,
    segment_id_str,
    is_document_completed=True,
):
    """
    Get the directory and filename for a specific segment image.

    Args:
        app_config: Flask app configuration
        user_id: The ID of the user
        document_folder_name: Name of the document folder
        page_image_filename_base: Base filename of the page (e.g., "page_001" or "page_001_crop_001")
        segment_id_str: Segment ID as string (e.g., "000", "001")
        is_document_completed: Whether document is in TOOCR folder

    Returns:
        tuple: (segment_dir, segment_image_filename) or (None, None) if not found
    """
    try:
        if is_document_completed:
            base_dir = get_document_completed_dir(
                app_config, user_id, document_folder_name
            )
        else:
            base_dir = get_document_png_dir(app_config, user_id, document_folder_name)

        # Segments are under PNG subdirectory with page name
        # Pattern: /uploads/kostas/<document_name>/TOOCR/PNG/page_004_crop_002/
        png_subdir = os.path.join(base_dir, "PNG")
        segments_dir = os.path.join(png_subdir, page_image_filename_base)

        # Segment image filename pattern: 000.png, 001.png, etc.
        segment_image_filename = f"{segment_id_str}.png"
        segment_image_path = os.path.join(segments_dir, segment_image_filename)

        if os.path.exists(segment_image_path):
            logger.info(f"Found segment image: {segment_image_path}")
            return segments_dir, segment_image_filename
        else:
            logger.warning(f"Segment image not found: {segment_image_path}")
            return None, None

    except Exception as e:
        logger.error(f"Error getting segment image file details: {str(e)}")
        return None, None
