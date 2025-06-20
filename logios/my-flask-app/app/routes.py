from flask import (
    Blueprint,
    jsonify,
    render_template,
    request,
    redirect,
    url_for,
    session,
    current_app,
    flash,
    send_from_directory,
)
from flask_login import login_required, current_user
from werkzeug.utils import secure_filename

import os
import re  # Still used in process_ocr for data from OCR service
import requests
import shutil

from loguru import logger
from collections import defaultdict

from . import file_operations

app = Blueprint("app", __name__)

# ALLOWED_EXTENSIONS can remain or be moved to app config
ALLOWED_EXTENSIONS = {"pdf", "png", "jpg", "jpeg"}


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/show_upload_page", methods=["POST", "GET"])
@login_required
def show_upload_page():
    if request.method == "POST":
        if "file" not in request.files:
            flash("No file part in the request.", "danger")
            return redirect(request.url)

        uploaded_file_storage = request.files["file"]
        if uploaded_file_storage.filename == "":
            flash("No selected file.", "warning")
            return redirect(request.url)

        if uploaded_file_storage and allowed_file(uploaded_file_storage.filename):
            from app.models import UploadedFile, db
            from datetime import datetime
            import mimetypes

            user_id = current_user.get_user_folder_name()
            original_filename = uploaded_file_storage.filename  # No sanitization

            # Task 1: Handle PDF Upload and Folder Creation with automatic folder name
            doc_folder_path, saved_pdf_path, doc_folder_name = (
                file_operations.handle_pdf_upload_with_auto_folder_no_sanitize(
                    current_app.config,
                    user_id,
                    uploaded_file_storage,
                )
            )

            if not doc_folder_path:
                flash(
                    "Error saving uploaded file or creating document folder.", "danger"
                )
                return redirect(request.url)

            current_app.logger.info(
                f"File uploaded to {saved_pdf_path} in document folder {doc_folder_name}"
            )

            # Get file information for database
            file_size = os.path.getsize(saved_pdf_path)
            file_type = "pdf" if original_filename.lower().endswith(".pdf") else "image"
            mime_type, _ = mimetypes.guess_type(original_filename)
            stored_filename = os.path.basename(saved_pdf_path)
            relative_file_path = os.path.relpath(
                saved_pdf_path, current_app.config["UPLOAD_FOLDER"]
            )

            # Create database record for uploaded file
            try:
                uploaded_file_record = UploadedFile(
                    user_id=current_user.id,
                    original_filename=original_filename,
                    stored_filename=stored_filename,
                    file_type=file_type,
                    file_size=file_size,
                    mime_type=mime_type,
                    document_folder=doc_folder_name,
                    file_path=relative_file_path,
                    processing_status="uploaded",
                )
                db.session.add(uploaded_file_record)
                db.session.commit()
                current_app.logger.info(
                    f"Created database record for uploaded file: {original_filename}"
                )
            except Exception as e:
                current_app.logger.error(f"Error creating database record: {e}")
                flash("File uploaded but database record creation failed.", "warning")

            # Task 2: PDF to PNG Conversion (if it's a PDF)
            if original_filename.lower().endswith(".pdf"):
                current_app.logger.info(
                    f"Starting PDF to PNG conversion for: {saved_pdf_path}"
                )

                # Update processing status
                try:
                    uploaded_file_record.update_processing_status("processing")
                except:
                    pass

                success_conversion, png_message_or_paths = (
                    file_operations.convert_pdf_to_png_pages(
                        current_app.config, user_id, doc_folder_name, saved_pdf_path
                    )
                )
                if success_conversion:
                    current_app.logger.info(
                        f"Successfully converted PDF to {len(png_message_or_paths)} PNGs for document {doc_folder_name}."
                    )

                    # Update processing status and page count
                    try:
                        uploaded_file_record.update_processing_status("completed")
                        uploaded_file_record.page_count = len(png_message_or_paths)
                        db.session.commit()
                    except:
                        pass

                    flash(
                        f"'{original_filename}' uploaded and converted to PNGs successfully!",
                        "success",
                    )
                else:
                    current_app.logger.error(
                        f"Error converting PDF to PNG for {doc_folder_name}: {png_message_or_paths}"
                    )

                    # Update processing status to failed
                    try:
                        uploaded_file_record.update_processing_status("failed")
                    except:
                        pass

                    flash(
                        f"File uploaded, but error during PDF to PNG conversion: {png_message_or_paths}",
                        "danger",
                    )
                    return redirect(request.url)
            else:
                # For images, mark as completed immediately
                try:
                    uploaded_file_record.update_processing_status("completed")
                except:
                    pass

                flash(
                    f"Image '{original_filename}' uploaded successfully!",
                    "success",
                )

            return redirect(url_for("app.crop_image", selected_folder=doc_folder_name))
        else:
            flash("File type not allowed.", "warning")
            return redirect(request.url)

    return render_template("show_upload_page.html")


@app.route("/crop_image", methods=["GET"])
@login_required
def crop_image():
    user_id = current_user.get_user_folder_name()

    user_workspace_info = file_operations.get_user_workspace_info(
        current_app.config, user_id
    )
    logger.info(user_workspace_info)

    # Get all documents for the user (both in-progress and completed can be cropped)
    documents_data = file_operations.get_user_documents_list(
        current_app.config, user_id, include_completed=True, include_in_progress=True
    )

    # Get all documents with their detailed info
    documents_with_info = []
    for doc_name in documents_data["all"]:
        doc_info = file_operations.get_document_info(
            current_app.config, user_id, doc_name
        )
        documents_with_info.append(doc_info)

    selected_document_folder = request.args.get("selected_folder")
    image_to_crop_filename = request.args.get("image_file")

    available_png_files = []
    selected_doc_info = None
    cropped_images = []

    if selected_document_folder:
        # Get the info for the selected document
        for doc_info in documents_with_info:
            if doc_info["name"] == selected_document_folder:
                selected_doc_info = doc_info
                break

        if selected_doc_info:
            # PNGs are in PNG subfolder for all documents
            current_png_dir = file_operations.get_document_png_dir(
                current_app.config, user_id, selected_document_folder
            )
            if os.path.exists(current_png_dir):
                try:
                    all_files_in_png_dir = os.listdir(current_png_dir)
                    available_png_files = [
                        f for f in all_files_in_png_dir if f.lower().endswith(".png")
                    ]
                    available_png_files.sort()
                    current_app.logger.info(
                        f"Found {len(available_png_files)} PNG files in {current_png_dir}"
                    )

                    # If a specific image is selected, find crops for that page
                    if image_to_crop_filename:
                        # Extract base page name (e.g., "page_001" from "page_001.png")
                        page_base = os.path.splitext(image_to_crop_filename)[0]
                        if (
                            not "_crop_" in image_to_crop_filename
                        ):  # Only for original pages
                            cropped_images = [
                                f
                                for f in available_png_files
                                if f.startswith(f"{page_base}_crop_")
                                and f.endswith(".png")
                            ]
                            current_app.logger.info(
                                f"Found {len(cropped_images)} crops for {page_base}"
                            )

                except Exception as e:
                    current_app.logger.error(
                        f"Error listing PNG files from {current_png_dir}: {str(e)}"
                    )
            else:
                current_app.logger.warning(
                    f"PNG directory does not exist: {current_png_dir}"
                )

    return render_template(
        "crop_menu.html",
        documents=documents_with_info,
        folders=[doc["name"] for doc in documents_with_info],  # For compatibility
        selected_folder=selected_document_folder,
        selected_doc_info=selected_doc_info,
        available_png_files=available_png_files,
        png_files_in_selected_folder=available_png_files,  # For compatibility
        image_to_crop_filename=image_to_crop_filename,
        cropped_images=cropped_images,
        user_workspace_info=user_workspace_info,
    )


@app.route("/image_preview")
@login_required
def image_preview():
    user_id = current_user.get_user_folder_name()

    # Get only completed documents for OCR/preview
    documents_data = file_operations.get_user_documents_list(
        current_app.config, user_id, include_completed=True, include_in_progress=False
    )
    completed_document_folders = documents_data["completed"]

    selected_document_folder = request.args.get("selected_folder")
    selected_png_filename = request.args.get("selected_png")

    png_files_in_completed_doc = []
    segments_data = []

    if (
        selected_document_folder
        and selected_document_folder in completed_document_folders
    ):
        # Get document info to verify it's completed
        doc_info = file_operations.get_document_info(
            current_app.config, user_id, selected_document_folder
        )

        if doc_info["is_completed"]:
            # PNGs for completed documents are in /<user_id>/<selected_folder>/TOOCR/PNG/
            completed_png_dir = os.path.join(
                file_operations.get_document_completed_dir(
                    current_app.config, user_id, selected_document_folder
                ),
                "PNG",
            )
            if os.path.exists(completed_png_dir):
                try:
                    all_files_in_completed_png_dir = os.listdir(completed_png_dir)
                    png_files_in_completed_doc = [
                        f
                        for f in all_files_in_completed_png_dir
                        if f.lower().endswith(".png")
                    ]
                    png_files_in_completed_doc.sort()
                except Exception as e:
                    current_app.logger.error(
                        f"Error listing PNG files from {completed_png_dir}: {str(e)}"
                    )

            # If a PNG is selected, load its segment data
            if selected_png_filename:
                # Extract base filename (remove .png extension)
                page_image_filename_base = os.path.splitext(selected_png_filename)[0]
                try:
                    segments_data = file_operations.fetch_page_segments_data(
                        current_app.config,
                        user_id,
                        selected_document_folder,
                        page_image_filename_base,
                        is_document_completed=True,
                    )
                    if segments_data is None:
                        segments_data = []
                except Exception as e:
                    current_app.logger.error(
                        f"Error loading segments for {selected_png_filename}: {str(e)}"
                    )
                    segments_data = []

    return render_template(
        "image_preview.html",
        folders=completed_document_folders,
        selected_folder=selected_document_folder,
        png_files=png_files_in_completed_doc,
        selected_png=selected_png_filename,
        selected_segment=request.args.get("selected_segment"),
        segments_data=segments_data,
    )


@app.route(
    "/crop_1", methods=["GET"]
)  # Assuming GET for now, POST would handle crop submission
def crop_image_1():
    user_id = current_user.get_user_folder_name()

    # Get all document folders for the user for the dropdown
    user_upload_dir = file_operations.get_user_upload_dir(current_app.config, user_id)
    document_folders = []
    if os.path.exists(user_upload_dir):
        document_folders = [
            name
            for name in os.listdir(user_upload_dir)
            if os.path.isdir(os.path.join(user_upload_dir, name))
            and not os.path.exists(
                os.path.join(user_upload_dir, name, "TOOCR")
            )  # List only non-completed folders
        ]
        document_folders.sort()

    selected_document_folder = request.args.get("selected_folder")
    image_to_crop_filename = request.args.get("image_file")  # e.g., page_001.png

    original_pngs_for_cropping = []  # PNGs from <selected_document_folder>/PNG/
    # Cropped images are also in <selected_document_folder>/PNG/ but named like page_00X_crop_YYY.png
    # The template might need to distinguish these or list all and allow selection.
    # For simplicity, let's list all PNGs from the PNG subfolder.

    if selected_document_folder:
        # PNGs for a document *before* completion are in <user_id>/<selected_document_folder>/PNG/
        current_png_dir = file_operations.get_document_png_dir(
            current_app.config, user_id, selected_document_folder
        )
        if os.path.exists(current_png_dir):
            try:
                all_files_in_png_dir = os.listdir(current_png_dir)
                original_pngs_for_cropping = [
                    f for f in all_files_in_png_dir if f.lower().endswith(".png")
                ]
                # A natural sort might be better if page numbers can exceed 999 or have inconsistent padding
                original_pngs_for_cropping.sort()
            except Exception as e:
                current_app.logger.error(
                    f"Error listing PNG files from {current_png_dir}: {str(e)}"
                )

    # The `cropped_images` list in the old template was for a separate `cropped` dir.
    # Now, crops are mixed in the `PNG` dir. The template needs to handle this.
    # We can pass `original_pngs_for_cropping` which contains both originals and their crops.

    return render_template(
        "crop_menu.html",  # Assuming crop_menu.html is adapted
        document_folders=document_folders,
        selected_folder=selected_document_folder,
        # Pass all PNGs (originals and crops) from the PNG subfolder
        available_png_files=original_pngs_for_cropping,
        image_to_crop_filename=image_to_crop_filename,
    )


@app.route("/uploads/<user_id>/<folder>/<filename>")
def uploaded_file(user_id, folder, filename):
    # First log exactly what we received
    current_app.logger.info(
        f"File request received - user_id: '{user_id}', folder: '{folder}', filename: '{filename}'"
    )

    # UPLOAD_FOLDER should be the path *inside the container*, e.g., /app/uploads
    upload_dir_base = current_app.config["UPLOAD_FOLDER"]

    # Add debug info about config
    current_app.logger.info(f"UPLOAD_FOLDER config: {upload_dir_base}")

    # Construct the path to the directory containing the user's specific folder of images
    # This is the directory from which send_from_directory will serve 'filename'
    directory_to_serve_from = os.path.join(upload_dir_base, str(user_id), folder)

    current_app.logger.info(f"Attempting to serve file: {filename}")
    current_app.logger.info(f"From directory: {directory_to_serve_from}")
    current_app.logger.info(
        f"Full expected path: {os.path.join(directory_to_serve_from, filename)}"
    )

    # Check if directory exists and list contents for debugging
    if os.path.exists(directory_to_serve_from):
        current_app.logger.info(
            f"Directory exists. Contents: {os.listdir(directory_to_serve_from)}"
        )
    else:
        current_app.logger.error(f"Directory does not exist: {directory_to_serve_from}")
        # Check parent directories
        parent_dir = os.path.dirname(directory_to_serve_from)
        if os.path.exists(parent_dir):
            current_app.logger.info(
                f"Parent directory exists. Contents: {os.listdir(parent_dir)}"
            )

    if not os.path.exists(os.path.join(directory_to_serve_from, filename)):
        current_app.logger.error(
            f"File NOT FOUND at: {os.path.join(directory_to_serve_from, filename)}"
        )
        return "File not found", 404

    try:
        return send_from_directory(directory_to_serve_from, filename)
    except Exception as e:
        current_app.logger.error(f"Error in send_from_directory: {str(e)}")
        import traceback

        current_app.logger.error(f"Traceback: {traceback.format_exc()}")
        return "Error serving file", 500


@app.route("/uploads/<user_id>/<folder>/cropped/<filename>")
def uploaded_cropped_file(user_id, folder, filename):
    """Serve cropped image files from the cropped subdirectory"""
    upload_dir_base = current_app.config["UPLOAD_FOLDER"]
    # Path to the cropped folder containing the requested file
    directory_to_serve_from = os.path.join(
        upload_dir_base, str(user_id), folder, "cropped"
    )

    current_app.logger.info(f"Attempting to serve cropped file: {filename}")
    current_app.logger.info(f"From cropped directory: {directory_to_serve_from}")

    try:
        return send_from_directory(directory_to_serve_from, filename)
    except Exception as e:
        current_app.logger.error(f"Error serving cropped file: {str(e)}")
        return "Error serving cropped file", 500


@app.route("/uploads/<user_id>/<folder>/TOOCR/<filename>")
def uploaded_toocr_file(user_id, folder, filename):
    """Serve OCR-ready image files from the TOOCR subdirectory"""
    upload_dir_base = current_app.config["UPLOAD_FOLDER"]
    # Path to the TOOCR folder containing the requested file
    directory_to_serve_from = os.path.join(
        upload_dir_base, str(user_id), folder, "TOOCR"
    )

    current_app.logger.info(f"Attempting to serve TOOCR file: {filename}")
    current_app.logger.info(f"From TOOCR directory: {directory_to_serve_from}")

    try:
        return send_from_directory(directory_to_serve_from, filename)
    except Exception as e:
        current_app.logger.error(f"Error serving TOOCR file: {str(e)}")
        return "Error serving OCR file", 500


@app.route("/admin")
@login_required
def admin():
    # Check if user has admin privileges
    if not current_user.is_admin:
        flash("You do not have permission to access the admin panel.", "danger")
        return redirect(url_for("app.index"))

    from app.models import User

    # Get all users from database
    db_users = User.query.all()
    upload_root = current_app.config["UPLOAD_FOLDER"]
    users_data = []
    total_storage = 0

    for user in db_users:
        try:
            user_folder = user.get_user_folder_name()  # This should return the username
            user_path = os.path.join(upload_root, user_folder)

            # Calculate storage and documents if user folder exists
            user_size = 0
            total_documents = 0
            completed_documents = 0
            in_progress_documents = 0
            user_documents = []

            if os.path.exists(user_path):
                # Calculate user storage
                user_size = sum(
                    os.path.getsize(os.path.join(dirpath, filename))
                    for dirpath, _, filenames in os.walk(user_path)
                    for filename in filenames
                )
                total_storage += user_size

                # Get user documents and info
                documents_data = file_operations.get_user_documents_list(
                    current_app.config,
                    user_folder,
                    include_completed=True,
                    include_in_progress=True,
                )

                # Get document details
                for doc_name in documents_data.get("all", []):
                    doc_info = file_operations.get_document_info(
                        current_app.config, user_folder, doc_name
                    )
                    if doc_info:
                        user_documents.append(doc_info)

                total_documents = len(user_documents)
                completed_documents = len(documents_data.get("completed", []))
                in_progress_documents = len(documents_data.get("in_progress", []))

            users_data.append(
                {
                    "user_id": user.username,  # Use username as user_id for file operations
                    "db_user_id": user.id,  # Include database ID
                    "username": user.username,
                    "email": user.email,
                    "full_name": user.full_name,
                    "display_name": user.display_name,
                    "is_admin": user.is_admin,
                    "is_active": user.is_active,
                    "created_at": user.created_at,
                    "last_login": user.last_login,
                    "documents": user_documents,
                    "total_documents": total_documents,
                    "completed_documents": completed_documents,
                    "in_progress_documents": in_progress_documents,
                    "storage_size": user_size,
                    "storage_size_mb": round(user_size / (1024 * 1024), 2),
                    "has_folder": os.path.exists(user_path),
                }
            )

        except Exception as e:
            current_app.logger.error(f"Error processing user {user.username}: {str(e)}")
            # Add user with basic info even if detailed info fails
            users_data.append(
                {
                    "user_id": user.username,
                    "db_user_id": user.id,
                    "username": user.username,
                    "email": user.email,
                    "full_name": user.full_name,
                    "display_name": user.display_name,
                    "is_admin": user.is_admin,
                    "is_active": user.is_active,
                    "created_at": user.created_at,
                    "last_login": user.last_login,
                    "documents": [],
                    "total_documents": 0,
                    "completed_documents": 0,
                    "in_progress_documents": 0,
                    "storage_size": 0,
                    "storage_size_mb": 0,
                    "has_folder": False,
                    "error": str(e),
                }
            )

    return render_template(
        "admin.html",
        users_data=users_data,
        total_storage=total_storage,
        total_storage_mb=round(total_storage / (1024 * 1024), 2),
        total_users=len(db_users),  # Total users from database
    )


@app.route("/process_cropped_image", methods=["POST"])
def process_cropped_image():
    original_folder = request.form.get(
        "original_folder"
    )  # This is document_folder_name
    original_filename = request.form.get("original_filename")  # e.g., page_001.png
    cropped_image_data_url = request.form.get("cropped_image_data")

    if not all([original_folder, original_filename, cropped_image_data_url]):
        flash("Missing required crop parameters.", "danger")
        return redirect(url_for("app.crop_image"))

    user_id = current_user.get_user_folder_name()
    original_page_filename_base = os.path.splitext(original_filename)[
        0
    ]  # e.g., page_001

    # Task 3: Save Cropped Image
    # This saves to /<user_id>/<original_folder>/PNG/page_00X_crop_YYY.png
    success, new_crop_filename_or_error = file_operations.save_cropped_page_image(
        current_app.config,
        user_id,
        original_folder,  # document_folder_name
        original_page_filename_base,
        cropped_image_data_url,
    )

    if success:
        current_app.logger.info(
            f"Saved cropped image as {new_crop_filename_or_error} in {original_folder}/PNG"
        )
        flash(
            f"Successfully saved cropped image as {new_crop_filename_or_error}",
            "success",
        )
    else:
        current_app.logger.error(
            f"Error saving cropped image: {new_crop_filename_or_error}"
        )
        flash(f"Error saving cropped image: {new_crop_filename_or_error}", "danger")

    return redirect(
        url_for(
            "app.crop_image",
            selected_folder=original_folder,
            image_file=original_filename,  # Keep the original page selected for further cropping
        )
    )


@app.route("/move_to_ocr", methods=["POST"])
def move_to_ocr():
    """Copy cropped images to a TOOCR folder with standardized naming"""
    # Get the selected folder
    folder = request.form.get("folder")

    if not folder:
        flash("No folder specified for OCR processing", "danger")
        return redirect(url_for("app.crop_image"))

    # Get user ID
    user_id = session.get("user_id", "anonymous")

    # Define paths
    folder_path = os.path.join(
        current_app.config["UPLOAD_FOLDER"], str(user_id), folder
    )
    cropped_path = os.path.join(folder_path, "cropped")
    toocr_path = os.path.join(folder_path, "TOOCR")

    # Create TOOCR directory if it doesn't exist
    os.makedirs(toocr_path, exist_ok=True)

    try:
        # Group existing cropped images by page number
        page_crops = defaultdict(list)

        if os.path.exists(cropped_path):
            # Process cropped images if they exist
            for filename in os.listdir(cropped_path):
                if not filename.endswith(".png"):
                    continue

                # Extract page base number from cropped filename (e.g. "001_cropped_002.png" -> "001")
                match = re.match(r"(\d+)_cropped_\d+\.png", filename)
                if match:
                    page_num = match.group(1)  # Original page number
                    page_crops[page_num].append(filename)

        # Track number of processed files
        copied_files = 0

        # Process all original images
        for filename in os.listdir(folder_path):
            if not filename.endswith(".png") or filename.startswith("."):
                continue

            # Get original page number (without extension)
            page_base = os.path.splitext(filename)[0]

            if page_crops[page_base]:
                # This page has crops - copy each with new naming convention
                for i, crop_file in enumerate(sorted(page_crops[page_base]), 1):
                    src_path = os.path.join(cropped_path, crop_file)
                    # Format: page_001_crop_001.png
                    dest_filename = f"page_{page_base}_crop_{i:03d}.png"
                    dest_path = os.path.join(toocr_path, dest_filename)

                    shutil.copy2(src_path, dest_path)
                    copied_files += 1
                    current_app.logger.info(
                        f"Copied cropped file: {crop_file} -> {dest_filename}"
                    )
            else:
                # No crops for this page - copy the original image
                src_path = os.path.join(folder_path, filename)
                # Format: page_001.png
                dest_filename = f"page_{page_base}.png"
                dest_path = os.path.join(toocr_path, dest_filename)

                shutil.copy2(src_path, dest_path)
                copied_files += 1
                current_app.logger.info(
                    f"Copied original file: {filename} -> {dest_filename}"
                )

        flash(f"Successfully copied {copied_files} files to the OCR folder", "success")

        # Optional: Redirect to a new OCR processing page if you have one
        # return redirect(url_for("app.process_ocr", folder=folder))

        # For now, redirect back to the crop page
        return redirect(url_for("app.crop_image", selected_folder=folder))

    except Exception as e:
        current_app.logger.error(f"Error preparing files for OCR: {str(e)}")
        import traceback

        current_app.logger.error(f"Traceback: {traceback.format_exc()}")
        flash(f"Error preparing files for OCR: {str(e)}", "danger")
        return redirect(url_for("app.crop_image", selected_folder=folder))


@app.route("/process_ocr", methods=["POST"])
@login_required
def process_ocr():
    data = request.json
    if (
        not data or not data.get("folder") or not data.get("filename")
    ):  # folder is document_folder_name
        return jsonify({"error": "Missing folder or filename"}), 400

    user_id = current_user.get_user_folder_name()
    document_folder_name = data.get("folder")
    page_image_to_ocr = data.get(
        "filename"
    )  # e.g., page_001.png or page_001_crop_001.png

    # Image for OCR is from the "completed" structure: /<user_id>/<doc_folder_name>/TOOCR/PNG/
    # The `is_completed` flag for path helpers should be True.
    ocr_source_png_directory = file_operations.get_ocr_source_png_dir(
        current_app.config, user_id, document_folder_name, is_completed=True
    )
    full_image_path_for_ocr_service = os.path.join(
        ocr_source_png_directory, page_image_to_ocr
    )

    current_app.logger.info(
        f"Processing OCR request for file: {full_image_path_for_ocr_service}"
    )
    if not os.path.exists(full_image_path_for_ocr_service):
        current_app.logger.error(
            f"OCR source image not found: {full_image_path_for_ocr_service}"
        )
        return jsonify({"success": False, "error": "OCR source image not found"}), 404

    try:
        ocr_service_url = current_app.config.get("OCR_SERVICE_URL", "http://ocr:8000")
        response = requests.post(
            f"{ocr_service_url}/process_image/",
            json={"value": full_image_path_for_ocr_service},
        )
        response.raise_for_status()  # Raise an exception for HTTP errors

        ocr_payload_from_service = response.json()
        current_app.logger.info(
            f"OCR service raw response: {str(ocr_payload_from_service)[:500]}..."
        )

        # The actual OCR result is often in a nested key like 'ret'
        actual_ocr_result_data = ocr_payload_from_service.get(
            "ret", ocr_payload_from_service
        )

        page_image_filename_base = os.path.splitext(page_image_to_ocr)[0]

        # Save the processed OCR data using the new file_operations function
        # This will save segments to /<user_id>/<doc_folder_name>/TOOCR/PNG/<page_image_filename_base>/
        # With individual segment files: 000.json, 001.json, etc. and summary.txt
        text_summary, saved_segments_data = file_operations.save_ocr_processed_data(
            current_app.config,
            user_id,
            document_folder_name,
            page_image_filename_base,
            actual_ocr_result_data,  # Pass the actual data part
            is_document_completed=True,
        )

        return jsonify(
            {
                "success": True,
                "text": text_summary,
                "segments": saved_segments_data,  # This is the list of segment dicts suitable for client
                "page_id": page_image_filename_base,
            }
        )

    except requests.RequestException as e:
        current_app.logger.error(f"Error connecting to OCR service: {str(e)}")
        return (
            jsonify(
                {
                    "success": False,
                    "error": f"Error connecting to OCR service: {str(e)}",
                }
            ),
            500,
        )
    except Exception as e:
        current_app.logger.error(f"Unexpected error during OCR processing: {str(e)}")
        import traceback

        current_app.logger.error(f"Traceback: {traceback.format_exc()}")
        return jsonify({"success": False, "error": f"Unexpected error: {str(e)}"}), 500


@app.route("/get_segments/<user_id>/<document_folder_name>/<page_image_filename_base>")
def get_segments(user_id, document_folder_name, page_image_filename_base):
    """
    Return segments for a specific page image that has been OCRd.
    page_image_filename_base is e.g., "page_001" or "page_001_crop_001".
    """
    try:
        # Assumes document is completed for OCR segments to exist
        segments_data = file_operations.fetch_page_segments_data(
            current_app.config,
            user_id,
            document_folder_name,
            page_image_filename_base,
            is_document_completed=True,
        )
        if (
            segments_data is None
        ):  # fetch_page_segments_data might return None on error or empty list
            return jsonify(
                {
                    "success": False,
                    "error": "Segments data not found or error fetching.",
                    "segments": [],
                }
            )
        return jsonify({"success": True, "segments": segments_data})
    except Exception as e:
        current_app.logger.error(
            f"Error retrieving segments for {page_image_filename_base}: {str(e)}"
        )
        return jsonify({"success": False, "error": str(e)})


@app.route(
    "/get_segment_image/<user_id>/<document_folder_name>/<page_image_filename_base>/<segment_id_str>"
)
def get_segment_image(
    user_id, document_folder_name, page_image_filename_base, segment_id_str
):
    """Serve a specific segment image."""
    # Assumes document is completed
    segment_image_dir, segment_image_filename = (
        file_operations.get_segment_image_file_details(
            current_app.config,
            user_id,
            document_folder_name,
            page_image_filename_base,
            segment_id_str,
            is_document_completed=True,
        )
    )

    if not segment_image_dir or not segment_image_filename:
        current_app.logger.error(
            f"Segment image not found for {page_image_filename_base}, segment {segment_id_str}"
        )
        return "Segment image not found", 404

    current_app.logger.info(
        f"Serving segment image {segment_image_filename} from {segment_image_dir}"
    )
    return send_from_directory(segment_image_dir, segment_image_filename)


@app.route(
    "/get_segment_json/<user_id>/<document_folder_name>/<page_image_filename_base>/<segment_id_str>"
)
def get_segment_json(
    user_id, document_folder_name, page_image_filename_base, segment_id_str
):
    """Serve a specific segment's JSON data."""
    # Assumes document is completed
    segments_storage_dir = file_operations.get_ocr_output_segments_base_dir(
        current_app.config,
        user_id,
        document_folder_name,
        page_image_filename_base,
        is_document_completed=True,
    )

    try:
        # Ensure segment_id_str is formatted if it's an index, or use as is if it's a full name.
        # The file_operations.save_ocr_processed_data saves JSON as 000.json, 001.json etc.
        json_filename = f"{int(segment_id_str):03d}.json"
    except ValueError:
        # If segment_id_str is not purely numeric, it might be a direct filename (less likely for JSONs here)
        if segment_id_str.lower().endswith(".json"):
            json_filename = segment_id_str
        else:
            current_app.logger.error(
                f"Invalid segment_id format for JSON: '{segment_id_str}'."
            )
            return "Invalid segment ID format", 400

    json_path = os.path.join(segments_storage_dir, json_filename)
    current_app.logger.info(f"Attempting to serve segment JSON: {json_path}")

    if not os.path.exists(segments_storage_dir):
        current_app.logger.error(
            f"Segments JSON directory for page not found: {segments_storage_dir}"
        )
        return "Page segments JSON directory not found", 404
    if not os.path.exists(json_path):
        current_app.logger.error(f"Segment JSON file not found: {json_path}")
        return "Segment JSON not found", 404

    return send_from_directory(segments_storage_dir, json_filename)


# Remove get_segments_1 if it's a duplicate or test route.
# The get_segment_preview route seems to be a client-side helper, ensure its URL generations are correct.


@app.route("/api/get_document_images/<user_id>/<document_folder_name>")
def api_get_document_images(user_id, document_folder_name):
    """
    AJAX endpoint to get PNG files for a specific document.
    Returns JSON data for dynamic loading without page refresh.
    """
    try:
        # Check if this is a completed document (has TOOCR folder)
        doc_info = file_operations.get_document_info(
            current_app.config, user_id, document_folder_name
        )

        if doc_info["is_completed"]:
            # For completed documents, look in TOOCR/PNG directory
            current_png_dir = os.path.join(
                file_operations.get_document_completed_dir(
                    current_app.config, user_id, document_folder_name
                ),
                "PNG",
            )
        else:
            # For in-progress documents, look in regular PNG directory
            current_png_dir = file_operations.get_document_png_dir(
                current_app.config, user_id, document_folder_name
            )

        images_data = {"success": True, "images": [], "message": ""}

        if os.path.exists(current_png_dir):
            all_files_in_png_dir = os.listdir(current_png_dir)
            png_files = [f for f in all_files_in_png_dir if f.lower().endswith(".png")]
            png_files.sort()

            for png_file in png_files:
                # Create display name
                if "_crop_" in png_file:
                    display_name = png_file.replace("_crop_", " (Crop ").replace(
                        ".png", ")"
                    )
                else:
                    display_name = png_file.replace("page_", "Page ").replace(
                        ".png", ""
                    )

                images_data["images"].append(
                    {"filename": png_file, "display_name": display_name}
                )

            images_data["message"] = f"Found {len(png_files)} images"
            current_app.logger.info(
                f"Successfully loaded {len(png_files)} images from {current_png_dir}"
            )
        else:
            images_data["message"] = f"PNG directory does not exist: {current_png_dir}"
            current_app.logger.warning(f"PNG directory not found: {current_png_dir}")

        return jsonify(images_data)

    except Exception as e:
        current_app.logger.error(
            f"Error getting images for document {document_folder_name}: {str(e)}"
        )
        return (
            jsonify({"success": False, "images": [], "message": f"Error: {str(e)}"}),
            500,
        )


@app.route("/api/get_page_crops/<user_id>/<document_folder_name>/<page_filename>")
def api_get_page_crops(user_id, document_folder_name, page_filename):
    """
    AJAX endpoint to get crop images for a specific page.
    Returns JSON data for dynamic loading without page refresh.
    """
    try:
        # Get PNG directory
        current_png_dir = file_operations.get_document_png_dir(
            current_app.config, user_id, document_folder_name
        )

        crops_data = {"success": True, "crops": [], "message": ""}

        if os.path.exists(current_png_dir):
            # Extract base page name (e.g., "page_001" from "page_001.png")
            page_base = os.path.splitext(page_filename)[0]

            if not "_crop_" in page_filename:  # Only for original pages
                all_files = os.listdir(current_png_dir)
                crop_files = [
                    f
                    for f in all_files
                    if f.startswith(f"{page_base}_crop_") and f.endswith(".png")
                ]
                crop_files.sort()

                for crop_file in crop_files:
                    display_name = crop_file.replace("_crop_", " Crop ").replace(
                        ".png", ""
                    )
                    crops_data["crops"].append(
                        {
                            "filename": crop_file,
                            "display_name": display_name,
                            "url": url_for(
                                "app.serve_document_png_file",
                                user_id=user_id,
                                document_folder_name=document_folder_name,
                                image_filename=crop_file,
                            ),
                        }
                    )

                crops_data["message"] = f"Found {len(crop_files)} crops for {page_base}"
            else:
                crops_data["message"] = "Crops not shown for crop images"
        else:
            crops_data["message"] = "PNG directory does not exist"

        return jsonify(crops_data)

    except Exception as e:
        current_app.logger.error(
            f"Error getting crops for page {page_filename}: {str(e)}"
        )
        return (
            jsonify({"success": False, "crops": [], "message": f"Error: {str(e)}"}),
            500,
        )


@app.route("/api/get_user_documents/<user_id>")
def api_get_user_documents(user_id):
    """
    AJAX endpoint to get all documents for a user for sidebar display.
    Returns JSON data with document metadata for dynamic loading.
    """
    try:
        # Get all documents for the user
        documents_data = file_operations.get_user_documents_list(
            current_app.config,
            user_id,
            include_completed=True,
            include_in_progress=True,
        )

        # Get detailed info for each document
        detailed_documents = []
        for doc_name in documents_data.get("all", []):
            doc_info = file_operations.get_document_info(
                current_app.config, user_id, doc_name
            )
            if doc_info:
                detailed_documents.append(
                    {
                        "name": doc_info.get("name"),
                        "is_completed": doc_info.get("is_completed", False),
                        "total_png_count": doc_info.get("total_png_count", 0),
                        "crop_count": doc_info.get("crop_count", 0),
                        "pdf_files": doc_info.get("pdf_files", []),
                        "status": (
                            "Completed"
                            if doc_info.get("is_completed")
                            else "In Progress"
                        ),
                        "creation_date": doc_info.get("creation_date", ""),
                    }
                )

        return jsonify(
            {
                "success": True,
                "documents": detailed_documents,
                "message": f"Found {len(detailed_documents)} documents",
            }
        )

    except Exception as e:
        current_app.logger.error(
            f"Error getting documents for user {user_id}: {str(e)}"
        )
        return (
            jsonify({"success": False, "documents": [], "message": f"Error: {str(e)}"}),
            500,
        )


@app.route("/api/process_crop", methods=["POST"])
@login_required
def api_process_crop():
    """
    AJAX endpoint to process cropped image without page reload.
    Returns JSON data with success/error information.
    """
    try:
        # Get form data
        original_folder = request.form.get("original_folder")
        original_filename = request.form.get("original_filename")
        cropped_image_data_url = request.form.get("cropped_image_data")

        if not all([original_folder, original_filename, cropped_image_data_url]):
            return (
                jsonify(
                    {"success": False, "message": "Missing required crop parameters"}
                ),
                400,
            )

        user_id = current_user.get_user_folder_name()
        original_page_filename_base = os.path.splitext(original_filename)[0]

        # Save cropped image
        success, new_crop_filename_or_error = file_operations.save_cropped_page_image(
            current_app.config,
            user_id,
            original_folder,
            original_page_filename_base,
            cropped_image_data_url,
        )

        if success:
            current_app.logger.info(
                f"Saved cropped image as {new_crop_filename_or_error} in {original_folder}/PNG"
            )

            # Get the URL for the new crop image
            crop_image_url = url_for(
                "app.serve_document_png_file",
                user_id=user_id,
                document_folder_name=original_folder,
                image_filename=new_crop_filename_or_error,
            )

            return jsonify(
                {
                    "success": True,
                    "message": f"Successfully saved cropped image as {new_crop_filename_or_error}",
                    "crop_filename": new_crop_filename_or_error,
                    "crop_url": crop_image_url,
                    "crop_display_name": new_crop_filename_or_error.replace(
                        "_crop_", " Crop "
                    ).replace(".png", ""),
                }
            )
        else:
            current_app.logger.error(
                f"Error saving cropped image: {new_crop_filename_or_error}"
            )
            return (
                jsonify(
                    {
                        "success": False,
                        "message": f"Error saving cropped image: {new_crop_filename_or_error}",
                    }
                ),
                500,
            )

    except Exception as e:
        current_app.logger.error(f"Error processing crop: {str(e)}")
        return jsonify({"success": False, "message": f"Error: {str(e)}"}), 500


@app.route("/api/admin/user_documents/<user_id>")
@login_required
def api_admin_get_user_documents(user_id):
    """
    AJAX endpoint to get documents for a specific user in admin interface.
    Returns JSON data with user's documents and metadata.
    """
    # Check admin privileges
    if not current_user.is_admin:
        return jsonify({"success": False, "message": "Access denied"}), 403

    try:
        # Get user documents
        documents_data = file_operations.get_user_documents_list(
            current_app.config,
            user_id,
            include_completed=True,
            include_in_progress=True,
        )

        # Get detailed info for each document
        detailed_documents = []
        for doc_name in documents_data.get("all", []):
            doc_info = file_operations.get_document_info(
                current_app.config, user_id, doc_name
            )
            if doc_info:
                detailed_documents.append(doc_info)

        # Calculate user storage
        upload_root = current_app.config["UPLOAD_FOLDER"]
        user_path = os.path.join(upload_root, user_id)
        user_size = 0
        if os.path.exists(user_path):
            try:
                documents_data = file_operations.get_user_documents_list(
                    current_app.config,
                    user_id,
                    include_completed=True,
                    include_in_progress=True,
                )
                user_size = sum(
                    os.path.getsize(os.path.join(dirpath, filename))
                    for dirpath, _, filenames in os.walk(user_path)
                    for filename in filenames
                )
            except:
                user_size = "unknown"

        return jsonify(
            {
                "success": True,
                "user_id": user_id,
                "documents": detailed_documents,
                "total_documents": len(detailed_documents),
                "completed_documents": len(documents_data.get("completed", [])),
                "in_progress_documents": len(documents_data.get("in_progress", [])),
                "storage_size": user_size,
                "storage_size_mb": round(user_size / (1024 * 1024), 2),
            }
        )

    except Exception as e:
        current_app.logger.error(
            f"Error getting documents for user {user_id}: {str(e)}"
        )
        return jsonify({"success": False, "message": f"Error: {str(e)}"}), 500


@app.route("/api/delete_page", methods=["POST"])
@login_required
def api_delete_page():
    """
    AJAX endpoint to delete a specific page but keep its cropped files.
    Returns JSON data with success/error information.
    """
    try:
        user_id = request.form.get("user_id")
        document_name = request.form.get("document_name")
        page_name = request.form.get("page_name")

        if not user_id or not document_name or not page_name:
            return (
                jsonify(
                    {
                        "success": False,
                        "message": "Missing user_id, document_name, or page_name",
                    }
                ),
                400,
            )

        # Verify user authorization (user can delete their own pages, admin can delete any)
        if user_id != session.get("user_id") and not current_user.is_admin:
            return jsonify({"success": False, "message": "Access denied"}), 403

        # Get PNG directory path
        png_directory = file_operations.get_document_png_dir(
            current_app.config, user_id, document_name
        )

        if not os.path.exists(png_directory):
            return (
                jsonify(
                    {
                        "success": False,
                        "message": f"Document '{document_name}' not found for user '{user_id}'",
                    }
                ),
                404,
            )

        # Get the full path to the page file
        page_file_path = os.path.join(png_directory, page_name)

        if not os.path.exists(page_file_path):
            return (
                jsonify(
                    {
                        "success": False,
                        "message": f"Page '{page_name}' not found",
                    }
                ),
                404,
            )

        # Check if this is a crop file (don't allow deleting crops this way)
        if "_crop_" in page_name:
            return (
                jsonify(
                    {
                        "success": False,
                        "message": "Cannot delete crop files using this endpoint. Use the crop management interface instead.",
                    }
                ),
                400,
            )

        # Delete only the original page file, keeping all crop files
        os.remove(page_file_path)

        # Also remove from TOOCR/PNG directory if it exists
        toocr_directory = os.path.join(
            file_operations.get_document_completed_dir(
                current_app.config, user_id, document_name
            ),
            "PNG",
        )
        toocr_page_path = os.path.join(toocr_directory, page_name)
        if os.path.exists(toocr_page_path):
            os.remove(toocr_page_path)
            current_app.logger.info(
                f"Also removed page from TOOCR directory: {toocr_page_path}"
            )

        # Get page base name to count remaining crops
        page_base = os.path.splitext(page_name)[0]
        all_files = os.listdir(png_directory)
        remaining_crops = [
            f
            for f in all_files
            if f.startswith(f"{page_base}_crop_") and f.endswith(".png")
        ]

        success_message = f"Page '{page_name}' deleted successfully"
        if remaining_crops:
            success_message += f" (kept {len(remaining_crops)} associated crop files)"

        current_app.logger.info(
            f"User '{user_id}' deleted page '{page_name}' from document '{document_name}' (kept {len(remaining_crops)} crops)"
        )

        return jsonify(
            {
                "success": True,
                "message": success_message,
                "remaining_crops": len(remaining_crops),
            }
        )

    except Exception as e:
        current_app.logger.error(f"Error deleting page: {str(e)}")
        return (
            jsonify({"success": False, "message": f"Error deleting page: {str(e)}"}),
            500,
        )


@app.route("/api/admin/delete_document", methods=["POST"])
@login_required
def api_admin_delete_document():
    """
    AJAX endpoint to delete a specific document folder.
    Returns JSON data with success/error information.
    """
    # Check admin privileges
    if not current_user.is_admin:
        return jsonify({"success": False, "message": "Access denied"}), 403

    try:
        user_id = request.form.get("user_id")
        document_name = request.form.get("document_name")

        if not user_id or not document_name:
            return (
                jsonify(
                    {"success": False, "message": "Missing user_id or document_name"}
                ),
                400,
            )

        # Get document path
        upload_root = current_app.config["UPLOAD_FOLDER"]
        document_path = os.path.join(upload_root, user_id, document_name)

        if not os.path.exists(document_path):
            return (
                jsonify(
                    {
                        "success": False,
                        "message": f"Document '{document_name}' not found for user '{user_id}'",
                    }
                ),
                404,
            )

        # Delete document folder
        shutil.rmtree(document_path)

        # Delete associated uploaded file records from database
        from app.models import UploadedFile, User
        from app import db

        user_obj = User.query.filter_by(username=user_id).first()
        if user_obj:
            deleted_files_count = UploadedFile.query.filter_by(
                user_id=user_obj.id, document_folder=document_name
            ).delete()
            db.session.commit()
            current_app.logger.info(
                f"Deleted {deleted_files_count} uploaded file records from database for document '{document_name}'"
            )

        current_app.logger.info(
            f"Admin deleted document '{document_name}' for user '{user_id}'"
        )

        return jsonify(
            {
                "success": True,
                "message": f"Document '{document_name}' deleted successfully",
            }
        )

    except Exception as e:
        current_app.logger.error(f"Error deleting document: {str(e)}")
        return (
            jsonify(
                {"success": False, "message": f"Error deleting document: {str(e)}"}
            ),
            500,
        )


@app.route("/api/admin/delete_user", methods=["POST"])
@login_required
def api_admin_delete_user():
    """
    AJAX endpoint to delete an entire user and all their documents.
    Returns JSON data with success/error information.
    """
    # Check admin privileges
    if not current_user.is_admin:
        return jsonify({"success": False, "message": "Access denied"}), 403

    try:
        from app.models import User
        from app import db

        user_id = request.form.get("user_id")  # This is the username

        if not user_id:
            return jsonify({"success": False, "message": "Missing user_id"}), 400

        # Prevent admin from deleting themselves
        if user_id == current_user.username:
            return (
                jsonify(
                    {"success": False, "message": "You cannot delete your own account"}
                ),
                400,
            )

        # Find user in database
        user_to_delete = User.query.filter_by(username=user_id).first()
        if not user_to_delete:
            return (
                jsonify(
                    {
                        "success": False,
                        "message": f"User '{user_id}' not found in database",
                    }
                ),
                404,
            )

        # Count documents before deletion for logging
        document_count = 0
        upload_root = current_app.config["UPLOAD_FOLDER"]
        user_path = os.path.join(upload_root, user_id)

        if os.path.exists(user_path):
            try:
                documents_data = file_operations.get_user_documents_list(
                    current_app.config,
                    user_id,
                    include_completed=True,
                    include_in_progress=True,
                )
                document_count = len(documents_data.get("all", []))
            except:
                document_count = "unknown"

            # Delete user folder and all documents
            shutil.rmtree(user_path)
            current_app.logger.info(
                f"Deleted user folder for '{user_id}' with {document_count} documents"
            )

        # Delete user's uploaded files from database (cascade should handle this, but let's be explicit)
        from app.models import UploadedFile

        uploaded_files_count = UploadedFile.query.filter_by(
            user_id=user_to_delete.id
        ).count()
        UploadedFile.query.filter_by(user_id=user_to_delete.id).delete()

        # Delete user from database
        db.session.delete(user_to_delete)
        db.session.commit()

        current_app.logger.info(
            f"Deleted {uploaded_files_count} uploaded file records from database for user '{user_id}'"
        )

        current_app.logger.warning(
            f"Admin '{current_user.username}' deleted user '{user_id}' (DB ID: {user_to_delete.id}) with {document_count} documents"
        )

        return jsonify(
            {
                "success": True,
                "message": f"User '{user_to_delete.display_name}' deleted successfully from database and filesystem",
            }
        )

    except Exception as e:
        db.session.rollback()
        current_app.logger.error(f"Error deleting user: {str(e)}", exc_info=True)
        return (
            jsonify({"success": False, "message": f"Error deleting user: {str(e)}"}),
            500,
        )


@app.route("/api/admin/create_user", methods=["POST"])
@login_required
def api_admin_create_user():
    """
    AJAX endpoint to create a new user.
    Returns JSON data with success/error information.
    """
    # Check admin privileges
    if not current_user.is_admin:
        current_app.logger.warning(
            f"Non-admin user {current_user.username} attempted to create user"
        )
        return jsonify({"success": False, "message": "Access denied"}), 403

    current_app.logger.info(
        f"Admin {current_user.username} attempting to create new user"
    )

    try:
        from app.models import User
        from app import db
        from werkzeug.security import generate_password_hash
        from email_validator import validate_email, EmailNotValidError

        username = request.form.get("username", "").strip()
        email = request.form.get("email", "").strip()
        first_name = request.form.get("first_name", "").strip()
        last_name = request.form.get("last_name", "").strip()
        password = request.form.get("password", "")
        is_admin = request.form.get("is_admin") == "on"

        current_app.logger.info(
            f"Received form data - username: {username}, email: {email}, is_admin: {is_admin}"
        )

        # Validation
        errors = {}

        if not username:
            errors["username"] = "Username is required"
        elif len(username) < 3:
            errors["username"] = "Username must be at least 3 characters"
        elif User.query.filter_by(username=username).first():
            errors["username"] = "Username already exists"

        if not email:
            errors["email"] = "Email is required"
        else:
            try:
                validate_email(email)
                if User.query.filter_by(email=email).first():
                    errors["email"] = "Email already registered"
            except EmailNotValidError:
                errors["email"] = "Invalid email format"

        if not password:
            errors["password"] = "Password is required"
        elif len(password) < 6:
            errors["password"] = "Password must be at least 6 characters"

        if errors:
            current_app.logger.warning(f"User creation validation failed: {errors}")
            return (
                jsonify(
                    {
                        "success": False,
                        "message": "Validation errors occurred",
                        "errors": errors,
                    }
                ),
                400,
            )

        current_app.logger.info(f"Validation passed, creating user: {username}")

        # Create new user
        new_user = User(
            username=username,
            email=email,
            first_name=first_name if first_name else None,
            last_name=last_name if last_name else None,
            password_hash=generate_password_hash(password),
            is_admin=is_admin,
            is_active=True,
        )

        db.session.add(new_user)
        db.session.commit()

        current_app.logger.info(
            f"Successfully created new user: {username} (ID: {new_user.id}, Admin: {is_admin})"
        )

        return jsonify(
            {
                "success": True,
                "message": f"User '{username}' created successfully",
                "username": username,
                "user_id": new_user.id,
            }
        )

    except Exception as e:
        db.session.rollback()
        current_app.logger.error(f"Error creating user: {str(e)}", exc_info=True)
        return (
            jsonify({"success": False, "message": f"Error creating user: {str(e)}"}),
            500,
        )


@app.route("/api/admin/test", methods=["GET", "POST"])
@login_required
def api_admin_test():
    """Test endpoint to verify admin functionality"""
    if not current_user.is_admin:
        return jsonify({"success": False, "message": "Access denied"}), 403

    if request.method == "POST":
        data = dict(request.form)
        return jsonify(
            {
                "success": True,
                "message": "Test endpoint working",
                "method": "POST",
                "form_data": data,
                "user": current_user.username,
            }
        )
    else:
        return jsonify(
            {
                "success": True,
                "message": "Test endpoint working",
                "method": "GET",
                "user": current_user.username,
            }
        )


@app.route("/api/admin/uploaded_files", methods=["GET"])
@login_required
def api_admin_get_uploaded_files():
    """
    API endpoint to get all uploaded files from database with filtering options.
    Returns JSON data with uploaded files information.
    """
    # Check admin privileges
    if not current_user.is_admin:
        return jsonify({"success": False, "message": "Access denied"}), 403

    try:
        from app.models import UploadedFile, User
        from sqlalchemy import func

        # Get query parameters for filtering
        user_id = request.args.get("user_id")  # Filter by specific user ID
        file_type = request.args.get("file_type")  # Filter by file type (pdf/image)
        status = request.args.get("status")  # Filter by processing status
        limit = request.args.get("limit", 100, type=int)  # Limit results
        offset = request.args.get("offset", 0, type=int)  # Pagination offset

        # Build query
        query = UploadedFile.query.join(User)

        # Apply filters
        if user_id:
            query = query.filter(User.username == user_id)
        if file_type:
            query = query.filter(UploadedFile.file_type == file_type)
        if status:
            query = query.filter(UploadedFile.processing_status == status)

        # Get total count before pagination
        total_count = query.count()

        # Apply pagination and ordering
        uploaded_files = (
            query.order_by(UploadedFile.uploaded_at.desc())
            .offset(offset)
            .limit(limit)
            .all()
        )

        # Format response data
        files_data = []
        for uploaded_file in uploaded_files:
            files_data.append(
                {
                    "id": uploaded_file.id,
                    "user_id": uploaded_file.user.id,
                    "username": uploaded_file.user.username,
                    "user_display_name": uploaded_file.user.display_name,
                    "original_filename": uploaded_file.original_filename,
                    "stored_filename": uploaded_file.stored_filename,
                    "file_type": uploaded_file.file_type,
                    "file_size": uploaded_file.file_size,
                    "file_size_mb": uploaded_file.file_size_mb,
                    "mime_type": uploaded_file.mime_type,
                    "document_folder": uploaded_file.document_folder,
                    "processing_status": uploaded_file.processing_status,
                    "page_count": uploaded_file.page_count,
                    "ocr_completed": uploaded_file.ocr_completed,
                    "uploaded_at": uploaded_file.uploaded_at.isoformat(),
                    "updated_at": uploaded_file.updated_at.isoformat(),
                    "file_path": uploaded_file.file_path,
                }
            )

        # Get summary statistics
        stats = {
            "total_files": UploadedFile.query.count(),
            "pdf_files": UploadedFile.query.filter_by(file_type="pdf").count(),
            "image_files": UploadedFile.query.filter_by(file_type="image").count(),
            "completed_files": UploadedFile.query.filter_by(
                processing_status="completed"
            ).count(),
            "failed_files": UploadedFile.query.filter_by(
                processing_status="failed"
            ).count(),
            "total_size_bytes": db.session.query(
                func.sum(UploadedFile.file_size)
            ).scalar()
            or 0,
        }
        stats["total_size_mb"] = round(stats["total_size_bytes"] / (1024 * 1024), 2)

        return jsonify(
            {
                "success": True,
                "files": files_data,
                "pagination": {
                    "total_count": total_count,
                    "limit": limit,
                    "offset": offset,
                    "has_more": offset + limit < total_count,
                },
                "stats": stats,
                "filters": {
                    "user_id": user_id,
                    "file_type": file_type,
                    "status": status,
                },
            }
        )

    except Exception as e:
        current_app.logger.error(
            f"Error getting uploaded files: {str(e)}", exc_info=True
        )
        return (
            jsonify(
                {
                    "success": False,
                    "message": f"Error retrieving uploaded files: {str(e)}",
                }
            ),
            500,
        )


@app.route("/api/admin/uploaded_files/<int:file_id>", methods=["GET"])
@login_required
def api_admin_get_uploaded_file(file_id):
    """
    API endpoint to get specific uploaded file details from database.
    Returns JSON data with file information.
    """
    # Check admin privileges
    if not current_user.is_admin:
        return jsonify({"success": False, "message": "Access denied"}), 403

    try:
        from app.models import UploadedFile, User

        uploaded_file = UploadedFile.query.filter_by(id=file_id).first()
        if not uploaded_file:
            return (
                jsonify(
                    {"success": False, "message": f"File with ID {file_id} not found"}
                ),
                404,
            )

        # Check if file exists on filesystem
        full_path = uploaded_file.get_full_path(current_app.config)
        file_exists = os.path.exists(full_path)

        file_data = {
            "id": uploaded_file.id,
            "user_id": uploaded_file.user.id,
            "username": uploaded_file.user.username,
            "user_display_name": uploaded_file.user.display_name,
            "user_email": uploaded_file.user.email,
            "original_filename": uploaded_file.original_filename,
            "stored_filename": uploaded_file.stored_filename,
            "file_type": uploaded_file.file_type,
            "file_size": uploaded_file.file_size,
            "file_size_mb": uploaded_file.file_size_mb,
            "mime_type": uploaded_file.mime_type,
            "document_folder": uploaded_file.document_folder,
            "processing_status": uploaded_file.processing_status,
            "page_count": uploaded_file.page_count,
            "ocr_completed": uploaded_file.ocr_completed,
            "uploaded_at": uploaded_file.uploaded_at.isoformat(),
            "updated_at": uploaded_file.updated_at.isoformat(),
            "file_path": uploaded_file.file_path,
            "full_path": full_path,
            "file_exists": file_exists,
        }

        return jsonify({"success": True, "file": file_data})

    except Exception as e:
        current_app.logger.error(
            f"Error getting uploaded file {file_id}: {str(e)}", exc_info=True
        )
        return (
            jsonify({"success": False, "message": f"Error retrieving file: {str(e)}"}),
            500,
        )


# Global dictionary to store upload progress (in production, use Redis or database)
upload_progress_store = {}


def cleanup_old_progress_entries():
    """
    Cleanup old progress entries to prevent memory leaks.
    Removes entries older than 1 hour.
    """
    import time

    current_time = time.time()
    keys_to_remove = []

    for upload_id, progress_data in upload_progress_store.items():
        # Add timestamp if not present
        if "timestamp" not in progress_data:
            progress_data["timestamp"] = current_time

        # Remove entries older than 1 hour
        if current_time - progress_data.get("timestamp", current_time) > 3600:
            keys_to_remove.append(upload_id)

    for key in keys_to_remove:
        del upload_progress_store[key]


# Schedule cleanup every 30 minutes (in production, use a proper scheduler)
import threading
import time


def periodic_cleanup():
    while True:
        time.sleep(1800)  # 30 minutes
        cleanup_old_progress_entries()


cleanup_thread = threading.Thread(target=periodic_cleanup, daemon=True)
cleanup_thread.start()


@app.route("/api/upload_with_progress", methods=["POST"])
@login_required
def api_upload_with_progress():
    """
    AJAX endpoint for file upload with progress tracking.
    Returns JSON response with upload_id for progress polling.
    """
    import uuid
    import threading

    try:
        if "file" not in request.files:
            return (
                jsonify({"success": False, "message": "No file part in the request"}),
                400,
            )

        uploaded_file_storage = request.files["file"]
        if uploaded_file_storage.filename == "":
            return jsonify({"success": False, "message": "No selected file"}), 400

        if not (uploaded_file_storage and allowed_file(uploaded_file_storage.filename)):
            return jsonify({"success": False, "message": "File type not allowed"}), 400

        # Generate unique upload ID
        upload_id = str(uuid.uuid4())

        # Initialize progress tracking
        import time

        upload_progress_store[upload_id] = {
            "status": "uploading",
            "total_pages": 0,
            "converted_pages": 0,
            "current_message": "File uploaded, starting processing...",
            "document_folder": None,
            "error_message": None,
            "user_id": current_user.get_user_folder_name(),
            "filename": uploaded_file_storage.filename,
            "timestamp": time.time(),
        }

        # Start upload processing in background thread
        thread = threading.Thread(
            target=process_upload_with_progress,
            args=(upload_id, uploaded_file_storage, current_user, current_app.config),
        )
        thread.daemon = True
        thread.start()

        return jsonify(
            {
                "success": True,
                "upload_id": upload_id,
                "message": "Upload started, processing in background",
            }
        )

    except Exception as e:
        current_app.logger.error(f"Error starting upload: {str(e)}", exc_info=True)
        return (
            jsonify({"success": False, "message": f"Error starting upload: {str(e)}"}),
            500,
        )


@app.route("/api/upload_progress/<upload_id>", methods=["GET"])
@login_required
def api_get_upload_progress(upload_id):
    """
    API endpoint to get upload progress information.
    Returns JSON data with current progress status.
    """
    try:
        if upload_id not in upload_progress_store:
            return jsonify({"success": False, "message": "Upload ID not found"}), 404

        progress_data = upload_progress_store[upload_id]

        # Check if this upload belongs to the current user
        if progress_data.get("user_id") != current_user.get_user_folder_name():
            return jsonify({"success": False, "message": "Access denied"}), 403

        return jsonify({"success": True, "progress": progress_data})

    except Exception as e:
        current_app.logger.error(
            f"Error getting upload progress: {str(e)}", exc_info=True
        )
        return (
            jsonify({"success": False, "message": f"Error getting progress: {str(e)}"}),
            500,
        )


def process_upload_with_progress(upload_id, uploaded_file_storage, user, app_config):
    """
    Background function to process file upload with progress tracking.
    Updates progress in upload_progress_store.
    """
    from app.models import UploadedFile, db
    from app import create_app
    import mimetypes

    # Create app context for background thread
    app = create_app()

    with app.app_context():
        try:
            # Get progress tracker
            progress = upload_progress_store[upload_id]

            user_id = user.get_user_folder_name()
            original_filename = uploaded_file_storage.filename

            progress["current_message"] = "Creating document folder..."

            # Handle PDF Upload and Folder Creation
            doc_folder_path, saved_pdf_path, doc_folder_name = (
                file_operations.handle_pdf_upload_with_auto_folder_no_sanitize(
                    app_config, user_id, uploaded_file_storage
                )
            )

            if not doc_folder_path:
                progress["status"] = "failed"
                progress["error_message"] = (
                    "Error saving uploaded file or creating document folder"
                )
                return

            progress["document_folder"] = doc_folder_name
            progress["current_message"] = (
                f"File saved to document folder: {doc_folder_name}"
            )

            # Get file information for database
            file_size = os.path.getsize(saved_pdf_path)
            file_type = "pdf" if original_filename.lower().endswith(".pdf") else "image"
            mime_type, _ = mimetypes.guess_type(original_filename)
            stored_filename = os.path.basename(saved_pdf_path)
            relative_file_path = os.path.relpath(
                saved_pdf_path, app_config["UPLOAD_FOLDER"]
            )

            # Create database record for uploaded file
            try:
                uploaded_file_record = UploadedFile(
                    user_id=user.id,
                    original_filename=original_filename,
                    stored_filename=stored_filename,
                    file_type=file_type,
                    file_size=file_size,
                    mime_type=mime_type,
                    document_folder=doc_folder_name,
                    file_path=relative_file_path,
                    processing_status="uploaded",
                )
                db.session.add(uploaded_file_record)
                db.session.commit()
                progress["current_message"] = "Database record created"
            except Exception as e:
                app.logger.error(f"Error creating database record: {e}")
                progress["current_message"] = (
                    "File uploaded but database record creation failed"
                )

            # PDF to PNG Conversion (if it's a PDF)
            if original_filename.lower().endswith(".pdf"):
                progress["current_message"] = "Starting PDF page extraction..."

                # Update processing status
                try:
                    uploaded_file_record.update_processing_status("processing")
                except:
                    pass

                # Custom PDF conversion with progress tracking
                success_conversion, png_paths_or_message = (
                    convert_pdf_with_progress_tracking(
                        app_config, user_id, doc_folder_name, saved_pdf_path, upload_id
                    )
                )

                if success_conversion:
                    app.logger.info(
                        f"Successfully converted PDF to {len(png_paths_or_message)} PNGs"
                    )

                    # Update processing status and page count
                    try:
                        uploaded_file_record.update_processing_status("completed")
                        uploaded_file_record.page_count = len(png_paths_or_message)
                        db.session.commit()
                    except:
                        pass

                    progress["status"] = "completed"
                    progress["current_message"] = (
                        f"PDF conversion completed! Created {len(png_paths_or_message)} images"
                    )
                else:
                    app.logger.error(f"Error converting PDF: {png_paths_or_message}")

                    # Update processing status to failed
                    try:
                        uploaded_file_record.update_processing_status("failed")
                    except:
                        pass

                    progress["status"] = "failed"
                    progress["error_message"] = (
                        f"PDF conversion failed: {png_paths_or_message}"
                    )
            else:
                # For images, mark as completed immediately
                try:
                    uploaded_file_record.update_processing_status("completed")
                except:
                    pass

                progress["status"] = "completed"
                progress["current_message"] = "Image upload completed successfully"

        except Exception as e:
            app.logger.error(
                f"Error in background upload processing: {str(e)}", exc_info=True
            )
            progress["status"] = "failed"
            progress["error_message"] = str(e)


def convert_pdf_with_progress_tracking(
    app_config, user_id, document_folder_name, pdf_file_path, upload_id
):
    """
    Custom PDF conversion function with progress tracking.
    Updates progress in upload_progress_store during conversion.
    """
    try:
        from pdf2image import convert_from_path
        import tempfile

        progress = upload_progress_store[upload_id]

        # Get the PNG directory for the document
        png_output_dir = file_operations.get_document_png_dir(
            app_config, user_id, document_folder_name
        )

        progress["current_message"] = "Analyzing PDF structure..."

        # First, get total page count
        with tempfile.TemporaryDirectory() as temp_dir:
            try:
                # Convert first page to get total count
                test_images = convert_from_path(
                    pdf_file_path, first_page=1, last_page=1
                )

                # Get total page count using pdf2image's internal method
                from pdf2image.pdf2image import pdfinfo_from_path

                info = pdfinfo_from_path(pdf_file_path)
                total_pages = info.get("Pages", 1)

                progress["total_pages"] = total_pages
                progress["current_message"] = f"Found {total_pages} pages to convert"

            except Exception as e:
                # Fallback: convert all at once if we can't get page count
                total_pages = 1
                progress["total_pages"] = total_pages

        # Convert pages one by one for progress tracking
        converted_files = []

        for page_num in range(1, total_pages + 1):
            try:
                progress["current_message"] = (
                    f"Converting page {page_num} of {total_pages}..."
                )

                # Convert single page
                images = convert_from_path(
                    pdf_file_path,
                    first_page=page_num,
                    last_page=page_num,
                    dpi=200,
                    fmt="PNG",
                )

                if images:
                    # Save the page image
                    page_filename = f"{page_num:03d}.png"
                    page_path = os.path.join(png_output_dir, page_filename)
                    images[0].save(page_path, "PNG")
                    converted_files.append(page_path)

                    # Update progress
                    progress["converted_pages"] = page_num
                    progress["current_message"] = (
                        f"Converted page {page_num} of {total_pages}"
                    )

            except Exception as e:
                current_app.logger.error(f"Error converting page {page_num}: {str(e)}")
                return False, f"Error converting page {page_num}: {str(e)}"

        return True, converted_files

    except ImportError:
        return False, "pdf2image library not available"
    except Exception as e:
        current_app.logger.error(f"Error in PDF conversion with progress: {str(e)}")
        return False, str(e)


@app.route("/api/move_to_ocr", methods=["POST"])
@login_required
def api_move_to_ocr():
    """
    AJAX endpoint to move cropped images to OCR processing folder.
    Returns JSON response with success/error information.
    """
    try:
        folder_name = request.form.get("folder")

        if not folder_name:
            return jsonify({"success": False, "message": "No folder specified"}), 400

        user_id = current_user.get_user_folder_name()

        # Call the existing move_to_ocr function
        success, message = file_operations.mark_document_editing_completed(
            current_app.config, user_id, folder_name
        )

        if success:
            current_app.logger.info(
                f"User {user_id} moved document {folder_name} to OCR via AJAX"
            )

            # Generate redirect URL for OCR page
            redirect_url = url_for("app.ocr", selected_folder=folder_name)

            return jsonify(
                {
                    "success": True,
                    "message": message,
                    "redirect_url": redirect_url,
                    "document_folder": folder_name,
                }
            )
        else:
            current_app.logger.error(
                f"Failed to move document {folder_name} to OCR for user {user_id}: {message}"
            )
            return jsonify({"success": False, "message": message}), 500

    except Exception as e:
        current_app.logger.error(f"Error in api_move_to_ocr: {str(e)}", exc_info=True)
        return (
            jsonify(
                {"success": False, "message": f"Error moving files to OCR: {str(e)}"}
            ),
            500,
        )


# General file serving from the root of a document folder (e.g., the PDF before completion)
@app.route("/uploads/<user_id>/<document_folder_name>/<filename>")
def serve_document_root_file(user_id, document_folder_name, filename):
    doc_dir = file_operations.get_document_dir(
        current_app.config, user_id, document_folder_name
    )
    current_app.logger.info(f"Attempting to serve root file: {filename} from {doc_dir}")
    if not os.path.exists(os.path.join(doc_dir, filename)):
        current_app.logger.error(
            f"File NOT FOUND at: {os.path.join(doc_dir, filename)}"
        )
        return "File not found", 404
    return send_from_directory(doc_dir, filename)


# Serving PNGs (originals/crops) before completion
@app.route("/uploads/<user_id>/<document_folder_name>/PNG/<image_filename>")
def serve_document_png_file(user_id, document_folder_name, image_filename):
    png_dir = file_operations.get_document_png_dir(
        current_app.config, user_id, document_folder_name
    )
    current_app.logger.info(
        f"Attempting to serve PNG file: {image_filename} from {png_dir}"
    )
    if not os.path.exists(os.path.join(png_dir, image_filename)):
        current_app.logger.error(
            f"File NOT FOUND at: {os.path.join(png_dir, image_filename)}"
        )
        return "File not found", 404
    return send_from_directory(png_dir, image_filename)


# Serving files from the root of TOOCR folder (e.g., PDF after completion)
@app.route("/uploads/<user_id>/<document_folder_name>/TOOCR/<filename>")
def serve_completed_document_root_file(user_id, document_folder_name, filename):
    toocr_dir = file_operations.get_document_completed_dir(
        current_app.config, user_id, document_folder_name
    )
    current_app.logger.info(
        f"Attempting to serve TOOCR root file: {filename} from {toocr_dir}"
    )
    if not os.path.exists(os.path.join(toocr_dir, filename)):
        current_app.logger.error(
            f"File NOT FOUND at: {os.path.join(toocr_dir, filename)}"
        )
        return "File not found", 404
    return send_from_directory(toocr_dir, filename)


# Serving PNGs (originals/crops) after completion
@app.route("/uploads/<user_id>/<document_folder_name>/TOOCR/PNG/<image_filename>")
def serve_completed_document_png_file(user_id, document_folder_name, image_filename):
    completed_png_dir = os.path.join(
        file_operations.get_document_completed_dir(
            current_app.config, user_id, document_folder_name
        ),
        "PNG",
    )
    current_app.logger.info(
        f"Attempting to serve TOOCR PNG file: {image_filename} from {completed_png_dir}"
    )
    if not os.path.exists(os.path.join(completed_png_dir, image_filename)):
        current_app.logger.error(
            f"File NOT FOUND at: {os.path.join(completed_png_dir, image_filename)}"
        )
        return "File not found", 404
    return send_from_directory(completed_png_dir, image_filename)


@app.route("/ocr", methods=["GET", "POST"])
@login_required
def ocr():
    """Render the OCR page with user documents and handle file uploads"""
    if request.method == "POST":
        # Handle OCR file upload logic here
        # This is where you'd process uploaded files
        pass

    user_id = current_user.get_user_folder_name()

    # Get user documents for sidebar
    documents_data = file_operations.get_user_documents_list(
        current_app.config, user_id, include_completed=True, include_in_progress=True
    )

    # Get detailed info for each document
    documents_with_info = []
    for doc_name in documents_data["all"]:
        doc_info = file_operations.get_document_info(
            current_app.config, user_id, doc_name
        )
        if doc_info:
            documents_with_info.append(doc_info)

    context = {
        "page_title": "OCR - Optical Character Recognition",
        "supported_formats": ["PDF", "PNG", "JPG", "JPEG", "TIFF", "BMP"],
        "max_file_size": "10MB",
        "documents": documents_with_info,
        "user_id": user_id,
    }
    return render_template("ocr.html", **context)
