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
from werkzeug.utils import secure_filename
from pdf2image import convert_from_path
import os
import base64
import re
import shutil
from collections import defaultdict
import requests
import json


from loguru import logger

app = Blueprint(
    "app", __name__
)  # Assuming you renamed 'app' to 'main' or vice-versa consistently

UPLOAD_FOLDER = "uploads"
ALLOWED_EXTENSIONS = {"pdf", "png", "jpg", "jpeg"}


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/show_upload_page", methods=["POST", "GET"])
def show_upload_page():
    if request.method == "POST":
        if "file" not in request.files:
            flash("No file part in the request.", "danger")
            return redirect(request.url)
        file = request.files["file"]
        if file.filename == "":
            flash("No selected file.", "warning")
            return redirect(request.url)

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            user_id = str(session.get("user_id", "anonymous"))

            # Create a folder name from the original filename (without extension)
            original_filename_basename = os.path.splitext(filename)[0]

            # Define the base path for this specific upload
            # e.g., /app/uploads/user_id/original_pdf_name/
            specific_upload_path = os.path.join(
                current_app.config["UPLOAD_FOLDER"],
                user_id,
                original_filename_basename,
            )

            try:
                os.makedirs(specific_upload_path, exist_ok=True)
                current_app.logger.info(f"Created directory: {specific_upload_path}")

                # Save the original file (e.g., mydocument.pdf)
                original_file_path = os.path.join(specific_upload_path, filename)
                file.save(original_file_path)
                current_app.logger.info(f"Saved original file to: {original_file_path}")

                # If it's a PDF, convert to PNGs
                if filename.lower().endswith(".pdf"):
                    current_app.logger.info(
                        f"Starting PDF to PNG conversion for: {original_file_path}"
                    )
                    try:
                        images = convert_from_path(
                            original_file_path, dpi=200
                        )  # Adjust DPI as needed
                        for i, image in enumerate(images):
                            image_filename = f"{i+1:03d}.png"  # e.g., 001.png, 002.png
                            image_save_path = os.path.join(
                                specific_upload_path, image_filename
                            )
                            image.save(image_save_path, "PNG")
                            current_app.logger.info(
                                f"Saved PNG page: {image_save_path}"
                            )
                        flash(
                            f"'{filename}' uploaded and processed successfully!",
                            "success",
                        )
                    except Exception as e:
                        current_app.logger.error(f"Error converting PDF to PNG: {e}")
                        flash(
                            f"File uploaded, but error during PDF to PNG conversion: {e}",
                            "danger",
                        )
                        # Optionally, clean up the original file if conversion fails critically
                        # os.remove(original_file_path)
                        # shutil.rmtree(specific_upload_path)
                        return redirect(request.url)
                else:
                    # If it's an image, it's already "processed" in a way
                    # You might want to rename it to a standard format like 001.png if it's a single image upload
                    # For now, we assume PDF is the primary multi-page document
                    flash(f"Image '{filename}' uploaded successfully!", "success")

                # Redirect to the image preview page for the newly created folder
                return redirect(
                    url_for(
                        "app.image_preview", selected_folder=original_filename_basename
                    )
                )

            except OSError as e:
                current_app.logger.error(f"OSError during file upload processing: {e}")
                flash(f"Error creating directory or saving file: {e}", "danger")
                return redirect(request.url)
            except Exception as e:
                current_app.logger.error(
                    f"General error during file upload processing: {e}"
                )
                flash(f"An unexpected error occurred: {e}", "danger")
                return redirect(request.url)

        else:
            flash("File type not allowed.", "warning")
            return redirect(request.url)

    # For GET request, just render the upload page
    return render_template("show_upload_page.html")


@app.route("/image_preview")
def image_preview():
    user_id = session.get("user_id", "anonymous")
    upload_root = os.path.join(current_app.config["UPLOAD_FOLDER"], str(user_id))

    # Initialize variables
    all_folders = []
    ocr_ready_folders = []
    png_files = []
    selected_folder = request.args.get("selected_folder")
    selected_png = request.args.get("selected_png")
    selected_segment = request.args.get("selected_segment")

    # Get OCR-ready folders
    if os.path.exists(upload_root):
        all_folders = [
            name
            for name in os.listdir(upload_root)
            if os.path.isdir(os.path.join(upload_root, name))
        ]

        # Filter for folders that have a TOOCR subdirectory
        for folder in all_folders:
            toocr_path = os.path.join(upload_root, folder, "TOOCR")
            if os.path.exists(toocr_path) and os.path.isdir(toocr_path):
                ocr_ready_folders.append(folder)

        ocr_ready_folders.sort()

    # Get PNG files if a folder is selected
    if selected_folder and selected_folder in ocr_ready_folders:
        toocr_path = os.path.join(upload_root, selected_folder, "TOOCR")

        if os.path.exists(toocr_path):
            # Get all PNG files in the TOOCR folder
            png_files = [
                f for f in os.listdir(toocr_path) if f.lower().endswith(".png")
            ]
            png_files.sort()

    # Get segment data if a PNG and segment are selected
    segment_data = None
    if selected_folder and selected_png and selected_segment:
        # Get the base filename without extension for finding segments
        base_filename = os.path.splitext(selected_png)[0]

        # Check if segment exists and get its data
        segment_json_path = os.path.join(
            upload_root,
            selected_folder,
            "TOOCR",
            "segments",
            base_filename,
            f"{selected_segment}.json",
        )

        if os.path.exists(segment_json_path):
            try:
                with open(segment_json_path, "r") as f:
                    segment_data = json.load(f)
            except Exception as e:
                current_app.logger.error(f"Error loading segment data: {str(e)}")

    # New logic to determine image filename using selected_segment as an INDEX
    resolved_segment_image_filename = None
    selected_segment_id_display = selected_segment  # What to show in the H5 tag

    if selected_segment is not None:  # selected_segment is from the dropdown
        page_id = request.args.get("selected_png").split(".")[0]

        # Logic to load segment_data (JSON) - this might use actual_segment_data_to_load
        # ... ensure segment_data is loaded ...

        segment_image_directory = os.path.join(
            current_app.config["UPLOAD_FOLDER"],
            session.get("user_id", "anonymous"),
            selected_folder,
            "TOOCR",
            page_id,
        )
        if os.path.isdir(segment_image_directory):
            try:
                # Ensure PNGs are sorted consistently (e.g., numerically if names are "0.png", "1.png", "10.png")
                # A natural sort function might be needed for robust sorting of names like "1.png", "2.png", "10.png".
                # For simple "0.png", "1.png" ... "9.png", "10.png", basic sort is fine.
                png_filenames = sorted(
                    [
                        f
                        for f in os.listdir(segment_image_directory)
                        if f.lower().endswith(".png")
                    ]
                )

                segment_index_from_dropdown = int(
                    selected_segment
                )  # Assumes selected_segment is a 0-based index string

                if 0 <= segment_index_from_dropdown < len(png_filenames):
                    resolved_segment_image_filename = png_filenames[
                        segment_index_from_dropdown
                    ]
                    # If you also want to display the original ID (if it was different from index)
                    # selected_segment_id_display = os.path.splitext(resolved_segment_image_filename)[0]
                else:
                    current_app.logger.warning(
                        f"Segment index {segment_index_from_dropdown} out of bounds."
                    )
                    # Fallback: try to use selected_segment as a direct name if index fails
                    if f"{selected_segment}.png" in png_filenames:
                        resolved_segment_image_filename = f"{selected_segment}.png"
                    else:
                        resolved_segment_image_filename = (
                            "error_image_not_found.png"  # Placeholder
                        )
            except ValueError:
                current_app.logger.error(
                    f"Could not convert selected_segment '{selected_segment}' to an integer index. Trying as direct filename."
                )
                # Fallback: treat selected_segment as a direct filename base
                if os.path.exists(
                    os.path.join(segment_image_directory, f"{selected_segment}.png")
                ):
                    resolved_segment_image_filename = f"{selected_segment}.png"
                else:
                    resolved_segment_image_filename = (
                        "error_filename_invalid.png"  # Placeholder
                    )
            except Exception as e:
                current_app.logger.error(f"Error resolving segment image filename: {e}")
                resolved_segment_image_filename = (
                    "error_processing_image_list.png"  # Placeholder
                )
        else:
            current_app.logger.warning(
                f"Segment image directory not found: {segment_image_directory}"
            )
            resolved_segment_image_filename = (
                "error_directory_not_found.png"  # Placeholder
            )

    return render_template(
        "image_preview.html",
        folders=ocr_ready_folders,
        selected_folder=selected_folder,
        png_files=png_files,
        selected_segment=selected_segment,
        segment_data=segment_data,
        resolved_segment_image_filename=resolved_segment_image_filename,
        selected_segment_id_display=selected_segment_id_display,
    )


@app.route(
    "/crop", methods=["GET"]
)  # Assuming GET for now, POST would handle crop submission
def crop_image():
    user_id = session.get("user_id", "anonymous")
    upload_root = os.path.join(current_app.config["UPLOAD_FOLDER"], str(user_id))

    # Get folders
    folders = []
    if os.path.exists(upload_root):
        folders = [
            name
            for name in os.listdir(upload_root)
            if os.path.isdir(os.path.join(upload_root, name))
        ]
        folders.sort()

    # Get selected folder from query parameters
    selected_folder = request.args.get("selected_folder")
    image_to_crop_filename = request.args.get("image_file")

    # Initialize png_files list and cropped_images list
    png_files_in_selected_folder = []
    cropped_images = []

    # If a folder is selected, get the PNG files in that folder
    if selected_folder:
        folder_path = os.path.join(upload_root, selected_folder)

        if os.path.exists(folder_path):
            try:
                # List all files in the folder
                all_files = os.listdir(folder_path)

                # Filter for PNG files only (exclude cropped subfolder)
                png_files_in_selected_folder = [
                    f for f in all_files if f.lower().endswith(".png")
                ]

                # Sort the files by filename
                png_files_in_selected_folder.sort(
                    key=lambda x: (
                        int(os.path.splitext(x)[0])
                        if os.path.splitext(x)[0].isdigit()
                        else 0
                    )
                )

                # Check for cropped images if an image is selected
                if image_to_crop_filename:
                    cropped_dir = os.path.join(folder_path, "cropped")
                    original_page_base = os.path.splitext(image_to_crop_filename)[0]

                    if os.path.exists(cropped_dir):
                        # Get all cropped images for the selected page
                        cropped_images = [
                            f
                            for f in os.listdir(cropped_dir)
                            if f.startswith(f"{original_page_base}_cropped_")
                            and f.endswith(".png")
                        ]
                        cropped_images.sort()

            except Exception as e:
                current_app.logger.error(f"Error listing files: {str(e)}")

    return render_template(
        "crop_menu.html",
        folders=folders,
        selected_folder=selected_folder,
        png_files_in_selected_folder=png_files_in_selected_folder,
        image_to_crop_filename=image_to_crop_filename,
        cropped_images=cropped_images,
    )


@app.route("/uploads/<user_id>/<folder>/<filename>")
@app.route("/s/<user_id>/<folder>/<filename>")  # Add this for the malformed URL
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
def admin():
    # Check if user has admin privileges - implement your authentication logic here
    upload_root = current_app.config["UPLOAD_FOLDER"]
    users = []
    total_storage = 0

    if os.path.exists(upload_root):
        users = [
            name
            for name in os.listdir(upload_root)
            if os.path.isdir(os.path.join(upload_root, name))
        ]
        for user in users:
            user_path = os.path.join(upload_root, user)
            user_size = sum(
                os.path.getsize(os.path.join(dirpath, filename))
                for dirpath, _, filenames in os.walk(user_path)
                for filename in filenames
            )
            total_storage += user_size

    return render_template(
        "admin.html", users=users, total_storage=total_storage, total_users=len(users)
    )


@app.route("/fs-check")
def fs_check():
    """Diagnostic endpoint to check file system"""
    upload_dir = current_app.config["UPLOAD_FOLDER"]
    results = []

    # Check if upload dir exists
    results.append(f"Upload dir ({upload_dir}) exists: {os.path.exists(upload_dir)}")

    # List upload dir contents
    if os.path.exists(upload_dir):
        results.append(f"Upload dir contents: {os.listdir(upload_dir)}")

        # Check anonymous dir
        anon_dir = os.path.join(upload_dir, "anonymous")
        results.append(f"Anonymous dir ({anon_dir}) exists: {os.path.exists(anon_dir)}")

        if os.path.exists(anon_dir):
            results.append(f"Anonymous dir contents: {os.listdir(anon_dir)}")

            # Try to find the boarding pass dir
            boarding_dirs = [d for d in os.listdir(anon_dir) if "boarding" in d.lower()]
            results.append(f"Found boarding dirs: {boarding_dirs}")

            # List boarding dir contents if found
            for bd in boarding_dirs:
                bd_path = os.path.join(anon_dir, bd)
                if os.path.exists(bd_path):
                    results.append(f"Dir {bd_path} contents: {os.listdir(bd_path)}")

    return "<br>".join(results), 200, {"Content-Type": "text/html"}


@app.route("/process_cropped_image", methods=["POST"])
def process_cropped_image():
    # Get form data
    original_folder = request.form.get("original_folder")
    original_filename = request.form.get("original_filename")
    cropped_image_data = request.form.get("cropped_image_data")

    # Validate input
    if not all([original_folder, original_filename, cropped_image_data]):
        flash("Missing required crop parameters", "danger")
        return redirect(url_for("app.crop_image"))

    # Get the user ID (or use anonymous)
    user_id = session.get("user_id", "anonymous")

    # Define the directory where the original file is stored
    original_dir = os.path.join(
        current_app.config["UPLOAD_FOLDER"],
        str(user_id),
        original_folder,
    )

    # Create a 'cropped' subfolder if it doesn't exist
    cropped_dir = os.path.join(original_dir, "cropped")
    os.makedirs(cropped_dir, exist_ok=True)

    # Get the base filename without extension (e.g., "001" from "001.png")
    original_page_base = os.path.splitext(original_filename)[0]

    # Find existing cropped images to determine the next number
    existing_crops = [
        f
        for f in os.listdir(cropped_dir)
        if f.startswith(f"{original_page_base}_cropped_")
    ]

    # Extract numbers from existing crop files using regex
    crop_numbers = []
    for crop_file in existing_crops:
        match = re.search(r"_cropped_(\d{3})\.png$", crop_file)
        if match:
            crop_numbers.append(int(match.group(1)))

    # Determine the next crop number
    next_crop_num = 1
    if crop_numbers:
        next_crop_num = max(crop_numbers) + 1

    # Format the new filename: original_page_cropped_XXX.png
    cropped_filename = f"{original_page_base}_cropped_{next_crop_num:03d}.png"
    cropped_file_path = os.path.join(cropped_dir, cropped_filename)

    try:
        # Remove the data:image/png;base64, part from the data URL
        image_data = re.sub(r"^data:image/\w+;base64,", "", cropped_image_data)

        # Decode the base64 data
        binary_data = base64.b64decode(image_data)

        # Save the cropped image
        with open(cropped_file_path, "wb") as f:
            f.write(binary_data)

        current_app.logger.info(f"Saved cropped image to {cropped_file_path}")
        flash(f"Successfully saved cropped image as {cropped_filename}", "success")

    except Exception as e:
        current_app.logger.error(f"Error saving cropped image: {str(e)}")
        flash(f"Error saving cropped image: {str(e)}", "danger")

    # Redirect back to the crop page with the same folder and file selected
    return redirect(
        url_for(
            "app.crop_image",
            selected_folder=original_folder,
            image_file=original_filename,
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
def process_ocr():
    """Proxy request to OCR service and return the results"""
    data = request.json

    if not data or not data.get("folder") or not data.get("filename"):
        return jsonify({"error": "Missing folder or filename"}), 400

    # Get the file path
    user_id = session.get("user_id", "anonymous")
    folder = data.get("folder")
    filename = data.get("filename")

    # Get the base filename (without extension) for directory structure
    base_filename = os.path.splitext(filename)[0]

    # Construct the full path to the image file
    full_path = os.path.join(
        "/app/uploads",  # This is the path inside both containers
        str(user_id),
        folder,
        "TOOCR",
        filename,
    )

    current_app.logger.info(f"Processing OCR request for file: {full_path}")

    try:
        # Call the OCR service
        ocr_service_url = current_app.config.get("OCR_SERVICE_URL", "http://ocr:8000")

        # Use the /process_image/ endpoint of the OCR service
        response = requests.post(
            f"{ocr_service_url}/process_image/", json={"value": full_path}
        )

        # Log the response status
        current_app.logger.info(f"OCR service response status: {response.status_code}")

        # Debug the raw response
        raw_response = response.text
        current_app.logger.info(
            f"OCR service raw response: {raw_response[:500]}..."
        )  # Log first 500 chars

        # Check for successful response
        if response.status_code == 200:
            try:
                ocr_result = response.json()
                current_app.logger.info(f"OCR result keys: {list(ocr_result.keys())}")

                # Process the result - handling different structures
                if "ret" in ocr_result:
                    ret_value = ocr_result["ret"]

                    # Handle the nested dictionary case with 'status' and 'file' keys
                    if isinstance(ret_value, dict) and "status" in ret_value:
                        # Handle the dictionary response format
                        status = ret_value.get("status")
                        file_path = ret_value.get("file", "")

                        current_app.logger.info(
                            f"OCR returned status: {status} for file: {file_path}"
                        )

                        # Create segments parent directory
                        segments_parent_dir = os.path.join(
                            current_app.config["UPLOAD_FOLDER"],
                            str(user_id),
                            folder,
                            "TOOCR",
                            "segments",
                        )
                        os.makedirs(segments_parent_dir, exist_ok=True)

                        # Create a single segment with the file info
                        segment = {
                            "id": "000",
                            "text": f"OCR status: {status}",
                            "has_image": False,
                            "file": file_path,
                        }

                        # Save combined segments JSON
                        combined_json_path = os.path.join(
                            segments_parent_dir, f"{base_filename}.json"
                        )
                        with open(combined_json_path, "w") as f:
                            json.dump([segment], f)

                        return jsonify(
                            {
                                "success": True,
                                "status": status,
                                "file": file_path,
                                "segments": [segment],
                                "page_id": base_filename,
                            }
                        )

                    # Handle list of segments (your existing logic)
                    elif isinstance(ret_value, list):
                        # Format the result for display
                        result_text = ""
                        segments = ret_value

                        # Your existing code for handling segment list
                        current_app.logger.info(f"Found {len(segments)} segments")

                        # Create parent segments directory if it doesn't exist
                        segments_parent_dir = os.path.join(
                            current_app.config["UPLOAD_FOLDER"],
                            str(user_id),
                            folder,
                            "TOOCR",
                            "segments",
                        )
                        os.makedirs(segments_parent_dir, exist_ok=True)

                        # Create the page-specific segments directory
                        segments_dir = os.path.join(segments_parent_dir, base_filename)
                        os.makedirs(segments_dir, exist_ok=True)

                        # Create a combined segments JSON file at the parent level for the page
                        all_segments = []

                        # Save segment info and line images
                        for idx, segment in enumerate(segments):
                            if not isinstance(segment, dict):
                                current_app.logger.warning(
                                    f"Segment {idx} is not a dictionary: {type(segment)}"
                                )
                                continue

                            segment_text = segment.get("text", "")
                            result_text += f"{segment_text}\n"
                            segment_id = f"{idx:03d}"

                            # Add ID to the segment for future reference
                            segment["id"] = segment_id
                            segment["has_image"] = (
                                "image_data" in segment
                            )  # Flag to indicate image exists

                            all_segments.append(segment)

                            # Save individual segment info to JSON file in page directory
                            segment_info = {
                                "id": segment_id,
                                "coords": segment.get("coords", []),
                                "text": segment_text,
                                "has_image": "image_data" in segment,
                            }
                            json_path = os.path.join(segments_dir, f"{segment_id}.json")
                            with open(json_path, "w") as f:
                                json.dump(segment_info, f)

                            # Save segment image if available
                            if "image_data" in segment and segment["image_data"]:
                                try:
                                    # Convert base64 to image and save
                                    img_data = re.sub(
                                        r"^data:image/\w+;base64,",
                                        "",
                                        segment["image_data"],
                                    )
                                    img_bytes = base64.b64decode(img_data)
                                    img_path = os.path.join(
                                        segments_dir, f"{segment_id}.png"
                                    )
                                    with open(img_path, "wb") as img_file:
                                        img_file.write(img_bytes)
                                    current_app.logger.info(
                                        f"Saved segment image: {img_path}"
                                    )
                                except Exception as img_error:
                                    current_app.logger.error(
                                        f"Error saving segment image: {str(img_error)}"
                                    )
                                    segment["has_image"] = False

                        # Save combined segments JSON for the page
                        combined_json_path = os.path.join(
                            segments_parent_dir, f"{base_filename}.json"
                        )
                        with open(combined_json_path, "w") as f:
                            json.dump(all_segments, f)

                        current_app.logger.info(
                            f"Saved combined segments JSON: {combined_json_path}"
                        )

                        return jsonify(
                            {
                                "success": True,
                                "text": result_text,
                                "segments": all_segments,
                                "page_id": base_filename,
                            }
                        )

                    # Handle string result
                    elif isinstance(ret_value, str):
                        # Handle case where "ret" is a string (e.g., plain text OCR without segments)
                        current_app.logger.info(
                            "OCR returned plain text without segments"
                        )
                        result_text = ret_value

                        # Create an artificial segment for the whole page
                        segments_parent_dir = os.path.join(
                            current_app.config["UPLOAD_FOLDER"],
                            str(user_id),
                            folder,
                            "TOOCR",
                            "segments",
                        )
                        os.makedirs(segments_parent_dir, exist_ok=True)

                        # Create a single segment
                        segment = {"id": "000", "text": result_text, "has_image": False}

                        # Save combined segments JSON
                        combined_json_path = os.path.join(
                            segments_parent_dir, f"{base_filename}.json"
                        )
                        with open(combined_json_path, "w") as f:
                            json.dump([segment], f)

                        return jsonify(
                            {
                                "success": True,
                                "text": result_text,
                                "segments": [segment],
                                "page_id": base_filename,
                            }
                        )

                    else:
                        # Unknown format
                        current_app.logger.error(
                            f"Unhandled OCR response type: {type(ret_value)}"
                        )
                        return (
                            jsonify(
                                {
                                    "error": f"Unhandled OCR response type: {type(ret_value)}"
                                }
                            ),
                            500,
                        )
                else:
                    # No "ret" key
                    current_app.logger.error(
                        f"OCR response missing 'ret' key: {ocr_result}"
                    )
                    return (
                        jsonify(
                            {"error": "OCR response missing expected data structure"}
                        ),
                        500,
                    )

            except Exception as parse_error:
                current_app.logger.error(
                    f"Error parsing OCR JSON response: {str(parse_error)}"
                )
                import traceback

                current_app.logger.error(f"Traceback: {traceback.format_exc()}")
                return (
                    jsonify(
                        {"error": f"Error parsing OCR response: {str(parse_error)}"}
                    ),
                    500,
                )
        else:
            # Non-200 response
            current_app.logger.error(f"OCR service error: {response.text}")
            return (
                jsonify(
                    {"error": f"OCR service returned status {response.status_code}"}
                ),
                response.status_code,
            )

    except requests.RequestException as e:
        current_app.logger.error(f"Error connecting to OCR service: {str(e)}")
        return jsonify({"error": f"Error connecting to OCR service: {str(e)}"}), 500
    except Exception as e:
        current_app.logger.error(f"Unexpected error during OCR processing: {str(e)}")
        import traceback

        current_app.logger.error(f"Traceback: {traceback.format_exc()}")
        return jsonify({"error": f"Unexpected error: {str(e)}"}), 500


@app.route("/get_segments/<user_id>/<folder>/<page_id>")
def get_segments(user_id, folder, page_id):
    """Return segments for a specific page"""
    try:
        # First try to find the combined JSON file (stored at the parent level)

        # Then check for a directory of individual segment files
        segments_dir = os.path.join(
            current_app.config["UPLOAD_FOLDER"],
            user_id,
            folder,
            "TOOCR",
            page_id,
        )

        import glob

        pngs = glob.glob(segments_dir + "/*.png")
        pngs = sorted(pngs)
        for png in pngs:
            logger.info(f"PNG FILE IS {png}")

        logger.info(f"Segments path = {segments_dir}")
        # Try the directory of individual segment files
        if os.path.exists(segments_dir) and os.path.isdir(segments_dir):
            segments_data = []
            json_files = [f for f in os.listdir(segments_dir) if f.endswith(".json")]
            json_files.sort()  # Sort to maintain order

            for json_file in json_files:
                try:
                    with open(os.path.join(segments_dir, json_file), "r") as f:
                        segment_data = json.load(f)
                    segments_data.append(segment_data)
                except Exception as e:
                    current_app.logger.error(
                        f"Error loading segment file {json_file}: {str(e)}"
                    )

            current_app.logger.info(
                f"Found {len(segments_data)} individual segment files"
            )
            return jsonify({"success": True, "segments": segments_data})

    except Exception as e:
        current_app.logger.error(f"Error retrieving segments: {str(e)}")
        import traceback

        current_app.logger.error(f"Traceback: {traceback.format_exc()}")
        return jsonify({"success": False, "error": str(e)})


@app.route("/get_segment_image/<user_id>/<folder>/<page_id>/<segment_id>")
def get_segment_image(user_id, folder, page_id, segment_id):
    """Serve a specific segment image"""
    # Corrected path: remove the extra "segments" directory
    segments_dir = os.path.join(
        current_app.config["UPLOAD_FOLDER"],
        user_id,
        folder,
        "TOOCR",
        page_id,  # Segment images are directly under the page_id directory
    )

    logger.info(f"SEGMENTS DIR: {segments_dir}")

    image_path = os.path.join(segments_dir, f"{int(segment_id):03d}.png")
    current_app.logger.info(f"Attempting to serve segment image: {image_path}")

    if not os.path.exists(segments_dir):
        current_app.logger.error(
            f"Segments directory for page not found: {segments_dir}"
        )
        return "Page segments directory not found", 404

    if not os.path.exists(image_path):
        current_app.logger.error(f"Segment image file not found: {image_path}")
        return "Segment image not found", 404

    try:
        return send_from_directory(segments_dir, f"{int(segment_id):03d}.png")
    except Exception as e:
        current_app.logger.error(f"Error serving segment image: {str(e)}")
        return f"Error serving segment image: {str(e)}", 500


@app.route("/get_segments_1/<user_id>/<folder>/<page_id>")
def get_segments_1(user_id, folder, page_id):
    """Return segments for a specific page"""
    try:
        # First try to find the combined JSON file (stored at the parent level)

        # Then check for a directory of individual segment files
        segments_dir = os.path.join(
            current_app.config["UPLOAD_FOLDER"],
            user_id,
            folder,
            "TOOCR",
            page_id,
        )

        import glob

        pngs = glob.glob(segments_dir + "/*.png")
        pngs = sorted(pngs)
        for png in pngs:
            logger.info(f"PNG FILE IS {png}")

        logger.info(f"Segments path = {segments_dir}")
        # Try the directory of individual segment files
        if os.path.exists(segments_dir) and os.path.isdir(segments_dir):
            segments_data = []
            json_files = [f for f in os.listdir(segments_dir) if f.endswith(".json")]
            json_files.sort()  # Sort to maintain order

            for json_file in json_files:
                try:
                    with open(os.path.join(segments_dir, json_file), "r") as f:
                        segment_data = json.load(f)
                    segments_data.append(segment_data)
                except Exception as e:
                    current_app.logger.error(
                        f"Error loading segment file {json_file}: {str(e)}"
                    )

            current_app.logger.info(
                f"Found {len(segments_data)} individual segment files"
            )
            return jsonify({"success": True, "segments": segments_data})

    except Exception as e:
        current_app.logger.error(f"Error retrieving segments: {str(e)}")
        import traceback

        current_app.logger.error(f"Traceback: {traceback.format_exc()}")
        return jsonify({"success": False, "error": str(e)})


@app.route("/get_segment_json/<user_id>/<folder>/<page_id>/<segment_id>")
def get_segment_json(user_id, folder, page_id, segment_id):
    """Serve a specific segment image"""
    # Corrected path: remove the extra "segments" directory
    segments_dir = os.path.join(
        current_app.config["UPLOAD_FOLDER"],
        user_id,
        folder,
        "TOOCR",
        page_id,  # Segment images are directly under the page_id directory
    )

    logger.info(f"SEGMENTS DIR: {segments_dir}")

    json_path = os.path.join(segments_dir, f"{int(segment_id):03d}.json")
    current_app.logger.info(f"Attempting to serve segment image: {json_path}")

    if not os.path.exists(segments_dir):
        current_app.logger.error(
            f"Segments directory for page not found: {segments_dir}"
        )
        return "Page segments directory not found", 404

    if not os.path.exists(json_path):
        current_app.logger.error(f"Segment image file not found: {json_path}")
        return "Segment image not found", 404

    try:
        return send_from_directory(segments_dir, f"{int(segment_id):03d}.json")
    except Exception as e:
        current_app.logger.error(f"Error serving segment json: {str(e)}")
        return f"Error serving segment json: {str(e)}", 500


@app.route("/get_segment_preview/<user_id>/<folder>/<page_id>/<segment_index>")
def get_segment_preview(user_id, folder, page_id, segment_index):
    try:
        segments_dir = os.path.join(
            current_app.config["UPLOAD_FOLDER"], user_id, folder, "TOOCR", page_id
        )

        if not os.path.exists(segments_dir):
            return (
                jsonify({"success": False, "error": "Segments directory not found"}),
                404,
            )

        segment_files = sorted(
            [f for f in os.listdir(segments_dir) if f.endswith(".json")]
        )
        segment_idx = int(segment_index)

        if not (0 <= segment_idx < len(segment_files)):
            return (
                jsonify({"success": False, "error": "Segment index out of bounds"}),
                404,
            )

        json_filename = segment_files[segment_idx]
        with open(os.path.join(segments_dir, json_filename), "r") as f:
            segment_data = json.load(f)

        # Generate image URL
        image_url = url_for(
            "app.get_segment_image",
            user_id=user_id,
            folder=folder,
            page_id=page_id,
            segment_id=segment_data.get("id", segment_idx),
        )

        return jsonify(
            {
                "success": True,
                "data": {
                    "text": segment_data.get("text", ""),
                    "imageUrl": image_url,
                    "confidence": segment_data.get("confidence"),
                    "id": segment_data.get("id"),
                    "coords": segment_data.get("coords", []),
                },
            }
        )

    except Exception as e:
        current_app.logger.error(f"Error getting segment preview: {str(e)}")
        return jsonify({"success": False, "error": str(e)}), 500
