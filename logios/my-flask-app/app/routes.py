from flask import (
    Blueprint,
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
    logger.info(f"Upload path  = {upload_root}")
    folders = []
    png_files = []
    selected_folder = request.args.get("selected_folder")
    if os.path.exists(upload_root):
        folders = [
            name
            for name in os.listdir(upload_root)
            if os.path.isdir(os.path.join(upload_root, name))
        ]
    if selected_folder and selected_folder in folders:
        folder_path = os.path.join(upload_root, selected_folder)
        png_files = [f for f in os.listdir(folder_path) if f.lower().endswith(".png")]

    for file in png_files:
        full_path = os.path.join(folder_path, file)
        if not os.path.exists(full_path):
            raise FileNotFoundError
        else:
            logger.info(f"SELECTED FILE = {full_path}")

    logger.info(f"Png files = {png_files}")
    return render_template(
        "image_preview.html",
        folders=folders,
        selected_folder=selected_folder,
        png_files=png_files,
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

    # Initialize png_files list
    png_files_in_selected_folder = []

    # If a folder is selected, get the PNG files in that folder
    if selected_folder:
        folder_path = os.path.join(upload_root, selected_folder)

        # Debug output for server logs
        current_app.logger.info(f"Looking for PNG files in: {folder_path}")

        if os.path.exists(folder_path):
            try:
                # List all files in the folder
                all_files = os.listdir(folder_path)
                current_app.logger.info(f"All files in folder: {all_files}")

                # Filter for PNG files only
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

                current_app.logger.info(
                    f"PNG files found: {png_files_in_selected_folder}"
                )
            except Exception as e:
                current_app.logger.error(f"Error listing PNG files: {str(e)}")
                png_files_in_selected_folder = []
        else:
            current_app.logger.error(f"Selected folder does not exist: {folder_path}")

    return render_template(
        "crop_menu.html",
        folders=folders,
        selected_folder=selected_folder,
        png_files_in_selected_folder=png_files_in_selected_folder,
        image_to_crop_filename=image_to_crop_filename,
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
