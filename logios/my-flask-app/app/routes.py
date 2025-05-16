from flask import (
    Blueprint,
    render_template,
    request,
    redirect,
    url_for,
    flash,
    session,
    current_app,
    send_from_directory,
)
from werkzeug.utils import secure_filename
import os
from app.forms import UploadForm
from app.utils import allowed_file, process_image
from pdf2image import convert_from_path
from loguru import logger

app = Blueprint("app", __name__)

UPLOAD_FOLDER = "uploads"
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "gif", "pdf"}


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/upload_file", methods=["POST", "GET"])
def upload_file():

    return render_template("upload_file.html")

    if "file" not in request.files:
        # flash("No file part")
        return redirect(request.url)

    file = request.files["file"]

    if file.filename == "":
        flash("No selected file")
        return redirect(request.url)

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        user_id = session.get(
            "user_id", "anonymous"
        )  # Replace with real user_id if available
        base_name, ext = os.path.splitext(filename)
        user_folder = os.path.join(
            current_app.config["UPLOAD_FOLDER"], str(user_id), base_name
        )
        os.makedirs(user_folder, exist_ok=True)
        file_path = os.path.join(user_folder, filename)
        file.save(file_path)

        if ext.lower() == ".pdf":
            # Convert PDF to PNGs
            images = convert_from_path(file_path)
            for idx, image in enumerate(images, start=1):
                out_path = os.path.join(user_folder, f"{idx:03d}.png")
                image.save(out_path, "PNG")
            flash(f"PDF converted to {len(images)} PNG images.")
        else:
            # Save image as-is
            out_path = os.path.join(user_folder, filename)
            file.save(out_path)
            flash("Image uploaded.")

        session["uploaded_file"] = filename
        session["user_folder"] = user_folder
        return redirect(url_for("app.image_preview"))

    flash("File type not allowed")
    return redirect(request.url)


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

    folders = []
    if os.path.exists(upload_root):
        folders = [
            name
            for name in os.listdir(upload_root)
            if os.path.isdir(os.path.join(upload_root, name))
        ]
        folders.sort()  # Optional: sort folders

    selected_folder = request.args.get("selected_folder")
    logger.info(f"SELECTED FOLDER = {selected_folder}")
    # image_to_crop will be the filename of the image to crop.
    # This would typically be set after selecting a folder and then a file from that folder.
    # For now, let's assume it might come from another parameter or be None.
    image_to_crop_filename = request.args.get(
        "image_file"
    )  # Example: if you add a file selector later

    return render_template(
        "crop_menu.html",
        folders=folders,
        selected_folder=selected_folder,
        image_to_crop_filename=image_to_crop_filename,
    )


@app.route("/uploads/<user_id>/<folder>/<filename>")
def uploaded_file(user_id, folder, filename):
    # Ensure UPLOAD_FOLDER is the path *inside the container*
    upload_dir_base = current_app.config["UPLOAD_FOLDER"]

    # Construct the path to the directory containing the user's specific folder of images
    user_specific_folder_path = os.path.join(upload_dir_base, str(user_id), folder)
    logger.info(
        f"USER_SPECIFIC DIR = {user_specific_folder_path}, FILENAME = {filename}"
    )

    try:
        ret = send_from_directory(user_specific_folder_path, filename)
        logger.info(f"RET = {ret}")
        return ret
    except ValueError:  # FileNotFoundError:
        # Log the error or flash a message if you want to handle it more gracefully
        logger.error(
            f"File not found: {os.path.join(user_specific_folder_path, filename)}"
        )
        return "File not found", 404


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
