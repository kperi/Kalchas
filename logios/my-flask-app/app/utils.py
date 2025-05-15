def allowed_file(filename):
    allowed_extensions = {"png", "jpg", "jpeg", "gif", "pdf"}
    return "." in filename and filename.rsplit(".", 1)[1].lower() in allowed_extensions


def process_image(image):

    from PIL import Image
    from io import BytesIO

    img = Image.open(image)
    return img


def resize_image(image, target_size):
    from PIL import Image
    from io import BytesIO

    img = Image.open(image)
    img.thumbnail(target_size)
    img_io = BytesIO()
    img.save(img_io, format="JPEG")
    img_io.seek(0)
    return img_io


def crop_image(image, crop_box):
    from PIL import Image
    from io import BytesIO

    img = Image.open(image)
    cropped_img = img.crop(crop_box)
    img_io = BytesIO()
    cropped_img.save(img_io, format="JPEG")
    img_io.seek(0)
    return img_io


def save_file(file, upload_folder):
    import os

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(upload_folder, filename)
        file.save(file_path)
        return filename
    return None
