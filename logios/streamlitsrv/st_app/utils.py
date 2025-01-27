

import requests
def process_image(path_to_file, url):
    """
    Send a POST request to the /submit/ endpoint of a FastAPI server.

    :param value: The string value to be sent in the request body.
    :param url: The URL of the FastAPI endpoint.
    :return: The response from the server.
    """
    payload = {"value": path_to_file}
    response = requests.post(url, json=payload)
    return response


def post_image_to_fastapi(image_path, url):
    """
    Post an image to a FastAPI server.

    :param image_path: Path to the image file to be uploaded.
    :param url: The URL of the FastAPI endpoint to which the image will be posted.
    :return: The response from the server.
    """
    with open(image_path, "rb") as image_file:
        files = {"file": image_file}
        response = requests.post(url, files=files)

    return response