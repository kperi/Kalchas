import streamlit as st
from auth_utils import do_login, system_loop
import os
import glob
import requests
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
import numpy as np


def get_overlap_area(rect1, rect2):
    x1 = max(rect1[1], rect2[1])
    y1 = max(rect1[0], rect2[0])
    x2 = min(rect1[3], rect2[3])
    y2 = min(rect1[2], rect2[2])
    overlap_width = max(0, x2 - x1)
    overlap_height = max(0, y2 - y1)
    return overlap_width * overlap_height


def filter_overlapping_rectangles(df, threshold=0.95):
    filtered_df = df.copy()
    for i, row1 in df.iterrows():
        for j, row2 in df.iterrows():
            if i >= j:
                continue
            rect1 = row1[["y1", "x1", "y2", "x2"]].values
            rect2 = row2[["y1", "x1", "y2", "x2"]].values
            overlap_area = get_overlap_area(rect1, rect2)
            area1 = (rect1[2] - rect1[0]) * (rect1[3] - rect1[1])
            area2 = (rect2[2] - rect2[0]) * (rect2[3] - rect2[1])
            if overlap_area / min(area1, area2) > threshold:
                if area1 < area2:
                    filtered_df.drop(i, inplace=True)
                else:
                    filtered_df.drop(j, inplace=True)
    return filtered_df


def process_image(path_to_file, threshold=0.5):
    """
    Send a POST request to the /submit/ endpoint of a FastAPI server.

    :param value: The string value to be sent in the request body.
    :param url: The URL of the FastAPI endpoint.
    :return: The response from the server.
    """
    url = "http://doclayout:8001/process_image"
    payload = {"value": path_to_file, "threshold": threshold}
    response = requests.post(url, json=payload)
    return response


login_status, user_workspace, _ = do_login()
active_user = st.session_state["name"]
st.sidebar.markdown(f"User: {active_user}")

books_path = user_workspace
# st.write(f"Workspace: {books_path }")

book_folders = sorted(glob.glob(os.path.join(books_path, "*")))
book_folders = [
    os.path.basename(book_folder)
    for book_folder in book_folders
    if os.path.isdir(book_folder)
]


def set_state():
    st.session_state.index = 0


confidence = st.sidebar.slider("Detection confidence", 0.0, 1.0, step=0.1, value=0.5)

book = st.selectbox("Κείμενο: ", book_folders, on_change=set_state)
book_pages = sorted(glob.glob(os.path.join(books_path, book, "*.png")))

select_pages = [os.path.basename(page) for page in book_pages]

page_selector = st.selectbox("Σελίδα: ", select_pages, on_change=set_state)
page_no = page_selector.split("/")[-1].replace(".png", "")
page_path_selected = books_path + "/" + book + "/" + page_no + ".png"

segments_files_path = books_path + "/" + book + "/" + page_no + "/*.png"
files = sorted(glob.glob(segments_files_path))
with st.container() as p:
    if page_path_selected is not None:
        col1, col2 = st.columns(spec=[0.5, 0.5])
        with col1:
            st.image([])  # Clear all images before processing
            with st.spinner("Layout detection in progress..."):
                ret = process_image(page_path_selected, threshold=confidence)
                if ret.status_code != 200:
                    error_message = ret.json().get("detail", "Unknown error")
                    st.error(f"Error : {error_message}")
                else:
                    j_resp = ret.json()
                    coords = j_resp["block_coords"]
                    df = pd.DataFrame(
                        data=coords, columns=["y1", "x1", "y2", "x2", "cls_name"]
                    ).sort_values(by=["y1", "x1"])

                    df = filter_overlapping_rectangles(df)
                    df = df.sort_values(by=["x1"])

                    np_image = np.array(Image.open(page_path_selected))
                    image_with_rectangle = Image.fromarray(np_image)
                    image_with_rectangle = image_with_rectangle.convert("RGB")

                    abs_file_path = os.path.dirname(__file__)

                    font_name = f"{abs_file_path}/verdana.ttf"
                    font = ImageFont.truetype(font_name, size=30)
                    draw = ImageDraw.Draw(image_with_rectangle)

                    for idx, row in df.iterrows():

                        y1, x1, y2, x2, cls_name = row

                        draw.rectangle([x1, y1, x2, y2], outline="red", width=5)

                        text_size = draw.textsize(cls_name, font=font)
                        text_x1, text_y1 = x1, y1 - text_size[1]
                        text_x2, text_y2 = x1 + text_size[0], y1

                        draw.rectangle([text_x1, text_y1, text_x2, text_y2], fill="red")
                        draw.text(
                            (x1, y1 - text_size[1]), cls_name, fill="green", font=font
                        )

                        index_text = str(idx)
                        index_text_size = draw.textsize(index_text, font=font)
                        index_text_x1, index_text_y1 = (
                            x2 - index_text_size[0],
                            y1 - index_text_size[1],
                        )
                        index_text_x2, index_text_y2 = x2, y1

                        draw.rectangle(
                            [
                                index_text_x1,
                                index_text_y1,
                                index_text_x2,
                                index_text_y2,
                            ],
                            fill="blue",
                        )
                        draw.text(
                            (index_text_x1, index_text_y1),
                            index_text,
                            fill="cyan",
                            font=font,
                        )

                    st.image(image_with_rectangle)
                with col2:
                    pass
            # st.image(page_path_selected, width=100, use_column_width=True)
            # st.image(crop)

            # selected_images = st.multiselect("Select segments to display", select_pages)
            # for selected_image in selected_images:
            #    segment_path = os.path.join(books_path, book, selected_image)
            #    segment_image = Image.open(segment_path)
            #    st.image(segment_image, width=100, use_column_width=True)

    # st.button("hello", key="match_width")
