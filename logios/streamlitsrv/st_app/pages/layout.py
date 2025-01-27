import streamlit as st
from auth_utils import do_login, system_loop
import os
import glob
import requests
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
import numpy as np

class_colors = {
    "plain text": "#FF5722",  # Blue
    "title": "#009688",  # Red
    "figure": "#4CAF50",  # Green
    "table": "#9C27B0",  # Purple
    "list": "#2196F3",  # Orange
    "header": "#795548",  # Brown
    "footer": "#607D8B",  # Blue Grey
    "page_number": "#FF9800",  # Deep Orange
    "caption": "#009688",  # Teal
    "footnote": "#795548",  # Brown
}
default_color = "#9E9E9E"  # Grey for unknown classes


def draw_block_rectangle(
    draw, x1, y1, x2, y2, cls_name, class_colors, default_color, font
):
    """Draw a rectangle around a detected block with its class label."""
    # Get color for this class
    border_color = class_colors.get(cls_name.lower(), default_color)

    # Draw the bounding box
    draw.rectangle([x1, y1, x2, y2], outline=border_color, width=5)

    # Calculate text dimensions for the class label
    text_size = draw.textlength(cls_name, font=font)
    padding = 10

    # Draw background rectangle for class label
    label_x1, label_y1 = x1, y1 - font.size - (padding * 2)
    label_x2 = x1 + text_size + (padding * 2)
    label_y2 = y1
    draw.rectangle([label_x1, label_y1, label_x2, label_y2], fill=border_color)

    # Draw class name text
    text_x = x1 + padding
    text_y = y1 - font.size - padding
    draw.text((text_x, text_y), cls_name, fill="white", font=font)


def draw_block_index(draw, x1, x2, y1, idx, font):
    """Draw the block index number in the top right corner."""
    padding = 10
    index_text = str(idx)
    index_text_size = draw.textlength(index_text, font=font)

    # Draw background rectangle for index
    index_x1 = x2 - index_text_size - (padding * 2)
    index_y1 = y1 - font.size - (padding * 2)
    index_x2 = x2
    index_y2 = y1
    draw.rectangle([index_x1, index_y1, index_x2, index_y2], fill="blue")

    # Draw index number
    draw.text(
        (index_x1 + padding, index_y1 + padding), index_text, fill="white", font=font
    )


def ocr_image_segment(path_to_file: str, x1: int, y1: int, x2: int, y2: int):
    url = "http://ocr:8000/ocr_image_segment"

    payload = {"path": path_to_file, "x1": x1, "y1": y1, "x2": x2, "y2": y2}
    response = requests.post(url, json=payload)
    return response


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


login_status, user_todo, user_uploads, user_workspace = do_login()
active_user = st.session_state["name"]

st.sidebar.markdown(f"User: {active_user}")

books_path = user_workspace + "/TODO/"
# st.write(f"books Workspace: {books_path }")

book_folders = sorted(glob.glob(os.path.join(books_path, "*")))
book_folders = [
    os.path.basename(book_folder)
    for book_folder in book_folders
    if os.path.isdir(book_folder)
]


def set_state():
    st.session_state.index = 0


confidence = st.sidebar.slider("Detection confidence", 0.0, 1.0, step=0.1, value=0.5)

book = st.selectbox("Κείμενο: ", book_folders, on_change=set_state, index=None)
if book is not None:
    book_pages = sorted(glob.glob(os.path.join(books_path, book, "*.png")))

    select_pages = [os.path.basename(page) for page in book_pages]

    page_selector = st.selectbox("Σελίδα: ", select_pages, on_change=set_state)

    page_no = page_selector.split("/")[-1].replace(".png", "")
    page_path_selected = books_path + "/" + book + "/" + page_no + ".png"
    page_path_selected = page_path_selected.replace("/*", "")

    segments_files_path = books_path + "/" + book + "/" + page_no + "/*.png"
    files = sorted(glob.glob(segments_files_path))

    tabs = st.tabs(["Layout Detection", "OCR Results"])

    with tabs[0]:
        if page_path_selected is not None:
            with st.spinner(f"Layout detection in progress... "):
                # Define color scheme for different block types

                ret = process_image(page_path_selected, threshold=confidence)
                if ret.status_code != 200:
                    error_message = ret.text
                    st.error(f"Error : {ret} - {error_message}")
                else:
                    j_resp = ret.json()
                    coords = j_resp["block_coords"]
                    df = pd.DataFrame(
                        data=coords, columns=["y1", "x1", "y2", "x2", "cls_name"]
                    )

                    df = filter_overlapping_rectangles(df)

                    # Sort blocks by reading order:
                    # First divide the page into rows by grouping blocks with overlapping y-coordinates
                    # Then sort within each row by x-coordinate

                    # Calculate the center point of each block
                    df["center_y"] = (df["y1"] + df["y2"]) / 2
                    df["center_x"] = (df["x1"] + df["x2"]) / 2

                    # Define a threshold for considering blocks to be in the same row
                    # (adjust this value based on your specific needs)
                    row_threshold = (df["y2"] - df["y1"]).mean() * 0.5

                    # Assign row numbers to blocks
                    current_row = 0
                    row_assignments = []
                    sorted_by_y = df.sort_values("center_y")

                    current_row_y = float("-inf")
                    for _, block in sorted_by_y.iterrows():
                        if block["center_y"] > current_row_y + row_threshold:
                            current_row += 1
                            current_row_y = block["center_y"]
                        row_assignments.append(current_row)

                    df["row"] = row_assignments

                    # Sort first by row number, then by x position within each row
                    df = df.sort_values(["center_y", "row"]).reset_index(drop=True)

                    # Drop the temporary columns
                    df = df.drop(["center_y", "center_x", "row"], axis=1)

                    # Display the annotated image
                    np_image = np.array(Image.open(page_path_selected))
                    image_with_rectangle = Image.fromarray(np_image)
                    image_with_rectangle = image_with_rectangle.convert("RGB")

                    abs_file_path = os.path.dirname(__file__)
                    font_name = f"{abs_file_path}/verdana.ttf"
                    font = ImageFont.truetype(font_name, size=24)
                    draw = ImageDraw.Draw(image_with_rectangle)

                    idx = 0  # df.shape[0]
                    for _, row in df.iterrows():
                        y1, x1, y2, x2, cls_name = row

                        # Draw the block rectangle and its class label
                        draw_block_rectangle(
                            draw,
                            x1,
                            y1,
                            x2,
                            y2,
                            cls_name,
                            class_colors,
                            default_color,
                            font,
                        )

                        # Draw the block index
                        draw_block_index(draw, x1, x2, y1, idx, font)
                        idx += 1

                    st.image(image_with_rectangle, use_container_width=True)

    with tabs[1]:
        if page_path_selected is not None:
            with st.spinner("Performing OCR"):
                # Create an expander for each text block
                for idx, row in df.iterrows():
                    y1, x1, y2, x2, cls_name = row
                    with st.expander(f"Block {idx} ({cls_name})"):
                        ret = ocr_image_segment(
                            path_to_file=page_path_selected,
                            x1=x1,
                            y1=y1,
                            x2=x2,
                            y2=y2,
                        )
                        ocred_text = ret.json()["ret"]["recognized_text"]
                        st.text_area(
                            label=f"Block {idx}",
                            value=ocred_text,
                            height=100,
                            key=f"text_{idx}",
                        )

                # Add a full text view
                with st.expander("View All Text"):
                    all_text = "\n\n".join(
                        [st.session_state[f"text_{idx}"] for idx in range(len(df))]
                    )
                    st.text_area(label="Complete Text", value=all_text, height=300)
