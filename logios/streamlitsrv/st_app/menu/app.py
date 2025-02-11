import streamlit as st
import glob
import json
import yaml
import os
import cv2
from typing import List, Optional, Tuple

# from st_app.page import segmentation_and_recognition
from st_app.utils import process_image, post_image_to_fastapi
from st_app.page_management import (
    get_max_final,
    get_finalized_lines,
    get_all_lines,
    get_all_texts,
)

from yaml.loader import SafeLoader

from auth_utils import system_loop, init_session_state

init_session_state()
# st.set_page_config(layout="wide")

# if "authentication_status" not in st.session_state:
#    login_status, user_workspace, _, _, _ = do_login()


with open("./config.yaml") as file:
    config = yaml.load(file, Loader=SafeLoader)


with open("/app/st_app/vowel_table.txt") as f:
    VOWELS_TABLE = f.read()

# with open("/app/st_app/style.css") as css:
#    st.markdown(f"<style>{css.read()}</style>", unsafe_allow_html=True)

segment_files = []


def set_state() -> None:
    """Reset the index in session state to 0."""
    st.write("On page change")
    st.session_state.page_index = 0
    st.session_state.segment_index = 0


def book_page_change() -> None:
    """Reset the index in session state to 0."""
    # st.write("On page change")
    st.session_state.segment_index = 0


def navigate(step: int) -> None:
    """
    Update the current index in session state by the given step.

    Args:
        step (int): The number of steps to move (positive or negative)
    """
    st.session_state.segment_index += step
    st.session_state.segment_index = st.session_state.segment_index % (
        len(segment_files) - 1
    )


def get_finals(book_page_path: str) -> int:
    """
    Count the number of finalized files in the given directory.

    Args:
        book_page_path (str): Path to the directory containing the files

    Returns:
        int: Number of files with .final extension
    """
    book_page_path = book_page_path.replace("*.png", "")
    finals = glob.glob(os.path.join(book_page_path, "*.final"))
    return len(finals)


def get_ocred_text(file: str) -> str:
    """
    Extract OCR text from the corresponding JSON file.

    Args:
        file (str): Path to the image file

    Returns:
        str: OCRed text from the corresponding JSON file
    """
    jfile = file.replace(".png", ".json")
    jcontent = json.load(open(jfile))
    ocred_text = jcontent["text"][0]
    return ocred_text


def has_final(file: str) -> bool:
    """
    Check if a finalized version exists for the given file.

    Args:
        file (str): Path to the image file

    Returns:
        bool: True if a .final file exists, False otherwise
    """
    return os.path.exists(file.replace(".png", ".final"))


def do_ocr() -> None:
    """
    Process the current page through the OCR service.
    Displays a spinner during processing and writes the result.
    """
    with st.spinner("Running OCR on the page..."):
        ret = process_image(
            st.session_state.page_path_selected, "http://ocr:8000/process_image"
        )


def get_book_folders(books_path: str) -> List[str]:
    """
    Get a sorted list of book folder names from the given path.

    Args:
        books_path (str): Path to the directory containing book folders

    Returns:
        List[str]: Sorted list of book folder names
    """
    book_folders = sorted(glob.glob(os.path.join(books_path, "*")))
    book_folders = [
        os.path.basename(book_folder)
        for book_folder in book_folders
        if os.path.isdir(book_folder)
    ]
    return book_folders


segments: List[str] = []


def on_segment_change() -> None:
    """
    Callback function for segment selection changes.
    Updates the session state index based on the selected segment.
    """
    global segments
    # st.session_state.segment_index = segments.index(st.session_state.segment_select)
    st.write(f"Segment index after change : {st.session_state.segment_index}")


def save() -> None:
    """
    Save the current text area content to a .final file.
    Called automatically when text area content changes.
    """
    print("Text changed, saving!")
    text = st.session_state.text_area
    open(
        segment_files[st.session_state.segment_index].replace(".png", ".final"),
        "w",
    ).write(text)


def render_sidebar():
    user_workspace = st.session_state["user_todo"]
    active_user = st.session_state["name"]
    books_path = user_workspace
    book_folders = get_book_folders(books_path)

    st.markdown("<div class='sidebar-header'>", unsafe_allow_html=True)
    st.header("📚 Book Navigation")
    st.markdown(f"👤 User: {active_user}", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # Book selection section with improved visuals
    st.subheader("📁 Select Book")
    book = st.selectbox(
        "Book",
        book_folders,
        on_change=set_state,
        index=None,
        placeholder="Choose a book...",
        help="Select a book to process",
        label_visibility="collapsed",
    )

    if not book:
        st.info("👆 Please select a book to begin")
        return None, None

    # Page selection section
    st.subheader("📄 Select Page")
    book_pages = sorted(glob.glob(os.path.join(books_path, book, "*.png")))
    select_pages = [os.path.basename(page) for page in book_pages]

    pages = st.selectbox(
        "Page",
        select_pages,
        on_change=book_page_change,
        placeholder="Choose a page...",
        help="Select a page to process",
        label_visibility="collapsed",
        # index=st.session_state.page_index,
        index=0,
    )

    page_no = pages.split("/")[-1].replace(".png", "")
    page_path_selected = books_path + "/" + book + "/" + page_no + ".png"
    st.session_state.page_path_selected = page_path_selected
    st.session_state.page_index = int(page_no)

    segments_files_path = books_path + "/" + book + "/" + page_no + "/*.png"
    segment_files = sorted(glob.glob(segments_files_path))

    if len(segment_files) == 0:
        st.warning("No segments found", icon="⚠️")
        st.button("🔄 Run OCR", on_click=do_ocr, use_container_width=True)
        return None, None

    # Settings section
    with st.expander("⚙️ Settings", expanded=False):
        st.toggle("Enable Auto-Save", value=True, key="auto_save")
        st.divider()
        st.subheader("Character Table")
        document = VOWELS_TABLE.replace(" ", "  ")
        st.markdown(
            f'<div style="color:#FF9B9B; font-family: Courier New;font-size: small">{document}</div>',
            unsafe_allow_html=True,
        )

    return segment_files, segments_files_path


def draw_segment(index):
    page_path = "/".join(segment_files[index].split("/")[0:-1]) + ".png"
    json_path = segment_files[index].replace(".png", ".json")
    job = json.load(open(json_path))
    coords = job["coords"]

    original_image = cv2.imread(page_path)
    x1, y1, x2, y2 = coords
    cv2.rectangle(original_image, (x1, y1), (x2, y2), color=(255, 0, 0), thickness=3)

    st.image(original_image)


def render_app() -> None:
    """
    Main application rendering function.
    Handles the complete UI layout including:
    - Sidebar navigation
    - Book and page selection
    - Image preview
    - Text editing
    - Progress tracking
    - Navigation controls
    """
    global segments
    global segment_files

    # books_path = st.session_state["user_todo"]
    # book_folders = get_book_folders(books_path)

    # Add page configuration

    # Custom CSS for better styling
    st.markdown(
        """
        <style>
        .stButton button {
            width: 100%;
            border-radius: 5px;
            height: 45px;
        }
        .stTextArea textarea {
            font-size: 16px;
            font-family: 'Courier New', monospace;
        }
        .sidebar-header {
            margin-bottom: 20px;
        }
        </style>
    """,
        unsafe_allow_html=True,
    )

    with st.sidebar:
        segment_files, segments_files_path = render_sidebar()
        if not segment_files:
            return

    with st.container():
        # Add a header for the main content area
        st.markdown("### 📝 Text Editor")

        col1, col2 = st.columns(spec=[0.3, 0.7])
        with col1:
            # Add a subheader for the page preview
            st.markdown("#### Page Preview")
            draw_segment(index=st.session_state.segment_index)
        with col2:
            # Progress tracking
            if get_finals(segments_files_path) == len(segment_files):
                st.success("✅ All lines completed!")
            else:

                progress = get_finals(segments_files_path) / len(segment_files)
                st.progress(progress, f"Progress: {int(progress * 100)}%")

            # Navigation controls with better styling
            segments = [
                s.split("/")[-1] for s in segment_files
            ]  # Update global segments

            # st.write(f"Segments=> {segments}, len = {len(segments)}")

            selected_segment = st.selectbox(
                options=segments,
                label="📄 Line Segment",
                on_change=on_segment_change,
                index=st.session_state.segment_index or 0,
            )
            st.session_state.segment_index = segments.index(selected_segment)

            if get_finals(segments_files_path) == len(segment_files):
                st.success("All segments are now completed")

            ocred_text = ""

            if st.session_state.segment_index is not None:
                st.image(
                    segment_files[st.session_state.segment_index],
                    caption="Image",
                    width=700,
                    use_container_width=True,
                )

                is_finalized = has_final(segment_files[st.session_state.segment_index])
                if is_finalized:
                    ocred_text = open(
                        segment_files[st.session_state.segment_index or 0].replace(
                            ".png", ".final"
                        )
                    ).read()
                else:
                    ocred_text = get_ocred_text(
                        segment_files[st.session_state.segment_index]
                    )
            else:
                st.write("Segment index is NONE!")

            # Text editing area with better labeling
            st.markdown("#### Edit Text")
            st.text_area(
                label="Edit recognized text below:",
                value=ocred_text,
                max_chars=1000,
                key="text_area",
                height=120,  # Increased height
                on_change=save,
                help="Edit the recognized text and it will auto-save when you make changes",
            )

            # Navigation buttons with better layout
            with st.container():
                col1, col2, col3 = st.columns([1, 2, 1])
                with col1:
                    st.button(
                        "⬅️ Previous",
                        on_click=lambda: navigate(-1),
                        use_container_width=True,
                    )
                with col2:
                    st.markdown(
                        f"<div style='text-align: center'>Line {(st.session_state.segment_index or 0) + 1} of {len(segment_files)}</div>",
                        unsafe_allow_html=True,
                    )
                with col3:
                    st.button(
                        "Next ➡️", on_click=lambda: navigate(1), use_container_width=True
                    )

                # Full text preview
                with st.expander("📄 View Full Text", expanded=False):
                    st.code(
                        get_all_texts(segments_files_path.replace("*.png", "")),
                        language="markdown",
                    )


system_loop(render_app)
