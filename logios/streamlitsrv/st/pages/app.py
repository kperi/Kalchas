import streamlit as st
import glob
import json
import yaml
import os
import cv2
import requests

from yaml.loader import SafeLoader
from auth_utils import do_login, system_loop

# from segmenter import segmentation_and_recognition

# st.set_page_config(layout="wide")
from auth_utils import do_login, system_loop

login_status, user_workspace, _, _ = do_login()

with open("./config.yaml") as file:
    config = yaml.load(file, Loader=SafeLoader)


with open("/app/st/vowel_table.txt") as f:
    VOWELS_TABLE = f.read()


with open("/app/st/style.css") as css:
    st.markdown(f"<style>{css.read()}</style>", unsafe_allow_html=True)


def get_all_texts(book_page_path):

    finals = glob.glob(os.path.join(book_page_path, "*.json"))
    finals = sorted(finals)
    lines = []
    for f in finals:
        final_f = f.replace(".json", ".final")
        if os.path.exists(final_f):
            line = open(final_f).read()
        else:
            jcontent = json.load(open(f))
            line = ">>> " + jcontent["text"][0]

        lines.append(line)

    # lines = [open(f).read() for f in finals]

    return "\n".join(lines)


def navigate(step):
    # st.warning('Βεβαιωθείτε ότι αποθηκεύσατε τυχόν αλλαγές στο προηγούμενο βήμα', icon="⚠️")
    st.session_state.index += step


def get_max_final(book_page_path):
    finals = glob.glob(os.path.join(book_page_path, "*.final"))
    finals = [int(f.split("/")[-1].split(".")[0]) for f in finals]
    return max(finals) if finals else 0


def get_finalized_lines(book_page_path):
    finals = glob.glob(os.path.join(book_page_path, "*.final"))
    finals = [int(f.split("/")[-1].split(".")[0]) for f in finals]
    finals = sorted(finals)
    finals = [str(f) for f in finals]
    return ",".join(finals)


def get_all_lines(book_page_path):
    all_lines = glob.glob(os.path.join(book_page_path, "*.png"))
    all_lines = [int(f.split("/")[-1].split(".")[0]) for f in all_lines]

    return all_lines


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


def render_app():

    st.markdown(
        f"""
                #### Current user:  *{st.session_state["name"]}*
                -----
                """
    )
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

    def get_finals(book_page_path):
        finals = glob.glob(os.path.join(book_page_path, "*.final"))
        return len(finals)

    def get_ocred_text(file):
        jfile = file.replace(".png", ".json")
        jcontent = json.load(open(jfile))
        ocred_text = jcontent["text"][0]
        return ocred_text

    def has_final(file):
        return os.path.exists(file.replace(".png", ".final"))

    with st.container():

        def set_state():
            st.session_state.index = 0

        book = st.selectbox("Κείμενο: ", book_folders, on_change=set_state)
        if not book:
            return

        book_pages = sorted(glob.glob(os.path.join(books_path, book, "*.png")))

        select_pages = [os.path.basename(page) for page in book_pages]
        pages = st.selectbox("Σελίδα: ", select_pages, on_change=set_state)

        page_no = pages.split("/")[-1].replace(".png", "")

        page_path_selected = books_path + "/" + book + "/" + page_no + ".png"

        segments_files_path = books_path + "/" + book + "/" + page_no + "/*.png"
        files = sorted(glob.glob(segments_files_path))

        def do_ocr():
            ret = process_image(page_path_selected, "http://ocr:8000/process_image")
            st.write(ret)

            # ret = segmentation_and_recognition(page_path_selected)
            # st.write(ret)

        if len(files) == 0:
            st.write("Δε βρέθηκαν γραμμές - πρέπει να γίνει OCR στη σελίδα")
            st.button("Εκτέλεση OCR", on_click=do_ocr)
            return

        # current book path

        book_page_path = segments_files_path = books_path + "/" + book + "/" + page_no

        # find the max .final file and move the index to the next one
        max_final = get_max_final(book_page_path)
        if "index" not in st.session_state:
            st.session_state.index = max_final
            index = max_final
        else:
            index = st.session_state.index

        index = index % len(files)

        col1, col2 = st.columns(spec=[0.3, 0.7])

        with col1:

            # st.write(files[index])

            page_path = "/".join(files[index].split("/")[0:-1]) + ".png"
            json_path = files[index].replace(".png", ".json")
            job = json.load(open(json_path))
            coords = job["coords"]

            original_image = cv2.imread(page_path)
            x1, y1, x2, y2 = coords
            cv2.rectangle(
                original_image, (x1, y1), (x2, y2), color=(255, 0, 0), thickness=3
            )

            st.image(original_image)

        with col2:

            def on_segment_change():
                key = st.session_state.segment_select
                page = key.replace(".png", "")
                print(page)
                page = int(page)
                st.session_state.index = page

            segments = [s.split("/")[-1] for s in files]
            st.selectbox(
                options=segments,
                label="segment",
                on_change=on_segment_change,
                key="segment_select",
                index=index,
            )

            finalized_lines = get_finalized_lines(book_page_path)
            # st.info(
            #        f"Τρέχουσα γραμμή: {index+1}. Έχουν ολοκληρωθεί  οι γραμμές {finalized_lines} ( {get_finals(book_page_path)} από {len(files)} γραμμές)"
            #    )
            progress_percent = get_finals(book_page_path) / len(files)

            if get_finals(book_page_path) == len(files):
                st.success("Εχουν ολοκληρωθεί όλες οι γραμμές")
            else:
                # st.progress(
                #    progress_percent, f"Εχουν ολοκληρωθεί οι γραμμές {finalized_lines}"
                # )
                pass
            st.image(files[index], caption="Image", width=700, use_container_width=True)

            is_finalized = has_final(files[index])
            if is_finalized:
                ocred_text = open(files[index].replace(".png", ".final")).read()
            else:
                ocred_text = get_ocred_text(files[index])

            def save():
                print("Text changed, saving!")
                text = st.session_state.text_area
                open(files[index].replace(".png", ".final"), "w").write(text)
                # st.rerun()

            text = st.text_area(
                label="Recognized text",
                value=ocred_text,
                # on_change=lambda v: text_change(v),
                max_chars=1000,
                key="text_area",
                height=80,
                on_change=save,
            )

            if text != ocred_text:
                print("Text changed, saving!")
                open(files[index].replace(".png", ".final"), "w").write(text)

            with st.container():

                col1, col2, col3 = st.columns(3)

                with col1:
                    btn_prev = st.button(
                        "Προηγούμενο",
                        on_click=lambda: navigate(-1),
                    )
                with col2:
                    pass
                    # btn_save = st.button("Save", )
                with col3:
                    btn_next = st.button(
                        "Επόμενο",
                        on_click=lambda: navigate(1),
                    )

                st.markdown(
                    """
                            #### Πίνακας χαρακτήρων
                    """
                )

                document = VOWELS_TABLE.replace(" ", "  ")

                st.markdown(
                    f'<div style="color:#8B0000; font-family: Courier New;font-size: small">{document}</div>',
                    unsafe_allow_html=True,
                )
                # st.markdown("```" + VOWELS_TABLE + "```")
                st.info("Κείμενο: ")
                st.code(get_all_texts(book_page_path), language="markdown")


system_loop(render_app)
if False:
    if st.session_state["authentication_status"]:
        render_app()
    elif st.session_state["authentication_status"] is False:
        st.error("Username/password is incorrect")
    elif st.session_state["authentication_status"] is None:
        st.warning("Please enter your username and password")
