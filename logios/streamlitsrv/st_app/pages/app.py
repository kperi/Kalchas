import streamlit as st
import glob
import json
import yaml
import os
import cv2
import requests

# from st_app.page import segmentation_and_recognition
from st_app.utils import process_image, post_image_to_fastapi
from st_app.page_management import (
    get_max_final,
    get_finalized_lines,
    get_all_lines,
    get_all_texts,
)

from yaml.loader import SafeLoader
from auth_utils import do_login, system_loop
from auth_utils import do_login, system_loop


st.set_page_config(layout="wide")

login_status, user_workspace, _, _ = do_login()

with open("./config.yaml") as file:
    config = yaml.load(file, Loader=SafeLoader)


with open("/app/st_app/vowel_table.txt") as f:
    VOWELS_TABLE = f.read()


with open("/app/st_app/style.css") as css:
    st.markdown(f"<style>{css.read()}</style>", unsafe_allow_html=True)


def navigate(step):
    # st.warning('Βεβαιωθείτε ότι αποθηκεύσατε τυχόν αλλαγές στο προηγούμενο βήμα', icon="⚠️")
    st.session_state.index += step


def render_app():

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

        book = st.selectbox("Book: ", book_folders, on_change=set_state, index=None)
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
            with st.spinner("Running OCR on the page..."):
                ret = process_image(page_path_selected, "http://ocr:8000/process_image")
            st.write(ret)

        if len(files) == 0:
            st.write("No segments and text found - run OCR on the page")
            st.button("Run OCR", on_click=do_ocr)
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
                label="line segment",
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
                        "Previous",
                        on_click=lambda: navigate(-1),
                    )
                with col2:
                    pass
                    # btn_save = st.button("Save", )
                with col3:
                    btn_next = st.button(
                        "Next",
                        on_click=lambda: navigate(1),
                    )

                st.markdown(
                    """
                            #### Character table
                    """
                )

                document = VOWELS_TABLE.replace(" ", "  ")

                st.markdown(
                    f'<div style="color:#FF9B9B; font-family: Courier New;font-size: small">{document}</div>',
                    unsafe_allow_html=True,
                )
                # st.markdown("```" + VOWELS_TABLE + "```")
                st.info("Full text: ")
                st.code(get_all_texts(book_page_path), language="markdown")


system_loop(render_app)
if False:
    if st.session_state["authentication_status"]:
        render_app()
    elif st.session_state["authentication_status"] is False:
        st.error("Username/password is incorrect")
    elif st.session_state["authentication_status"] is None:
        st.warning("Please enter your username and password")
