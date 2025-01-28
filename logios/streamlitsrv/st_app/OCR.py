import streamlit as st
import streamlit_authenticator as stauth
from yaml.loader import SafeLoader
import yaml
from auth_utils import do_login
import os
import glob

st.set_page_config(
    page_title="Logios - Greek Polytonic OCR",
    page_icon="👋",
    layout="wide",
    # initial_sidebar_state="collapsed",
)


with open("./config.yaml") as file:
    config = yaml.load(file, Loader=SafeLoader)


login_status, user_todo, user_uploads, user_workspace, authenticator = do_login()

active_user = st.session_state["name"]


if login_status == False:
    st.error("Username/password is incorrect")

if login_status == None:
    st.warning("Please enter your username and password")


def render_main(login_status):
    name = st.session_state["name"]
    st.sidebar.title(f"Welcome {name}")

    hide_st_style = """
                <style>
                #MainMenu {visibility: hidden;}
                footer {visibility: hidden;}
                header {visibility: hidden;}
                </style>
                """
    st.markdown(hide_st_style, unsafe_allow_html=True)
    st.write("## Logios : A Greek Polytonic OCR Platform")
    # st.image("./st_app/images/scholar.png")

    st.markdown(
        """
           ##### `Logios` is an an OCR engine developed by University of Athens. 
        """
    )

    if login_status:
        layout_detection = st.Page(
            "menu/layout.py",
            title="Page layout detection",
            icon=":material/document_scanner:",
        )
        ocr = st.Page(
            "menu/app.py",
            title="OCR",
            icon=":material/menu_book:",
            default=True,
        )
        page_editing = st.Page(
            "menu/editing.py",
            title="Page editing",
            icon=":material/edit:",
        )
        pdf_upload = st.Page(
            "menu/pdf_upload.py",
            title="File upload",
            icon=":material/upload:",
        )
        pg = st.navigation(
            {
                # "Account": [logout_page],
                "Menu": [
                    ocr,
                    pdf_upload,
                    page_editing,
                    layout_detection,
                ],
            }
        )
        pg.run()
        # pg.title("Logios")


render_main(login_status=login_status)
