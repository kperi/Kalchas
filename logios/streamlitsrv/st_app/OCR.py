import streamlit as st
import streamlit_authenticator as stauth
from yaml.loader import SafeLoader
import yaml
from auth_utils import do_login
import os
import glob

st.set_page_config(
    page_title="Logios - Greek Polytonic OCR",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
    # initial_sidebar_state="collapsed",
)


# Remove streamlit deploy button


st.markdown(
    """
    <style>
        .reportview-container {
            margin-top: -2em;
        }
        #MainMenu {visibility: hidden;}
        .stDeployButton {display:none;}
        footer {visibility: hidden;}
        #stDecoration {display:none;}
    </style>
""",
    unsafe_allow_html=True,
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

    hide_st_style = """
                <style>
                #MainMenu {visibility: hidden;}
                footer {visibility: hidden;}
                header {visibility: hidden;}
                </style>
                """

    if not login_status:
        st.markdown(
            """
            ## Logios : A Greek Polytonic OCR Platform
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            """
            ##### `Logios` is an an OCR engine developed by University of Athens. 
            """
        )

    else:
        name = st.session_state["name"]
        # st.sidebar.markdown(f"## Welcome {name}")

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

        admin = st.Page(
            "menu/admin.py",
            title="Admin",
            icon=":material/admin_panel_settings:",
        )

        if name == "Kostas":
            menu = [ocr, pdf_upload, page_editing, layout_detection, admin]
        else:
            menu = [ocr, pdf_upload, page_editing, layout_detection]

        pg = st.navigation(
            {
                "Menu": menu,
            }
        )
        # st.sidebar.image("./st_app/images/scholar.png", width=200)
        pg.run()
        # pg.title("Logios")


render_main(login_status=login_status)
