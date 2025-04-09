import streamlit as st
from menu.login import get_authenticator

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
            ##### `Logios` is an OCR engine developed by the National and Kapodistrian University of Athens.
            """
        )

        login = st.Page(
            "menu/login.py",
            title="Login",
            icon=":material/document_scanner:",
        )
        register = st.Page(
            "menu/register.py",
            title="Register",
            icon=":material/document_scanner:",
        )
        pg = st.navigation(
            {
                "Menu": [login]#, register],
            }
        )
        pg.run()

    else:
        authenticator = get_authenticator()
        if authenticator is not None:
            authenticator.logout(location="sidebar", key="side_logout")

        name = st.session_state["name"]
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
            title="PDF upload",
            icon=":material/upload:",
        )

        admin = st.Page(
            "menu/admin.py",
            title="Admin",
            icon=":material/admin_panel_settings:",
        )

        if name == "Kostas":
            menu = [pdf_upload, page_editing, ocr, layout_detection, admin]
        else:
            menu = [
                pdf_upload,
                page_editing,
                ocr,
            ]

        pg = st.navigation(
            {
                "Menu": menu,
            }
        )
        pg.run()


login_status = (
    "authentication_status" in st.session_state
    and st.session_state["authentication_status"] is not None
)

#   login_status = False
render_main(login_status=login_status)  # login status
