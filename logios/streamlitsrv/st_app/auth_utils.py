import streamlit_authenticator as stauth
from yaml.loader import SafeLoader
import yaml
import streamlit as st
import os
from loguru import logger


def login_cb():
    logger.info(f"loggin callback:")
    pass


def do_login():

    with open("./config.yaml") as file:
        config = yaml.load(file, Loader=SafeLoader)

    # st.write(config)
    # st.set_page_config(
    #    page_title="Hello",
    #    page_icon="👋",
    # )

    authenticator = stauth.Authenticate(
        config["credentials"],
        config["cookie"]["name"],
        config["cookie"]["key"],
        config["cookie"]["expiry_days"],
        # config["pre-authorized"],
        # provider="google",
    )

    # if st.session_state.get("authentication_status") is  None:
    authenticator.login(
        # location="sidebar",
    )

    name = st.session_state["name"]
    login_status = st.session_state["authentication_status"]
    username = st.session_state["username"]

    user_todo = None
    user_uploads = None
    user_workspace = None

    if login_status:
        st.session_state["name"] = name
        st.session_state["username"] = username

        st.session_state["authentication_status"] = True
        authenticator.logout(location="sidebar")

        system_workspace = config["system_workspace"]["path"]

        # create user paths if they don't exist
        user_workspace = os.path.join(system_workspace, name)
        user_todo = os.path.join(user_workspace, "TODO")
        user_uploads = os.path.join(user_workspace, "uploads")

        st.session_state.user_workspace = user_workspace
        st.session_state.user_todo = user_todo
        st.session_state.user_uploads = user_uploads

        os.makedirs(user_todo, exist_ok=True)
        os.makedirs(user_uploads, exist_ok=True)

        logger.info(user_todo)
        logger.info(user_uploads)

    return login_status, user_todo, user_uploads, user_workspace, authenticator


def system_loop(render_function):
    if st.session_state.get("authentication_status") is None:
        st.warning("Please enter your username and password")
    elif st.session_state["authentication_status"]:
        render_function()
    elif st.session_state["authentication_status"] is False:
        st.error("Username/password is incorrect")
    elif st.session_state["authentication_status"] is None:
        st.warning("Please enter your username and password")


def init_session_state() -> None:
    """
    Initialize all required Streamlit session state variables if they don't exist.
    Sets default values for index, authentication_status, name, user_todo,
    segment_select, and text_area.
    """
    if "page_index" not in st.session_state or st.session_state.page_index is None:
        st.session_state.page_index = 0

    if (
        "segment_index" not in st.session_state
        or st.session_state.segment_index is None
    ):
        st.session_state.segment_index = 0

    if "authentication_status" not in st.session_state:
        st.session_state.authentication_status = None
    if "name" not in st.session_state:
        st.session_state.name = None
    if "user_todo" not in st.session_state:
        st.session_state.user_todo = None
    if "segment_select" not in st.session_state:
        st.session_state.segment_select = None
    if "text_area" not in st.session_state:
        st.session_state.text_area = ""
