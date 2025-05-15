import streamlit as st
import streamlit_authenticator as stauth
from yaml.loader import SafeLoader
import yaml
from auth_utils import do_login
import loguru

st.info(
    "For registration information, please contact dgoutsos or kperifanos at phil.uoa.gr"
)

st.title("Login")
authenticator = None


def get_authenticator():
    global authenticator
    return authenticator


with open("./config.yaml") as file:
    config = yaml.load(file, Loader=SafeLoader)


# loguru.logger.info(config["credentials"]["usernames"])
loguru.logger.info("Attempting login")
login_status, user_todo, user_uploads, user_workspace, authenticator = do_login()
active_user = st.session_state["name"]
loguru.logger.info(f"User is {active_user}, login status is {login_status}")

if not login_status:
    st.error("Username or password is incorrect")

if login_status is None:
    st.warning("Please enter your username and password")
