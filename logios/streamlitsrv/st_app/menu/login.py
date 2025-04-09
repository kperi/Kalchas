import streamlit as st
import streamlit_authenticator as stauth
from yaml.loader import SafeLoader
import yaml
from auth_utils import do_login

st.info( "For registration information, please contact dgoutsos or kperifanos at phil.uoa.gr")

st.title(f"Login")
authenticator = None


def get_authenticator():
    global authenticator
    return authenticator


with open("./config.yaml") as file:
    config = yaml.load(file, Loader=SafeLoader)

login_status, user_todo, user_uploads, user_workspace, authenticator = do_login()
active_user = st.session_state["name"]

if login_status == False:
    st.error("Username/password is incorrect")

if login_status == None:
    st.warning("Please enter your username and password")
