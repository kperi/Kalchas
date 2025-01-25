import streamlit as st
import streamlit_authenticator as stauth
from yaml.loader import SafeLoader
import yaml
import os 
import requests
    

#def get_file_directory():
#    return os.path.dirname(os.path.abspath(__file__))

#directory = get_file_directory()
#st.write(f"Current file directory: {directory}")

hide_bar= """
    <style>
    [data-testid="stSidebar"][aria-expanded="true"] > div:first-child {
        visibility:hidden;
        width: 0px;
    }
    [data-testid="stSidebar"][aria-expanded="false"] > div:first-child {
        visibility:hidden;
    }
    </style>
"""

with open("./config.yaml") as file:
    config = yaml.load(file, Loader=SafeLoader) 

st.set_page_config(
    page_title="Logios - Greek Polytonic OCR",
    page_icon="👋",
)

authenticator = stauth.Authenticate(
    config["credentials"],
    config["cookie"]["name"],
    config["cookie"]["key"],
    config["cookie"]["expiry_days"],
    config["pre-authorized"],
)


name, auth_status, username =  authenticator.login()
if auth_status == False:
    st.error("Username/password is incorrect")
    st.markdown(hide_bar, unsafe_allow_html=True)

if auth_status == None:
    st.warning("Please enter your username and password")
    st.markdown(hide_bar, unsafe_allow_html=True)


def render_main():
    st.write("## Logios : A Greek Polytonic OCR Platform")

    st.markdown(
        """
           #### `Logios` is an an OCR engine developed by University of Athens. 
        """
    )

if auth_status:
    # # ---- SIDEBAR ----
    st.sidebar.title(f"Welcome {name}")
    

    hide_st_style = """
                <style>
                #MainMenu {visibility: hidden;}
                footer {visibility: hidden;}
                header {visibility: hidden;}
                </style>
                """
    st.markdown(hide_st_style, unsafe_allow_html=True)
    render_main()

    authenticator.logout("Logout", "sidebar")


#    render_main()
#if st.session_state["authentication_status"]:
#if st.session_state["authentication_status"] is False:
#elif st.session_state["authentication_status"] is None:
##    st.error("Username/password is incorrect")
#    st.warning("Please enter your username and password")

