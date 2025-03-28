import streamlit_authenticator as stauth
from yaml.loader import SafeLoader
import yaml
import streamlit as st
import os
from loguru import logger


st.title("Logios - Register")


# Load config file
config_path = "./config.yaml"
if not os.path.exists(config_path):
    st.error(f"Configuration file not found at {config_path}")
    st.stop()

with open(config_path) as file:
    config = yaml.load(file, Loader=SafeLoader)

authenticator = stauth.Authenticate(
    config["credentials"],
    config["cookie"]["name"],
    config["cookie"]["key"],
    config["cookie"]["expiry_days"],
)

try:
    # Attempt to register new user with unique form keys
    (
        email_of_registered_user,
        username_of_registered_user,
        name_of_registered_user,
    ) = authenticator.register_user(
        # pre_authorized=config.get("pre-authorized", {}).get("emails", []),
        captcha=False,
        location="main",
        pre_authorized=None,
    )

    if email_of_registered_user:
        # Save the updated config back to file
        with open(config_path, "w") as file:
            yaml.dump(config, file, default_flow_style=False)
        st.success("User registered successfully!")
        logger.info(f"New user registered: {email_of_registered_user}")

except Exception as e:
    st.error(f"Registration error: {str(e)}")
    logger.error(f"Registration failed: {str(e)}")
