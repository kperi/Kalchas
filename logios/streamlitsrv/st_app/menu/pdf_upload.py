import streamlit as st
from PIL import Image
import fitz
from auth_utils import system_loop, do_login
import os
from streamlit_cropper import st_cropper
from PIL import Image
import io


def render_app():

    login_status = st.session_state["authentication_status"]
    if not login_status:
        st.warning("Πρέπει να είστε συνδεδεμένος για να μπορέσετε να ανεβάσετε ένα pdf")
        st.stop()

    uploaded_file = st.file_uploader("Choose a file", type="pdf")

    user_uploads_dir = st.session_state["user_uploads"]
    user_uploads_dir = st.session_state["user_uploads"]
    # workspace_dir = st.session_state["user_workspace"]

    if uploaded_file is not None:
        # To read file as bytes:
        bytes_data = uploaded_file.getvalue()

        dest_folder = f"{user_uploads_dir}/{uploaded_file.name}".replace(".pdf", "")
        if not os.path.isdir(dest_folder):
            os.makedirs(dest_folder, exist_ok=True)

        dest_filename = f"{user_uploads_dir}/{uploaded_file.name}"
        with open(dest_filename, "wb") as fh:
            fh.write(bytes_data)

        pbar = st.progress(0.0)
        doc = fitz.open(dest_filename)
        for page_num in range(doc.page_count):

            # st.markdown("## Page: " + str(page_num + 1))
            page = doc.load_page(page_num)
            pixmap = page.get_pixmap(dpi=300)
            img_data = pixmap.tobytes()
            image = Image.open(io.BytesIO(img_data))
            image = image.convert("L")

            if not os.path.isdir(dest_folder):
                os.makedirs(dest_folder, exist_ok=True)

            dest_file = os.path.join(dest_folder, str(page_num).rjust(3, "0")) + ".png"
            image.save(dest_file)
            pbar.progress(
                float(page_num + 1) / doc.page_count,
                "Μετατροπή σελίδων σε εικόνες png...",
            )
            continue

        st.success("Η μετατροπή ολοκληρώθηκε")


system_loop(render_app)
