import streamlit as st
from PIL import Image
from streamlit_cropper import st_cropper
from PIL import Image
import glob
import os
from loguru import logger


def get_cropped(page_path):
    page_path = page_path.replace(".png", "")
    cropped_files = glob.glob(f"{page_path}*_cropped.png")

    # cropped_files = [f for f in cropped_files if f.split("/")[-1].split("_")[0] == str(page_num)]
    return sorted(cropped_files)


login_status = st.session_state["authentication_status"]
if login_status:

    upload_dir = st.session_state["user_uploads"]
    workspace_dir = st.session_state["user_workspace"]
    # user workspace folders
    folders = glob.glob(upload_dir + "/*")
    folders = [f for f in folders if os.path.isdir(f)]
    folders = sorted(folders)
    if len(folders) > 0:
        base_folder = "/".join(folders[0].split("/")[:-1])

    folders = [f.split("/")[-1] for f in folders]

    selected_folder = st.selectbox("Επιλέξτε φάκελο", folders, index=None)

    if selected_folder:
        selected_folder = base_folder + "/" + selected_folder

        def move_folder_to_TODO():

            ## move the selected folder to the editing workspace [also, do OCR]
            folder_name = os.path.basename(selected_folder)
            dest_path = os.path.join(workspace_dir, folder_name)
            os.rename(selected_folder, dest_path)

        st.button("Μετακίνηση στον φάκελο επεξεργασίας:", on_click=move_folder_to_TODO)

        box_color = st.sidebar.color_picker(label="Box Color", value="#0000FF")
        box_color_2 = st.sidebar.color_picker(label="Box Color 2", value="#FF00FF")

        files = glob.glob(selected_folder + "/*.png")
        files = [f for f in files if "_cropped" not in f]
        files = sorted(files)
        if len(files) > 0:
            base_file_path = os.path.dirname(files[0])
            files = [os.path.basename(f) for f in files]

        selected_page = st.selectbox("Επιλέξτε εικόνα", files, index=None)
        if selected_page:
            # st.image(selected_page)
            selected_page = os.path.join(base_file_path, selected_page)

            image = Image.open(selected_page)
            im_w, im_h = image.size

            page_num = selected_page.split("/")[-1].split(".")[0]
            page_num = int(page_num)

            cropped_files = get_cropped(selected_page)
            num_cropped = len(cropped_files)

            st.write(f"#### `Επιλεγμένη σελίδα: {page_num}`")

            def on_delete():
                dirname = os.path.dirname(selected_page)
                filename = os.path.basename(selected_page)

                dest = os.path.join(dirname, "backup")
                os.makedirs(dest, exist_ok=True)
                dest_file = os.path.join(dest, filename)
                os.rename(selected_page, dest_file)

                for file in cropped_files:
                    new_name = file.replace("_cropped", "")
                    logger.info(f"Cropping new name: {new_name}")
                    os.rename(file, new_name)

                # st.write( f"Delete this: {selected_page}")

            st.button("## Διαγραφή αρχικής εικόνας", on_click=on_delete)

            # st.write( f"### Cropped images: {num_cropped}")
            st.markdown("#### αποκομμένα αρχεία ####")
            st.table(cropped_files)

            # cropped_files = pd.DataFrame( cropped_files, columns = ["Αποκομμένες εικόνες"])
            # cropped_files["Delete"] = False
            def delete_row():
                os.remove(cropped_files[-1])

            st.button(
                "Διαγραφή τελευταίας εικόνας", key="btn_delete", on_click=delete_row
            )

            st.markdown("""-----------------""")

            col1, col2 = st.columns((0.7, 0.3))
            realtime_update = True
            with col1:
                cropped_img = st_cropper(
                    image,
                    # default_coords=( int(im_w/2)-50, int(im_h/2)-50, int(im_w/2)+100, int(im_h/2)+50),
                    realtime_update=realtime_update,
                    box_color=box_color,
                    aspect_ratio=None,
                    stroke_width=3,
                )

            with col2:
                # Manipulate cropped image at will
                st.write("Προεπισκόπηση")
                # _ = cropped_img.thumbnail((150, 150))
                st.image(cropped_img)

                next_crop_index = num_cropped + 1

                cropped_img_name = selected_page.replace(
                    ".png", f"_{next_crop_index}_cropped.png"
                )

                def crop_and_save():
                    cropped_img.save(f"{cropped_img_name}")

                filename = os.path.basename(selected_page)

                st.markdown(
                    f"""- Ονομα αρχείου:  
                    `-- {cropped_img_name}`"""
                )

                st.button(
                    "Αποκοπή και αποθήκευση", key=f"{page_num}", on_click=crop_and_save
                )
    st.markdown("-------------")
