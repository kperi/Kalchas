import streamlit as st
from PIL import Image
from streamlit_cropper import st_cropper
from PIL import Image
import glob
import os
from loguru import logger
import shutil


def initialize_session_state():
    if "current_file_index" not in st.session_state:
        st.session_state.current_file_index = 0
    if "current_segment_index" not in st.session_state:
        st.session_state.current_segment_index = 0
    if "last_action" not in st.session_state:
        st.session_state.last_action = None
    if "shortcuts_enabled" not in st.session_state:
        st.session_state.shortcuts_enabled = True


def reset_indexes():
    st.session_state.current_file_index = 0
    st.session_state.current_segment_index = 0


def get_workspace_folders():
    base_folder = None
    upload_dir = st.session_state["user_uploads"]
    workspace_dir = st.session_state["user_workspace"]
    # user workspace folders
    folders = glob.glob(upload_dir + "/*")
    folders = [f for f in folders if os.path.isdir(f)]
    folders = sorted(folders)
    if len(folders) > 0:
        base_folder = "/".join(folders[0].split("/")[:-1])

    folders = [f.split("/")[-1] for f in folders]
    return workspace_dir, upload_dir, folders, base_folder


def get_folder_files(selected_folder):
    base_file_path = None
    files = glob.glob(selected_folder + "/*.png")
    files = [f for f in files if "_cropped" not in f]
    files = sorted(files)
    if len(files) > 0:
        base_file_path = os.path.dirname(files[0])
        files = [os.path.basename(f) for f in files]

    return base_file_path, files


def get_cropped(page_path):
    page_path = page_path.replace(".png", "")
    cropped_files = glob.glob(f"{page_path}*_cropped.png")

    # cropped_files = [f for f in cropped_files if f.split("/")[-1].split("_")[0] == str(page_num)]
    return sorted(cropped_files)


def move_folder_to_workspace():
    """
    Moves the selected folder from uploads directory to workspace directory
    """
    if "selected_folder" not in st.session_state:
        st.error("No folder selected")
        return

    try:
        source_path = st.session_state["selected_folder"]
        folder_name = os.path.basename(source_path)
        dest_path = os.path.join(st.session_state["user_todo"], folder_name)

        # Create todo workspace directory if it doesn't exist
        os.makedirs(st.session_state["user_todo"], exist_ok=True)

        # Move the folder
        shutil.move(source_path, dest_path)

        st.success(f"Moved {folder_name} to workspace")
        st.session_state.last_action = "move"

    except Exception as e:
        logger.error(f"Error moving folder: {str(e)}")
        st.error("Failed to move folder to workspace")


def on_delete():
    """
    Handles deletion of original image:
    1. Creates a backup of the original image
    2. Moves the original to a backup folder
    3. Keeps the cropped regions in place
    """
    try:
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

    except Exception as e:
        logger.error(f"Error during deletion: {str(e)}")
        st.error("Failed to delete original image")


login_status = st.session_state["authentication_status"]
if login_status:
    # Initialize session state for keyboard shortcuts
    initialize_session_state()

    # Sidebar organization
    with st.sidebar:
        st.header("🛠️ Controls")

        # Folder selection section
        st.subheader("📁 Project Folders")
        workspace_dir, upload_dir, folders, base_folder = get_workspace_folders()

        selected_folder = st.selectbox(
            "Select Folder",
            folders,
            index=None,
            placeholder="Choose a folder...",
            help="Select a folder containing images to process",
        )

        if selected_folder:
            if (
                "previous_folder" not in st.session_state
                or st.session_state.previous_folder != selected_folder
            ):
                reset_indexes()
                st.session_state.previous_folder = selected_folder

            selected_folder = base_folder + "/" + selected_folder
            st.session_state.selected_folder = selected_folder

            st.checkbox("Remove Original Pages", value=True)
            st.button(
                "📥 Move to Workspace",
                help="Move folder to editing workspace (Shortcut: Ctrl+M)",
                on_click=move_folder_to_workspace,
                use_container_width=True,
            )

        # Settings section
        with st.expander("⚙️ Settings", expanded=False):
            box_color = st.color_picker("Box Color", value="#0000FF")
            st.toggle("Enable Keyboard Shortcuts", value=True, key="shortcuts_enabled")
            st.caption(
                """
            Keyboard Shortcuts:
            - Save crop: Ctrl+S
            - Next image: →
            - Previous image: ←
            - Delete crop: Del
            - Move to workspace: Ctrl+M
            """
            )

    # Main content area
    if selected_folder:
        base_file_path, files = get_folder_files(selected_folder)

        if files:
            # File navigation
            col1, col2, col3 = st.columns([1, 4, 1])
            with col1:
                prev_disabled = st.session_state.current_file_index == 0
                if st.button("⬅️", disabled=prev_disabled, use_container_width=True):
                    st.session_state.current_file_index = max(
                        0, st.session_state.current_file_index - 1
                    )
                    st.session_state.current_segment_index = 0
                    st.session_state.last_action = "navigation"

            with col2:
                current_index = st.session_state.current_file_index
                selected_page = st.selectbox(
                    "Current Image",
                    files,
                    index=current_index,
                    key="file_selector",
                    label_visibility="collapsed",
                )

            with col3:
                next_disabled = st.session_state.current_file_index == len(files) - 1
                if st.button("➡️", disabled=next_disabled, use_container_width=True):
                    st.session_state.current_file_index = min(
                        len(files) - 1, st.session_state.current_file_index + 1
                    )
                    st.session_state.current_segment_index = 0
                    st.session_state.last_action = "navigation"

            if selected_page:
                selected_page = os.path.join(base_file_path, selected_page)
                image = Image.open(selected_page)
                page_num = int(selected_page.split("/")[-1].split(".")[0])
                cropped_files = get_cropped(selected_page)

                # Progress indicator
                total_files = len(files)
                current_file = st.session_state.current_file_index + 1
                st.progress(
                    current_file / total_files,
                    f"Processing image {current_file} of {total_files}",
                )

                # Main workspace tabs
                tab1, tab2 = st.tabs(["🔍 Image Editor", "📋 Cropped Regions"])

                with tab1:
                    editor_col1, editor_col2 = st.columns([0.7, 0.3])
                    with editor_col1:
                        cropped_img = st_cropper(
                            image,
                            realtime_update=True,
                            box_color=box_color,
                            aspect_ratio=None,
                            stroke_width=3,
                            return_type="image",
                        )

                    with editor_col2:
                        st.subheader("Preview")
                        st.image(cropped_img)

                        next_crop_index = len(cropped_files) + 1
                        cropped_img_name = selected_page.replace(
                            ".png", f"_{next_crop_index}_cropped.png"
                        )

                        def save_crop():
                            with st.spinner("Saving crop..."):
                                cropped_img.save(cropped_img_name)
                                # Update segment index after saving
                                st.session_state.current_segment_index = len(
                                    cropped_files
                                )
                            st.success("Crop saved!", icon="✅")
                            st.session_state.last_action = "save"

                        st.button(
                            "💾 Save Region (Ctrl+S)",
                            key=f"save_{page_num}",
                            on_click=save_crop,
                            use_container_width=True,
                        )

                with tab2:
                    if cropped_files:
                        st.caption(f"{len(cropped_files)} cropped regions found")
                        for idx, crop in enumerate(cropped_files, 1):
                            with st.container():
                                crop_col1, crop_col2 = st.columns([4, 1])
                                with crop_col1:
                                    st.image(crop, use_container_width=True)
                                with crop_col2:
                                    st.caption(f"Region {idx}")
                                    if st.button("🗑️", key=f"delete_{idx}"):
                                        os.remove(crop)
                                        st.session_state.current_segment_index = max(
                                            0, len(cropped_files) - 1
                                        )
                                        st.session_state.last_action = "delete"
                    else:
                        st.info("No cropped regions available for this image")

                # Danger zone
                with st.expander("⚠️ Advanced Options", expanded=False):
                    col1, col2 = st.columns(2)
                    with col1:
                        if st.button(
                            "🗑️ Delete Original",
                            type="primary",
                            help="Move original image to backup and promote crops",
                            use_container_width=True,
                        ):
                            on_delete()
                            st.success("Original image moved to backup")
                    with col2:
                        if st.button(
                            "↩️ Reset All",
                            type="secondary",
                            help="Clear all crops for this image",
                            use_container_width=True,
                        ):
                            for crop in cropped_files:
                                os.remove(crop)
                            # st.rerun()

        else:
            st.info("No images found in the selected folder")
    else:
        st.info("👈 Please select a folder from the sidebar to begin")
