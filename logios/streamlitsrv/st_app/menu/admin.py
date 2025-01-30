import streamlit as st
import os
from pathlib import Path
import shutil


def get_disk_space(path):
    """Calculate available disk space in bytes"""
    try:
        stats = os.statvfs(path)
        # Calculate total available space in bytes
        # statvfs returns blocks * block size
        available_space = stats.f_bavail * stats.f_frsize
        return available_space
    except OSError:
        return 0


def get_dir_size(path):
    """Calculate total size of directory in bytes"""
    total = 0
    with os.scandir(path) as it:
        for entry in it:
            if entry.is_file():
                total += entry.stat().st_size
            elif entry.is_dir():
                total += get_dir_size(entry.path)
    return total


def bytes_to_gb(bytes_size):
    """Convert bytes to gigabytes"""
    return bytes_size / (1024 * 1024 * 1024)


def render_admin():
    st.title("Storage Usage")

    if "user_workspace" in st.session_state:
        workspace = "/app/data"
        size_bytes = get_dir_size(workspace)
        size_gb = bytes_to_gb(size_bytes)

        st.metric(label="Total Storage Used", value=f"{size_gb:.2f} GB")

        # Show breakdown by user folder
        st.subheader("Storage by Directory")
        for entry in os.scandir(workspace):
            if entry.is_dir():
                dir_size = get_dir_size(entry.path)
                dir_size_gb = bytes_to_gb(dir_size)
                st.metric(label=entry.name, value=f"{dir_size_gb:.2f} GB")

        available_space_gb = bytes_to_gb(get_disk_space(workspace))
        st.metric(label="Available Space", value=f"{available_space_gb:.2f} GB")
    else:
        st.error("No workspace found. Please log in first.")


def dfs(root):
    import glob

    stack = [root]

    all_nodes = []
    while stack:
        node = stack.pop()
        all_nodes.append(node)
        files = [x[0] for x in os.walk(node)]
        for file in files:
            stack.append(node + file)
        # return all_nodes
    return all_nodes


def render_tree():
    # from streamlit_tree_select import tree_select

    st.spinner("Scanning data dir...")
    all_nodes = dfs("/app/data")
    st.write(all_nodes)

    st.title("🐙 Streamlit-tree-select")
    st.subheader("A simple and elegant checkbox tree for Streamlit.")
    return
    # Create nodes to display
    nodes = [
        {"label": "Folder A", "value": "folder_a"},
        {
            "label": "Folder B",
            "value": "folder_b",
            "children": [
                {"label": "Sub-folder A", "value": "sub_a"},
                {"label": "Sub-folder B", "value": "sub_b"},
                {"label": "Sub-folder C", "value": "sub_c"},
            ],
        },
        {
            "label": "Folder C",
            "value": "folder_c",
            "children": [
                {"label": "Sub-folder D", "value": "sub_d"},
                {
                    "label": "Sub-folder E",
                    "value": "sub_e",
                    "children": [
                        {"label": "Sub-sub-folder A", "value": "sub_sub_a"},
                        {"label": "Sub-sub-folder B", "value": "sub_sub_b"},
                    ],
                },
                {"label": "Sub-folder F", "value": "sub_f"},
            ],
        },
    ]

    return_select = tree_select(nodes)
    st.write(return_select)


def render_cleanup():
    st.title("Cleanup")

    def delete_all_files():
        st.write(f"Deleting all files in workspace for user {st.session_state['name']}")
        # st.write(st.session_state["user_workspace"])

        # Use shutil to recursively remove directory contents
        for item in os.listdir(st.session_state["user_workspace"]):
            item_path = os.path.join(st.session_state["user_workspace"], item)
            if os.path.isfile(item_path):
                os.remove(item_path)
            elif os.path.isdir(item_path):
                shutil.rmtree(item_path)

        st.success("All files and folders deleted")

    st.button("Delete all files in workspace for user", on_click=delete_all_files)


def import_export_workspace():
    st.subheader("Import/Export workspace")
    
    workspace = "/app/data"
    if os.path.exists(workspace):
        if st.button("Export All Workspaces"):
            try:
                # Create a timestamp for the zip file name
                from datetime import datetime
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                zip_filename = f"workspace_backup_{timestamp}.zip"
                
                # Create zip file in memory
                with st.spinner("Creating backup..."):
                    shutil.make_archive(
                        base_name=os.path.join(workspace, zip_filename[:-4]),  # Remove .zip extension
                        format='zip',
                        root_dir=workspace
                    )
                    
                    # Create download button for the zip file
                    zip_path = os.path.join(workspace, zip_filename)
                    with open(zip_path, 'rb') as f:
                        st.download_button(
                            label="Download Workspace Backup",
                            data=f,
                            file_name=zip_filename,
                            mime="application/zip"
                        )
                    
                    # Clean up the zip file after creating download button
                    if os.path.exists(zip_path):
                        os.remove(zip_path)
                        
                st.success("Backup created successfully!")
            except Exception as e:
                st.error(f"Error creating backup: {str(e)}")
    else:
        st.error("Workspace directory not found")


if st.session_state.get("authentication_status"):
    tab1, tab2, tab3, tab4 = st.tabs(["Disk Usage", "File System", "Cleanup", "Export/Import Workspace"])
    with tab1:
        render_admin()
    with tab2:
        render_tree()
    with tab3:
        render_cleanup()
    with tab4:
        import_export_workspace()
else:
    st.warning("Please log in to view admin panel")
