import streamlit as st
import os
from pathlib import Path


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
    stack  = [root] 

    all_nodes = []
    while stack:
        node = stack.pop()
        all_nodes.append(node)
        files =  [x[0] for x in os.walk(node)]
        for file in files:
            stack.append(node + file)
        #return all_nodes
    return all_nodes


def render_tree():
    from streamlit_tree_select import tree_select

    st.spinner( "Scanning data dir...")
    all_nodes  = dfs( "/app/data")
    st.write( all_nodes)

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






if st.session_state.get("authentication_status"):
    tab1, tab2 = st.tabs( ["Disk Usage", "File System"] )
    with tab1:
        render_admin()
    with tab2:
        render_tree()
else:
    st.warning("Please log in to view admin panel")
