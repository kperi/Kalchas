import os
import json
import base64
import re
import shutil
from collections import defaultdict
from loguru import logger  # Or pass the Flask app's logger

# --- Path Helper Functions ---


def get_upload_root(app_config):
    return app_config["UPLOAD_FOLDER"]


def get_user_dir(app_config, user_id):
    return os.path.join(get_upload_root(app_config), str(user_id))


def get_document_dir(app_config, user_id, folder_name):
    return os.path.join(get_user_dir(app_config, user_id), folder_name)


def get_toocr_dir(app_config, user_id, folder_name):
    return os.path.join(get_document_dir(app_config, user_id, folder_name), "TOOCR")


def get_page_segments_dir(app_config, user_id, folder_name, page_id_base):
    """Directory where individual segment .json and .png files for a page are stored."""
    return os.path.join(get_toocr_dir(app_config, user_id, folder_name), page_id_base)


def get_combined_segments_parent_dir(app_config, user_id, folder_name):
    """Directory where the combined <page_id_base>.json (summary for a page) is stored."""
    return os.path.join(get_toocr_dir(app_config, user_id, folder_name), "segments")


def get_cropped_images_dir(app_config, user_id, folder_name):
    return os.path.join(get_document_dir(app_config, user_id, folder_name), "cropped")


# --- Segment and Image File Operations ---


def get_segment_image_file_details(
    app_config, user_id, folder_name, page_id_base, segment_id_str
):
    """
    Determines the directory and filename for a specific segment image.
    Validates existence of the directory and the image file.
    Args:
        segment_id_str (str): The segment ID, typically an integer string.
    Returns:
        tuple: (directory_path, image_filename_with_ext) if found and valid.
               (None, None) if any path component is missing or file not found.
    """
    segments_storage_dir = get_page_segments_dir(
        app_config, user_id, folder_name, page_id_base
    )

    if not os.path.isdir(segments_storage_dir):
        logger.error(f"Segments directory for page not found: {segments_storage_dir}")
        return None, None

    try:
        # Segment images are typically named like 001.png, 002.png etc.
        image_filename = f"{int(segment_id_str):03d}.png"
    except ValueError:
        logger.error(
            f"Invalid segment_id format: '{segment_id_str}'. Must be an integer string."
        )
        # Attempt to use segment_id_str directly if it already looks like a filename
        if segment_id_str.lower().endswith(".png") and os.path.exists(
            os.path.join(segments_storage_dir, segment_id_str)
        ):
            image_filename = segment_id_str
        else:
            return None, None  # Invalid format and not a direct file match

    image_path = os.path.join(segments_storage_dir, image_filename)
    logger.info(f"Attempting to locate segment image: {image_path}")

    if not os.path.exists(image_path):
        logger.error(f"Segment image file not found at: {image_path}")
        return None, None

    return segments_storage_dir, image_filename


def fetch_page_segments_data(app_config, user_id, folder_name, page_id_base):
    """
    Retrieves all segment data (from .json files) for a specific page.
    """
    segments_storage_dir = get_page_segments_dir(
        app_config, user_id, folder_name, page_id_base
    )
    if not os.path.isdir(segments_storage_dir):
        logger.warning(
            f"Segments directory not found or invalid for page {page_id_base}"
        )
        return []

    segments_data_list = []
    try:
        json_files = sorted(
            [f for f in os.listdir(segments_storage_dir) if f.lower().endswith(".json")]
        )
        for json_file in json_files:
            try:
                with open(os.path.join(segments_storage_dir, json_file), "r") as f:
                    segment_data = json.load(f)
                segments_data_list.append(segment_data)
            except Exception as e:
                logger.error(
                    f"Error loading segment file {json_file} in {segments_storage_dir}: {str(e)}"
                )
        logger.info(
            f"Found {len(segments_data_list)} segment JSON files in {segments_storage_dir}"
        )
    except Exception as e:
        logger.error(
            f"Error listing/processing segment JSON files in {segments_storage_dir}: {str(e)}"
        )
        return []

    return segments_data_list


def save_ocr_processed_data(
    app_config, user_id, folder_name, page_id_base, ocr_payload
):
    """
    Saves OCR results, including individual segment JSONs, segment images,
    and a combined page JSON.
    Returns:
        tuple: (final_text_summary, list_of_all_segments_for_response)
               Returns (error_message_str, None) on critical failure.
    """
    # Dir for combined page summary JSON (e.g., TOOCR/segments/page_001_crop_001.json)
    combined_json_parent_dir = get_combined_segments_parent_dir(
        app_config, user_id, folder_name
    )
    os.makedirs(combined_json_parent_dir, exist_ok=True)

    # Dir for individual segment files (e.g., TOOCR/page_001_crop_001/000.json, 000.png)
    individual_segments_dir = get_page_segments_dir(
        app_config, user_id, folder_name, page_id_base
    )
    os.makedirs(individual_segments_dir, exist_ok=True)

    all_segments_for_response = []
    final_text_summary = ""

    if isinstance(ocr_payload, dict) and "status" in ocr_payload:
        status = ocr_payload.get("status")
        file_path_from_ocr = ocr_payload.get("file", "")
        final_text_summary = f"OCR status: {status}"
        segment_for_response = {
            "id": "000",
            "text": final_text_summary,
            "has_image": False,
            "file": file_path_from_ocr,
        }
        all_segments_for_response.append(segment_for_response)
        # No individual segment files typically saved for this case by default

    elif isinstance(ocr_payload, list):  # List of segment dictionaries
        for idx, segment_info_from_ocr in enumerate(ocr_payload):
            if not isinstance(segment_info_from_ocr, dict):
                logger.warning(
                    f"Segment {idx} is not a dictionary: {type(segment_info_from_ocr)}"
                )
                continue

            segment_text = segment_info_from_ocr.get("text", "")
            final_text_summary += f"{segment_text}\n"
            segment_id_str = f"{idx:03d}"

            current_segment_data_for_json_and_response = {
                "id": segment_id_str,
                "coords": segment_info_from_ocr.get("coords", []),
                "text": segment_text,
                "has_image": "image_data" in segment_info_from_ocr
                and bool(segment_info_from_ocr["image_data"]),
            }
            all_segments_for_response.append(current_segment_data_for_json_and_response)

            # Save individual segment JSON
            json_path = os.path.join(individual_segments_dir, f"{segment_id_str}.json")
            try:
                with open(json_path, "w") as f:
                    json.dump(current_segment_data_for_json_and_response, f)
            except Exception as e:
                logger.error(f"Error saving individual segment JSON {json_path}: {e}")

            # Save segment image if available
            if current_segment_data_for_json_and_response["has_image"]:
                try:
                    img_data_b64 = re.sub(
                        r"^data:image/\w+;base64,",
                        "",
                        segment_info_from_ocr["image_data"],
                    )
                    img_bytes = base64.b64decode(img_data_b64)
                    img_path = os.path.join(
                        individual_segments_dir, f"{segment_id_str}.png"
                    )
                    with open(img_path, "wb") as img_file:
                        img_file.write(img_bytes)
                    logger.info(f"Saved segment image: {img_path}")
                except Exception as img_error:
                    logger.error(
                        f"Error saving segment image for {segment_id_str}: {str(img_error)}"
                    )
                    # Update has_image flag if saving failed
                    all_segments_for_response[-1]["has_image"] = False
                    current_segment_data_for_json_and_response["has_image"] = False
                    try:  # Attempt to re-save JSON with updated has_image flag
                        with open(json_path, "w") as f:
                            json.dump(current_segment_data_for_json_and_response, f)
                    except Exception as e_resave:
                        logger.error(
                            f"Error re-saving segment JSON {json_path} after image error: {e_resave}"
                        )

    elif isinstance(ocr_payload, str):  # Plain text OCR result
        final_text_summary = ocr_payload
        segment_for_response = {
            "id": "000",
            "text": final_text_summary,
            "has_image": False,
        }
        all_segments_for_response.append(segment_for_response)

    else:
        err_msg = f"Unhandled OCR payload type for saving: {type(ocr_payload)}"
        logger.error(err_msg)
        return err_msg, None

    # Save combined segments JSON for the page (summary file)
    combined_json_path = os.path.join(combined_json_parent_dir, f"{page_id_base}.json")
    try:
        with open(combined_json_path, "w") as f:
            json.dump(
                all_segments_for_response, f
            )  # Save the list that will be sent in HTTP response
        logger.info(f"Saved combined segments summary JSON: {combined_json_path}")
    except Exception as e:
        logger.error(f"Error saving combined segments JSON {combined_json_path}: {e}")
        # This is not critical if individual files are saved, but good to log

    return final_text_summary.strip(), all_segments_for_response


def save_uploaded_cropped_image(
    app_config, user_id, original_folder, original_filename, cropped_image_data_url
):
    """Saves a base64 encoded cropped image to the filesystem."""
    cropped_dir = get_cropped_images_dir(app_config, user_id, original_folder)
    os.makedirs(cropped_dir, exist_ok=True)

    original_page_base = os.path.splitext(original_filename)[0]
    existing_crops = [
        f
        for f in os.listdir(cropped_dir)
        if f.startswith(f"{original_page_base}_cropped_")
    ]
    crop_numbers = [
        int(re.search(r"_cropped_(\d{3})\.png$", cf).group(1))
        for cf in existing_crops
        if re.search(r"_cropped_(\d{3})\.png$", cf)
    ]
    next_crop_num = max(crop_numbers) + 1 if crop_numbers else 1

    cropped_filename = f"{original_page_base}_cropped_{next_crop_num:03d}.png"
    cropped_file_path = os.path.join(cropped_dir, cropped_filename)

    try:
        image_data = re.sub(r"^data:image/\w+;base64,", "", cropped_image_data_url)
        binary_data = base64.b64decode(image_data)
        with open(cropped_file_path, "wb") as f:
            f.write(binary_data)
        logger.info(f"Saved cropped image to {cropped_file_path}")
        return True, cropped_filename
    except Exception as e:
        logger.error(f"Error saving cropped image: {str(e)}")
        return False, str(e)


def copy_files_to_ocr_folder(app_config, user_id, folder_name):
    """Copies original or cropped images to the TOOCR folder with standardized names."""
    source_document_dir = get_document_dir(app_config, user_id, folder_name)
    source_cropped_dir = get_cropped_images_dir(app_config, user_id, folder_name)
    destination_toocr_dir = get_toocr_dir(app_config, user_id, folder_name)
    os.makedirs(destination_toocr_dir, exist_ok=True)

    copied_files_count = 0
    page_crops_map = defaultdict(list)

    if os.path.exists(source_cropped_dir):
        for fname in os.listdir(source_cropped_dir):
            if fname.lower().endswith(".png"):
                match = re.match(r"(\d+)_cropped_\d+\.png", fname)
                if match:
                    page_num_base = match.group(1)
                    page_crops_map[page_num_base].append(fname)

    for original_fname in os.listdir(source_document_dir):
        if not original_fname.lower().endswith(".png") or original_fname.startswith(
            "."
        ):
            continue

        page_base_num = os.path.splitext(original_fname)[0]

        if page_crops_map[page_base_num]:
            for i, crop_file_name in enumerate(
                sorted(page_crops_map[page_base_num]), 1
            ):
                src_path = os.path.join(source_cropped_dir, crop_file_name)
                dest_filename = f"page_{page_base_num}_crop_{i:03d}.png"
                dest_path = os.path.join(destination_toocr_dir, dest_filename)
                shutil.copy2(src_path, dest_path)
                copied_files_count += 1
        else:
            src_path = os.path.join(source_document_dir, original_fname)
            dest_filename = f"page_{page_base_num}.png"  # Original page if no crops
            dest_path = os.path.join(destination_toocr_dir, dest_filename)
            shutil.copy2(src_path, dest_path)
            copied_files_count += 1

    return True, copied_files_count


def resolve_segment_image_filename_from_index(
    app_config, user_id, folder_name, page_id_base, selected_segment_index_str
):
    """Resolves segment image filename by index from the page's segment directory."""
    segment_storage_dir = get_page_segments_dir(
        app_config, user_id, folder_name, page_id_base
    )
    resolved_filename = None

    if not os.path.isdir(segment_storage_dir):
        logger.warning(f"Segment image directory not found: {segment_storage_dir}")
        return "error_directory_not_found.png"

    try:
        # Assuming filenames like "001.png", "002.png" which sort correctly alphanumerically.
        # For natural sort (1.png, 2.png, 10.png), a more complex sort key would be needed.
        png_files = sorted(
            [f for f in os.listdir(segment_storage_dir) if f.lower().endswith(".png")]
        )

        segment_index = int(selected_segment_index_str)

        if 0 <= segment_index < len(png_files):
            resolved_filename = png_files[segment_index]
        else:  # Index out of bounds
            logger.warning(
                f"Segment index {segment_index} out of bounds for {page_id_base}. PNGs: {len(png_files)}"
            )
            # Fallback: try to use selected_segment_index_str as a direct name (e.g., "021")
            potential_direct_filename = f"{selected_segment_index_str}.png"
            if potential_direct_filename in png_files:
                resolved_filename = potential_direct_filename
            else:  # Check if it's a name without extension
                if os.path.exists(
                    os.path.join(
                        segment_storage_dir, f"{selected_segment_index_str}.png"
                    )
                ):
                    resolved_filename = f"{selected_segment_index_str}.png"
                else:
                    resolved_filename = "error_image_not_found.png"
    except ValueError:  # selected_segment_index_str is not an int
        logger.error(
            f"Could not convert selected_segment '{selected_segment_index_str}' to int for {page_id_base}. Trying as direct filename."
        )
        potential_filename = f"{selected_segment_index_str}.png"  # e.g. "021.png"
        if os.path.exists(os.path.join(segment_storage_dir, potential_filename)):
            resolved_filename = potential_filename
        else:  # Try without .png if it was just "021"
            if os.path.exists(
                os.path.join(segment_storage_dir, f"{selected_segment_index_str}.png")
            ):
                resolved_filename = f"{selected_segment_index_str}.png"
            else:
                resolved_filename = "error_filename_invalid.png"
    except Exception as e:
        logger.error(
            f"Error resolving segment image for {page_id_base} by index '{selected_segment_index_str}': {e}"
        )
        resolved_filename = "error_processing_image_list.png"

    return resolved_filename
