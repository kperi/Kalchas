import numpy as np
import cv2
import os
import torch
import json
from kraken import pageseg, binarization
from PIL import Image
from ocr.predict import TextRegognizer

ARTIFACTS = "/src/ocr/artifacts/model7_artifacts.pt"
# MODEL_PATH = "ocr/artifacts/best_cpu_model_7.pth"
MODEL_PATH = "/src/ocr/artifacts/88_model_best_loss_0.7971.pt"


import numpy as np
from skimage import io
from skimage.transform import rotate
from skimage.color import rgb2gray

# from deskew import determine_skew


def deskew(grayscale):
    return grayscale
    # image = io.imread(_img)
    # grayscale = rgb2gray(image)
    angle = determine_skew(grayscale)
    rotated = rotate(grayscale, angle, resize=True) * 255
    return rotated.astype(np.uint8)


def segment(image):
    """
    image: np.array
    """
    # binarize

    if isinstance(image, (np.ndarray, np.generic)):
        image = Image.fromarray(image)

    image = binarization.nlbin(image)
    # segment
    regions = pageseg.segment(image)
    return regions


def segmentation_and_recognition(src_page):

    char2idx, idx2char, width, height, MAX_TEXT_LENGTH = torch.load(ARTIFACTS)

    # pbar = st.progress(0.0)

    device = torch.device("cpu")

    model = TextRegognizer(
        device=device,
        width=width,
        height=height,
        num_class=len(idx2char),
        model_path=MODEL_PATH,
        idx2char=idx2char,
        char2idx=char2idx,
    )

    img = cv2.imread(src_page)
    if img is None:
        # st.error("Could not read image")
        return {"status": "error", "file": src_page, "message": "Could not read image"}

    img = deskew(img)
    # save back deskewed image - TODO: move this to edit
    cv2.imwrite(src_page, img)

    sections = segment(img)

    # boxes
    boxes = [
        ((x1, y1, x2, y2), f"line {idx}")
        for idx, (x1, y1, x2, y2) in enumerate(sections["boxes"])
    ]

    # get pages and create a dir per page to store segments
    page_number = os.path.basename(src_page).split(".")[0]
    base_dir = os.path.dirname(src_page)
    out_dir = os.path.join(base_dir, page_number)
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    # for all segments, ocr and save
    for index in range(len(boxes)):

        x1, y1, x2, y2 = boxes[index][0]
        # print( "coords:", (x1,x2), (y1,y2 ))
        cropped_im = img[y1:y2, x1:x2, :]

        gray_image = cv2.cvtColor(cropped_im, cv2.COLOR_BGR2GRAY)

        # plt.imshow(gray_image, cmap="gray")
        # plt.show()
        out_file = os.path.join(out_dir, f"{index:>03d}.png")
        out_txt = os.path.join(out_dir, f"{index:>03d}.json")

        # print(gray_image.shape)
        # break
        # write coords
        ocred_text = model.ocr(gray_image)

        job = {"coords": [x1, y1, x2, y2], "text": ocred_text}
        json.dump(job, open(out_txt, "w"))

        # write image
        # gray_image = deskew(gray_image)
        print(f"Outfile = {out_file}")

        cv2.imwrite(out_file, gray_image)

        # pbar.progress( (index+1) / len(boxes), "OCR..." )
        # st.image(gray_image, caption=ocred_text,)

    return {
        "status": "ok",
        "file": src_page,
        "message": "Segmentation and recognition completed",
    }


def segmentation_and_recognition_ii(src_page):

    char2idx, idx2char, width, height, MAX_TEXT_LENGTH = torch.load(ARTIFACTS)
    device = torch.device("cpu")

    model = TextRegognizer(
        device=device,
        width=width,
        height=height,
        num_class=len(idx2char),
        model_path=MODEL_PATH,
        idx2char=idx2char,
        char2idx=char2idx,
    )

    img = cv2.imread(src_page)
    if img is None:
        # st.error("Could not read image")
        return {"status": "error", "file": src_page, "message": "Could not read image"}

    img = deskew(img)
    # save back deskewed image - TODO: move this to edit
    cv2.imwrite(src_page, img)

    sections = segment(img)

    # boxes
    boxes = [
        ((x1, y1, x2, y2), f"line {idx}")
        for idx, (x1, y1, x2, y2) in enumerate(sections["boxes"])
    ]

    # get pages and create a dir per page to store segments
    # page_number = os.path.basename(src_page).split(".")[0]
    # base_dir = os.path.dirname(src_page)
    # out_dir = os.path.join(base_dir, page_number)
    # if not os.path.exists(out_dir):
    #    os.mkdir(out_dir)
    # for all segments, ocr and save

    ret_text = ""
    for index in range(len(boxes)):

        x1, y1, x2, y2 = boxes[index][0]
        cropped_im = img[y1:y2, x1:x2, :]

        gray_image = cv2.cvtColor(cropped_im, cv2.COLOR_BGR2GRAY)

        # plt.imshow(gray_image, cmap="gray")
        # plt.show()
        # out_file = os.path.join(out_dir, f"{index:>03d}.png")
        # out_txt = os.path.join(out_dir, f"{index:>03d}.json")

        # print(gray_image.shape)
        # break
        # write coords
        ocred_text = model.ocr(gray_image)

        job = {"coords": [x1, y1, x2, y2], "text": ocred_text}
        # json.dump(job, open(out_txt, "w"))

        # write image
        # gray_image = deskew(gray_image)
        # print(f"Outfile = {out_file}")

        # cv2.imwrite(out_file, gray_image)

        # pbar.progress( (index+1) / len(boxes), "OCR..." )
        # st.image(gray_image, caption=ocred_text,)
        ret_text += ocred_text[0] + "\n"

    return {
        "status": "ok",
        "file": src_page,
        "message": "Segmentation and recognition completed",
        "recognized_text": ret_text,
    }
