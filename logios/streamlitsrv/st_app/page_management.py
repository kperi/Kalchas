import glob
import os
import json


def get_max_final(book_page_path):
    finals = glob.glob(os.path.join(book_page_path, "*.final"))
    finals = [int(f.split("/")[-1].split(".")[0]) for f in finals]
    return max(finals) if finals else 0


def get_finalized_lines(book_page_path):
    finals = glob.glob(os.path.join(book_page_path, "*.final"))
    finals = [int(f.split("/")[-1].split(".")[0]) for f in finals]
    finals = sorted(finals)
    finals = [str(f) for f in finals]
    return ",".join(finals)


def get_all_lines(book_page_path):
    all_lines = glob.glob(os.path.join(book_page_path, "*.png"))
    all_lines = [int(f.split("/")[-1].split(".")[0]) for f in all_lines]

    return all_lines


def get_all_texts(book_page_path):

    finals = glob.glob(os.path.join(book_page_path, "*.json"))
    finals = sorted(finals)
    lines = []
    for f in finals:
        final_f = f.replace(".json", ".final")
        if os.path.exists(final_f):
            line = open(final_f).read()
        else:
            jcontent = json.load(open(f))
            line = ">>> " + jcontent["text"][0]

        lines.append(line)

    # lines = [open(f).read() for f in finals]

    return "\n".join(lines)
