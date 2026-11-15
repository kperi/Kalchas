import logging
from collections.abc import Callable
from pathlib import Path

from PIL import Image

from app.response_parser import NoRecognizedText
from app.segmentation import MAX_PIXELS, segment_lines


logger = logging.getLogger(__name__)


def process_coptic(
    image_path: Path,
    recognize: Callable[[Image.Image], list[dict]],
) -> list[dict]:
    with Image.open(image_path) as source:
        if source.width * source.height > MAX_PIXELS:
            raise ValueError("Image exceeds 24 megapixels; resize it before OCR.")
        # Logios previews the stored pixels without EXIF transposition.
        image = source.convert("RGB")
    lines = segment_lines(image)
    if not lines:
        raise ValueError("No text lines detected; crop the text region before OCR.")

    segments = []
    for line in lines:
        try:
            recognized = recognize(line.image)
            text = "\n".join(segment["text"] for segment in recognized).strip()
        except NoRecognizedText:
            text = ""
            logger.warning("No text recognized for Coptic line %d in %s", len(segments), image_path)
        segments.append({"text": text, "bbox": line.bbox, "coords": line.bbox})

    if not any(segment["text"] for segment in segments):
        raise NoRecognizedText("Model response contains no recognized text on any line")

    output_dir = image_path.parent / image_path.stem
    output_dir.mkdir(exist_ok=True)
    for existing_crop in output_dir.glob("*.png"):
        if len(existing_crop.stem) == 3 and existing_crop.stem.isdigit():
            existing_crop.unlink()
    for index, line in enumerate(lines):
        line.image.save(output_dir / f"{index:03d}.png", format="PNG")
    logger.info("Segmented %s into %d Coptic lines", image_path, len(segments))
    return segments
