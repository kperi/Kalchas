import json
from dataclasses import dataclass
from typing import Any


class InvalidModelResponse(ValueError):
    pass


@dataclass(frozen=True)
class ImageSize:
    width: int
    height: int


def _extract_json(value: str) -> Any:
    content = value.strip()
    if content.startswith("```"):
        first_newline = content.find("\n")
        last_fence = content.rfind("```")
        if first_newline == -1 or last_fence <= first_newline:
            raise InvalidModelResponse("Model response contains an invalid code fence")
        content = content[first_newline + 1 : last_fence].strip()

    try:
        return json.loads(content)
    except json.JSONDecodeError as exc:
        raise InvalidModelResponse(
            f"Model response is not valid JSON: {exc.msg}"
        ) from exc


def _validate_bbox(value: Any, image_size: ImageSize, index: int) -> list[int]:
    if not isinstance(value, list) or len(value) != 4:
        raise InvalidModelResponse(
            f"Segment {index} must contain a four-number bbox"
        )
    if not all(isinstance(coordinate, (int, float)) for coordinate in value):
        raise InvalidModelResponse(f"Segment {index} bbox must contain only numbers")

    x1, y1, x2, y2 = (round(coordinate) for coordinate in value)
    if not (0 <= x1 < x2 <= image_size.width):
        raise InvalidModelResponse(
            f"Segment {index} bbox has invalid horizontal coordinates"
        )
    if not (0 <= y1 < y2 <= image_size.height):
        raise InvalidModelResponse(
            f"Segment {index} bbox has invalid vertical coordinates"
        )
    return [x1, y1, x2, y2]


def parse_segments(value: str, image_size: ImageSize) -> list[dict[str, Any]]:
    payload = _extract_json(value)
    if isinstance(payload, dict):
        payload = payload.get("segments")
    if not isinstance(payload, list):
        raise InvalidModelResponse(
            "Model response must be a JSON array or an object with a segments array"
        )

    segments = []
    for index, segment in enumerate(payload):
        if not isinstance(segment, dict):
            raise InvalidModelResponse(f"Segment {index} must be a JSON object")

        text = segment.get("text")
        if not isinstance(text, str) or not text.strip():
            raise InvalidModelResponse(f"Segment {index} must contain non-empty text")

        segments.append(
            {
                "text": text.strip(),
                "bbox": _validate_bbox(segment.get("bbox"), image_size, index),
            }
        )
    return segments

