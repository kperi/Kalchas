import os
import threading
from pathlib import Path

import torch
from PIL import Image
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

from app.response_parser import ImageSize, parse_segments


DEFAULT_MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"
DEFAULT_PROMPT = """Transcribe every visible text line in this document image.
Preserve the original script, spelling, capitalization, punctuation, and polytonic
Greek diacritics exactly. Return only valid JSON in this shape:
{"segments":[{"text":"line text","bbox":[x1,y1,x2,y2]}]}
Use one segment per text line in natural reading order. Bounding boxes must use
absolute image pixel coordinates. Do not add explanations or Markdown fences."""


def _env_int(name: str, default: int) -> int:
    value = int(os.environ.get(name, default))
    if value <= 0:
        raise ValueError(f"{name} must be greater than zero")
    return value


class QwenOCRModel:
    def __init__(self) -> None:
        self.model_id = os.environ.get("VLM_MODEL_ID", DEFAULT_MODEL_ID)
        self.max_new_tokens = _env_int("VLM_MAX_NEW_TOKENS", 2048)
        self.prompt = os.environ.get("VLM_OCR_PROMPT", DEFAULT_PROMPT)
        self._generation_lock = threading.Lock()

        if torch.cuda.is_available():
            dtype = (
                torch.bfloat16
                if torch.cuda.is_bf16_supported()
                else torch.float16
            )
        else:
            dtype = torch.float32

        self.processor = AutoProcessor.from_pretrained(
            self.model_id,
            min_pixels=_env_int("VLM_MIN_PIXELS", 256 * 28 * 28),
            max_pixels=_env_int("VLM_MAX_PIXELS", 1280 * 28 * 28),
        )
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.model_id,
            torch_dtype=dtype,
            device_map="auto",
            low_cpu_mem_usage=True,
        )
        self.model.eval()

    def process(self, image_path: Path) -> list[dict]:
        with Image.open(image_path) as source_image:
            image = source_image.convert("RGB")
            image_size = ImageSize(*image.size)

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": self.prompt},
                ],
            }
        ]
        prompt = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[prompt],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(self.model.device)

        with self._generation_lock, torch.inference_mode():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
            )

        trimmed_ids = [
            output_ids[len(input_ids) :]
            for input_ids, output_ids in zip(inputs.input_ids, generated_ids)
        ]
        response = self.processor.batch_decode(
            trimmed_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
        return parse_segments(response, image_size)

