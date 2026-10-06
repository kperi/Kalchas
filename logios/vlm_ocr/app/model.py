import os
import threading
import logging
from pathlib import Path

import torch
from peft import PeftModel
from PIL import Image
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

from app.response_parser import ImageSize, parse_ocr_response


logger = logging.getLogger(__name__)

DEFAULT_MODEL_ID = "Qwen/Qwen3-VL-2B-Instruct"
DEFAULT_MODEL_PATH = "/models/finetuned"
DEFAULT_PROMPT = "Extract text (OCR) from the image"


def _env_int(name: str, default: int) -> int:
    value = int(os.environ.get(name, default))
    if value <= 0:
        raise ValueError(f"{name} must be greater than zero")
    return value


class QwenOCRModel:
    def __init__(self) -> None:
        self.model_id = os.environ.get("VLM_MODEL_ID", DEFAULT_MODEL_ID)
        self.model_path = Path(
            os.environ.get("VLM_MODEL_PATH", DEFAULT_MODEL_PATH)
        ).resolve()
        self.max_new_tokens = _env_int("VLM_MAX_NEW_TOKENS", 128)
        self.prompt = os.environ.get("VLM_OCR_PROMPT", DEFAULT_PROMPT)
        self._generation_lock = threading.Lock()
        if not (self.model_path / "adapter_config.json").is_file():
            raise FileNotFoundError(
                f"Fine-tuned adapter not found at {self.model_path}"
            )

        if torch.cuda.is_available():
            dtype = (
                torch.bfloat16
                if torch.cuda.is_bf16_supported()
                else torch.float16
            )
        else:
            dtype = torch.float32

        self.processor = AutoProcessor.from_pretrained(
            self.model_path,
            min_pixels=_env_int("VLM_MIN_PIXELS", 256 * 28 * 28),
            max_pixels=_env_int("VLM_MAX_PIXELS", 512 * 28 * 28),
        )
        base_model = Qwen3VLForConditionalGeneration.from_pretrained(
            self.model_id,
            dtype=dtype,
            device_map="auto" if torch.cuda.is_available() else None,
            attn_implementation=os.environ.get("VLM_ATTN_IMPLEMENTATION", "sdpa"),
        )
        self.model = PeftModel.from_pretrained(base_model, self.model_path)
        if not torch.cuda.is_available():
            self.model = self.model.to("cpu")
        self.model.eval()

    def process(self, image_path: Path) -> list[dict]:
        with Image.open(image_path) as source_image:
            image = source_image.convert("RGB")
            image_size = ImageSize(*image.size)

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": self.prompt},
                ],
            }
        ]
        prompt = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self.processor(
            text=[prompt],
            images=[image],
            padding=True,
            return_tensors="pt",
        )
        inputs = {name: value.to(self.model.device) for name, value in inputs.items()}

        with self._generation_lock, torch.inference_mode():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
            )

        trimmed_ids = [
            output_ids[len(input_ids) :]
            for input_ids, output_ids in zip(inputs["input_ids"], generated_ids)
        ]
        response = self.processor.batch_decode(
            trimmed_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
        logger.debug("Raw Qwen OCR response: %s", response[:4000])
        return parse_ocr_response(response, image_size)
