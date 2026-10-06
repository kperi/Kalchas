# Qwen VLM OCR service

This service mirrors the existing OCR service's `POST /process_image/` contract
and runs alongside it. It loads `Qwen/Qwen3-VL-2B-Instruct` with the PEFT LoRA
adapter mounted from `~/code/projects/datasets-master/vlm/multiple_langs/`.
Fine-tuned plain-text output is returned as a page-level segment.

The first startup downloads the model into the `vlm_model_cache` Docker volume.
An NVIDIA GPU and NVIDIA Container Toolkit are expected for practical inference.

## API

```bash
curl -X POST http://localhost:8001/process_image/ \
  -H "Content-Type: application/json" \
  -d '{"value":"/app/uploads/user/document/TOOCR/PNG/page_001.png"}'
```

Health and loaded model information are available from `GET /health`.

## Configuration

- `VLM_MODEL_ID`: Hugging Face base model ID; defaults to
  `Qwen/Qwen3-VL-2B-Instruct`.
- `VLM_MODEL_PATH`: Fine-tuned PEFT adapter path; defaults to
  `/models/finetuned`.
- `VLM_MAX_NEW_TOKENS`: Maximum generated tokens; defaults to `128`.
- `VLM_MIN_PIXELS`: Processor minimum image pixels; defaults to `200704`.
- `VLM_MAX_PIXELS`: Processor maximum image pixels; defaults to `401408`.
- `VLM_OCR_PROMPT`: Complete OCR instruction override.
- `OCR_IMAGE_ROOT`: Allowed image root; defaults to `/app/uploads`.

The Flask UI routes Polytonic Greek requests to the existing `ocr` service and
routes Coptic, Old English, Latin, and Catalan requests to this service through
`VLM_OCR_SERVICE_URL`.
