# Qwen VLM OCR service

This service mirrors the existing OCR service's `POST /process_image/` contract
and runs alongside it. It loads `Qwen/Qwen3-VL-2B-Instruct` with the PEFT LoRA
adapter mounted from `~/code/projects/datasets-master/vlm/multiple_langs/`.
Fine-tuned plain-text output is returned as a page-level segment by default.
Coptic requests from Logios use manuscript line segmentation before recognition.

The first startup downloads the model into the `vlm_model_cache` Docker volume.
An NVIDIA GPU and NVIDIA Container Toolkit are expected for practical inference.

## API

```bash
curl -X POST http://localhost:8001/process_image/ \
  -H "Content-Type: application/json" \
  -d '{"value":"/app/uploads/user/document/TOOCR/PNG/page_001.png"}'
```

Health and loaded model information are available from `GET /health`.

## Coptic line segmentation

The Flask route adds `"segmentation": "manuscript"` for Coptic only. Polytonic
Greek continues to use the original Kraken segmentation and Greek OCR service,
with no change to its request or processing. Requests to this VLM service without
the new field retain whole-page recognition.

The CPU technique is adapted from
`~/code/projects/datasets-master/page_segmentation/segment.py`: green-channel
adaptive thresholding, projection peak detection, and minimum-cost curved
whitespace seams. Detection does not binarize the OCR crops: they retain RGB
pixels, mask neighboring lines, and have a five-pixel synthetic border.

The selected page or crop is one text region, processed top-to-bottom. For
multi-column images, grids, illustrations, or ornamental margins, use Logios's
existing cropping workspace to isolate text regions first. The source example
presets and their manually reviewed line centers are not automatically applied
to unrelated uploads. No text lines detected is an explicit processing error,
not a fallback to whole-page OCR.
An individual detected line for which the model returns no text is retained with
an empty `text` value so later lines can still be recognized. If every detected
line is empty, the request fails explicitly instead of reporting false success.

The response keeps the existing `received_value` / `ret` envelope with `status`,
`file`, `message`, `recognized_text`, and `segments`. Each detected line yields
one segment with string `text`, `bbox`, and legacy-compatible `coords`, both
`[x1, y1, x2, y2]` in stored source pixels (not padded crop coordinates).
Model-produced crop-local boxes do not replace the detector's source boxes.
The original image is not overwritten or EXIF-transposed.

RGB crops are saved alongside the input under `<page-stem>/000.png`,
`001.png`, etc. Flask persists the matching numbered JSON files and summary
using its existing save path, so immediate preview and reload use the same
segment IDs and boxes. A successful reprocessing run replaces prior numbered
PNG crops and JSON segments while leaving unrelated files intact. The uploads
mount is writable for this purpose; model
weights remain read-only. Inputs larger than 24 megapixels are rejected.

```bash
curl -X POST http://localhost:8001/process_image/ \
  -H "Content-Type: application/json" \
  -d '{"value":"/app/uploads/user/document/TOOCR/PNG/page_001.png","segmentation":"manuscript"}'
```

Rebuild the VLM service after updating its segmentation dependencies.

The lightweight tests do not load model weights. From this directory, with
the service dependencies plus `pytest` and `httpx` installed, run:

```bash
python -m pytest tests -q
```

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
