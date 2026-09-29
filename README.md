# Receipt text detector pilot

This folder contains an isolated detector-plus-postprocessing pilot. It is
separate from both the production cropper and the thesis feature extractor.

The pilot uses PaddleOCR **text detection only**. It does not run OCR text
recognition and does not use word content. Detector polygons are perspective
rectified, trimmed to ink bounds, and padded proportionally to text height.
Every output is retained for review. Barcode-like geometry is a review flag,
not an automatic rejection, because long receipt text lines can look similar.

## Run the mixed pilot

The isolated dependencies are in `python_deps`. The first run may use the
cached PP-OCRv5 detector model under the user's local model cache.

```powershell
$py = 'C:\Users\dev\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe'
$deps = 'C:\Users\dev\Workspace\school\elective\programs\receipt_text_detector\python_deps'
$env:PYTHONPATH = $deps

& $py '.\run_detector_pilot.py' `
  --fake-root '..\..\data\receipt_dataset\raw\receipts\1_fake' `
  --real-root '..\..\data\receipt_dataset\raw\receipts\0_real' `
  --output-dir '..\..\data\receipt_dataset\processed\detector_pilot_v3' `
  --fake-count 10 `
  --real-count 10
```

## Outputs

Each receipt receives:

- `source_with_boxes.jpg` with detector polygons;
- `contact_sheet.jpg` with rectified crops;
- `manifest.csv` with polygon, score, trim, and review fields.

`aggregate_manifest.csv` combines all pilot detections. The current pilot
result is stored under `data/receipt_dataset/processed/detector_pilot_v3`.

This is an evaluation utility. We will only adopt it for the shared crop
pipeline after the mixed pilot is manually reviewed and compared against the
existing detector-free cropper.
