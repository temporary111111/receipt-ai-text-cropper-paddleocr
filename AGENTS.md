# Agent instructions: PaddleOCR detector cropper

## Purpose

This repository contains the PaddleOCR text-detection-only cropper. It uses
detector polygons, perspective correction, tight ink-bound trimming, and
proportional padding. It does not recognize text or use OCR content.

## Entrypoint

- `run_detector_pilot.py` — resumable detector and crop pipeline.
- Use `--resume` to skip complete receipt folders.
- Use `--device gpu:0` only after the GPU preflight reports CUDA support and
  the cuDNN versions are compatible.

## Environment

- CPU environment: `python_deps` with the bundled Python runtime.
- GPU environment: `python_gpu_env` with the matching Paddle GPU and cuDNN
  runtime. Never set the old CPU `PYTHONPATH` when launching the GPU venv.

## Outputs

Each receipt folder contains `crop_*.png`, `manifest.csv`,
`source_with_boxes.jpg`, and `contact_sheet.jpg`. The aggregate manifest is
for audit and downstream feature extraction.

Barcode-like detections are review flags only. They are retained by default;
do not automatically reject them without visual evidence.

## Safety

Do not delete raw receipts. Preserve resumability and do not mix CPU outputs
with GPU outputs produced under an incompatible runtime. Do not commit model
caches, environments, crops, or CSV outputs.
