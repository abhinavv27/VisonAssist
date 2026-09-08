# Models Directory

This directory stores pre-trained and fine-tuned weights for the VisionAssist perception pipeline.

## Default Weights
- **YOLOv8 Nano (`yolov8n.pt`)**: Downloaded automatically by Ultralytics on first run.
- **MiDaS Small (`dpt_swin2_tiny_256.pt` / `model-small.onnx`)**: Optional monocular depth estimation weights.

## Usage
When running `app.py` or the Streamlit dashboard, Ultralytics downloads `yolov8n.pt` into this directory or the working folder automatically if not already present.
