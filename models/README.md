# Models Directory

This directory stores pre-trained and fine-tuned weights for the VisionAssist perception pipeline.

## Fine-tuning workflow

The checked-in `yolov8m.pt` is a general COCO model. It has not been trained
on VisionAssist camera views, so collecting representative labeled frames is
required before expecting better stairs, chairs, doors, or obstacle accuracy.

1. Create `images/train`, `images/val`, `labels/train`, and `labels/val` under
	a dataset directory and copy `dataset.yaml.example` there as `dataset.yaml`.
2. Label every image in standard YOLO format. Include clear negatives that do
	not contain the target objects, especially ordinary floors and shelves.
3. Run the guarded trainer from the project root:

```powershell
python scripts/train_custom_model.py --data path\to\dataset\dataset.yaml --epochs 50
```

The trainer refuses to start when labels are missing. Review the validation
metrics and confusion matrix before replacing the production weights.

## Default Weights
- **YOLOv8 Nano (`yolov8n.pt`)**: Downloaded automatically by Ultralytics on first run.
- **MiDaS Small (`dpt_swin2_tiny_256.pt` / `model-small.onnx`)**: Optional monocular depth estimation weights.

## Usage
When running `app.py` or the Streamlit dashboard, Ultralytics downloads `yolov8n.pt` into this directory or the working folder automatically if not already present.
