# Model Weights Directory

Store trained model checkpoints here.

## Current Files

```
weights/
├── yolov8n.pt       # YOLOv8 nano (6.5MB) - Fast, lightweight
├── yolov8s.pt       # YOLOv8 small (22MB) - Balanced
└── yolo11n.pt       # YOLO11 nano (5.6MB) - Latest version
```

## Download Pretrained Weights

Pretrained YOLO weights will be downloaded automatically when you run training/inference for the first time.

Manual download:
```bash
# YOLOv8 nano
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt

# YOLOv8 small
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8s.pt

# YOLOv8 medium
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8m.pt
```

## Custom Weights

After training, your best model will be saved as `best.pt`.

To use custom weights:
```bash
python scripts/deploy_edge.py --weights weights/best.pt --source 0
```
