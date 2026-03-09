# Military Object Detection & Tracking (MODT)

Production-ready end-to-end object detection and tracking system for military applications.

## Features

- Real-time detection using YOLOv8 (50-60 FPS on GPU)
- Multi-tracker support: Kalman, ByteTrack, DeepSORT, IoU
- Edge optimization: TensorRT, ONNX, INT8/FP16 quantization
- Trained model: 12 military object classes, 26K+ images

## System Requirements

- **Python**: 3.10+
- **GPU**: NVIDIA GPU with CUDA support (RTX 3060+ recommended)
- **VRAM**: 6GB+ for training, 4GB+ for inference
- **Disk**: 50GB+ for datasets

## Supported Objects (12 Classes)

| ID | Class Name | ID | Class Name |
|----|------------|----|----|
| 0 | Camouflage Soldier | 6 | Soldier |
| 1 | Weapon | 7 | Civilian Vehicle |
| 2 | Military Tank | 8 | Military Artillery |
| 3 | Military Truck | 9 | Trench |
| 4 | Military Vehicle | 10 | Military Aircraft |
| 5 | Civilian | 11 | Military Warship |

## Project Structure

```
MODT/
├── config/                        # Configuration files
│   ├── default.yaml               # Default settings
│   ├── training.yaml              # Training parameters
│   └── edge_jetson.yaml           # Jetson deployment config
│
├── data/                          # Datasets
│   └── military/assets/           # Military datasets
│       └── military_object_dataset/  # Main dataset (26K images, 12 classes)
│
├── docs/                          # Documentation
│   ├── ARCHITECTURE.md            # System architecture
│   ├── TRAINING.md                # Training guide
│   └── DEPLOYMENT.md              # Deployment guide
│
├── models/                        # Model implementations
│   ├── __init__.py                # Package init
│   ├── dtea.py                    # Dynamic Temporal Enhanced Attention
│   ├── backbones/                 # Backbone networks
│   ├── detectors/                 # Detection models (YOLODetector)
│   └── trackers/                  # Tracking algorithms (5 trackers)
│
├── notebooks/                     # Jupyter notebooks
│   ├── 01_dataset_preparation.ipynb
│   ├── 02_model_training.ipynb
│   └── 03_edge_optimization.ipynb
│
├── outputs/                       # Training outputs
│   ├── demo_output.mp4            # Demo video
│   ├── military_train/            # Baseline YOLOv8n model
│   ├── yolov8s_train/             # Best model (YOLOv8s, 75 epochs)
│   └── trained_model_demo/        # Demo results
│
├── scripts/                       # Executable scripts (14 total)
│   ├── train_detector.py          # Model training
│   ├── demo.py                    # Synthetic demo
│   ├── demo_trained_model.py      # Run trained model on images
│   ├── demo_video.py              # Video demo
│   ├── deploy_edge.py             # Edge deployment
│   ├── evaluate_model.py          # Model evaluation
│   ├── export_onnx.py             # ONNX export
│   ├── optimize_tensorrt.py       # TensorRT optimization
│   ├── compare_trackers.py        # Tracker comparison
│   ├── download_datasets.py       # Dataset management
│   ├── run_military_inference.py  # Military inference
│   ├── convert_airbus_to_yolo.py  # Airbus ship converter
│   ├── convert_flir_to_yolo.py    # FLIR thermal converter
│   └── merge_datasets.py          # Multi-dataset merger
│
├── src/                           # Core source code
│   ├── __init__.py                # Package init
│   ├── analytics/                 # Analytics (zones, trajectories)
│   ├── data/                      # Data loading, augmentation
│   ├── deployment/                # Deployment pipeline
│   ├── detection/                 # Detection (trainer, evaluator)
│   ├── optimization/              # Model optimization
│   ├── tracking/                  # Tracking utilities
│   └── utils/                     # Config, logger utilities
│
├── tests/                         # Unit tests (3 files)
│   ├── test_detection.py          # Detection tests
│   ├── test_tracking.py           # Tracking tests
│   └── test_integration.py        # Integration tests
│
├── weights/                       # Pre-trained weights
│   └── yolov8n.pt                 # YOLOv8 nano base (6.5MB)
│
├── .env                           # Environment variables
├── .gitignore                     # Git ignore rules
├── requirements.txt               # Python dependencies
├── setup.py                       # Package setup
└── TECHNICAL_BRIEF.md             # Technical summary
```

## Installation

```bash
# Create virtual environment
python -m venv .venv
.venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

```bash
# Run demo with synthetic data
python scripts/demo.py

# Run with trained model on images
python scripts/demo_trained_model.py

# Real-time webcam detection
python -m src.deployment.pipeline --source 0

# Headless mode (server)
python -m src.deployment.pipeline --source 0 --no-display
```

## Training

```bash
# Train with YOLOv8s (recommended for best results)
.venv\Scripts\yolo.exe detect train \
    data=data/military/assets/military_object_dataset/military_dataset.yaml \
    model=yolov8s.pt \
    epochs=150 \
    batch=16 \
    imgsz=640 \
    device=0

# Or use training script
python scripts/train_detector.py --epochs 100
```

## Model Performance

| Model | Epochs | mAP@0.5 | mAP@0.5:0.95 | Precision | Recall |
|-------|--------|---------|--------------|-----------|--------|
| YOLOv8n | Baseline | 0.445 | 0.280 | - | - |
| **YOLOv8s** | **75** | **0.613** | **0.418** | **0.616** | **0.556** |

**Training Growth**:
- mAP@0.5 improved by **+38%** over baseline
- mAP@0.5:0.95 improved by **+49%** over baseline

Best performing classes: Military Aircraft, Military Tank, Soldier.

## Trained Model

The best trained model is located at:
```
outputs/yolov8s_train/military_75ep/weights/best.pt
```

Use it for inference:
```bash
.venv\Scripts\yolo.exe detect predict model=outputs/yolov8s_train/military_75ep/weights/best.pt source=your_image.jpg
```

## Dataset

- **Main Dataset**: Military Object Dataset (26,315 images, 12 classes)
- **Format**: YOLOv8 compatible
- **Split**: Train (21,978) / Val (2,941) / Test (1,396)

## Documentation

- [Architecture Guide](docs/ARCHITECTURE.md)
- [Training Guide](docs/TRAINING.md)
- [Deployment Guide](docs/DEPLOYMENT.md)

## License

Internal Project
