# Deployment Guide

## Overview

This guide covers deploying the MODT system to production environments, specifically focusing on NVIDIA Jetson edge devices and PC workstations.

## 1. Deployment Pipeline

The `src/deployment/pipeline.py` script manages the end-to-end inference process:
1. **Capture**: Reads frames from RTSP, USB, or File.
2. **Preprocess**: Resizes and normalizes (Letterbox).
3. **Inference**: Runs detection (YOLO) and tracking.
4. **Analytics**: Checks zones and updates trajectories.
5. **Output**: Visualizes and streams/saves results.

## 2. Configuration for Edge

Use `config/edge_jetson.yaml` for optimized settings.

### Tracker Selection
Choose the tracker based on your resource constraints:
- **IoU Tracker**: Fastest (>100 FPS), good for high frame rates.
- **ByteTrack**: Balanced accuracy/speed (Recommended).
- **Kalman**: Standard baseline.
- **DeepSORT**: Highest accuracy, heavy resource usage (requires ReID features).

Set in config:
```yaml
model:
  tracker:
    type: "bytetrack"
```

## 3. Edge Optimization (TensorRT)

For Jetson devices, optimizing to TensorRT is critical.

### Step 1: Export to ONNX
```bash
python scripts/export_onnx.py --weights weights/best.pt --output weights/model.onnx --simplify
```

### Step 2: Convert to TensorRT Engine
Use the newly created optimization script:
```bash
python scripts/optimize_tensorrt.py --onnx weights/model.onnx --output weights/model.engine --precision fp16
```
*Note: Run this ON the target Jetson device.*

### Step 3: Run Inference
Update config to point to engine:
```yaml
model:
  detector:
    path: "weights/model.engine"
```

## 4. Hardware Setup (Jetson)

### Prerequisites
- JetPack 5.0+
- PyTorch with CUDA support

### Installation
```bash
# Clone and install dependencies
git clone <repo>
cd MODT
pip install -r requirements.txt
```

### Power Mode
Set Jetson to max performance mode:
```bash
sudo nvpmodel -m 0
sudo jetson_clocks
```

## 5. Running the Pipeline

**Basic Camera Stream:**
```bash
python src/deployment/pipeline.py --source 0
```

**RTSP Stream with Edge Config:**
```bash
python src/deployment/pipeline.py --config config/edge_jetson.yaml --source rtsp://192.168.1.100:554/stream
```

**Headless Mode (No GUI):**
Set `deployment.output.display: false` in YAML.

## 6. Performance Tuning

| Parameter | Action | Effect |
|-----------|--------|--------|
| `input_size` | Reduce (e.g. 640->320) | Higher FPS, Lower Accuracy |
| `precision` | FP16 / INT8 | 2x-4x Speedup |
| `tracker` | Use IoU/ByteTrack | Reduces CPU load |
| `skip_frames` | Increase | Reduces load, lower temporal resolution |
