# System Architecture

## Overview

The Military Object Detection & Tracking System follows a modular, layered architecture designed for scalability, maintainability, and edge deployment.

```
┌─────────────────────────────────────────────────────────────┐
│                    Application Layer                         │
│  ┌─────────────┐  ┌──────────────┐  ┌──────────────┐       │
│  │   Training  │  │  Deployment  │  │  Analytics   │       │
│  │   Scripts   │  │   Pipeline   │  │   Modules    │       │
│  └─────────────┘  └──────────────┘  └──────────────┘       │
└─────────────────────────────────────────────────────────────┘
┌─────────────────────────────────────────────────────────────┐
│                     Core Modules                             │
│  ┌─────────────┐  ┌──────────────┐  ┌──────────────┐       │
│  │  Detection  │  │   Tracking   │  │Optimization  │       │
│  │   Module    │  │    Module    │  │   Module     │       │
│  └─────────────┘  └──────────────┘  └──────────────┘       │
└─────────────────────────────────────────────────────────────┘
┌─────────────────────────────────────────────────────────────┐
│                  Data & Utilities                           │
│  ┌─────────────┐  ┌──────────────┐  ┌──────────────┐       │
│  │    Data     │  │    Config    │  │   Logging    │       │
│  │  Processing │  │   Manager    │  │   System     │       │
│  └─────────────┘  └──────────────┘  └──────────────┘       │
└─────────────────────────────────────────────────────────────┘
┌─────────────────────────────────────────────────────────────┐
│                    Model Layer                               │
│  ┌─────────────────┐         ┌─────────────────┐           │
│  │  YOLOv8 Detector │        │ Kalman Tracker  │           │
│  │  (PyTorch/ONNX)  │        │ (DeepSORT)      │           │
│  └─────────────────┘         └─────────────────┘           │
└─────────────────────────────────────────────────────────────┘
```

## Component Descriptions

### 1. Detection Module

**Purpose**: Real-time object detection

**Components**:
- `YOLODetector`: Wrapper for YOLOv5/YOLOv8 models
- `DetectionTrainer`: Training pipeline with metrics
- `DetectionEvaluator`: mAP, precision, recall computation

**Key Features**:
- Multiple YOLO variants (nano → xlarge)
- Custom class heads
- Pretrained weight loading
- Multi-scale detection

### 2. Tracking Module

**Purpose**: Multi-object tracking with ID persistence

**Components**:
- `KalmanTracker`: Standard Kalman filter
- `IoUTracker`: High-speed intersection-over-union tracker
- `DeepSORTTracker`: Appearance-based tracking
- `ByteTracker`: Two-stage high/low confidence association
- `MOTMetrics`: MOTA, MOTP, IDF1 computation
- `TrackManager`: Track lifecycle management

**Key Features**:
- 8-state Kalman filter (position + velocity)
- Optimal data association
- Occlusion handling
- Class-aware tracking

### 3. Data Processing

**Purpose**: Dataset loading and augmentation

**Components**:
- `DetectionDataset`: YOLO/COCO format support
- `TrackingDataset`: Sequential frame loading
- `AugmentationPipeline`: Advanced augmentations

**Key Features**:
- Letterbox resizing
- Mosaic & MixUp augmentation
- Environmental effects (fog, rain, shadows)
- Thermal imaging support

### 4. Optimization Module

**Purpose**: Model export and edge optimization

**Components**:
- `ModelExporter`: ONNX/TensorRT export
- `Quantization`: INT8/FP16 quantization
- `Benchmarker`: Performance profiling

**Key Features**:
- ONNX simplification
- TensorRT optimization
- Precision calibration
- FPS/latency benchmarking

### 5. Deployment Pipeline

**Purpose**: Real-time inference and visualization

**Components**:
- `DeploymentPipeline`: Main inference loop
- `VideoStream`: Multi-threaded video capture
- `Visualizer`: Bounding box rendering

**Key Features**:
- Multi-threaded processing
- RTSP stream support
- Real-time FPS monitoring
- Video annotation

### 6. Analytics Module

**Purpose**: Spatial and temporal analytics

**Components**:
- `ZoneManager`: Polygon zones, intrusion detection
- `TrajectoryAnalyzer`: Path prediction
- `EventLogger`: Event tracking

**Key Features**:
- Entry/exit counting
- Zone-based alerts
- Speed estimation
- Dwell time analysis

## Data Flow

### Training Pipeline

```
Raw Data → Annotation → Augmentation → Detector Training → 
Evaluation → Checkpoint Saving → Export (ONNX/TensorRT)
```

### Deployment Pipeline

```
Video Stream → Preprocessing → Detection → Tracking → 
Analytics → Visualization → Output (Display/Save)
```

### Edge Deployment

```
PyTorch Model → ONNX Export → TensorRT Optimization → 
Jetson Deployment → Real-time Inference
```

## Configuration System

Hierarchical YAML configuration with three levels:

1. **default.yaml**: Base configuration
2. **training.yaml**: Training-specific overrides
3. **edge_jetson.yaml**: Jetson deployment settings

Configuration accessed via dot notation:
```python
config.get('model.detector.type')  # Returns: 'yolov8'
```

## Design Patterns

### 1. Strategy Pattern

Multiple detector/tracker implementations with unified interface:
```python
detector = YOLODetector(...)  # or FasterRCNN(...)
tracker = KalmanTracker(...)   # or DeepSORT(...)
```

### 2. Factory Pattern

Model creation based on configuration:
```python
model = create_detector(config.get('model.detector'))
```

### 3. Observer Pattern

Event-driven analytics:
```python
zone_manager.on_intrusion(callback)
```

## Performance Considerations

### Training
- Mixed precision (AMP) for 2x speedup
- Multi-GPU support via DataParallel
- Gradient accumulation for large batch sizes
- Early stopping to prevent overfitting

### Inference
- Model warmup for consistent FPS
- Multi-threaded video I/O
- Batch processing where applicable
- TensorRT for 3-5x speedup on Jetson

### Memory
- Image caching (optional)
- Gradient checkpointing in training
- INT8 quantization for edge devices

## Security & Privacy

- No data transmission to cloud (offline operation)
- Configurable data retention policies
- Encrypted model weights (optional)
- Access control for zone definitions

## Scalability

### Horizontal Scaling
- Multiple edge devices with central aggregation
- Load balancing across devices
- Distributed training

### Vertical Scaling
- Larger model variants for accuracy
- Higher resolutions for small objects
- Ensemble methods

## Future Enhancements

1. **Additional Trackers**: ByteTrack, OC-SORT
2. **3D Detection**: Depth estimation integration
3. **Action Recognition**: Behavior analysis
4. **Cloud Integration**: Optional cloud backup
5. **Mobile Deployment**: iOS/Android support

---

For implementation details, see individual module documentation.
