# Training Guide

## Overview

This guide details how to train the object detection model using the Military Object Detection & Tracking (MODT) system. The system supports YOLOv8 (Nano to X-Large) and provides advanced features like mixed precision training, multi-step scheduling, and quantization-aware training (QAT).

## 1. Dataset Preparation

### Format
The system expects data in standard YOLO format:
```
data/
├── train/
│   ├── images/
│   └── labels/
├── val/
│   ├── images/
│   └── labels/
└── dataset.yaml  # Optional ultralytics style config
```

### Labels
Labels should be calibrated to 0-1 range:
`class_id x_center y_center width height`

## 2. Configuration

Configure training parameters in `config/training.yaml`.

### Key Settings

**Model Selection:**
```yaml
model:
  detector:
    type: "yolov8"
    variant: "medium"  # nano, small, medium, large, xlarge
    pretrained: true
```

**Optimization:**
```yaml
training:
  batch_size: 32
  optimizer:
    type: "AdamW"  # SGD, Adam, AdamW
    lr: 0.001
  scheduler:
    type: "multistep"  # cosine, step, multistep, exponential
    milestones: [30, 60, 90]
```

## 3. Training Process

### Standard Training
Run the training script with your configuration:
```bash
python scripts/train_detector.py --config config/training.yaml
```

### Quantization Aware Training (QAT)
For edge deployment requiring maximum efficiency:
1. Enable QAT in config:
   ```yaml
   edge:
     quantization:
       enable: true
       method: "qat"
   ```
2. Run training. The trainer will perform standard training followed by QAT fine-tuning.

## 4. Monitoring

### TensorBoard
Real-time metrics visualization:
```bash
tensorboard --logdir outputs/logs/tensorboard
```

### Weights & Biases (WandB)
1. Set `monitoring.wandb.enable: true`
2. Configure project name and tags
3. Login via `wandb login`

## 5. Artifacts

Training produces the following outputs in `weights/`:
- `best.pt`: Model with highest mAP
- `last.pt`: Checkpoint from final epoch
- `checkpoint_epoch_N.pt`: Periodic checkpoints

## 6. Troubleshooting

- **OOM Error**: Reduce `batch_size` or enable `accumulate_grad`.
- **Low mAP**: Verify dataset labels, increase `epochs`, or try a larger model variant.
- **NaN Loss**: Lower `learning_rate` or switch optimizer to SGD.
