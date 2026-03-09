"""
Run Detection on Military Dataset
Batch processes images from military dataset and saves annotated output
"""
import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import cv2
import numpy as np
import torch
from models.detectors.yolo_detector import YOLODetector
from models.trackers.kalman_tracker import KalmanTracker
from src.utils.logger import setup_logger

def main():
    logger = setup_logger("MilitaryInference")
    logger.info("="*80)
    logger.info("Military Object Detection - Dataset Inference")
    logger.info("="*80)
    
    # Check GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    
    # Initialize detector
    logger.info("Loading YOLOv8 detector...")
    detector = YOLODetector(
        variant="yolov8n",
        num_classes=80,  # COCO pretrained
        pretrained=True,
        device=device
    )
    
    # Initialize tracker
    tracker = KalmanTracker(max_age=30, min_hits=3)
    
    # Dataset paths
    image_dir = project_root / "data" / "military" / "assets" / "military_object_dataset" / "val" / "images"
    output_dir = project_root / "outputs" / "military_inference"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get images
    images = list(image_dir.glob("*.jpg"))[:20]  # Process first 20 images
    logger.info(f"Processing {len(images)} images from {image_dir}")
    
    total_detections = 0
    
    for idx, img_path in enumerate(images):
        # Load image
        frame = cv2.imread(str(img_path))
        if frame is None:
            continue
            
        # Detect
        results = detector.predict(frame)
        
        # Get first result (single image)
        result = results[0] if results else {'boxes': [], 'scores': [], 'class_ids': []}
        
        # Get boxes
        boxes = result.get('boxes', np.array([]))
        scores = result.get('scores', np.array([]))
        classes = result.get('classes', np.array([]))
        
        # Track
        if len(boxes) > 0:
            tracks = tracker.update(boxes, classes, scores)
        else:
            tracks = np.empty((0, 7))
            
        total_detections += len(boxes)
        
        # Draw detections
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = map(int, box)
            score = scores[i]
            cls = int(classes[i])
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"cls:{cls} {score:.2f}"
            cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Draw tracks
        for track in tracks:
            x1, y1, x2, y2 = map(int, track[:4])
            track_id = int(track[4])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 3)
            cv2.putText(frame, f"ID:{track_id}", (x1, y2+20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        
        # Save output
        output_path = output_dir / f"detected_{img_path.name}"
        cv2.imwrite(str(output_path), frame)
        
        if (idx + 1) % 5 == 0:
            logger.info(f"Processed {idx+1}/{len(images)} images")
    
    logger.info("="*80)
    logger.info(f"Completed! Total detections: {total_detections}")
    logger.info(f"Output saved to: {output_dir}")
    logger.info("="*80)

if __name__ == "__main__":
    main()
