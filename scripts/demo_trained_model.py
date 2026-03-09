"""
Demo: Run Trained Military Model on Images
Shows detection + tracking pipeline using trained weights
"""
import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import cv2
import numpy as np
from ultralytics import YOLO
from models.trackers.kalman_tracker import KalmanTracker
from src.utils.logger import setup_logger

# Class names for military dataset
CLASS_NAMES = [
    'camouflage_soldier', 'weapon', 'military_tank', 'military_truck',
    'military_vehicle', 'civilian', 'soldier', 'civilian_vehicle',
    'military_artillery', 'trench', 'military_aircraft', 'military_warship'
]

# Colors for each class (BGR)
COLORS = [
    (0, 255, 0),    # camouflage_soldier - green
    (0, 0, 255),    # weapon - red
    (255, 0, 0),    # military_tank - blue
    (255, 165, 0),  # military_truck - orange
    (128, 0, 128),  # military_vehicle - purple
    (255, 255, 0),  # civilian - cyan
    (0, 255, 255),  # soldier - yellow
    (255, 192, 203),# civilian_vehicle - pink
    (139, 69, 19),  # military_artillery - brown
    (128, 128, 128),# trench - gray
    (0, 191, 255),  # military_aircraft - deep sky blue
    (0, 128, 128),  # military_warship - teal
]

def main():
    logger = setup_logger("TrainedModelDemo")
    logger.info("="*80)
    logger.info("MODT - Trained Military Model Demonstration")
    logger.info("="*80)
    
    # Load trained model
    weights_path = project_root / "outputs" / "yolov8s_train" / "military_75ep" / "weights" / "best.pt"
    logger.info(f"Loading trained model: {weights_path}")
    
    model = YOLO(str(weights_path))
    logger.info("Model loaded successfully!")
    
    # Initialize tracker
    tracker = KalmanTracker(max_age=30, min_hits=3, iou_threshold=0.3)
    
    # Input images
    image_dir = project_root / "data" / "military" / "assets" / "military_object_dataset" / "val" / "images"
    output_dir = project_root / "outputs" / "trained_model_demo"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    images = sorted(list(image_dir.glob("*.jpg")))[:10]  # Process 10 images
    logger.info(f"Processing {len(images)} images from validation set")
    
    total_detections = 0
    total_tracks = 0
    
    for idx, img_path in enumerate(images):
        # Load image
        frame = cv2.imread(str(img_path))
        if frame is None:
            continue
        
        h, w = frame.shape[:2]
        
        # Run inference
        results = model.predict(frame, conf=0.25, verbose=False)
        
        # Extract detections
        boxes = results[0].boxes.xyxy.cpu().numpy()
        scores = results[0].boxes.conf.cpu().numpy()
        classes = results[0].boxes.cls.cpu().numpy().astype(int)
        
        total_detections += len(boxes)
        
        # Update tracker
        if len(boxes) > 0:
            tracks = tracker.update(boxes, classes, scores)
        else:
            tracks = np.empty((0, 7))
        
        total_tracks += len(tracks)
        
        # Draw detections with class-specific colors
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = map(int, box)
            cls_id = classes[i]
            score = scores[i]
            color = COLORS[cls_id % len(COLORS)]
            
            # Draw box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Draw label background
            label = f"{CLASS_NAMES[cls_id]}: {score:.2f}"
            (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            cv2.rectangle(frame, (x1, y1-label_h-10), (x1+label_w, y1), color, -1)
            cv2.putText(frame, label, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        # Draw tracks
        for track in tracks:
            tx1, ty1, tx2, ty2 = map(int, track[:4])
            track_id = int(track[4])
            
            # Draw track ID
            cv2.putText(frame, f"ID:{track_id}", (tx1, ty2+20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        # Add info overlay
        info_text = f"Frame: {idx+1}/{len(images)} | Detections: {len(boxes)} | Tracks: {len(tracks)}"
        cv2.rectangle(frame, (10, 10), (500, 50), (0, 0, 0), -1)
        cv2.putText(frame, info_text, (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Save output
        output_path = output_dir / f"demo_{idx:03d}_{img_path.name}"
        cv2.imwrite(str(output_path), frame)
        
        logger.info(f"[{idx+1}/{len(images)}] {img_path.name}: {len(boxes)} detections, {len(tracks)} tracks")
    
    logger.info("="*80)
    logger.info(f"Demo Complete!")
    logger.info(f"Total Detections: {total_detections}")
    logger.info(f"Total Tracks Created: {total_tracks}")
    logger.info(f"Output saved to: {output_dir}")
    logger.info("="*80)

if __name__ == "__main__":
    main()
