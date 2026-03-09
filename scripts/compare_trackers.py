"""
Tracker Comparison Script
Compares Kalman, DeepSORT, and ByteTracker on military dataset images
"""
import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import cv2
import numpy as np
import time
from ultralytics import YOLO
from models.trackers.kalman_tracker import KalmanTracker
from models.trackers.bytetrack_tracker import ByteTracker
from models.trackers.deep_sort import DeepSort
from src.utils.logger import setup_logger

# Class names
CLASS_NAMES = [
    'camouflage_soldier', 'weapon', 'military_tank', 'military_truck',
    'military_vehicle', 'civilian', 'soldier', 'civilian_vehicle',
    'military_artillery', 'trench', 'military_aircraft', 'military_warship'
]

# Colors for trackers
TRACKER_COLORS = {
    'kalman': (0, 255, 0),      # Green
    'deepsort': (255, 0, 0),    # Blue
    'bytetrack': (0, 0, 255),   # Red
}

def run_tracker_test(tracker_name: str, tracker, model, images: list, output_dir: Path, logger):
    """Run a single tracker on all images"""
    
    tracker_output = output_dir / tracker_name
    tracker_output.mkdir(parents=True, exist_ok=True)
    
    total_time = 0
    total_tracks = 0
    total_detections = 0
    
    # Reset tracker for each run
    if hasattr(tracker, 'reset'):
        tracker.reset()
    
    for idx, img_path in enumerate(images):
        frame = cv2.imread(str(img_path))
        if frame is None:
            continue
        
        # Detect
        results = model.predict(frame, conf=0.25, verbose=False)
        boxes = results[0].boxes.xyxy.cpu().numpy()
        scores = results[0].boxes.conf.cpu().numpy()
        classes = results[0].boxes.cls.cpu().numpy().astype(int)
        
        total_detections += len(boxes)
        
        # Track with timing
        start = time.time()
        if len(boxes) > 0:
            try:
                if tracker_name == 'deepsort':
                    # DeepSORT expects (boxes_xywh, confidences, classes, frame)
                    # Convert xyxy to xywh
                    boxes_xywh = boxes.copy()
                    boxes_xywh[:, 2] = boxes[:, 2] - boxes[:, 0]  # width
                    boxes_xywh[:, 3] = boxes[:, 3] - boxes[:, 1]  # height
                    tracks = tracker.update(boxes_xywh, scores, classes, frame)
                else:
                    # Kalman and ByteTrack
                    tracks = tracker.update(boxes, classes, scores)
            except Exception as e:
                logger.warning(f"{tracker_name} error: {e}")
                tracks = np.empty((0, 7))
        else:
            tracks = np.empty((0, 7))
        elapsed = time.time() - start
        total_time += elapsed
        total_tracks += len(tracks)
        
        # Draw on frame
        color = TRACKER_COLORS[tracker_name]
        
        # Draw detections
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (128, 128, 128), 1)
        
        # Draw tracks
        for track in tracks:
            if len(track) >= 5:
                x1, y1, x2, y2 = map(int, track[:4])
                track_id = int(track[4])
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
                cv2.putText(frame, f"ID:{track_id}", (x1, y1-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # Add info
        info = f"{tracker_name.upper()} | Frame {idx+1} | Tracks: {len(tracks)}"
        cv2.rectangle(frame, (5, 5), (400, 40), (0, 0, 0), -1)
        cv2.putText(frame, info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        # Save
        cv2.imwrite(str(tracker_output / f"{idx:03d}_{img_path.name}"), frame)
    
    avg_time = (total_time / len(images)) * 1000  # ms
    
    logger.info(f"{tracker_name.upper():10} | Avg: {avg_time:.1f}ms | Total Tracks: {total_tracks}")
    
    return {
        'name': tracker_name,
        'avg_time_ms': avg_time,
        'total_tracks': total_tracks,
        'total_detections': total_detections
    }


def main():
    logger = setup_logger("TrackerComparison")
    logger.info("="*80)
    logger.info("TRACKER COMPARISON: Kalman vs DeepSORT vs ByteTracker")
    logger.info("="*80)
    
    # Load model
    weights_path = project_root / "outputs" / "yolov8s_train" / "military_75ep" / "weights" / "best.pt"
    logger.info(f"Loading model: {weights_path}")
    model = YOLO(str(weights_path))
    
    # Get images
    image_dir = project_root / "data" / "military" / "assets" / "military_object_dataset" / "val" / "images"
    images = sorted(list(image_dir.glob("*.jpg")))[:15]  # 15 images for comparison
    logger.info(f"Testing on {len(images)} images")
    
    # Output directory
    output_dir = project_root / "outputs" / "tracker_comparison"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize trackers
    trackers = {
        'kalman': KalmanTracker(max_age=30, min_hits=3, iou_threshold=0.3),
        'bytetrack': ByteTracker(max_age=30, min_hits=3),
        'deepsort': DeepSort(max_age=70, n_init=3, use_cuda=True),
    }
    
    logger.info("-"*80)
    logger.info("Running tracker tests...")
    logger.info("-"*80)
    
    results = []
    for name, tracker in trackers.items():
        result = run_tracker_test(name, tracker, model, images, output_dir, logger)
        results.append(result)
    
    # Summary
    logger.info("="*80)
    logger.info("COMPARISON SUMMARY")
    logger.info("="*80)
    logger.info(f"{'Tracker':<12} | {'Avg Time (ms)':<15} | {'Total Tracks':<12}")
    logger.info("-"*45)
    for r in results:
        logger.info(f"{r['name']:<12} | {r['avg_time_ms']:<15.2f} | {r['total_tracks']:<12}")
    
    logger.info("="*80)
    logger.info(f"Output saved to: {output_dir}")
    logger.info("="*80)
    
    # Create comparison image
    create_comparison_image(output_dir, images[5].name if len(images) > 5 else images[0].name, output_dir)

def create_comparison_image(output_dir: Path, sample_name: str, save_dir: Path):
    """Create side-by-side comparison image"""
    try:
        kalman_img = cv2.imread(str(output_dir / "kalman" / f"005_{sample_name}"))
        deepsort_img = cv2.imread(str(output_dir / "deepsort" / f"005_{sample_name}"))
        bytetrack_img = cv2.imread(str(output_dir / "bytetrack" / f"005_{sample_name}"))
        
        if all(img is not None for img in [kalman_img, deepsort_img, bytetrack_img]):
            # Stack horizontally
            comparison = np.hstack([kalman_img, deepsort_img, bytetrack_img])
            cv2.imwrite(str(save_dir / "comparison_side_by_side.jpg"), comparison)
    except Exception as e:
        print(f"Could not create comparison image: {e}")


if __name__ == "__main__":
    main()
