"""
Quick Demo Script
Demonstrates the object detection and tracking system with a synthetic demo
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


def create_demo_frame(width=1280, height=720, frame_number=0):
    """Create a synthetic frame with moving objects"""
    frame = np.random.randint(50, 100, (height, width, 3), dtype=np.uint8)
    
    # Add some moving rectangles (simulated objects)
    num_objects = 3
    objects = []
    
    for i in range(num_objects):
        # Simulate moving objects
        x = int(100 + i * 200 + (frame_number * 5) % 400)
        y = int(200 + i * 100 + np.sin(frame_number * 0.1 + i) * 50)
        w, h = 100 + i * 20, 80 + i * 15
        
        # Draw rectangle on frame
        color = (0, 255, 0) if i % 2 == 0 else (0, 0, 255)
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, -1)
        
        # Store bbox in x1, y1, x2, y2 format
        objects.append([x, y, x + w, y + h, i % 2])  # Last value is class_id
    
    return frame, objects


def main():
    logger = setup_logger("DemoScript")
    logger.info("="*80)
    logger.info("Military Object Detection & Tracking - DEMO")
    logger.info("="*80)
    
    # Check if CUDA is available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    
    # Initialize tracker (we'll use synthetic detections instead of actual detector)
    logger.info("Initializing tracker...")
    tracker = KalmanTracker(max_age=30, min_hits=3, iou_threshold=0.3)
    
    # Demo video parameters
    width, height = 1280, 720
    fps = 30
    duration_seconds = 10
    total_frames = fps * duration_seconds
    
    # Create video writer
    output_path = project_root / "outputs" / "demo_output.mp4"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    
    logger.info(f"Generating demo video: {width}x{height} @ {fps} FPS")
    logger.info(f"Duration: {duration_seconds} seconds ({total_frames} frames)")
    logger.info("-"*80)
    
    # Process frames
    for frame_idx in range(total_frames):
        # Create synthetic frame with moving objects
        frame, detections = create_demo_frame(width, height, frame_idx)
        
        # Convert detections to numpy arrays
        detection_boxes = np.array([obj[:4] for obj in detections])
        detection_classes = np.array([obj[4] for obj in detections])
        detection_scores = np.ones(len(detections)) * 0.95
        
        # Update tracker
        tracks = tracker.update(detection_boxes, detection_classes, detection_scores)
        
        # Visualize tracks
        colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255)]
        
        for track in tracks:
            x1, y1, x2, y2, track_id, class_id, score = track
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            track_id = int(track_id)
            class_id = int(class_id)
            
            # Draw bounding box
            color = colors[track_id % len(colors)]
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
            
            # Draw label
            class_name = "Vehicle" if class_id == 0 else "Soldier"
            label = f"ID:{track_id} {class_name} {score:.2f}"
            (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            
            cv2.rectangle(frame, (x1, y1 - label_h - 10), (x1 + label_w, y1), color, -1)
            cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Add info text
        info_text = f"Frame: {frame_idx+1}/{total_frames} | Tracks: {len(tracks)}"
        cv2.putText(frame, info_text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(frame, "DEMO MODE - Synthetic Objects", (10, height - 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        
        # Write frame
        video_writer.write(frame)
        
        # Display progress
        if (frame_idx + 1) % 30 == 0:
            progress = (frame_idx + 1) / total_frames * 100
            logger.info(f"Progress: {progress:.1f}% | Tracks: {len(tracks)}")
    
    # Cleanup
    video_writer.release()
    
    logger.info("-"*80)
    logger.info(f"Demo video saved to: {output_path}")
    logger.info("="*80)
    logger.info("Demo completed successfully!")
    logger.info("="*80)
    logger.info("")
    logger.info("To view the demo video:")
    logger.info(f"  Open: {output_path}")
    logger.info("")
    logger.info("To run on real video:")
    logger.info("  python scripts/deploy_edge.py --source your_video.mp4")
    logger.info("")


if __name__ == "__main__":
    main()
