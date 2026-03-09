"""
End-to-end Deployment Pipeline
Integrates detection, tracking, and analytics for real-time object detection and tracking
"""

import cv2
import torch
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any, Tuple, List
from collections import deque
import time
from threading import Thread
from queue import Queue

from src.utils.config import load_config
from src.utils.logger import setup_logger


from src.deployment.video_stream import VideoStream
from src.deployment.preprocessor import Preprocessor
from src.deployment.event_handler import EventHandler
from src.analytics.zone_manager import ZoneManager


class DeploymentPipeline:
    """
    Main deployment pipeline for detection and tracking
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize deployment pipeline
        
        Args:
            config_path: Path to configuration file
        """
        # Load configuration
        self.config = load_config(config_path) if config_path else load_config()
        
        # Setup logger
        self.logger = setup_logger(
            "Deployment",
            log_file=Path(self.config.get("monitoring.checkpoints.save_dir", "./outputs/logs")) / "deployment.log"
        )
        
        # Device setup
        self.device = self.config.get("system.device", "cuda" if torch.cuda.is_available() else "cpu")
        self.logger.info(f"Using device: {self.device}")
        
        # Initialize models
        self._init_detector()
        self._init_tracker()
        
        # Performance metrics
        self.fps_counter = deque(maxlen=30)
        self.frame_count = 0
        
        # Output setup
        self.output_dir = Path(self.config.get("deployment.output.output_dir", "./outputs/results"))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.save_video = self.config.get("deployment.output.save_video", True)
        self.video_writer = None
    
    def _init_detector(self):
        """Initialize object detector"""
        try:
            from models.detectors.yolo_detector import YOLODetector
            
            variant = self.config.get("model.detector.variant", "nano")
            model_type = self.config.get("model.detector.type", "yolov8")
            model_name = f"{model_type}{variant[0]}"  # e.g., yolov8n
            
            self.detector = YOLODetector(
                variant=model_name,
                num_classes=self.config.get("model.detector.num_classes", 5),
                pretrained=self.config.get("model.detector.pretrained", True),
                img_size=tuple(self.config.get("model.detector.input_size", [640, 640])),
                device=self.device
            )
            
            # Set class names
            classes = self.config.get("dataset.classes", {})
            class_names = [classes.get(i, f"class_{i}") for i in sorted(classes.keys())]
            self.detector.set_class_names(class_names)
            
            self.logger.info(f"Detector initialized: {model_name}")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize detector: {e}")
            raise
    
    def _init_tracker(self):
        """Initialize multi-object tracker"""
        try:
            from models.trackers.kalman_tracker import KalmanTracker
            
            tracker_type = self.config.get("model.tracker.type", "kalman")
            
            if tracker_type == "kalman":
                self.tracker = KalmanTracker(
                    max_age=self.config.get("model.tracker.max_age", 30),
                    min_hits=self.config.get("model.tracker.min_hits", 3),
                    iou_threshold=self.config.get("model.tracker.iou_threshold", 0.3)
                )
            elif tracker_type == "iou":
                from models.trackers.iou_tracker import IoUTracker
                self.tracker = IoUTracker(
                    max_age=self.config.get("model.tracker.max_age", 5),
                    min_hits=self.config.get("model.tracker.min_hits", 2),
                    iou_threshold=self.config.get("model.tracker.iou_threshold", 0.3)
                )
            elif tracker_type == "deepsort":
                from models.trackers.deepsort_tracker import DeepSORTTracker
                self.tracker = DeepSORTTracker(
                    max_age=self.config.get("model.tracker.max_age", 30),
                    min_hits=self.config.get("model.tracker.min_hits", 3),
                    iou_threshold=self.config.get("model.tracker.iou_threshold", 0.3)
                )
            elif tracker_type == "bytetrack":
                from models.trackers.bytetrack_tracker import ByteTracker
                self.tracker = ByteTracker(
                    max_age=self.config.get("model.tracker.max_age", 30),
                    min_hits=self.config.get("model.tracker.min_hits", 3),
                    iou_threshold=self.config.get("model.tracker.iou_threshold", 0.3)
                )
            else:
                raise ValueError(f"Unknown tracker type: {tracker_type}")
            
            self.logger.info(f"Tracker initialized: {tracker_type}")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize tracker: {e}")
            raise
    
    def preprocess(self, frame: np.ndarray) -> np.ndarray:
        """
        Preprocess frame for inference
        
        Args:
            frame: Input frame (H, W, C)
        
        Returns:
            Preprocessed frame
        """
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return frame_rgb
    
    def visualize(
        self,
        frame: np.ndarray,
        detections: Dict[str, np.ndarray],
        tracks: np.ndarray
    ) -> np.ndarray:
        """
        Visualize detections and tracks on frame
        
        Args:
            frame: Input frame
            detections: Detection dictionary
            tracks: Track array (N, 7) [x1, y1, x2, y2, track_id, class_id, score]
        
        Returns:
            Annotated frame
        """
        frame_vis = frame.copy()
        
        # Color map for different classes
        colors = [
            (255, 0, 0),    # Red
            (0, 255, 0),    # Green
            (0, 0, 255),    # Blue
            (255, 255, 0),  # Cyan
            (255, 0, 255),  # Magenta
        ]
        
        # Draw tracks
        for track in tracks:
            x1, y1, x2, y2, track_id, class_id, score = track
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            class_id = int(class_id)
            
            # Get color for class
            color = colors[class_id % len(colors)]
            
            # Draw bounding box
            cv2.rectangle(frame_vis, (x1, y1), (x2, y2), color, 2)
            
            # Draw label
            label = f"ID:{int(track_id)} {self.detector.class_names[class_id]} {score:.2f}"
            (label_w, label_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            
            # Draw label background
            cv2.rectangle(frame_vis, (x1, y1 - label_h - 10), (x1 + label_w, y1), color, -1)
            cv2.putText(frame_vis, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Draw FPS
        if len(self.fps_counter) > 0:
            fps = np.mean(self.fps_counter)
            cv2.putText(frame_vis, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Draw object count
        cv2.putText(frame_vis, f"Objects: {len(tracks)}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        return frame_vis
    
    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """
        Process single frame through detection and tracking
        
        Args:
            frame: Input frame
        
        Returns:
            Annotated frame and metadata
        """
        start_time = time.time()
        
        # Preprocess
        frame_rgb = self.preprocess(frame)
        
        # Run detection
        detections_list = self.detector.predict(
            frame_rgb,
            conf_threshold=self.config.get("inference.confidence_threshold", 0.25),
            iou_threshold=self.config.get("inference.iou_threshold", 0.45)
        )
        
        detections = detections_list[0] if detections_list else {
            'boxes': np.zeros((0, 4)),
            'scores': np.zeros(0),
            'classes': np.zeros(0, dtype=int)
        }
        
        # Run tracking
        tracks = self.tracker.update(
            detections['boxes'],
            detections['classes'],
            detections['scores']
        )
        
        # Visualize
        frame_vis = self.visualize(frame, detections, tracks)
        
        # Calculate FPS
        elapsed = time.time() - start_time
        self.fps_counter.append(1.0 / elapsed if elapsed > 0 else 0)
        
        # Metadata
        metadata = {
            'frame_count': self.frame_count,
            'num_detections': len(detections['boxes']),
            'num_tracks': len(tracks),
            'fps': np.mean(self.fps_counter) if len(self.fps_counter) > 0 else 0,
            'latency_ms': elapsed * 1000
        }
        
        self.frame_count += 1
        
        return frame_vis, metadata
    
    def run(self, source: str = "0", no_display: bool = False):
        """
        Run deployment pipeline on video source
        
        Args:
            source: Video source (file, RTSP URL, or webcam index)
        """
        self.logger.info(f"Starting deployment pipeline on source: {source}")
        
        # Initialize video stream
        stream = VideoStream(source)
        self.logger.info(f"Video: {stream.width}x{stream.height} @ {stream.fps} FPS")
        
        # Initialize video writer if saving
        if self.save_video:
            output_video = self.output_dir / "output.mp4"
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.video_writer = cv2.VideoWriter(
                str(output_video),
                fourcc,
                stream.fps,
                (stream.width, stream.height)
            )
            self.logger.info(f"Saving output to: {output_video}")
        
        try:
            while True:
                # Read frame
                ret, frame = stream.read()
                if not ret:
                    break
                
                # Process frame
                frame_vis, metadata = self.process_frame(frame)
                
                # Log periodically
                if self.frame_count % 30 == 0:
                    self.logger.info(
                        f"Frame {metadata['frame_count']}: "
                        f"{metadata['num_tracks']} tracks, "
                        f"{metadata['fps']:.1f} FPS, "
                        f"{metadata['latency_ms']:.1f} ms"
                    )
                
                # Display (skip if --no-display or no display available)
                if not no_display and self.config.get("deployment.output.display", True):
                    try:
                        cv2.imshow("Military Object Detection & Tracking", frame_vis)
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            break
                    except cv2.error:
                        no_display = True  # Disable display on error
                        self.logger.warning("Display not available, running headless")
                
                # Save to video
                if self.video_writer:
                    self.video_writer.write(frame_vis)
                
        except KeyboardInterrupt:
            self.logger.info("Interrupted by user")
        
        finally:
            # Cleanup
            stream.stop()
            if self.video_writer:
                self.video_writer.release()
            try:
                cv2.destroyAllWindows()
            except cv2.error:
                pass  # Ignore if no display was created
            
            self.logger.info(f"Processed {self.frame_count} frames")
            self.logger.info("Pipeline stopped")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Military Object Detection & Tracking Deployment")
    parser.add_argument("--config", type=str, default=None, help="Path to config file")
    parser.add_argument("--source", type=str, default="0", help="Video source")
    parser.add_argument("--no-display", action="store_true", help="Run without display (headless mode)")
    
    args = parser.parse_args()
    
    # Run pipeline
    pipeline = DeploymentPipeline(config_path=args.config)
    pipeline.run(source=args.source, no_display=args.no_display)
