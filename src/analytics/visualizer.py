"""
Visualization utilities for detection and tracking
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional, Dict
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


class Visualizer:
    """
    Visualization utilities for object detection and tracking
    """
    
    def __init__(self, class_names: Optional[List[str]] = None):
        """
        Initialize visualizer
        
        Args:
            class_names: List of class names
        """
        self.class_names = class_names or []
        
        # Color palette
        self.colors = [
            (255, 0, 0), (0, 255, 0), (0, 0, 255),
            (255, 255, 0), (255, 0, 255), (0, 255, 255),
            (128, 0, 0), (0, 128, 0), (0, 0, 128)
        ]
    
    def draw_detections(
        self,
        image: np.ndarray,
        boxes: np.ndarray,
        scores: np.ndarray,
        classes: np.ndarray,
        thickness: int = 2
    ) -> np.ndarray:
        """
        Draw detection bounding boxes on image
        
        Args:
            image: Input image
            boxes: Boxes (N, 4) [x1, y1, x2, y2]
            scores: Confidence scores (N,)
            classes: Class IDs (N,)
            thickness: Box thickness
        
        Returns:
            Annotated image
        """
        result = image.copy()
        
        for box, score, cls in zip(boxes, scores, classes):
            x1, y1, x2, y2 = map(int, box)
            cls = int(cls)
            
            # Get color
            color = self.colors[cls % len(self.colors)]
            
            # Draw box
            cv2.rectangle(result, (x1, y1), (x2, y2), color, thickness)
            
            # Draw label
            label = f"{self.class_names[cls] if cls < len(self.class_names) else cls}: {score:.2f}"
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(result, (x1, y1 - h - 4), (x1 + w, y1), color, -1)
            cv2.putText(result, label, (x1, y1 - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return result
    
    def draw_tracks(
        self,
        image: np.ndarray,
        tracks: np.ndarray,
        trajectories: Optional[Dict[int, List]] = None,
        thickness: int = 2
    ) -> np.ndarray:
        """
        Draw tracking results with trajectories
        
        Args:
            image: Input image
            tracks: Tracks (N, 7) [x1, y1, x2, y2, track_id, class_id, score]
            trajectories: Dictionary mapping track_id to list of past positions
            thickness: Line thickness
        
        Returns:
            Annotated image
        """
        result = image.copy()
        
        for track in tracks:
            x1, y1, x2, y2, track_id, class_id, score = track
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
            track_id = int(track_id)
            class_id = int(class_id)
            
            # Get color based on track ID
            color = self.colors[track_id % len(self.colors)]
            
            # Draw box
            cv2.rectangle(result, (x1, y1), (x2, y2), color, thickness)
            
            # Draw label
            label = f"ID:{track_id} {self.class_names[class_id] if class_id < len(self.class_names) else class_id}"
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(result, (x1, y1 - h -4), (x1 + w, y1), color, -1)
            cv2.putText(result, label, (x1, y1 - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Draw trajectory if provided
            if trajectories and track_id in trajectories:
                points = trajectories[track_id]
                for i in range(1, len(points)):
                    if points[i-1] is None or points[i] is None:
                        continue
                    cv2.line(result, tuple(map(int, points[i-1])), tuple(map(int, points[i])), color, 2)
        
        return result
    
    def create_heatmap(
        self,
        detections_history: List[np.ndarray],
        image_shape: Tuple[int, int],
        grid_size: int = 50
    ) -> np.ndarray:
        """
        Create detection heatmap
        
        Args:
            detections_history: List of detection arrays from multiple frames
            image_shape: (height, width) of images
            grid_size: Grid cell size
        
        Returns:
            Heatmap image
        """
        h, w = image_shape
        
        # Create grid
        grid_h = h // grid_size + 1
        grid_w = w // grid_size + 1
        heatmap = np.zeros((grid_h, grid_w))
        
        # Accumulate detections
        for dets in detections_history:
            if len(dets) == 0:
                continue
            centers = (dets[:, :2] + dets[:, 2:4]) / 2  # Compute centers
            for center in centers:
                x, y = center
                grid_x = int(x) // grid_size
                grid_y = int(y) // grid_size
                if 0 <= grid_x < grid_w and 0 <= grid_y < grid_h:
                    heatmap[grid_y, grid_x] += 1
        
        # Normalize and colorize
        heatmap = (heatmap / (heatmap.max() + 1e-6) * 255).astype(np.uint8)
        heatmap_colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        heatmap_resized = cv2.resize(heatmap_colored, (w, h), interpolation=cv2.INTER_LINEAR)
        
        return heatmap_resized
    
    def plot_metrics(
        self,
        metrics: Dict[str, List[float]],
        output_path: Optional[str] = None
    ):
        """
        Plot training/evaluation metrics
        
        Args:
            metrics: Dictionary mapping metric names to lists of values
            output_path: Path to save plot
        """
        fig, axes = plt.subplots(len(metrics), 1, figsize=(10, 4 * len(metrics)))
        
        if len(metrics) == 1:
            axes = [axes]
        
        for ax, (name, values) in zip(axes, metrics.items()):
            ax.plot(values)
            ax.set_title(name)
            ax.set_xlabel('Epoch' if 'loss' in name.lower() else 'Step')
            ax.set_ylabel(name)
            ax.grid(True)
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
        else:
            plt.show()


if __name__ == "__main__":
    # Example usage
    visualizer = Visualizer(class_names=["vehicle", "person", "drone"])
    
    # Dummy image
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    
    # Dummy detections
    boxes = np.array([[100, 100, 200, 200], [300, 300, 400, 400]])
    scores = np.array([0.95, 0.88])
    classes = np.array([0, 1])
    
    # Visualize
    result = visualizer.draw_detections(image, boxes, scores, classes)
    
    # Display would happen here
    print("Visualization created successfully")
