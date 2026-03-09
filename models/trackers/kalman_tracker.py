"""
Kalman Filter-based Multi-Object Tracker
Uses constant velocity motion model with Hungarian algorithm for data association
"""

import numpy as np
from typing import List, Tuple, Optional, Dict
from filterpy.kalman import KalmanFilter
from scipy.optimize import linear_sum_assignment


class KalmanBoxTracker:
    """
    Kalman filter for tracking bounding boxes with constant velocity model
    State: [x_center, y_center, width, height, vx, vy, vw, vh]
    """
    
    count = 0  # Global track ID counter
    
    def __init__(self, bbox: np.ndarray, class_id: int = 0, score: float = 1.0):
        """
        Initialize Kalman filter for a bounding box
        
        Args:
            bbox: Initial bounding box [x1, y1, x2, y2]
            class_id: Object class ID
            score: Detection confidence score
        """
        # Initialize ID
        KalmanBoxTracker.count += 1
        self.id = KalmanBoxTracker.count
        
        # Store class and score
        self.class_id = class_id
        self.score = score
        
        # Initialize Kalman filter
        # State: [x_center, y_center, width, height, vx, vy, vw, vh]
        self.kf = KalmanFilter(dim_x=8, dim_z=4)
        
        # State transition matrix (constant velocity model)
        self.kf.F = np.array([
            [1, 0, 0, 0, 1, 0, 0, 0],  # x += vx
            [0, 1, 0, 0, 0, 1, 0, 0],  # y += vy
            [0, 0, 1, 0, 0, 0, 1, 0],  # w += vw
            [0, 0, 0, 1, 0, 0, 0, 1],  # h += vh
            [0, 0, 0, 0, 1, 0, 0, 0],  # vx
            [0, 0, 0, 0, 0, 1, 0, 0],  # vy
            [0, 0, 0, 0, 0, 0, 1, 0],  # vw
            [0, 0, 0, 0, 0, 0, 0, 1],  # vh
        ])
        
        # Measurement matrix (observe position and size only)
        self.kf.H = np.array([
            [1, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0],
        ])
        
        # Measurement noise covariance
        self.kf.R *= 10.0
        
        # Process noise covariance
        self.kf.Q[4:, 4:] *= 0.01  # Low process noise for velocities
        
        # Initial state covariance
        self.kf.P[4:, 4:] *= 1000.0  # High uncertainty for velocities initially
        self.kf.P *= 10.0
        
        # Initialize state
        self.kf.x[:4, 0] = self._bbox_to_state(bbox)
        
        # Track history
        self.time_since_update = 0
        self.hits = 1
        self.hit_streak = 1
        self.age = 0
        self.history = []
    
    def _bbox_to_state(self, bbox: np.ndarray) -> np.ndarray:
        """
        Convert bounding box to state representation
        
        Args:
            bbox: [x1, y1, x2, y2]
        
        Returns:
            state: [x_center, y_center, width, height]
        """
        x1, y1, x2, y2 = bbox
        w = x2 - x1
        h = y2 - y1
        x = x1 + w / 2
        y = y1 + h / 2
        return np.array([x, y, w, h])
    
    def _state_to_bbox(self, state: np.ndarray) -> np.ndarray:
        """
        Convert state to bounding box
        
        Args:
            state: [x_center, y_center, width, height, ...]
        
        Returns:
            bbox: [x1, y1, x2, y2]
        """
        x, y, w, h = state[:4]
        x1 = x - w / 2
        y1 = y - h / 2
        x2 = x + w / 2
        y2 = y + h / 2
        return np.array([x1, y1, x2, y2])
    
    def predict(self) -> np.ndarray:
        """
        Predict next state
        
        Returns:
            Predicted bounding box [x1, y1, x2, y2]
        """
        # Prevent negative width/height
        if self.kf.x[2] + self.kf.x[6] <= 0:
            self.kf.x[6] = 0
        if self.kf.x[3] + self.kf.x[7] <= 0:
            self.kf.x[7] = 0
        
        self.kf.predict()
        self.age += 1
        
        if self.time_since_update > 0:
            self.hit_streak = 0
        
        self.time_since_update += 1
        self.history.append(self._state_to_bbox(self.kf.x))
        
        return self.history[-1]
    
    def update(self, bbox: np.ndarray, score: float = 1.0):
        """
        Update Kalman filter with new detection
        
        Args:
            bbox: Detected bounding box [x1, y1, x2, y2]
            score: Detection confidence
        """
        self.time_since_update = 0
        self.hits += 1
        self.hit_streak += 1
        self.score = score
        
        measurement = self._bbox_to_state(bbox)
        self.kf.update(measurement)
    
    def get_state(self) -> np.ndarray:
        """
        Get current bounding box state
        
        Returns:
            Current bounding box [x1, y1, x2, y2]
        """
        return self._state_to_bbox(self.kf.x)


class KalmanTracker:
    """
    Multi-object tracker using Kalman filter and Hungarian algorithm
    """
    
    def __init__(
        self,
        max_age: int = 30,
        min_hits: int = 3,
        iou_threshold: float = 0.3
    ):
        """
        Initialize tracker
        
        Args:
            max_age: Maximum frames to keep track without detections
            min_hits: Minimum hits before confirming track
            iou_threshold: IoU threshold for matching
        """
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        
        self.trackers: List[KalmanBoxTracker] = []
        self.frame_count = 0
    
    @staticmethod
    def compute_iou(bbox1: np.ndarray, bbox2: np.ndarray) -> float:
        """
        Compute IoU between two bounding boxes
        
        Args:
            bbox1: First box [x1, y1, x2, y2]
            bbox2: Second box [x1, y1, x2, y2]
        
        Returns:
            IoU value
        """
        x1 = max(bbox1[0], bbox2[0])
        y1 = max(bbox1[1], bbox2[1])
        x2 = min(bbox1[2], bbox2[2])
        y2 = min(bbox1[3], bbox2[3])
        
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        
        area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
        area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
        union = area1 + area2 - intersection
        
        return intersection / (union + 1e-6)
    
    def update(
        self,
        detections: np.ndarray,
        classes: Optional[np.ndarray] = None,
        scores: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Update tracker with new detections
        
        Args:
            detections: Array of detections (N, 4) [x1, y1, x2, y2]
            classes: Array of class IDs (N,)
            scores: Array of confidence scores (N,)
        
        Returns:
            Array of active tracks (M, 6) [x1, y1, x2, y2, track_id, class_id]
        """
        self.frame_count += 1
        
        # Predict new locations of existing tracks
        for tracker in self.trackers:
            tracker.predict()
        
        # Match detections to tracks using Hungarian algorithm
        if len(detections) > 0:
            matched, unmatched_dets, unmatched_trks = self._associate_detections(
                detections, classes, scores
            )
            
            # Update matched tracks
            for det_idx, trk_idx in matched:
                self.trackers[trk_idx].update(
                    detections[det_idx],
                    scores[det_idx] if scores is not None else 1.0
                )
            
            # Create new tracks for unmatched detections
            for det_idx in unmatched_dets:
                class_id = classes[det_idx] if classes is not None else 0
                score = scores[det_idx] if scores is not None else 1.0
                tracker = KalmanBoxTracker(detections[det_idx], class_id, score)
                self.trackers.append(tracker)
        
        # Remove dead tracks
        self.trackers = [
            t for t in self.trackers
            if t.time_since_update <= self.max_age
        ]
        
        # Return active tracks
        active_tracks = []
        for tracker in self.trackers:
            # Only return confirmed tracks
            if tracker.time_since_update < 1 and \
               (tracker.hit_streak >= self.min_hits or self.frame_count <= self.min_hits):
                bbox = tracker.get_state()
                # Ensure bbox is a 1D array with 4 elements
                if bbox.ndim > 1:
                    bbox = bbox.flatten()
                track = np.concatenate([bbox, [tracker.id, tracker.class_id, tracker.score]])
                active_tracks.append(track)
        
        if len(active_tracks) > 0:
            return np.array(active_tracks)
        return np.empty((0, 7))
    
    def _associate_detections(
        self,
        detections: np.ndarray,
        classes: Optional[np.ndarray],
        scores: Optional[np.ndarray]
    ) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        """
        Associate detections to tracks using IoU and Hungarian algorithm
        
        Returns:
            matched: List of (detection_idx, tracker_idx) pairs
            unmatched_detections: List of unmatched detection indices
            unmatched_trackers: List of unmatched tracker indices
        """
        if len(self.trackers) == 0:
            return [], list(range(len(detections))), []
        
        # Compute IoU matrix
        iou_matrix = np.zeros((len(detections), len(self.trackers)))
        
        for d, det in enumerate(detections):
            for t, tracker in enumerate(self.trackers):
                trk_bbox = tracker.get_state()
                iou_matrix[d, t] = self.compute_iou(det, trk_bbox)
                
                # Penalize class mismatch
                if classes is not None:
                    if classes[d] != tracker.class_id:
                        iou_matrix[d, t] *= 0.5
        
        # Hungarian algorithm
        matched_indices = []
        if min(iou_matrix.shape) > 0:
            # Convert to cost matrix
            cost_matrix = 1 - iou_matrix
            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            
            for r, c in zip(row_ind, col_ind):
                if iou_matrix[r, c] >= self.iou_threshold:
                    matched_indices.append((r, c))
        
        # Get unmatched detections and trackers
        matched_dets = {m[0] for m in matched_indices}
        matched_trks = {m[1] for m in matched_indices}
        
        unmatched_dets = [d for d in range(len(detections)) if d not in matched_dets]
        unmatched_trks = [t for t in range(len(self.trackers)) if t not in matched_trks]
        
        return matched_indices, unmatched_dets, unmatched_trks
    
    def reset(self):
        """Reset tracker"""
        self.trackers = []
        self.frame_count = 0
        KalmanBoxTracker.count = 0


if __name__ == "__main__":
    # Example usage
    tracker = KalmanTracker(max_age=30, min_hits=3, iou_threshold=0.3)
    
    # Simulated detections over multiple frames
    for frame_idx in range(10):
        # Generate dummy detections
        num_dets = np.random.randint(1, 5)
        detections = np.random.rand(num_dets, 4) * 100
        detections[:, 2:] += detections[:, :2]  # Convert to x2, y2
        classes = np.random.randint(0, 5, num_dets)
        scores = np.random.rand(num_dets)
        
        # Update tracker
        tracks = tracker.update(detections, classes, scores)
        
        print(f"Frame {frame_idx}: {len(detections)} detections, {len(tracks)} tracks")
        if len(tracks) > 0:
            print(f"  Track IDs: {tracks[:, 4].astype(int)}")
