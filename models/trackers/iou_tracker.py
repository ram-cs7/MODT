"""
IoU Tracker
Simple, high-speed tracking based on Intersection over Union
"""

import numpy as np
from typing import List, Tuple, Optional, Dict
from scipy.optimize import linear_sum_assignment

class IoUTracker:
    """
    Simple IoU-based Tracker
    Associates detections across frames based on spatial overlap
    """
    
    def __init__(
        self,
        max_age: int = 5,
        min_hits: int = 2,
        iou_threshold: float = 0.3
    ):
        """
        Initialize IoU Tracker
        
        Args:
            max_age: Maximum frames to keep lost track
            min_hits: Minimum hits to confirm track
            iou_threshold: Minimum IoU for matching
        """
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        
        self.tracks = []  # List of dicts
        self.frame_count = 0
        self.next_id = 0
    
    def update(
        self,
        detections: np.ndarray,
        classes: Optional[np.ndarray] = None,
        scores: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Update tracker with new detections
        
        Args:
            detections: (N, 4) [x1, y1, x2, y2]
            classes: (N,) class IDs
            scores: (N,) confidence scores
            
        Returns:
            Active tracks (M, 7) [x1, y1, x2, y2, id, class, score]
        """
        self.frame_count += 1
        
        # Handle empty detections
        if len(detections) == 0:
            for track in self.tracks:
                track['time_since_update'] += 1
            self._cleanup_tracks()
            return self._get_active_tracks()
            
        # Match tracks to detections
        unmatched_tracks, unmatched_dets, matches = self._associate(detections)
        
        # Update matched tracks
        for track_idx, det_idx in matches:
            track = self.tracks[track_idx]
            track['box'] = detections[det_idx]
            track['score'] = scores[det_idx] if scores is not None else 1.0
            track['class_id'] = classes[det_idx] if classes is not None else 0
            track['hits'] += 1
            track['time_since_update'] = 0
            track['history'].append(detections[det_idx])
            
        # Create new tracks
        for det_idx in unmatched_dets:
            self._init_track(
                detections[det_idx],
                classes[det_idx] if classes is not None else 0,
                scores[det_idx] if scores is not None else 1.0
            )
            
        # Mark lost tracks
        for track_idx in unmatched_tracks:
            self.tracks[track_idx]['time_since_update'] += 1
            
        # Remove dead tracks
        self._cleanup_tracks()
        
        return self._get_active_tracks()
    
    def _associate(self, detections: np.ndarray):
        """Associate detections to tracks using IoU"""
        if len(self.tracks) == 0:
            return [], list(range(len(detections))), []
            
        iou_matrix = np.zeros((len(self.tracks), len(detections)))
        
        for t, track in enumerate(self.tracks):
            iou_matrix[t, :] = self._compute_iou(track['box'], detections)
            
        # Hungarian algorithm
        cost_matrix = 1.0 - iou_matrix
        row_indices, col_indices = linear_sum_assignment(cost_matrix)
        
        matches = []
        unmatched_tracks = list(range(len(self.tracks)))
        unmatched_dets = list(range(len(detections)))
        
        for r, c in zip(row_indices, col_indices):
            if iou_matrix[r, c] >= self.iou_threshold:
                matches.append((r, c))
                unmatched_tracks.remove(r)
                unmatched_dets.remove(c)
                
        return unmatched_tracks, unmatched_dets, matches
    
    def _compute_iou(self, box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
        """Compute IoU between box and boxes"""
        # Box: [x1, y1, x2, y2]
        # Boxes: (N, 4)
        
        x1 = np.maximum(box[0], boxes[:, 0])
        y1 = np.maximum(box[1], boxes[:, 1])
        x2 = np.minimum(box[2], boxes[:, 2])
        y2 = np.minimum(box[3], boxes[:, 3])
        
        inter_area = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
        
        box_area = (box[2] - box[0]) * (box[3] - box[1])
        boxes_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        
        union_area = box_area + boxes_area - inter_area
        
        return inter_area / (union_area + 1e-6)
    
    def _init_track(self, box, class_id, score):
        """Initialize new track"""
        self.tracks.append({
            'box': box,
            'id': self.next_id,
            'hits': 1,
            'time_since_update': 0,
            'class_id': class_id,
            'score': score,
            'history': [box]
        })
        self.next_id += 1
        
    def _cleanup_tracks(self):
        """Remove dead tracks"""
        self.tracks = [t for t in self.tracks if t['time_since_update'] <= self.max_age]
        
    def _get_active_tracks(self):
        """Get array of active confirmed tracks"""
        active_tracks = []
        for t in self.tracks:
            if t['hits'] >= self.min_hits or self.frame_count <= self.min_hits:
                x1, y1, x2, y2 = t['box']
                active_tracks.append([x1, y1, x2, y2, t['id'], t['class_id'], t['score']])
                
        if len(active_tracks) > 0:
            return np.array(active_tracks)
        return np.empty((0, 7))
