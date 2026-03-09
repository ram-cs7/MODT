"""
Multi-Object Tracking Metrics
Computes MOTA, MOTP, IDF1, and other MOT metrics
"""

import numpy as np
from typing import Dict, List, Tuple
from collections import defaultdict


class MOTMetrics:
    """
    Multi-Object Tracking metrics calculator
    Implements CLEAR MOT metrics (MOTA, MOTP) and IDF1
    """
    
    def __init__(self, num_classes: int = 1):
        """
        Initialize MOT metrics calculator
        
        Args:
            num_classes: Number of object classes
        """
        self.num_classes = num_classes
        self.reset()
    
    def reset(self):
        """Reset all metrics"""
        self.num_frames = 0
        self.num_gt = 0
        self.num_pred = 0
        self.num_matches = 0
        self.num_false_positives = 0
        self.num_misses = 0
        self.num_id_switches = 0
        self.total_distance = 0.0
        
        # Track management
        self.gt_track_ids = set()
        self.pred_track_ids = set()
        self.track_mapping = {}  # gt_id -> pred_id mapping
    
    @staticmethod
    def compute_iou(box1: np.ndarray, box2: np.ndarray) -> float:
        """Compute IoU between two boxes"""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection
        
        return intersection / (union + 1e-6)
    
    def update(
        self,
        gt_boxes: np.ndarray,
        gt_ids: np.ndarray,
        pred_boxes: np.ndarray,
        pred_ids: np.ndarray,
        iou_threshold: float = 0.5
    ):
        """
        Update metrics for one frame
        
        Args:
            gt_boxes: Ground truth boxes (N, 4) [x1, y1, x2, y2]
            gt_ids: Ground truth track IDs (N,)
            pred_boxes: Predicted boxes (M, 4)
            pred_ids: Predicted track IDs (M,)
            iou_threshold: IoU threshold for matching
        """
        self.num_frames += 1
        self.num_gt += len(gt_boxes)
        self.num_pred += len(pred_boxes)
        
        # Update track sets
        self.gt_track_ids.update(gt_ids)
        self.pred_track_ids.update(pred_ids)
        
        if len(gt_boxes) == 0 or len(pred_boxes) == 0:
            if len(pred_boxes) > 0:
                self.num_false_positives += len(pred_boxes)
            if len(gt_boxes) > 0:
                self.num_misses += len(gt_boxes)
            return
        
        # Compute IoU matrix
        iou_matrix = np.zeros((len(gt_boxes), len(pred_boxes)))
        for i, gt_box in enumerate(gt_boxes):
            for j, pred_box in enumerate(pred_boxes):
                iou_matrix[i, j] = self.compute_iou(gt_box, pred_box)
        
        # Match using greedy assignment
        matched_gt = set()
        matched_pred = set()
        
        # Sort by IoU (descending)
        matches = []
        for i in range(len(gt_boxes)):
            for j in range(len(pred_boxes)):
                if iou_matrix[i, j] >= iou_threshold:
                    matches.append((i, j, iou_matrix[i, j]))
        
        matches.sort(key=lambda x: x[2], reverse=True)
        
        # Assign matches
        for gt_idx, pred_idx, iou in matches:
            if gt_idx not in matched_gt and pred_idx not in matched_pred:
                matched_gt.add(gt_idx)
                matched_pred.add(pred_idx)
                
                self.num_matches += 1
                self.total_distance += (1 - iou)  # Distance metric for MOTP
                
                # Check for ID switch
                gt_id = gt_ids[gt_idx]
                pred_id = pred_ids[pred_idx]
                
                if gt_id in self.track_mapping:
                    if self.track_mapping[gt_id] != pred_id:
                        self.num_id_switches += 1
                        self.track_mapping[gt_id] = pred_id
                else:
                    self.track_mapping[gt_id] = pred_id
        
        # Count false positives and misses
        self.num_false_positives += (len(pred_boxes) - len(matched_pred))
        self.num_misses += (len(gt_boxes) - len(matched_gt))
    
    def compute_mota(self) -> float:
        """
        Compute MOTA (Multiple Object Tracking Accuracy)
        
        MOTA = 1 - (FP + FN + IDSW) / GT
        
        Returns:
            MOTA score
        """
        if self.num_gt == 0:
            return 0.0
        
        mota = 1 - (self.num_false_positives + self.num_misses + self.num_id_switches) / self.num_gt
        return mota
    
    def compute_motp(self) -> float:
        """
        Compute MOTP (Multiple Object Tracking Precision)
        
        Average IoU of matched detections
        
        Returns:
            MOTP score
        """
        if self.num_matches == 0:
            return 0.0
        
        motp = 1 - (self.total_distance / self.num_matches)
        return motp
    
    def compute_idf1(self) -> float:
        """
        Compute IDF1 (ID F1 Score)
        
        F1 score based on ID matches
        
        Returns:
            IDF1 score
        """
        if self.num_matches == 0:
            return 0.0
        
        # Simplified IDF1 calculation
        # Proper IDF1 requires tracking identity consistency over time
        idtp = self.num_matches - self.num_id_switches
        idfp = self.num_false_positives + self.num_id_switches
        idfn = self.num_misses + self.num_id_switches
        
        idf1 = 2 * idtp / (2 * idtp + idfp + idfn + 1e-6)
        return idf1
    
    def get_metrics(self) -> Dict[str, float]:
        """
        Get all tracking metrics
        
        Returns:
            Dictionary of metrics
        """
        return {
            'MOTA': self.compute_mota(),
            'MOTP': self.compute_motp(),
            'IDF1': self.compute_idf1(),
            'num_frames': self.num_frames,
            'num_gt': self.num_gt,
            'num_pred': self.num_pred,
            'num_matches': self.num_matches,
            'num_false_positives': self.num_false_positives,
            'num_misses': self.num_misses,
            'num_id_switches': self.num_id_switches,
            'num_gt_tracks': len(self.gt_track_ids),
            'num_pred_tracks': len(self.pred_track_ids)
        }
    
    def print_metrics(self):
        """Print metrics in formatted table"""
        metrics = self.get_metrics()
        
        print("="*80)
        print("Multi-Object Tracking Metrics")
        print("="*80)
        print(f"MOTA:            {metrics['MOTA']:.4f}")
        print(f"MOTP:            {metrics['MOTP']:.4f}")
        print(f"IDF1:            {metrics['IDF1']:.4f}")
        print("-"*80)
        print(f"Frames:          {metrics['num_frames']}")
        print(f"GT Objects:      {metrics['num_gt']}")
        print(f"Pred Objects:    {metrics['num_pred']}")
        print(f"Matches:         {metrics['num_matches']}")
        print(f"False Positives: {metrics['num_false_positives']}")
        print(f"Misses:          {metrics['num_misses']}")
        print(f"ID Switches:     {metrics['num_id_switches']}")
        print(f"GT Tracks:       {metrics['num_gt_tracks']}")
        print(f"Pred Tracks:     {metrics['num_pred_tracks']}")
        print("="*80)


if __name__ == "__main__":
    # Example usage
    metrics = MOTMetrics()
    
    # Frame 1
    gt_boxes = np.array([[10, 10, 50, 50], [60, 60, 100, 100]])
    gt_ids = np.array([1, 2])
    pred_boxes = np.array([[12, 12, 48, 48], [65, 65, 95, 95]])
    pred_ids = np.array([1, 2])
    
    metrics.update(gt_boxes, gt_ids, pred_boxes, pred_ids)
    
    # Frame 2
    gt_boxes = np.array([[15, 15, 55, 55], [70, 70, 110, 110]])
    gt_ids = np.array([1, 2])
    pred_boxes = np.array([[17, 17, 53, 53], [68, 68, 108, 108]])
    pred_ids = np.array([1, 2])
    
    metrics.update(gt_boxes, gt_ids, pred_boxes, pred_ids)
    
    # Print results
    metrics.print_metrics()
