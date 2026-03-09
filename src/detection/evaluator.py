"""
Detection Model Evaluator
Computes mAP, precision, recall, and other detection metrics
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import json
from collections import defaultdict
from tqdm import tqdm

from src.utils.logger import setup_logger


class DetectionEvaluator:
    """
    Evaluator for object detection models
    Computes COCO-style metrics including mAP@0.5, mAP@0.5:0.95
    """
    
    def __init__(
        self,
        num_classes: int,
        iou_thresholds: List[float] = None,
        class_names: Optional[List[str]] = None
    ):
        """
        Initialize evaluator
        
        Args:
            num_classes: Number of object classes
            iou_thresholds: List of IoU thresholds for mAP computation
            class_names: List of class names
        """
        self.num_classes = num_classes
        self.iou_thresholds = iou_thresholds or [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
        self.class_names = class_names or [f"class_{i}" for i in range(num_classes)]
        
        self.logger = setup_logger("Evaluator")
        
        # Storage for predictions and ground truth
        self.predictions = []
        self.ground_truths = []
    
    def add_predictions(
        self,
        pred_boxes: np.ndarray,
        pred_scores: np.ndarray,
        pred_classes: np.ndarray,
        gt_boxes: np.ndarray,
        gt_classes: np.ndarray,
        image_id: int = 0
    ):
        """
        Add predictions and ground truth for an image
        
        Args:
            pred_boxes: Predicted boxes (N, 4) [x1, y1, x2, y2]
            pred_scores: Prediction scores (N,)
            pred_classes: Predicted classes (N,)
            gt_boxes: Ground truth boxes (M, 4)
            gt_classes: Ground truth classes (M,)
            image_id: Image identifier
        """
        self.predictions.append({
            'image_id': image_id,
            'boxes': pred_boxes,
            'scores': pred_scores,
            'classes': pred_classes
        })
        
        self.ground_truths.append({
            'image_id': image_id,
            'boxes': gt_boxes,
            'classes': gt_classes
        })
    
    @staticmethod
    def compute_iou(box1: np.ndarray, box2: np.ndarray) -> float:
        """
        Compute IoU between two boxes
        
        Args:
            box1: First box [x1, y1, x2, y2]
            box2: Second box [x1, y1, x2, y2]
        
        Returns:
            IoU value
        """
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection
        
        return intersection / (union + 1e-6)
    
    def compute_ap(
        self,
        predictions: List[Dict],
        ground_truths: List[Dict],
        class_id: int,
        iou_threshold: float = 0.5
    ) -> Tuple[float, float, float]:
        """
        Compute Average Precision for a single class
        
        Args:
            predictions: List of predictions
            ground_truths: List of ground truths
            class_id: Class to evaluate
            iou_threshold: IoU threshold for matching
        
        Returns:
            Tuple of (AP, precision, recall)
        """
        # Filter predictions for this class
        class_preds = []
        for pred in predictions:
            mask = pred['classes'] == class_id
            if mask.sum() > 0:
                class_preds.append({
                    'image_id': pred['image_id'],
                    'boxes': pred['boxes'][mask],
                    'scores': pred['scores'][mask]
                })
        
        # Filter ground truths for this class
        class_gts = []
        for gt in ground_truths:
            mask = gt['classes'] == class_id
            if mask.sum() > 0:
                class_gts.append({
                    'image_id': gt['image_id'],
                    'boxes': gt['boxes'][mask]
                })
        
        if len(class_preds) == 0 or len(class_gts) == 0:
            return 0.0, 0.0, 0.0
        
        # Collect all predictions with scores
        all_preds = []
        for pred in class_preds:
            for box, score in zip(pred['boxes'], pred['scores']):
                all_preds.append({
                    'image_id': pred['image_id'],
                    'box': box,
                    'score': score,
                    'matched': False
                })
        
        # Sort by confidence
        all_preds.sort(key=lambda x: x['score'], reverse=True)
        
        # Count total ground truths
        total_gts = sum(len(gt['boxes']) for gt in class_gts)
        
        # Match predictions to ground truths
        tp = np.zeros(len(all_preds))
        fp = np.zeros(len(all_preds))
        
        gt_matched = defaultdict(list)  # Track matched GTs per image
        
        for i, pred in enumerate(all_preds):
            image_id = pred['image_id']
            pred_box = pred['box']
            
            # Find ground truths for this image
            image_gts = next((gt for gt in class_gts if gt['image_id'] == image_id), None)
            
            if image_gts is None:
                fp[i] = 1
                continue
            
            # Find best matching GT
            best_iou = 0
            best_gt_idx = -1
            
            for gt_idx, gt_box in enumerate(image_gts['boxes']):
                if gt_idx in gt_matched[image_id]:
                    continue
                
                iou = self.compute_iou(pred_box, gt_box)
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_idx
            
            # Check if match is good enough
            if best_iou >= iou_threshold:
                tp[i] = 1
                gt_matched[image_id].append(best_gt_idx)
            else:
                fp[i] = 1
        
        # Compute precision and recall
        tp_cumsum = np.cumsum(tp)
        fp_cumsum = np.cumsum(fp)
        
        recalls = tp_cumsum / (total_gts + 1e-6)
        precisions = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-6)
        
        # Compute AP using 11-point interpolation
        ap = 0.0
        for t in np.linspace(0, 1, 11):
            if np.sum(recalls >= t) == 0:
                p = 0
            else:
                p = np.max(precisions[recalls >= t])
            ap += p / 11
        
        # Final precision and recall
        final_precision = precisions[-1] if len(precisions) > 0 else 0.0
        final_recall = recalls[-1] if len(recalls) > 0 else 0.0
        
        return ap, final_precision, final_recall
    
    def evaluate(self) -> Dict[str, float]:
        """
        Evaluate all predictions
        
        Returns:
            Dictionary of metrics
        """
        self.logger.info("Computing detection metrics...")
        
        metrics = {}
        
        # Compute per-class AP for different IoU thresholds
        class_aps = defaultdict(dict)
        
        for class_id in range(self.num_classes):
            for iou_thresh in self.iou_thresholds:
                ap, precision, recall = self.compute_ap(
                    self.predictions,
                    self.ground_truths,
                    class_id,
                    iou_thresh
                )
                class_aps[class_id][iou_thresh] = {
                    'AP': ap,
                    'precision': precision,
                    'recall': recall
                }
        
        # Compute mAP@0.5
        map_50 = np.mean([class_aps[c][0.5]['AP'] for c in range(self.num_classes)])
        metrics['mAP@0.5'] = map_50
        
        # Compute mAP@0.5:0.95
        map_50_95 = np.mean([
            np.mean([class_aps[c][t]['AP'] for t in self.iou_thresholds])
            for c in range(self.num_classes)
        ])
        metrics['mAP@0.5:0.95'] = map_50_95
        
        # Per-class metrics at IoU=0.5
        for class_id in range(self.num_classes):
            class_name = self.class_names[class_id]
            metrics[f'{class_name}_AP'] = class_aps[class_id][0.5]['AP']
            metrics[f'{class_name}_precision'] = class_aps[class_id][0.5]['precision']
            metrics[f'{class_name}_recall'] = class_aps[class_id][0.5]['recall']
        
        return metrics
    
    def print_metrics(self, metrics: Dict[str, float]):
        """Print metrics in formatted table"""
        self.logger.info("="*80)
        self.logger.info("Detection Evaluation Results")
        self.logger.info("="*80)
        self.logger.info(f"mAP@0.5:      {metrics['mAP@0.5']:.4f}")
        self.logger.info(f"mAP@0.5:0.95: {metrics['mAP@0.5:0.95']:.4f}")
        self.logger.info("-"*80)
        
        for class_id in range(self.num_classes):
            class_name = self.class_names[class_id]
            ap = metrics[f'{class_name}_AP']
            precision = metrics[f'{class_name}_precision']
            recall = metrics[f'{class_name}_recall']
            
            self.logger.info(
                f"{class_name:20s} | AP: {ap:.4f} | P: {precision:.4f} | R: {recall:.4f}"
            )
        
        self.logger.info("="*80)
    
    def save_metrics(self, output_path: str, metrics: Dict[str, float]):
        """Save metrics to JSON file"""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        self.logger.info(f"Metrics saved to: {output_path}")
    
    def reset(self):
        """Reset accumulated predictions and ground truths"""
        self.predictions = []
        self.ground_truths = []


if __name__ == "__main__":
    # Example usage
    evaluator = DetectionEvaluator(
        num_classes=5,
        class_names=["background", "soldier", "military_vehicle", "drone", "civilian_vehicle"]
    )
    
    # Dummy predictions and ground truths
    pred_boxes = np.array([[10, 10, 50, 50], [60, 60, 100, 100]])
    pred_scores = np.array([0.9, 0.85])
    pred_classes = np.array([1, 2])
    
    gt_boxes = np.array([[12, 12, 48, 48], [65, 65, 95, 95]])
    gt_classes = np.array([1, 2])
    
    evaluator.add_predictions(pred_boxes, pred_scores, pred_classes, gt_boxes, gt_classes, image_id=0)
    
    # Evaluate
    metrics = evaluator.evaluate()
    evaluator.print_metrics(metrics)
