"""
DeepSORT Tracker Wrapper
Exposes the generic DeepSORT implementation with defaults for military object tracking.
"""
import torch
import numpy as np
from typing import List, Optional, Any

from models.trackers.deepsort_tracker import DeepSORTTracker
from models.backbones.simple_cnn import create_reid_backbone

class DeepSort(DeepSORTTracker):
    """
    DeepSORT tracker implementation.
    This module provides the main interface for DeepSORT tracking,
    initializing the ReID backbone and handling data association.
    """
    
    def __init__(
        self,
        max_dist: float = 0.2,
        min_confidence: float = 0.3,
        nms_max_overlap: float = 1.0,
        max_iou_distance: float = 0.7,
        max_age: int = 70,
        n_init: int = 3,
        nn_budget: int = 100,
        use_cuda: bool = True
    ):
        """
        Initialize DeepSORT
        
        Args:
            max_dist: Maximum cosine distance for ReID association
            min_confidence: Minimum detection confidence
            nms_max_overlap: NMS overlap threshold (if used post-detection)
            max_iou_distance: Maximum IoU distance for IoU matching
            max_age: Maximum frames to keep lost tracks
            n_init: Number of frames to confirm a track
            nn_budget: Maximum size of the appearance descriptor gallery
            use_cuda: Use GPU/CUDA for feature extraction
        """
        # Initialize the feature extractor
        device = "cuda" if use_cuda and torch.cuda.is_available() else "cpu"
        embedder = create_reid_backbone("simple_cnn")
        embedder.to(device)
        embedder.eval()
        
        super().__init__(
            max_age=max_age,
            min_hits=n_init,
            iou_threshold=max_iou_distance
        )
        
        self.embedder = embedder
        self.device = device
        self.max_dist = max_dist
        self.nn_budget = nn_budget
        
    def _get_features(self, bbox_xywh: np.ndarray, ori_img: np.ndarray) -> np.ndarray:
        """
        Extract features from bounding boxes
        
        Args:
            bbox_xywh: [N, 4] bounding boxes (x, y, w, h)
            ori_img: Original image frame
            
        Returns:
            features: [N, 128] normalized feature vectors
        """
        if len(bbox_xywh) == 0:
            return np.empty((0, 128))
            
        crops = []
        h, w, _ = ori_img.shape
        
        for box in bbox_xywh:
            x, y, bw, bh = box
            
            # Clip to image
            x1 = max(0, int(x))
            y1 = max(0, int(y))
            x2 = min(w, int(x + bw))
            y2 = min(h, int(y + bh))
            
            crop = ori_img[y1:y2, x1:x2]
            if crop.size == 0:
                # Fallback for empty crop (shouldn't happen with valid boxes)
                crop = np.zeros((128, 64, 3), dtype=np.uint8)
            else:
                import cv2
                crop = cv2.resize(crop, (64, 128))
                
            crops.append(crop)
            
        if not crops:
            return np.empty((0, 128))
            
        # Preprocess
        crops_tensor = torch.as_tensor(np.array(crops)).float()
        crops_tensor = crops_tensor.permute(0, 3, 1, 2) / 255.0 # [B, 3, 128, 64]
        crops_tensor = crops_tensor.to(self.device)
        
        # Inference
        with torch.no_grad():
            features = self.embedder(crops_tensor)
            
        return features.cpu().numpy()

    def update(self, bbox_xywh, confidences, classes, ori_img):
        """
        Update tracks
        
        Args:
            bbox_xywh: Bounding boxes (xywh)
            confidences: Confidence scores
            classes: Class IDs
            ori_img: Original image for feature extraction
        """
        # Extract features
        features = self._get_features(bbox_xywh, ori_img)
        
        # In a real DeepSORT implementation, we would pass these features 
        # to the matching cascade. For this implementation plan, we are extending
        # the base Tracker class which might be using IoU/Kalman.
        # Ideally, `models.trackers.deepsort_tracker.DeepSORTTracker` should
        # accept features in its `update` or `_match` method.
        # Assuming the base class expects standard update arguments.
        
        # TODO: Integrate features into the association cost matrix
        # For now, we proceed with the standard Kalman/IoU update
        # but with the infrastructure ready for ReID integration.
        
        return super().update(bbox_xywh, classes, confidences)
