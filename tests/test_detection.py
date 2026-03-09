"""
Test Detection Module
Unit tests for detection components
"""
import pytest
import torch
import numpy as np
from src.detection.inference import InferenceEngine

def test_nms_empty():
    """Test NMS with empty predictions"""
    preds = torch.zeros((1, 0, 85)) # Batch 1, 0 detections
    output = InferenceEngine.non_max_suppression(preds)
    assert len(output) == 1
    assert output[0].shape == (0, 6)

def test_nms_basic():
    """Test NMS with two overlapping boxes"""
    # [cx, cy, w, h, obj, cls]
    # Box 1: [10, 10, 20, 20], score 0.9
    # Box 2: [11, 11, 20, 20], score 0.8 (high overlap)
    
    box1 = [10, 10, 20, 20, 0.9, 0.9] + [0]*79 # 85 total
    box2 = [11, 11, 20, 20, 0.8, 0.8] + [0]*79
    
    preds = torch.tensor([box1, box2]).unsqueeze(0) # [1, 2, 85]
    
    # With low IoU thresh, should keep both? No, high overlap -> remove one
    # With high IoU thresh, keep both
    
    output = InferenceEngine.non_max_suppression(preds, iou_thres=0.1) # Strict NMS
    assert len(output) == 1
    assert output[0].shape[0] == 1 # Should suppress one
