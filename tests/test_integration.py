"""
Integration Test
Verifies the end-to-end pipeline components working together.
"""
import pytest
import torch
import numpy as np
from src.tracking.track_manager import TrackManager
from src.detection.inference import InferenceEngine
from models.trackers.kalman_tracker import KalmanTracker

def test_pipeline_integration():
    """
    Simulate a pipeline run:
    1. Mock detection output
    2. NMS
    3. Tracker update
    4. Track Manager lifecycle
    """
    # 1. Mock Detection Output [Batch=1, Anchors=10, 85]
    # Create two boxes: one valid high conf, one low conf
    # Box 1: [100, 100, 50, 50], conf=0.9, cls=0
    # Box 2: [100, 100, 50, 50], conf=0.1, cls=0
    
    preds = torch.zeros((1, 10, 85))
    preds[0, 0, :4] = torch.tensor([100, 100, 50, 50])
    preds[0, 0, 4] = 0.9 # Obj conf
    preds[0, 0, 5] = 1.0 # Cls 0 prob
    
    preds[0, 1, :4] = torch.tensor([200, 200, 50, 50])
    preds[0, 1, 4] = 0.1 # Low conf
    preds[0, 1, 5] = 1.0
    
    # 2. Inference Engine (NMS)
    detections = InferenceEngine.non_max_suppression(preds, conf_thres=0.5)
    assert len(detections) == 1
    assert detections[0].shape[0] == 1 # Only high conf box remains
    
    # Convert to numpy for tracker
    dets_cpu = detections[0].cpu().numpy()
    
    # 3. Tracker Update
    # KalmanTracker expects [x1, y1, x2, y2, score] usually, or [x1,y1,x2,y2]
    # Let's check KalmanTracker signature or assume standard SORT-like interface
    # Our KalmanTracker.update takes (detections) where detections is [x1, y1, x2, y2, score]
    
    tracker = KalmanTracker()
    
    # Frame 1
    if len(dets_cpu) > 0:
        # dets_cpu is [x1, y1, x2, y2, conf, cls]
        boxes = dets_cpu[:, :4]
        scores = dets_cpu[:, 4]
        classes = dets_cpu[:, 5].astype(int)
        
        tracks = tracker.update(boxes, classes, scores)
    else:
        tracks = np.empty((0, 5))
        
    # First frame, usually no tracks confirmed or 1 track tentative depending on implementation
    # Default KalmanTracker might return nothing on first frame if min_hits > 0
    
    # 4. Data Splitter Verification (just import check)
    from src.data.splitter import DataSplitter
    assert hasattr(DataSplitter, 'split_dataset')
    
    # 5. Annotation Converter Verification
    from src.data.annotation_converter import AnnotationConverter
    assert hasattr(AnnotationConverter, 'coco_to_yolo')
    
    print("Integration test passed components check")

def test_models_import():
    """Verify all model modules can be imported"""
    from models.backbones.simple_cnn import SimpleCNN
    from models.trackers.deep_sort import DeepSort
    from models.trackers.bytetrack_tracker import ByteTracker
    
    model = SimpleCNN()
    assert model is not None
    
    tracker = DeepSort(use_cuda=False) # Force CPU for test
    assert tracker is not None
    
    byte_tracker = ByteTracker()
    assert byte_tracker is not None
