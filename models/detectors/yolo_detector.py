"""
YOLO Detector Wrapper
Supports YOLOv5 and YOLOv8 with custom heads for military object detection
"""

import torch
import torch.nn as nn
from typing import Tuple, List, Dict, Optional, Union
import numpy as np
from pathlib import Path


class YOLODetector(nn.Module):
    """
    Wrapper for YOLO detectors (YOLOv5/YOLOv8)
    Provides unified interface for training and inference
    """
    
    def __init__(
        self,
        variant: str = "yolov8n",
        num_classes: int = 5,
        pretrained: bool = True,
        img_size: Tuple[int, int] = (640, 640),
        device: str = "cuda"
    ):
        """
        Initialize YOLO detector
        
        Args:
            variant: Model variant ('yolov8n', 'yolov8s', 'yolov8m', 'yolov8l', 'yolov8x')
            num_classes: Number of object classes
            pretrained: Use pretrained weights
            img_size: Input image size
            device: Device to run model on
        """
        super().__init__()
        
        self.variant = variant
        self.num_classes = num_classes
        self.img_size = img_size
        self.device = device
        
        # Load model based on variant
        self.model = self._load_model(variant, num_classes, pretrained)
        self.model.to(device)
        
        # Model stride
        self.stride = 32
        
        # Class names (can be set later)
        self.class_names = [f"class_{i}" for i in range(num_classes)]
    
    def _load_model(self, variant: str, num_classes: int, pretrained: bool):
        """Load YOLO model from ultralytics"""
        try:
            from ultralytics import YOLO
            
            if pretrained:
                # Load pretrained model
                if variant.startswith("yolov8"):
                    model = YOLO(f"{variant}.pt")
                elif variant.startswith("yolov5"):
                    model = YOLO(f"{variant}.pt")
                else:
                    raise ValueError(f"Unknown variant: {variant}")
                
                # Modify output layer for custom number of classes if different from COCO
                if num_classes != 80:
                    # Create new model with custom classes
                    model_cfg = variant.replace("yolov", "yolov")
                    model = YOLO(f"{variant}.yaml")
                    # Load pretrained weights where possible
                    # This would require custom implementation for weight transfer
            else:
                # Create from scratch
                model = YOLO(f"{variant}.yaml")
            
            return model
            
        except ImportError:
            # Fallback: implement custom YOLO or use torchvision
            print("WARNING: ultralytics not installed. Using custom implementation.")
            return self._create_custom_yolo()
    
    def _create_custom_yolo(self):
        """
        Create custom YOLO implementation if ultralytics is not available
        This is a simplified version - production code should use ultralytics
        """
        # NOTE: This is a placeholder. In production, use ultralytics YOLO
        # or implement full YOLO architecture
        
        class SimpleYOLO(nn.Module):
            def __init__(self, num_classes):
                super().__init__()
                self.num_classes = num_classes
                
                # Placeholder backbone (should be CSPDarknet or similar)
                self.backbone = nn.Sequential(
                    nn.Conv2d(3, 64, 3, 1, 1),
                    nn.BatchNorm2d(64),
                    nn.ReLU(),
                    nn.MaxPool2d(2, 2),
                    
                    nn.Conv2d(64, 128, 3, 1, 1),
                    nn.BatchNorm2d(128),
                    nn.ReLU(),
                    nn.MaxPool2d(2, 2),
                    
                    nn.Conv2d(128, 256, 3, 1, 1),
                    nn.BatchNorm2d(256),
                    nn.ReLU(),
                    nn.MaxPool2d(2, 2),
                )
                
                # Detection head
                self.head = nn.Conv2d(256, (num_classes + 5) * 3, 1)  # 3 anchors per grid
            
            def forward(self, x):
                x = self.backbone(x)
                x = self.head(x)
                return x
        
        return SimpleYOLO(self.num_classes)
    
    def forward(self, x: torch.Tensor) -> Union[torch.Tensor, List]:
        """
        Forward pass
        
        Args:
            x: Input tensor of shape (B, C, H, W)
        
        Returns:
            Predictions (format depends on underlying model)
        """
        return self.model(x)
    
    def predict(
        self,
        images: Union[np.ndarray, torch.Tensor, List],
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        agnostic_nms: bool = False
    ) -> List[Dict[str, np.ndarray]]:
        """
        Run inference on images
        
        Args:
            images: Input images (numpy array, tensor, or list of paths)
            conf_threshold: Confidence threshold
            iou_threshold: IoU threshold for NMS
            agnostic_nms: Class-agnostic NMS
        
        Returns:
            List of detection dictionaries, one per image
            Each dict contains: 'boxes', 'scores', 'classes', 'labels'
        """
        self.model.eval()
        
        with torch.no_grad():
            # Run prediction
            if hasattr(self.model, 'predict'):
                # Using ultralytics YOLO
                results = self.model.predict(
                    images,
                    conf=conf_threshold,
                    iou=iou_threshold,
                    agnostic_nms=agnostic_nms,
                    verbose=False
                )
                
                # Parse results
                detections = []
                for result in results:
                    boxes = result.boxes.xyxy.cpu().numpy()  # x1, y1, x2, y2
                    scores = result.boxes.conf.cpu().numpy()
                    classes = result.boxes.cls.cpu().numpy().astype(int)
                    labels = [self.class_names[c] for c in classes]
                    
                    detections.append({
                        'boxes': boxes,
                        'scores': scores,
                        'classes': classes,
                        'labels': labels
                    })
                
                return detections
            else:
                # Custom implementation
                if isinstance(images, np.ndarray):
                    images = torch.from_numpy(images).float()
                
                if len(images.shape) == 3:
                    images = images.unsqueeze(0)
                
                images = images.to(self.device)
                
                # Forward pass
                outputs = self.model(images)
                
                # Post-process outputs (NMS, etc.)
                detections = self._post_process(outputs, conf_threshold, iou_threshold)
                
                return detections
    
    def _post_process(
        self,
        outputs: torch.Tensor,
        conf_threshold: float,
        iou_threshold: float
    ) -> List[Dict[str, np.ndarray]]:
        """Post-process model outputs"""
        # Placeholder - implement NMS and output parsing
        # This would extract boxes, scores, classes from raw outputs
        batch_size = outputs.shape[0]
        
        detections = []
        for i in range(batch_size):
            detections.append({
                'boxes': np.zeros((0, 4)),
                'scores': np.zeros(0),
                'classes': np.zeros(0, dtype=int),
                'labels': []
            })
        
        return detections
    
    def train_step(
        self,
        images: torch.Tensor,
        targets: List[Dict[str, torch.Tensor]]
    ) -> Dict[str, torch.Tensor]:
        """
        Single training step
        
        Args:
            images: Batch of images (B, C, H, W)
            targets: List of target dicts with 'boxes' and 'labels'
        
        Returns:
            Dict of losses
        """
        self.model.train()
        
        # Forward pass
        if hasattr(self.model, 'model'):
            # Ultralytics YOLO
            loss_dict = self.model.model(images, targets)
        else:
            # Custom implementation
            predictions = self.model(images)
            loss_dict = self._compute_loss(predictions, targets)
        
        return loss_dict
    
    def _compute_loss(
        self,
        predictions: torch.Tensor,
        targets: List[Dict]
    ) -> Dict[str, torch.Tensor]:
        """Compute YOLO losses"""
        # Placeholder - implement YOLO loss computation
        # Should include: box loss, objectness loss, classification loss
        
        total_loss = torch.tensor(0.0, device=self.device)
        
        return {
            'total_loss': total_loss,
            'box_loss': torch.tensor(0.0, device=self.device),
            'obj_loss': torch.tensor(0.0, device=self.device),
            'cls_loss': torch.tensor(0.0, device=self.device)
        }
    
    def save(self, path: Union[str, Path]):
        """Save model weights"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        if hasattr(self.model, 'save'):
            self.model.save(str(path))
        else:
            torch.save(self.model.state_dict(), path)
    
    def load(self, path: Union[str, Path]):
        """Load model weights"""
        path = Path(path)
        
        if hasattr(self.model, 'load'):
            self.model = self.model.load(str(path))
        else:
            self.model.load_state_dict(torch.load(path))
    
    def to(self, device: str):
        """Move model to device"""
        self.device = device
        self.model.to(device)
        return self
    
    def eval(self):
        """Set model to evaluation mode"""
        self.model.eval()
    
    def set_class_names(self, class_names: List[str]):
        """Set class names"""
        self.class_names = class_names


if __name__ == "__main__":
    # Example usage
    detector = YOLODetector(
        variant="yolov8n",
        num_classes=5,
        pretrained=True,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    
    detector.set_class_names(["background", "soldier", "military_vehicle", "drone", "civilian_vehicle"])
    
    # Dummy input
    dummy_input = torch.randn(1, 3, 640, 640).to(detector.device)
    
    # Forward pass
    with torch.no_grad():
        output = detector(dummy_input)
    
    print(f"Model: {detector.variant}")
    print(f"Device: {detector.device}")
    print(f"Classes: {detector.num_classes}")
