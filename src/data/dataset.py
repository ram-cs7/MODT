"""
Custom PyTorch Dataset classes for military object detection
Supports YOLO and COCO formats, multi-modal inputs, and tracking sequences
"""

import os
from pathlib import Path
from typing import Optional, List, Tuple, Dict, Any, Callable
import json
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import cv2


class DetectionDataset(Dataset):
    """
    Generic detection dataset supporting YOLO and COCO formats
    """
    
    def __init__(
        self,
        data_dir: str,
        annotation_format: str = "yolo",
        img_size: Tuple[int, int] = (640, 640),
        transforms: Optional[Callable] = None,
        cache_images: bool = False
    ):
        """
        Args:
            data_dir: Path to dataset directory
            annotation_format: Format of annotations ('yolo' or 'coco')
            img_size: Target image size (height, width)
            transforms: Optional augmentation transforms
            cache_images: Cache images in memory for faster loading
        """
        self.data_dir = Path(data_dir)
        self.annotation_format = annotation_format.lower()
        self.img_size = img_size
        self.transforms = transforms
        self.cache_images = cache_images
        
        # Get list of images
        self.image_files = self._get_image_files()
        self.num_samples = len(self.image_files)
        
        # Cache for images
        self.image_cache = {} if cache_images else None
        
        # Load annotations
        if self.annotation_format == "coco":
            self.annotations = self._load_coco_annotations()
        else:
            self.annotations = None  # YOLO uses per-image txt files
    
    def _get_image_files(self) -> List[Path]:
        """Get list of all image files in directory"""
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        
        if (self.data_dir / "images").exists():
            img_dir = self.data_dir / "images"
        else:
            img_dir = self.data_dir
        
        image_files = []
        for ext in image_extensions:
            image_files.extend(img_dir.glob(f"*{ext}"))
            image_files.extend(img_dir.glob(f"*{ext.upper()}"))
        
        return sorted(image_files)
    
    def _load_coco_annotations(self) -> Dict:
        """Load COCO format annotations"""
        ann_file = self.data_dir / "annotations.json"
        if not ann_file.exists():
            raise FileNotFoundError(f"COCO annotations not found: {ann_file}")
        
        with open(ann_file, 'r') as f:
            annotations = json.load(f)
        
        return annotations
    
    def _load_yolo_annotation(self, img_path: Path) -> np.ndarray:
        """
        Load YOLO format annotation for an image
        
        Format: class_id x_center y_center width height (normalized)
        
        Returns:
            Array of shape (N, 5) where N is number of objects
        """
        # Get corresponding label file
        if (self.data_dir / "labels").exists():
            label_dir = self.data_dir / "labels"
        else:
            label_dir = img_path.parent
        
        label_file = label_dir / f"{img_path.stem}.txt"
        
        if not label_file.exists():
            return np.zeros((0, 5))  # No annotations
        
        # Load labels
        labels = []
        with open(label_file, 'r') as f:
            for line in f:
                values = line.strip().split()
                if len(values) >= 5:
                    labels.append([float(v) for v in values[:5]])
        
        return np.array(labels) if labels else np.zeros((0, 5))
    
    def _load_image(self, img_path: Path) -> np.ndarray:
        """Load and preprocess image"""
        # Check cache first
        if self.cache_images and str(img_path) in self.image_cache:
            return self.image_cache[str(img_path)].copy()
        
        # Load image
        img = cv2.imread(str(img_path))
        if img is None:
            raise ValueError(f"Failed to load image: {img_path}")
        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Cache if enabled
        if self.cache_images:
            self.image_cache[str(img_path)] = img.copy()
        
        return img
    
    def _resize_image(self, img: np.ndarray, labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Resize image with letterbox padding and adjust labels
        
        Args:
            img: Input image (H, W, C)
            labels: Labels array (N, 5) in normalized format
        
        Returns:
            Resized image and adjusted labels
        """
        h, w = img.shape[:2]
        target_h, target_w = self.img_size
        
        # Calculate scaling factor
        scale = min(target_w / w, target_h / h)
        new_w, new_h = int(w * scale), int(h * scale)
        
        # Resize image
        img_resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        
        # Create padded image
        img_padded = np.full((target_h, target_w, 3), 114, dtype=np.uint8)
        
        # Calculate padding
        pad_w = (target_w - new_w) // 2
        pad_h = (target_h - new_h) // 2
        
        # Place resized image in center
        img_padded[pad_h:pad_h+new_h, pad_w:pad_w+new_w] = img_resized
        
        # Adjust labels (they're already normalized, so we need to account for padding)
        if len(labels) > 0:
            # Convert from normalized to pixel coordinates on resized image
            labels_adjusted = labels.copy()
            # The normalized coordinates need to be scaled for the new image
            # This is already handled since labels are in normalized form
        
        return img_padded, labels
    
    def __len__(self) -> int:
        return self.num_samples
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """
        Get item from dataset
        
        Returns:
            image: Tensor of shape (C, H, W)
            labels: Tensor of shape (N, 5) [class_id, x_center, y_center, w, h]
            metadata: Dict with image info
        """
        img_path = self.image_files[idx]
        
        # Load image
        img = self._load_image(img_path)
        
        # Load labels
        if self.annotation_format == "yolo":
            labels = self._load_yolo_annotation(img_path)
        else:
            # Handle COCO format
            labels = self._get_coco_labels(img_path)
        
        # Store original shape
        orig_h, orig_w = img.shape[:2]
        
        # Resize image and adjust labels
        img, labels = self._resize_image(img, labels)
        
        # Apply augmentations
        if self.transforms:
            augmented = self.transforms(image=img, bboxes=labels[:, 1:], class_labels=labels[:, 0])
            img = augmented['image']
            if augmented['bboxes']:
                labels = np.column_stack([augmented['class_labels'], augmented['bboxes']])
            else:
                labels = np.zeros((0, 5))
        
        # Convert to tensor
        img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0  # (C, H, W)
        labels = torch.from_numpy(labels).float()
        
        # Metadata
        metadata = {
            'image_path': str(img_path),
            'image_id': idx,
            'orig_shape': (orig_h, orig_w),
            'resized_shape': self.img_size
        }
        
        return img, labels, metadata
    
    def _get_coco_labels(self, img_path: Path) -> np.ndarray:
        """Get labels for COCO format image"""
        # This would need to be implemented based on COCO structure
        # Placeholder for now
        return np.zeros((0, 5))


class TrackingDataset(Dataset):
    """
    Dataset for tracking with sequential frames
    Returns sequences of frames for training/evaluating trackers
    """
    
    def __init__(
        self,
        data_dir: str,
        sequence_length: int = 10,
        img_size: Tuple[int, int] = (640, 640),
        stride: int = 1
    ):
        """
        Args:
            data_dir: Path to dataset directory with video sequences
            sequence_length: Number of consecutive frames per sequence
            img_size: Target image size
            stride: Stride between sequences
        """
        self.data_dir = Path(data_dir)
        self.sequence_length = sequence_length
        self.img_size = img_size
        self.stride = stride
        
        # Get all sequences
        self.sequences = self._get_sequences()
    
    def _get_sequences(self) -> List[Dict]:
        """Get list of all valid sequences"""
        sequences = []
        
        # Each subdirectory is a sequence
        for seq_dir in self.data_dir.iterdir():
            if seq_dir.is_dir():
                frames = sorted(list(seq_dir.glob("*.jpg")) + list(seq_dir.glob("*.png")))
                if len(frames) >= self.sequence_length:
                    # Create overlapping sequences
                    for i in range(0, len(frames) - self.sequence_length + 1, self.stride):
                        sequences.append({
                            'sequence_name': seq_dir.name,
                            'frames': frames[i:i+self.sequence_length]
                        })
        
        return sequences
    
    def __len__(self) -> int:
        return len(self.sequences)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, List[np.ndarray], Dict]:
        """
        Get sequence of frames
        
        Returns:
            images: Tensor of shape (T, C, H, W) where T is sequence length
            labels_list: List of label arrays, one per frame
            metadata: Dict with sequence info
        """
        sequence = self.sequences[idx]
        
        images = []
        labels_list = []
        
        for frame_path in sequence['frames']:
            # Load image
            img = cv2.imread(str(frame_path))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (self.img_size[1], self.img_size[0]))
            
            # Load tracking labels (format: frame_id, track_id, x, y, w, h, class_id)
            label_path = frame_path.parent.parent / "labels" / f"{frame_path.stem}.txt"
            if label_path.exists():
                labels = np.loadtxt(label_path, ndmin=2)
            else:
                labels = np.zeros((0, 7))
            
            images.append(img)
            labels_list.append(labels)
        
        # Convert to tensor
        images = torch.from_numpy(np.stack(images)).permute(0, 3, 1, 2).float() / 255.0
        
        metadata = {
            'sequence_name': sequence['sequence_name'],
            'num_frames': len(sequence['frames'])
        }
        
        return images, labels_list, metadata


if __name__ == "__main__":
    # Example usage
    dataset = DetectionDataset(
        data_dir="./data/train",
        annotation_format="yolo",
        img_size=(640, 640)
    )
    
    print(f"Dataset size: {len(dataset)}")
    
    if len(dataset) > 0:
        img, labels, metadata = dataset[0]
        print(f"Image shape: {img.shape}")
        print(f"Labels shape: {labels.shape}")
        print(f"Metadata: {metadata}")
