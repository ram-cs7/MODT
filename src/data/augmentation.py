"""
Advanced data augmentation pipeline for military object detection
Uses Albumentations for efficient and diverse augmentations
"""

import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import numpy as np
from typing import Tuple, Optional


class AugmentationPipeline:
    """
    Augmentation pipeline builder for detection tasks
    """
    
    @staticmethod
    def get_training_transforms(
        img_size: Tuple[int, int] = (640, 640),
        brightness_limit: float = 0.2,
        contrast_limit: float = 0.2,
        saturation_limit: float = 0.2,
        hue_limit: float = 0.1,
        flip_prob: float = 0.5,
        rotate_limit: int = 10,
        blur_prob: float = 0.1,
        noise_prob: float = 0.05
    ) -> A.Compose:
        """
        Get training augmentation pipeline
        
        Args:
            img_size: Target image size (height, width)
            brightness_limit: Max change in brightness
            contrast_limit: Max change in contrast
            saturation_limit: Max change in saturation
            hue_limit: Max change in hue
            flip_prob: Probability of horizontal flip
            rotate_limit: Max rotation angle in degrees
            blur_prob: Probability of applying blur
            noise_prob: Probability of adding noise
        
        Returns:
            Albumentations Compose object
        """
        return A.Compose([
            # Geometric transformations
            A.HorizontalFlip(p=flip_prob),
            A.ShiftScaleRotate(
                shift_limit=0.1,
                scale_limit=0.2,
                rotate_limit=rotate_limit,
                border_mode=cv2.BORDER_CONSTANT,
                value=114,
                p=0.5
            ),
            
            # Photometric transformations
            A.ColorJitter(
                brightness=brightness_limit,
                contrast=contrast_limit,
                saturation=saturation_limit,
                hue=hue_limit,
                p=0.5
            ),
            
            A.OneOf([
                A.MotionBlur(blur_limit=3, p=1.0),
                A.MedianBlur(blur_limit=3, p=1.0),
                A.GaussianBlur(blur_limit=3, p=1.0),
            ], p=blur_prob),
            
            # Noise
            A.OneOf([
                A.GaussNoise(var_limit=(10.0, 50.0), p=1.0),
                A.ISONoise(p=1.0),
            ], p=noise_prob),
            
            # Weather and environmental effects
            A.OneOf([
                A.RandomFog(fog_coef_lower=0.1, fog_coef_upper=0.3, p=1.0),
                A.RandomRain(
                    slant_lower=-10,
                    slant_upper=10,
                    drop_length=10,
                    drop_width=1,
                    p=1.0
                ),
                A.RandomShadow(p=1.0),
            ], p=0.1),
            
            # Cutout/Coarse Dropout for robustness
            A.CoarseDropout(
                max_holes=8,
                max_height=32,
                max_width=32,
                min_holes=1,
                fill_value=114,
                p=0.3
            ),
            
        ], bbox_params=A.BboxParams(
            format='yolo',
            label_fields=['class_labels'],
            min_area=16,
            min_visibility=0.3
        ))
    
    @staticmethod
    def get_validation_transforms(
        img_size: Tuple[int, int] = (640, 640)
    ) -> A.Compose:
        """
        Get validation/test augmentation pipeline (minimal/none)
        
        Args:
            img_size: Target image size
        
        Returns:
            Albumentations Compose object
        """
        return A.Compose([
            # No augmentation for validation, just normalization
        ], bbox_params=A.BboxParams(
            format='yolo',
            label_fields=['class_labels']
        ))
    
    @staticmethod
    def get_inference_transforms(
        img_size: Tuple[int, int] = (640, 640)
    ) -> A.Compose:
        """
        Get inference-time transforms
        
        Args:
            img_size: Target image size
        
        Returns:
            Albumentations Compose object
        """
        return A.Compose([
            # Minimal transforms for inference
        ])
    
    @staticmethod
    def get_thermal_transforms(
        img_size: Tuple[int, int] = (640, 640)
    ) -> A.Compose:
        """
        Get augmentation pipeline for thermal/infrared images
        
        Args:
            img_size: Target image size
        
        Returns:
            Albumentations Compose object
        """
        return A.Compose([
            A.HorizontalFlip(p=0.5),
            A.ShiftScaleRotate(
                shift_limit=0.1,
                scale_limit=0.2,
                rotate_limit=10,
                border_mode=cv2.BORDER_CONSTANT,
                value=0,
                p=0.5
            ),
            
            # Thermal-specific augmentations
            A.RandomBrightnessContrast(
                brightness_limit=0.3,
                contrast_limit=0.3,
                p=0.5
            ),
            
            A.GaussNoise(var_limit=(5.0, 25.0), p=0.3),
            
            # Simulate thermal sensor artifacts
            A.OneOf([
                A.MotionBlur(blur_limit=3, p=1.0),
                A.GaussianBlur(blur_limit=3, p=1.0),
            ], p=0.2),
            
        ], bbox_params=A.BboxParams(
            format='yolo',
            label_fields=['class_labels'],
            min_area=16,
            min_visibility=0.3
        ))


class MosaicAugmentation:
    """
    Mosaic augmentation - combines 4 images into one
    Popular in YOLO training for improving small object detection
    """
    
    def __init__(self, img_size: Tuple[int, int] = (640, 640), prob: float = 0.5):
        """
        Args:
            img_size: Target image size
            prob: Probability of applying mosaic
        """
        self.img_size = img_size
        self.prob = prob
    
    def __call__(
        self,
        images: list,
        labels_list: list
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply mosaic augmentation to 4 images
        
        Args:
            images: List of 4 images
            labels_list: List of 4 label arrays
        
        Returns:
            Mosaic image and combined labels
        """
        if np.random.random() > self.prob or len(images) < 4:
            return images[0], labels_list[0]
        
        h, w = self.img_size
        
        # Create mosaic image
        mosaic_img = np.full((h, w, 3), 114, dtype=np.uint8)
        
        # Divide into 4 quadrants
        center_x = w // 2
        center_y = h // 2
        
        mosaic_labels = []
        
        # Place 4 images in quadrants
        quadrants = [
            (0, 0, center_x, center_y),              # Top-left
            (center_x, 0, w, center_y),              # Top-right
            (0, center_y, center_x, h),              # Bottom-left
            (center_x, center_y, w, h)               # Bottom-right
        ]
        
        for i, (img, labels) in enumerate(zip(images[:4], labels_list[:4])):
            x1, y1, x2, y2 = quadrants[i]
            quad_w, quad_h = x2 - x1, y2 - y1
            
            # Resize image to fit quadrant
            img_resized = cv2.resize(img, (quad_w, quad_h))
            mosaic_img[y1:y2, x1:x2] = img_resized
            
            # Adjust labels for new position
            if len(labels) > 0:
                labels_adjusted = labels.copy()
                # Convert from normalized to pixel coordinates
                labels_adjusted[:, 1] = (labels[:, 1] * quad_w + x1) / w
                labels_adjusted[:, 2] = (labels[:, 2] * quad_h + y1) / h
                labels_adjusted[:, 3] = labels[:, 3] * quad_w / w
                labels_adjusted[:, 4] = labels[:, 4] * quad_h / h
                
                mosaic_labels.append(labels_adjusted)
        
        if mosaic_labels:
            mosaic_labels = np.vstack(mosaic_labels)
        else:
            mosaic_labels = np.zeros((0, 5))
        
        return mosaic_img, mosaic_labels


class MixUpAugmentation:
    """
    MixUp augmentation - blends two images together
    """
    
    def __init__(self, alpha: float = 0.5, prob: float = 0.1):
        """
        Args:
            alpha: Blending factor
            prob: Probability of applying mixup
        """
        self.alpha = alpha
        self.prob = prob
    
    def __call__(
        self,
        img1: np.ndarray,
        labels1: np.ndarray,
        img2: np.ndarray,
        labels2: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply mixup to two images
        
        Args:
            img1: First image
            labels1: First image labels
            img2: Second image
            labels2: Second image labels
        
        Returns:
            Mixed image and combined labels
        """
        if np.random.random() > self.prob:
            return img1, labels1
        
        # Random mixing coefficient
        lam = np.random.beta(self.alpha, self.alpha)
        
        # Mix images
        mixed_img = (lam * img1 + (1 - lam) * img2).astype(np.uint8)
        
        # Combine labels
        mixed_labels = np.vstack([labels1, labels2]) if len(labels1) > 0 and len(labels2) > 0 else labels1
        
        return mixed_img, mixed_labels


if __name__ == "__main__":
    # Example usage
    aug_pipeline = AugmentationPipeline()
    
    # Get training transforms
    train_transforms = aug_pipeline.get_training_transforms(img_size=(640, 640))
    
    # Example image and bounding boxes
    image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
    bboxes = [[0.5, 0.5, 0.2, 0.2]]  # YOLO format: x_center, y_center, width, height
    class_labels = [1]
    
    # Apply augmentation
    augmented = train_transforms(image=image, bboxes=bboxes, class_labels=class_labels)
    
    print("Original image shape:", image.shape)
    print("Augmented image shape:", augmented['image'].shape)
    print("Original bboxes:", bboxes)
    print("Augmented bboxes:", augmented['bboxes'])
