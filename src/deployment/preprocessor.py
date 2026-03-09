"""
Preprocessing Module
Handles image resizing, normalization, and batch preparation.
"""

import cv2
import numpy as np
from typing import Tuple

class Preprocessor:
    """Image preprocessing utilities"""
    
    @staticmethod
    def letterbox(img: np.ndarray, new_shape: Tuple[int, int] = (640, 640), color: Tuple[int, int, int] = (114, 114, 114)) -> Tuple[np.ndarray, float, Tuple[float, float]]:
        """
        Resize image to a 32-pixel-multiple rectangle using Letterbox method
        
        Args:
            img: Input image
            new_shape: Target size (h, w)
            color: Padding color
        
        Returns:
            Resized image, ratio, (dw, dh)
        """
        shape = img.shape[:2]  # current shape [height, width]
        
        # Scale ratio (new / old)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        
        # Compute padding
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
        
        dw /= 2  # divide padding into 2 sides
        dh /= 2
        
        if shape[::-1] != new_unpad:  # resize
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
            
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
        
        return img, r, (dw, dh)

    @staticmethod
    def normalize(img: np.ndarray) -> np.ndarray:
        """Normalize image 0-255 to 0-1 and BGR to RGB"""
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img, dtype=np.float32)  # uint8 to float32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        return img
