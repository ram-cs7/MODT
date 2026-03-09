"""
Dynamic Temporal Enhanced Attention (DTEA) Module
Advanced attention mechanism for video object detection with temporal awareness

Implements:
1. Temporal feature aggregation across frames
2. Dynamic attention weighting
3. Multi-scale temporal fusion
4. Motion-aware feature enhancement
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional
import numpy as np


class TemporalAttention(nn.Module):
    """
    Temporal attention mechanism for video sequences
    Learns to weight frames based on their relevance
    """
    
    def __init__(self, feature_dim: int, num_heads: int = 8):
        """
        Args:
            feature_dim: Dimension of input features
            num_heads: Number of attention heads
        """
        super().__init__()
        
        self.feature_dim = feature_dim
        self.num_heads = num_heads
        self.head_dim = feature_dim // num_heads
        
        assert feature_dim % num_heads == 0, "feature_dim must be divisible by num_heads"
        
        # Query, Key, Value projections
        self.query = nn.Linear(feature_dim, feature_dim)
        self.key = nn.Linear(feature_dim, feature_dim)
        self.value = nn.Linear(feature_dim, feature_dim)
        
        # Output projection
        self.out_proj = nn.Linear(feature_dim, feature_dim)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, features: torch.Tensor, temporal_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            features: (B, T, C, H, W) - Batch, Time, Channels, Height, Width
            temporal_mask: Optional mask for valid frames
        
        Returns:
            Attended features: (B, T, C, H, W)
        """
        B, T, C, H, W = features.shape
        
        # Reshape to (B*H*W, T, C) for temporal attention
        features_flat = features.permute(0, 3, 4, 1, 2).reshape(B*H*W, T, C)
        
        # Multi-head attention
        Q = self.query(features_flat).view(B*H*W, T, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.key(features_flat).view(B*H*W, T, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.value(features_flat).view(B*H*W, T, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.head_dim)
        
        # Apply mask if provided
        if temporal_mask is not None:
            scores = scores.masked_fill(temporal_mask == 0, float('-inf'))
        
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        attn_output = torch.matmul(attn_weights, V)
        
        # Concatenate heads and project
        attn_output = attn_output.transpose(1, 2).contiguous().view(B*H*W, T, C)
        output = self.out_proj(attn_output)
        
        # Reshape back to (B, T, C, H, W)
        output = output.view(B, H, W, T, C).permute(0, 3, 4, 1, 2)
        
        return output


class DynamicTemporalFusion(nn.Module):
    """
    Dynamically fuses temporal features based on learned importance
    """
    
    def __init__(self, feature_dim: int, num_frames: int = 5):
        """
        Args:
            feature_dim: Feature dimension
            num_frames: Number of temporal frames to process
        """
        super().__init__()
        
        self.feature_dim = feature_dim
        self.num_frames = num_frames
        
        # Temporal weight network
        self.weight_net = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(feature_dim * num_frames, num_frames * 2),
            nn.ReLU(),
            nn.Linear(num_frames * 2, num_frames),
            nn.Softmax(dim=1)
        )
        
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: (B, T, C, H, W)
        
        Returns:
            Fused features: (B, C, H, W)
        """
        B, T, C, H, W = features.shape
        
        # Compute temporal weights
        features_for_weights = features.view(B, T*C, H, W)
        weights = self.weight_net(features_for_weights)  # (B, T)
        
        # Apply weights
        weights = weights.view(B, T, 1, 1, 1)
        weighted_features = features * weights
        
        # Fuse across time dimension
        fused = weighted_features.sum(dim=1)  # (B, C, H, W)
        
        return fused


class MotionEnhancedFeatures(nn.Module):
    """
    Enhances features with motion information using frame differences
    """
    
    def __init__(self, feature_dim: int):
        """
        Args:
            feature_dim: Feature dimension
        """
        super().__init__()
        
        # Motion feature extraction
        self.motion_conv = nn.Sequential(
            nn.Conv2d(feature_dim * 2, feature_dim, 3, 1, 1),
            nn.BatchNorm2d(feature_dim),
            nn.ReLU(),
            nn.Conv2d(feature_dim, feature_dim, 3, 1, 1),
            nn.BatchNorm2d(feature_dim),
            nn.ReLU()
        )
        
        # Feature fusion
        self.fusion = nn.Conv2d(feature_dim * 2, feature_dim, 1)
        
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: (B, T, C, H, W)
        
        Returns:
            Motion-enhanced features: (B, T, C, H, W)
        """
        B, T, C, H, W = features.shape
        
        enhanced_features = []
        
        for t in range(T):
            current_frame = features[:, t]
            
            if t == 0:
                # No previous frame, use current twice
                prev_frame = current_frame
            else:
                prev_frame = features[:, t-1]
            
            # Compute motion
            motion_input = torch.cat([current_frame, current_frame - prev_frame], dim=1)
            motion_features = self.motion_conv(motion_input)
            
            # Fuse original and motion features
            fused = self.fusion(torch.cat([current_frame, motion_features], dim=1))
            enhanced_features.append(fused)
        
        return torch.stack(enhanced_features, dim=1)


class DTEA(nn.Module):
    """
    Dynamic Temporal Enhanced Attention Module
    
    Combines temporal attention, dynamic fusion, and motion enhancement
    for improved video object detection
    """
    
    def __init__(
        self,
        feature_dim: int = 256,
        num_frames: int = 5,
        num_heads: int = 8,
        use_motion: bool = True
    ):
        """
        Args:
            feature_dim: Feature dimension
            num_frames: Number of temporal frames
            num_heads: Number of attention heads
            use_motion: Use motion-enhanced features
        """
        super().__init__()
        
        self.feature_dim = feature_dim
        self.num_frames = num_frames
        self.use_motion = use_motion
        
        # Temporal attention module
        self.temporal_attention = TemporalAttention(feature_dim, num_heads)
        
        # Dynamic temporal fusion
        self.temporal_fusion = DynamicTemporalFusion(feature_dim, num_frames)
        
        # Motion enhancement (optional)
        if use_motion:
            self.motion_enhancer = MotionEnhancedFeatures(feature_dim)
        
        # Feature refinement
        self.refinement = nn.Sequential(
            nn.Conv2d(feature_dim, feature_dim, 3, 1, 1),
            nn.BatchNorm2d(feature_dim),
            nn.ReLU(),
            nn.Conv2d(feature_dim, feature_dim, 3, 1, 1),
            nn.BatchNorm2d(feature_dim)
        )
        
    def forward(
        self,
        features: torch.Tensor,
        temporal_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            features: (B, T, C, H, W) - Temporal feature sequence
            temporal_mask: Optional temporal mask
        
        Returns:
            Tuple of:
                - Fused features: (B, C, H, W)
                - Attention weights: (B, T)
        """
        # Motion enhancement
        if self.use_motion:
            features = self.motion_enhancer(features)
        
        # Temporal attention
        attended_features = self.temporal_attention(features, temporal_mask)
        
        # Dynamic temporal fusion
        fused_features = self.temporal_fusion(attended_features)
        
        # Feature refinement
        refined_features = self.refinement(fused_features)
        
        # Skip connection
        fused_features = fused_features + refined_features
        
        # Extract attention weights for visualization
        with torch.no_grad():
            B, T, C, H, W = features.shape
            features_flat = features.permute(0, 3, 4, 1, 2).reshape(B*H*W, T, C)
            features_pooled = features_flat.mean(dim=2)  # (B*H*W, T)
            attention_weights = F.softmax(features_pooled, dim=1).view(B, H, W, T).mean(dim=[1, 2])
        
        return fused_features, attention_weights


class DTEADetector(nn.Module):
    """
    DTEA-enhanced object detector
    Wraps base detector with temporal attention
    """
    
    def __init__(
        self,
        base_detector: nn.Module,
        feature_dim: int = 256,
        num_frames: int = 5
    ):
        """
        Args:
            base_detector: Base detection model
            feature_dim: Feature dimension from detector
            num_frames: Number of frames for temporal processing
        """
        super().__init__()
        
        self.base_detector = base_detector
        self.dtea = DTEA(feature_dim, num_frames)
        self.num_frames = num_frames
        
        # Frame buffer for temporal processing
        self.frame_buffer = []
        
    def forward(self, x: torch.Tensor, use_temporal: bool = True) -> torch.Tensor:
        """
        Args:
            x: Input frame(s)
            use_temporal: Whether to use temporal processing
        
        Returns:
            Detection predictions
        """
        # Extract features from base detector
        features = self.base_detector.extract_features(x)
        
        if use_temporal and len(self.frame_buffer) > 0:
            # Add to buffer
            self.frame_buffer.append(features)
            
            # Keep only last N frames
            if len(self.frame_buffer) > self.num_frames:
                self.frame_buffer.pop(0)
            
            # Stack temporal features
            if len(self.frame_buffer) == self.num_frames:
                temporal_features = torch.stack(self.frame_buffer, dim=1)
                
                # Apply DTEA
                enhanced_features, _ = self.dtea(temporal_features)
                
                # Use enhanced features for detection
                predictions = self.base_detector.predict_from_features(enhanced_features)
            else:
                # Not enough frames yet, use current features
                predictions = self.base_detector.predict_from_features(features)
        else:
            # Single frame mode
            predictions = self.base_detector.predict_from_features(features)
        
        return predictions
    
    def reset_buffer(self):
        """Reset temporal buffer"""
        self.frame_buffer = []


if __name__ == "__main__":
    # Example usage
    print("="*80)
    print("Dynamic Temporal Enhanced Attention (DTEA) Module")
    print("="*80)
    
    # Create dummy temporal features
    B, T, C, H, W = 2, 5, 256, 20, 20
    features = torch.randn(B, T, C, H, W)
    
    # Initialize DTEA
    dtea = DTEA(feature_dim=C, num_frames=T, num_heads=8)
    
    # Forward pass
    fused_features, attention_weights = dtea(features)
    
    print(f"Input shape: {features.shape}")
    print(f"Output shape: {fused_features.shape}")
    print(f"Attention weights shape: {attention_weights.shape}")
    print(f"Attention weights: {attention_weights[0]}")
    print("="*80)
    print("DTEA Module Test Successful!")
    print("="*80)
