"""
Model Quantization Module
Implements INT8 and FP16 quantization for model compression
"""

import torch
import torch.nn as nn
from pathlib import Path
from typing import Optional, Tuple
import numpy as np

from src.utils.logger import setup_logger


class ModelQuantizer:
    """
    Model quantization for edge deployment
    Supports INT8 and FP16 quantization
    """
    
    def __init__(self):
        """Initialize quantizer"""
        self.logger = setup_logger("Quantizer")
    
    def quantize_dynamic(
        self,
        model: nn.Module,
        output_path: str,
        dtype: torch.dtype = torch.qint8
    ) -> nn.Module:
        """
        Apply dynamic quantization (INT8)
        
        Args:
            model: PyTorch model
            output_path: Path to save quantized model
            dtype: Quantization dtype
        
        Returns:
            Quantized model
        """
        self.logger.info("Applying dynamic quantization...")
        
        # Dynamic quantization (runtime quantization)
        quantized_model = torch.quantization.quantize_dynamic(
            model,
            {nn.Linear, nn.Conv2d},
            dtype=dtype
        )
        
        # Save quantized model
        torch.save(quantized_model.state_dict(), output_path)
        self.logger.info(f"Quantized model saved: {output_path}")
        
        # Calculate compression ratio
        original_size = sum(p.numel() * p.element_size() for p in model.parameters())
        quantized_size = sum(p.numel() * p.element_size() for p in quantized_model.parameters())
        compression_ratio = original_size / quantized_size
        
        self.logger.info(f"Original size: {original_size / 1e6:.2f} MB")
        self.logger.info(f"Quantized size: {quantized_size / 1e6:.2f} MB")
        self.logger.info(f"Compression ratio: {compression_ratio:.2f}x")
        
        return quantized_model
    
    def calibrate_int8(
        self,
        model: nn.Module,
        calibration_data: torch.utils.data.DataLoader,
        num_batches: int = 100
    ) -> nn.Module:
        """
        Calibrate model for INT8 quantization
        
        Args:
            model: PyTorch model
            calibration_data: Calibration dataset
            num_batches: Number of batches for calibration
        
        Returns:
            Calibrated model
        """
        self.logger.info("Calibrating for INT8 quantization...")
        
        model.eval()
        
        # Prepare model for quantization
        model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
        torch.quantization.prepare(model, inplace=True)
        
        # Calibration
        with torch.no_grad():
            for i, (images, _, _) in enumerate(calibration_data):
                if i >= num_batches:
                    break
                model(images)
        
        # Convert to quantized model
        torch.quantization.convert(model, inplace=True)
        
        self.logger.info("INT8 calibration complete")
        return model
    
    def quantize_qat(
        self,
        model: nn.Module,
        train_loader: torch.utils.data.DataLoader,
        num_epochs: int = 1,
        learning_rate: float = 1e-4
    ) -> nn.Module:
        """
        Perform Quantization Aware Training (QAT)
        
        Args:
            model: PyTorch model
            train_loader: Training data
            num_epochs: Number of QAT epochs
            learning_rate: Learning rate for QAT
        
        Returns:
            Quantized model
        """
        self.logger.info("Starting Quantization Aware Training (QAT)...")
        
        # 1. Fuse modules (Conv+BN+ReLU) - Simplified for generic model
        model.eval()
        model.qconfig = torch.quantization.get_default_qat_qconfig('fbgemm')
        torch.quantization.prepare_qat(model, inplace=True)
        
        # 2. Training loop
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        criterion = nn.MSELoss() # Simplified loss
        
        model.train()
        for epoch in range(num_epochs):
            total_loss = 0
            for i, (images, _, _) in enumerate(train_loader):
                if i > 100: break # Limit for demo
                
                optimizer.zero_grad()
                outputs = model(images)
                
                # Dummy target for QAT demo (in real usage, pass labels)
                targets = torch.zeros_like(outputs) 
                loss = criterion(outputs, targets)
                
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                
            self.logger.info(f"QAT Epoch {epoch+1}/{num_epochs}, Loss: {total_loss:.4f}")
            
        # 3. Convert
        model.eval()
        quantized_model = torch.quantization.convert(model, inplace=True)
        
        self.logger.info("QAT complete")
        return quantized_model
    
    def convert_to_fp16(
        self,
        model: nn.Module,
        output_path: str
    ) -> nn.Module:
        """
        Convert model to FP16 precision
        
        Args:
            model: PyTorch model
            output_path: Path to save FP16 model
        
        Returns:
            FP16 model
        """
        self.logger.info("Converting to FP16...")
        
        # Convert to half precision
        model_fp16 = model.half()
        
        # Save
        torch.save(model_fp16.state_dict(), output_path)
        self.logger.info(f"FP16 model saved: {output_path}")
        
        return model_fp16


if __name__ == "__main__":
    # Example usage
    quantizer = ModelQuantizer()
    
    # Create dummy model
    model = nn.Sequential(
        nn.Conv2d(3, 64, 3, 1, 1),
        nn.ReLU(),
        nn.Conv2d(64, 128, 3, 1, 1),
        nn.ReLU()
    )
    
    # Dynamic quantization
    quantized = quantizer.quantize_dynamic(model, "model_int8.pth")
    print("Quantization test successful!")
