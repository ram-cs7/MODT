"""
Model Exporter
Export PyTorch models to ONNX, TensorRT, and other formats for edge deployment
"""

import torch
import torch.nn as nn
from pathlib import Path
from typing import Tuple, Optional, List
import subprocess

from src.utils.logger import setup_logger


class ModelExporter:
    """
    Export models to various formats for deployment
    """
    
    def __init__(self):
        """Initialize exporter"""
        self.logger = setup_logger("ModelExporter")
    
    def export_to_onnx(
        self,
        model: nn.Module,
        output_path: str,
        input_size: Tuple[int, int] = (640, 640),
        opset_version: int = 12,
        simplify: bool = True,
        dynamic_axes: bool = False
    ) -> bool:
        """
        Export PyTorch model to ONNX format
        
        Args:
            model: PyTorch model
            output_path: Path to save ONNX model
            input_size: Input image size (H, W)
            opset_version: ONNX opset version
            simplify: Simplify ONNX model using onnx-simplifier
            dynamic_axes: Enable dynamic batch size
        
        Returns:
            True if export successful
        """
        try:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            self.logger.info(f"Exporting model to ONNX: {output_path}")
            
            # Set model to eval mode
            model.eval()
            
            # Create dummy input
            dummy_input = torch.randn(1, 3, input_size[0], input_size[1])
            
            # Dynamic axes configuration
            if dynamic_axes:
                dynamic_axes_dict = {
                    'input': {0: 'batch_size'},
                    'output': {0: 'batch_size'}
                }
            else:
                dynamic_axes_dict = None
            
            # Export to ONNX
            torch.onnx.export(
                model,
                dummy_input,
                str(output_path),
                export_params=True,
                opset_version=opset_version,
                do_constant_folding=True,
                input_names=['input'],
                output_names=['output'],
                dynamic_axes=dynamic_axes_dict,
                verbose=False
            )
            
            self.logger.info(f"ONNX export successful: {output_path}")
            
            # Simplify ONNX model
            if simplify:
                self.simplify_onnx(output_path)
            
            # Verify ONNX model
            self.verify_onnx(output_path)
            
            return True
            
        except Exception as e:
            self.logger.error(f"ONNX export failed: {e}")
            return False
    
    def simplify_onnx(self, onnx_path: str) -> bool:
        """
        Simplify ONNX model using onnx-simplifier
        
        Args:
            onnx_path: Path to ONNX model
        
        Returns:
            True if simplification successful
        """
        try:
            import onnx
            from onnxsim import simplify
            
            self.logger.info("Simplifying ONNX model...")
            
            # Load ONNX model
            model = onnx.load(onnx_path)
            
            # Simplify
            model_simp, check = simplify(model)
            
            if not check:
                self.logger.warning("Simplified ONNX model validation failed")
                return False
            
            # Save simplified model
            onnx.save(model_simp, onnx_path)
            
            self.logger.info("ONNX model simplified successfully")
            return True
            
        except ImportError:
            self.logger.warning("onnx-simplifier not installed. Skipping simplification.")
            return False
        except Exception as e:
            self.logger.error(f"ONNX simplification failed: {e}")
            return False
    
    def verify_onnx(self, onnx_path: str) -> bool:
        """
        Verify ONNX model
        
        Args:
            onnx_path: Path to ONNX model
        
        Returns:
            True if model is valid
        """
        try:
            import onnx
            
            self.logger.info("Verifying ONNX model...")
            
            # Load and check model
            model = onnx.load(onnx_path)
            onnx.checker.check_model(model)
            
            self.logger.info("ONNX model is valid")
            return True
            
        except Exception as e:
            self.logger.error(f"ONNX verification failed: {e}")
            return False
    
    def export_to_tensorrt(
        self,
        onnx_path: str,
        output_path: str,
        precision: str = "fp16",
        workspace_size: int = 4,
        max_batch_size: int = 1
    ) -> bool:
        """
        Export ONNX model to TensorRT engine
        
        Args:
            onnx_path: Path to ONNX model
            output_path: Path to save TensorRT engine
            precision: Precision mode ('fp32', 'fp16', 'int8')
            workspace_size: Workspace size in GB
            max_batch_size: Maximum batch size
        
        Returns:
            True if export successful
        """
        try:
            self.logger.info(f"Converting ONNX to TensorRT ({precision})...")
            
            # Build trtexec command
            cmd = [
                'trtexec',
                f'--onnx={onnx_path}',
                f'--saveEngine={output_path}',
                f'--workspace={workspace_size * 1024}',  # Convert to MB
                f'--minShapes=input:1x3x640x640',
                f'--optShapes=input:{max_batch_size}x3x640x640',
                f'--maxShapes=input:{max_batch_size}x3x640x640',
            ]
            
            # Add precision flags
            if precision == 'fp16':
                cmd.append('--fp16')
            elif precision == 'int8':
                cmd.append('--int8')
            
            # Run trtexec
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                self.logger.info(f"TensorRT engine saved: {output_path}")
                return True
            else:
                self.logger.error(f"TensorRT conversion failed: {result.stderr}")
                return False
                
        except FileNotFoundError:
            self.logger.error("trtexec not found. Make sure TensorRT is installed.")
            return False
        except Exception as e:
            self.logger.error(f"TensorRT export failed: {e}")
            return False
    
    def export_to_tflite(
        self,
        model: nn.Module,
        output_path: str,
        input_size: Tuple[int, int] = (640, 640),
        quantize: bool = False
    ) -> bool:
        """
        Export to TensorFlow Lite format
        
        Args:
            model: PyTorch model
            output_path: Path to save TFLite model
            input_size: Input size
            quantize: Apply quantization
        
        Returns:
            True if export successful
        """
        self.logger.warning("TFLite export requires onnx2tf. Not implemented in this version.")
        return False


if __name__ == "__main__":
    # Example usage
    exporter = ModelExporter()
    
    # Create dummy model
    model = torch.nn.Sequential(
        torch.nn.Conv2d(3, 64, 3, 1, 1),
        torch.nn.ReLU(),
        torch.nn.AdaptiveAvgPool2d(1)
    )
    
    # Export to ONNX
    exporter.export_to_onnx(
        model,
        "model.onnx",
        input_size=(640, 640),
        simplify=True
    )
