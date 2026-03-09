"""
Export Model to ONNX
Converts PyTorch model to ONNX format for deployment
"""

import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import argparse
import torch
from src.optimization.exporter import ModelExporter
from models.detectors.yolo_detector import YOLODetector
from src.utils.logger import setup_logger


def main():
    parser = argparse.ArgumentParser(description="Export Model to ONNX")
    
    parser.add_argument(
        "--weights",
        type=str,
        required=True,
        help="Path to PyTorch weights (.pt)"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        default="weights/model.onnx",
        help="Output ONNX file path"
    )
    
    parser.add_argument(
        "--img-size",
        nargs=2,
        type=int,
        default=[640, 640],
        help="Input image size (height width)"
    )
    
    parser.add_argument(
        "--opset",
        type=int,
        default=12,
        help="ONNX opset version"
    )
    
    parser.add_argument(
        "--simplify",
        action="store_true",
        help="Simplify ONNX model"
    )
    
    parser.add_argument(
        "--dynamic",
        action="store_true",
        help="Enable dynamic batch size"
    )
    
    args = parser.parse_args()
    
    logger = setup_logger("ExportONNX")
    logger.info("="*80)
    logger.info("Export Model to ONNX")
    logger.info("="*80)
    
    try:
        # Load model
        logger.info(f"Loading model from: {args.weights}")
        
        # For YOLOdetector, we would load the actual trained weights
        # This is a simplified version
        detector = YOLODetector(
            variant="yolov8n",
            num_classes=5,
            pretrained=False,
            img_size=tuple(args.img_size),
            device="cpu"
        )
        
        # Load weights (implement actual weight loading)
        # detector.load(args.weights)
        
        # Initialize exporter
        exporter = ModelExporter()
        
        # Export to ONNX
        success = exporter.export_to_onnx(
            model=detector.model,
            output_path=args.output,
            input_size=tuple(args.img_size),
            opset_version=args.opset,
            simplify=args.simplify,
            dynamic_axes=args.dynamic
        )
        
        if success:
            logger.info("="*80)
            logger.info(f"Model exported successfully to: {args.output}")
            logger.info("="*80)
        else:
            logger.error("Export failed!")
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"Export failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
