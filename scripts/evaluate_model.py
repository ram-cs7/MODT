"""
Evaluate Model Script
Standalone script for comprehensive model evaluation (mAP, Precision, Recall).
"""
import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from src.utils.config import load_config
from src.detection.evaluator import DetectionEvaluator
from src.utils.logger import setup_logger

def main():
    parser = argparse.ArgumentParser(description="Evaluate Detector")
    parser.add_argument("--config", type=str, default="config/default.yaml", help="Path to config")
    parser.add_argument("--weights", type=str, required=True, help="Path to model weights")
    args = parser.parse_args()
    
    logger = setup_logger("Evaluator")
    config = load_config(args.config)
    
    # Override weights
    config.set("model.detector.path", args.weights)
    
    logger.info(f"Starting evaluation for {args.weights}")
    
    try:
        evaluator = DetectionEvaluator(config)
        metrics = evaluator.evaluate()
        
        logger.info("Evaluation Results:")
        for k, v in metrics.items():
            logger.info(f"  {k}: {v}")
            
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
