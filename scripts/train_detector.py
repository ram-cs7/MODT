"""
Train Detection Model
Main training script with CLI arguments
"""

import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import argparse
from src.utils.config import load_config
from src.detection.trainer import DetectionTrainer
from src.utils.logger import setup_logger


def main():
    parser = argparse.ArgumentParser(
        description="Train Object Detection Model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train with default config
  python scripts/train_detector.py

  # Train with custom config
  python scripts/train_detector.py --config config/training.yaml

  # Resume from checkpoint
  python scripts/train_detector.py --resume weights/last.pt
        """
    )
    
    parser.add_argument(
        "--config",
        type=str,
        default="config/training.yaml",
        help="Path to training configuration file"
    )
    
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume from"
    )
    
    parser.add_argument(
        "--data",
        type=str,
        default=None,
        help="Path to dataset (overrides config)"
    )
    
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Number of epochs (overrides config)"
    )
    
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Batch size (overrides config)"
    )
    
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (cuda/cpu, overrides config)"
    )
    
    args = parser.parse_args()
    
    # Setup logger
    logger = setup_logger("TrainScript")
    logger.info("="*80)
    logger.info("Military Object Detection - Training")
    logger.info("="*80)
    
    try:
        # Load configuration
        config = load_config(args.config)
        logger.info(f"Loaded config: {args.config}")
        
        # Override config with CLI arguments
        if args.data:
            config.set("dataset.paths.train", args.data + "/train")
            config.set("dataset.paths.val", args.data + "/val")
        
        if args.epochs:
            config.set("training.epochs", args.epochs)
        
        if args.batch_size:
            config.set("training.batch_size", args.batch_size)
        
        if args.device:
            config.set("system.device", args.device)
        
        # Initialize trainer
        trainer = DetectionTrainer(config)
        
        # Resume from checkpoint if specified
        if args.resume:
            logger.info(f"Resuming from checkpoint: {args.resume}")
            # Implement resume logic here
        
        # Start training
        trainer.train()
        
        logger.info("="*80)
        logger.info("Training completed successfully!")
        logger.info("="*80)
        
    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
