"""Deploy edge detection system - Main deployment script"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import argparse
from src.deployment.pipeline import DeploymentPipeline
from src.utils.logger import setup_logger


def main():
    parser = argparse.ArgumentParser(
        description="Military Object Detection & Tracking - Edge Deployment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Webcam
  python scripts/deploy_edge.py --source 0

  # Video file
  python scripts/deploy_edge.py --source video.mp4

  # RTSP stream
  python scripts/deploy_edge.py --source rtsp://192.168.1.100:554/stream

  # With custom config (Jetson)
  python scripts/deploy_edge.py --config config/edge_jetson.yaml --source 0
        """
    )
    
    parser.add_argument(
        "--config",
        type=str,
        default="config/default.yaml",
        help="Path to configuration file"
    )
    
    parser.add_argument(
        "--source",
        type=str,
        default="0",
        help="Video source: webcam index, video file, or RTSP URL"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for results (overrides config)"
    )
    
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Disable video display window"
    )
    
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Don't save output video"
    )
    
    args = parser.parse_args()
    
    # Setup pipeline
    logger = setup_logger("EdgeDeploy")
    logger.info("="*80)
    logger.info("Military Object Detection & Tracking System")
    logger.info("Edge Deployment Mode")
    logger.info("="*80)
    
    try:
        # Initialize pipeline
        pipeline = DeploymentPipeline(config_path=args.config)
        
        # Override config if needed
        if args.output_dir:
            pipeline.output_dir = Path(args.output_dir)
            pipeline.output_dir.mkdir(parents=True, exist_ok=True)
        
        if args.no_display:
            pipeline.config.set("deployment.output.display", False)
        
        if args.no_save:
            pipeline.save_video = False
        
        # Run pipeline
        logger.info(f"Configuration: {args.config}")
        logger.info(f"Source: {args.source}")
        logger.info(f"Output directory: {pipeline.output_dir}")
        logger.info("="*80)
        logger.info("Press 'q' to quit")
        logger.info("="*80)
        
        pipeline.run(source=args.source)
        
    except KeyboardInterrupt:
        logger.info("\nInterrupted by user")
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        sys.exit(1)
    
    logger.info("="*80)
    logger.info("Deployment completed")
    logger.info("="*80)


if __name__ == "__main__":
    main()
