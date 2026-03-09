"""Package initialization"""

__version__ = "1.0.0"
__author__ = "Your Name"
__description__ = "Military Object Detection and Tracking System"

from pathlib import Path

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent

# Default paths
CONFIG_DIR = PROJECT_ROOT / "config"
DATA_DIR = PROJECT_ROOT / "data"
WEIGHTS_DIR = PROJECT_ROOT / "weights"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
