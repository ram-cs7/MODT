"""Models package initialization"""

__version__ = "1.0.0"

# Import submodules for convenience
from . import detectors
from . import trackers
from . import backbones

# Import DTEA module
from .detectors.dtea import DTEA, DTEADetector, TemporalAttention, DynamicTemporalFusion, MotionEnhancedFeatures

__all__ = [
    'detectors',
    'trackers',
    'backbones',
    'DTEA',
    'DTEADetector',
    'TemporalAttention',
    'DynamicTemporalFusion',
    'MotionEnhancedFeatures',
]