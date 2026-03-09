"""
Analytics Logger
Specialized logger for analytics events with separate file handling and structured JSON output.
"""
import logging
from pathlib import Path
from src.utils.logger import setup_logger, MetricsLogger

class AnalyticsLogger:
    """Wrapper for analytics-specific logging requirements"""
    
    def __init__(self, output_dir: str = "./outputs/logs/analytics"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # General text logger
        self.logger = setup_logger(
            "Analytics", 
            log_file=str(self.output_dir / "analytics.log")
        )
        
        # Structured event logger
        self.metrics = MetricsLogger(str(self.output_dir))
        
    def log_intrusion(self, zone_name: str, track_id: int):
        msg = f"Intrusion detected in {zone_name} by ID {track_id}"
        self.logger.warning(msg)
        self.metrics.log_event("intrusion", {"zone": zone_name, "track_id": track_id})
        
    def log_trajectory(self, track_id: int, points: list):
        self.metrics.log_event("trajectory_end", {"track_id": track_id, "length": len(points)})
        
    def save(self):
        self.metrics.save()
