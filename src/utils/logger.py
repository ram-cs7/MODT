"""
Logging utilities for the detection and tracking system
Provides structured logging, metrics tracking, and event logging
"""

import logging
import sys
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime
import json
from colorama import Fore, Style, init

# Initialize colorama
init(autoreset=True)


class ColoredFormatter(logging.Formatter):
    """Custom formatter with color support"""
    
    COLORS = {
        'DEBUG': Fore.CYAN,
        'INFO': Fore.GREEN,
        'WARNING': Fore.YELLOW,
        'ERROR': Fore.RED,
        'CRITICAL': Fore.RED + Style.BRIGHT,
    }
    
    def format(self, record):
        log_color = self.COLORS.get(record.levelname, '')
        record.levelname = f"{log_color}{record.levelname}{Style.RESET_ALL}"
        return super().format(record)


def setup_logger(
    name: str = "MODT",
    level: int = logging.INFO,
    log_file: Optional[str] = None,
    use_color: bool = True
) -> logging.Logger:
    """
    Set up logger with file and console handlers
    
    Args:
        name: Logger name
        level: Logging level
        log_file: Path to log file (optional)
        use_color: Use colored output for console
    
    Returns:
        Configured logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Remove existing handlers
    logger.handlers.clear()
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    
    if use_color:
        console_format = ColoredFormatter(
            '%(levelname)s | %(asctime)s | %(name)s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
    else:
        console_format = logging.Formatter(
            '%(levelname)s | %(asctime)s | %(name)s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
    
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)
    
    # File handler (if log file is specified)
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_format = logging.Formatter(
            '%(levelname)s | %(asctime)s | %(name)s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_format)
        logger.addHandler(file_handler)
    
    return logger


class MetricsLogger:
    """Logger for tracking metrics and events"""
    
    def __init__(self, output_dir: str = "./outputs/logs"):
        """
        Initialize metrics logger
        
        Args:
            output_dir: Directory to save logs
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create timestamped log file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.metrics_file = self.output_dir / f"metrics_{timestamp}.json"
        self.events_file = self.output_dir / f"events_{timestamp}.json"
        
        self.metrics = []
        self.events = []
    
    def log_metric(self, step: int, metrics: Dict[str, Any]):
        """
        Log metrics for a specific step
        
        Args:
            step: Training step or frame number
            metrics: Dictionary of metrics
        """
        entry = {
            "step": step,
            "timestamp": datetime.now().isoformat(),
            **metrics
        }
        self.metrics.append(entry)
    
    def log_event(self, event_type: str, data: Dict[str, Any]):
        """
        Log an event
        
        Args:
            event_type: Type of event (e.g., "zone_intrusion", "track_started")
            data: Event data
        """
        entry = {
            "event_type": event_type,
            "timestamp": datetime.now().isoformat(),
            **data
        }
        self.events.append(entry)
    
    def save(self):
        """Save metrics and events to files"""
        # Save metrics
        with open(self.metrics_file, 'w') as f:
            json.dump(self.metrics, f, indent=2)
        
        # Save events
        with open(self.events_file, 'w') as f:
            json.dump(self.events, f, indent=2)
    
    def get_metrics(self) -> list:
        """Get all logged metrics"""
        return self.metrics
    
    def get_events(self) -> list:
        """Get all logged events"""
        return self.events
    
    def __del__(self):
        """Save logs on destruction"""
        if self.metrics or self.events:
            self.save()


# Create default logger
default_logger = setup_logger()


if __name__ == "__main__":
    # Example usage
    logger = setup_logger("TestLogger", logging.DEBUG, "test.log")
    
    logger.debug("Debug message")
    logger.info("Info message")
    logger.warning("Warning message")
    logger.error("Error message")
    
    # Metrics logger
    metrics_logger = MetricsLogger("./test_logs")
    metrics_logger.log_metric(1, {"loss": 0.5, "accuracy": 0.85})
    metrics_logger.log_event("detection", {"class": "soldier", "confidence": 0.92})
    metrics_logger.save()
