"""
Event Handler Module
Handles zone intrusion detection and alert generation.
"""

import time
from typing import List, Dict, Any
import numpy as np
from src.analytics.zone_manager import ZoneManager

class EventHandler:
    """
    Handles high-level events like Zone Intrusion or Loitering
    """
    
    def __init__(self, config: Dict[str, Any], zone_manager: ZoneManager):
        self.config = config
        self.zone_manager = zone_manager
        self.alerts = []
        
    def process(self, tracks: np.ndarray, frame_id: int):
        """
        Check for events in current frame tracks
        """
        # Zone Intrusions
        if self.config.get('zone_intrusion', True):
            intrusions = self.zone_manager.check_intrusions(tracks)
            for zone_name, track_ids in intrusions.items():
                if track_ids:
                    msg = f"Frame {frame_id}: ALERT - Intrusion in {zone_name} by IDs {track_ids}"
                    # In real system: send MQTT/HTTP
                    self.alerts.append(msg)
                    
        return self.alerts[-5:] # Return last 5 alerts
