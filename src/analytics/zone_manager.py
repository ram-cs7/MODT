"""
Zone Manager for spatial analytics
Supports polygon zones for intrusion detection and counting
"""

import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional
from shapely.geometry import Point, Polygon
from shapely import prepare

from src.utils.logger import setup_logger


class Zone:
    """Represents a polygonal zone"""
    
    def __init__(self, name: str, points: List[Tuple[int, int]], color: Tuple[int, int, int] = (0, 255, 0)):
        """
        Initialize zone
        
        Args:
            name: Zone name/ID
            points: List of (x, y) points defining polygon
            color: Zone color for visualization (B, G, R)
        """
        self.name = name
        self.points = np.array(points, dtype=np.int32)
        self.color = color
        
        # Create Shapely polygon for efficient point-in-polygon checks
        self.polygon = Polygon(points)
        prepare(self.polygon)  # Optimize for repeated contains() calls
        
        # Statistics
        self.entry_count = 0
        self.exit_count = 0
        self.current_count = 0
        self.objects_in_zone = set()
    
    def contains_point(self, point: Tuple[float, float]) -> bool:
        """
        Check if point is inside zone
        
        Args:
            point: (x, y) coordinate
        
        Returns:
            True if point is inside zone
        """
        return self.polygon.contains(Point(point))
    
    def draw(self, image: np.ndarray, filled: bool = False, alpha: float = 0.3) -> np.ndarray:
        """
        Draw zone on image
        
        Args:
            image: Input image
            filled: Draw filled polygon
            alpha: Transparency for filled polygon
        
        Returns:
            Image with zone drawn
        """
        result = image.copy()
        
        if filled:
            overlay = result.copy()
            cv2.fillPoly(overlay, [self.points], self.color)
            result = cv2.addWeighted(result, 1 - alpha, overlay, alpha, 0)
        
        cv2.polylines(result, [self.points], True, self.color, 2)
        
        # Draw zone label
        centroid = self.points.mean(axis=0).astype(int)
        label = f"{self.name}: {self.current_count}"
        cv2.putText(result, label, tuple(centroid), cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.color, 2)
        
        return result


class ZoneManager:
    """
    Manages multiple zones and tracks object interactions
    """
    
    def __init__(self, zones: Optional[List[Dict]] = None):
        """
        Initialize zone manager
        
        Args:
            zones: List of zone definitions
                  Each zone is a dict with 'name', 'points', and optional 'color'
        """
        self.logger = setup_logger("ZoneManager")
        self.zones: List[Zone] = []
        
        if zones:
            for zone_def in zones:
                self.add_zone(
                    zone_def['name'],
                    zone_def['points'],
                    zone_def.get('color', (0, 255, 0))
                )
        
        # Track previous states for entry/exit detection
        self.previous_states = {}  # track_id -> zone_name mapping
    
    def add_zone(self, name: str, points: List[Tuple[int, int]], color: Tuple[int, int, int] = (0, 255, 0)):
        """Add a zone"""
        zone = Zone(name, points, color)
        self.zones.append(zone)
        self.logger.info(f"Added zone: {name} with {len(points)} points")
    
    def update(self, tracks: np.ndarray) -> Dict[str, List]:
        """
        Update zones with current tracks
        
        Args:
            tracks: Array of tracks (N, 7) [x1, y1, x2, y2, track_id, class_id, score]
        
        Returns:
            Dictionary of events: {'entries': [...], 'exits': [...], 'intrusions': [...]}
        """
        events = {
            'entries': [],
            'exits': [],
            'intrusions': []
        }
        
        # Reset current counts
        for zone in self.zones:
            zone.objects_in_zone.clear()
            zone.current_count = 0
        
        # Check each track
        for track in tracks:
            x1, y1, x2, y2, track_id, class_id, score = track
            track_id = int(track_id)
            
            # Get center point of bounding box
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            point = (center_x, center_y)
            
            # Check which zone contains this track
            current_zone = None
            for zone in self.zones:
                if zone.contains_point(point):
                    current_zone = zone.name
                    zone.objects_in_zone.add(track_id)
                    zone.current_count += 1
                    
                    # Record intrusion
                    events['intrusions'].append({
                        'zone': zone.name,
                        'track_id': track_id,
                        'class_id': int(class_id),
                        'position': point
                    })
                    break
            
            # Check for entry/exit
            previous_zone = self.previous_states.get(track_id)
            
            if current_zone != previous_zone:
                # Exit from previous zone
                if previous_zone is not None:
                    prev_zone_obj = next((z for z in self.zones if z.name == previous_zone), None)
                    if prev_zone_obj:
                        prev_zone_obj.exit_count += 1
                        events['exits'].append({
                            'zone': previous_zone,
                            'track_id': track_id,
                            'class_id': int(class_id)
                        })
                
                # Entry to current zone
                if current_zone is not None:
                    curr_zone_obj = next((z for z in self.zones if z.name == current_zone), None)
                    if curr_zone_obj:
                        curr_zone_obj.entry_count += 1
                        events['entries'].append({
                            'zone': current_zone,
                            'track_id': track_id,
                            'class_id': int(class_id)
                        })
                
                # Update state
                self.previous_states[track_id] = current_zone
        
        return events
    
    def visualize(self, image: np.ndarray, filled: bool = True) -> np.ndarray:
        """
        Visualize all zones on image
        
        Args:
            image: Input image
            filled: Draw filled polygons
        
        Returns:
            Image with zones drawn
        """
        result = image.copy()
        
        for zone in self.zones:
            result = zone.draw(result, filled=filled)
        
        return result
    
    def get_statistics(self) -> Dict[str, Dict]:
        """Get statistics for all zones"""
        stats = {}
        for zone in self.zones:
            stats[zone.name] = {
                'current_count': zone.current_count,
                'entry_count': zone.entry_count,
                'exit_count': zone.exit_count,
                'net_count': zone.entry_count - zone.exit_count
            }
        return stats


if __name__ == "__main__":
    # Example usage
    zone_manager = ZoneManager()
    
    # Add a rectangular zone
    zone_manager.add_zone(
        "Restricted Area",
        [(100, 100), (400, 100), (400, 300), (100, 300)],
        color=(0, 0, 255)
    )
    
    # Dummy tracks
    tracks = np.array([
        [150, 150, 200, 200, 1, 0, 0.9],  # Inside zone
        [500, 500, 550, 550, 2, 1, 0.85]  # Outside zone
    ])
    
    # Update
    events = zone_manager.update(tracks)
    print("Events:", events)
    
    # Get statistics
    stats = zone_manager.get_statistics()
    print("Statistics:", stats)
