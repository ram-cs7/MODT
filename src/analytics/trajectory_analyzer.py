"""
Trajectory Analyzer
Speed calculation, direction estimation, and path prediction
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from collections import deque
import cv2


class TrajectoryAnalyzer:
    """
    Analyzes object trajectories for speed, direction, and predictions
    """
    
    def __init__(self, fps: float = 30.0, max_history: int = 30):
        """
        Args:
            fps: Video frame rate
            max_history: Maximum trajectory history length
        """
        self.fps = fps
        self.max_history = max_history
        
        # Track ID -> trajectory history
        self.trajectories: Dict[int, deque] = {}
        
    def update(self, track_id: int, position: Tuple[float, float], timestamp: Optional[float] = None):
        """
        Update trajectory for a track
        
        Args:
            track_id: Track ID
            position: (x, y) position
            timestamp: Optional timestamp
        """
        if track_id not in self.trajectories:
            self.trajectories[track_id] = deque(maxlen=self.max_history)
        
        self.trajectories[track_id].append({
            'position': position,
            'timestamp': timestamp if timestamp else len(self.trajectories[track_id]) / self.fps
        })
    
    def get_speed(self, track_id: int, pixels_per_meter: float = 10.0) -> Optional[float]:
        """
        Calculate speed in meters/second
        
        Args:
            track_id: Track ID
            pixels_per_meter: Pixels per meter conversion
        
        Returns:
            Speed in m/s or None if insufficient data
        """
        if track_id not in self.trajectories or len(self.trajectories[track_id]) < 2:
            return None
        
        traj = list(self.trajectories[track_id])
        
        # Get last two positions
        pos1 = np.array(traj[-2]['position'])
        pos2 = np.array(traj[-1]['position'])
        t1 = traj[-2]['timestamp']
        t2 = traj[-1]['timestamp']
        
        # Calculate distance in pixels
        distance_px = np.linalg.norm(pos2 - pos1)
        
        # Convert to meters
        distance_m = distance_px / pixels_per_meter
        
        # Calculate time difference
        time_diff = t2 - t1
        
        if time_diff == 0:
            return None
        
        # Speed in m/s
        speed = distance_m / time_diff
        
        return speed
    
    def get_direction(self, track_id: int) -> Optional[float]:
        """
        Get movement direction in degrees (0-360)
        
        Args:
            track_id: Track ID
        
        Returns:
            Direction in degrees or None
        """
        if track_id not in self.trajectories or len(self.trajectories[track_id]) < 2:
            return None
        
        traj = list(self.trajectories[track_id])
        
        # Get last two positions
        pos1 = np.array(traj[-2]['position'])
        pos2 = np.array(traj[-1]['position'])
        
        # Calculate direction vector
        direction_vector = pos2 - pos1
        
        # Calculate angle in degrees
        angle = np.arctan2(direction_vector[1], direction_vector[0])
        angle_deg = np.degrees(angle)
        
        # Normalize to 0-360
        if angle_deg < 0:
            angle_deg += 360
        
        return angle_deg
    
    def predict_position(
        self,
        track_id: int,
        frames_ahead: int = 5
    ) -> Optional[Tuple[float, float]]:
        """
        Predict future position using linear extrapolation
        
        Args:
            track_id: Track ID
            frames_ahead: Number of frames to predict ahead
        
        Returns:
            Predicted (x, y) position
        """
        if track_id not in self.trajectories or len(self.trajectories[track_id]) < 2:
            return None
        
        traj = list(self.trajectories[track_id])
        
        # Use last N positions for prediction
        n_positions = min(5, len(traj))
        recent_positions = [np.array(t['position']) for t in traj[-n_positions:]]
        
        # Calculate average velocity
        velocities = []
        for i in range(1, len(recent_positions)):
            vel = recent_positions[i] - recent_positions[i-1]
            velocities.append(vel)
        
        avg_velocity = np.mean(velocities, axis=0)
        
        # Predict
        last_position = recent_positions[-1]
        predicted_position = last_position + avg_velocity * frames_ahead
        
        return tuple(predicted_position)
    
    def get_trajectory_smoothness(self, track_id: int) -> Optional[float]:
        """
        Calculate trajectory smoothness (0-1, higher is smoother)
        
        Args:
            track_id: Track ID
        
        Returns:
            Smoothness score
        """
        if track_id not in self.trajectories or len(self.trajectories[track_id]) < 3:
            return None
        
        traj = list(self.trajectories[track_id])
        positions = np.array([t['position'] for t in traj])
        
        # Calculate second derivative (acceleration)
        velocities = np.diff(positions, axis=0)
        accelerations = np.diff(velocities, axis=0)
        
        # Smoothness is inverse of acceleration magnitude
        accel_magnitude = np.linalg.norm(accelerations, axis=1)
        smoothness = 1.0 / (1.0 + np.mean(accel_magnitude))
        
        return smoothness
    
    def detect_stop(self, track_id: int, threshold: float = 2.0) -> bool:
        """
        Detect if object has stopped moving
        
        Args:
            track_id: Track ID
            threshold: Movement threshold in pixels
        
        Returns:
            True if stopped
        """
        if track_id not in self.trajectories or len(self.trajectories[track_id]) < 5:
            return False
        
        traj = list(self.trajectories[track_id])
        recent = traj[-5:]
        
        positions = np.array([t['position'] for t in recent])
        
        # Calculate total movement
        movements = np.diff(positions, axis=0)
        total_movement = np.sum(np.linalg.norm(movements, axis=1))
        
        return total_movement < threshold
    
    def visualize_trajectory(
        self,
        image: np.ndarray,
        track_id: int,
        color: Tuple[int, int, int] = (0, 255, 0),
        thickness: int = 2
    ) -> np.ndarray:
        """
        Draw trajectory on image
        
        Args:
            image: Input image
            track_id: Track ID
            color: Line color
            thickness: Line thickness
        
        Returns:
            Image with trajectory
        """
        if track_id not in self.trajectories:
            return image
        
        result = image.copy()
        traj = list(self.trajectories[track_id])
        
        if len(traj) < 2:
            return result
        
        # Draw lines
        points = [t['position'] for t in traj]
        for i in range(1, len(points)):
            pt1 = tuple(map(int, points[i-1]))
            pt2 = tuple(map(int, points[i]))
            cv2.line(result, pt1, pt2, color, thickness)
        
        # Draw direction arrow at current position
        if len(points) >= 2:
            current = np.array(points[-1])
            previous = np.array(points[-2])
            direction = current - previous
            
            if np.linalg.norm(direction) > 0:
                direction = direction / np.linalg.norm(direction) * 20
                arrow_end = current + direction
                
                cv2.arrowedLine(
                    result,
                    tuple(map(int, current)),
                    tuple(map(int, arrow_end)),
                    color,
                    thickness + 1,
                    tipLength=0.3
                )
        
        return result
    
    def get_statistics(self, track_id: int) -> Dict:
        """Get all statistics for a track"""
        stats = {
            'speed': self.get_speed(track_id),
            'direction': self.get_direction(track_id),
            'smoothness': self.get_trajectory_smoothness(track_id),
            'stopped': self.detect_stop(track_id),
            'trajectory_length': len(self.trajectories.get(track_id, []))
        }
        
        return stats


if __name__ == "__main__":
    # Example usage
    analyzer = TrajectoryAnalyzer(fps=30.0)
    
    # Simulate trajectory
    for i in range(20):
        x = 100 + i * 5
        y = 200 + np.sin(i * 0.3) * 10
        analyzer.update(track_id=1, position=(x, y))
    
    # Get metrics
    speed = analyzer.get_speed(1)
    direction = analyzer.get_direction(1)
    prediction = analyzer.predict_position(1, frames_ahead=10)
    
    print(f"Speed: {speed:.2f} m/s" if speed else "Speed: N/A")
    print(f"Direction: {direction:.2f}°" if direction else "Direction: N/A")
    print(f"Predicted position: {prediction}")
    
    stats = analyzer.get_statistics(1)
    print(f"Statistics: {stats}")
