"""
Track Manager
Manages track lifecycle, ID persistence, state transitions, and cleanup.
"""
from typing import List, Dict, Any, Optional
import numpy as np
from enum import Enum

class TrackState(Enum):
    TENTATIVE = 0
    CONFIRMED = 1
    DELETED = 2

class ComponentTrack:
    """Represents a single track with state"""
    def __init__(self, track_id: int, initial_state: Any, max_age: int, min_hits: int):
        self.id = track_id
        self.state = TrackState.TENTATIVE
        self.hits = 1
        self.age = 1
        self.time_since_update = 0
        self.max_age = max_age
        self.min_hits = min_hits
        self.history = [initial_state]
        
        if self.hits >= self.min_hits:
            self.state = TrackState.CONFIRMED
        
    def  update(self, detection: Any):
        self.history.append(detection)
        self.hits += 1
        self.time_since_update = 0
        if self.state == TrackState.TENTATIVE and self.hits >= self.min_hits:
            self.state = TrackState.CONFIRMED
            
    def predict(self):
        self.age += 1
        self.time_since_update += 1
        
    def mark_missed(self):
        if self.state == TrackState.TENTATIVE:
            self.state = TrackState.DELETED
        elif self.time_since_update > self.max_age:
            self.state = TrackState.DELETED
            
    def is_confirmed(self):
        return self.state == TrackState.CONFIRMED
    
    def is_deleted(self):
        return self.state == TrackState.DELETED


class TrackManager:
    """
    Manages collection of tracks
    Wrapper around specific tracker implementations to handle uniform lifecycle management
    if not provided by the base tracker logic.
    """
    def __init__(self, max_age=30, min_hits=3):
        self.max_age = max_age
        self.min_hits = min_hits
        self.tracks: Dict[int, ComponentTrack] = {}
        
    def update_lifecycle(self, active_track_ids: List[int], all_detections: Any = None):
        """
        Update states for all managed tracks
        
        Args:
            active_track_ids: List of IDs currently matched by the core tracker
        """
        # Create set for fast lookup
        present_ids = set(active_track_ids)
        
        # Check existing tracks
        for tid in list(self.tracks.keys()):
            track = self.tracks[tid]
            
            if tid in present_ids:
                # Was matched
                track.update(None) # Content updated by core tracker
            else:
                # Was missed
                track.predict() # Age gracefully
                track.mark_missed()
                if track.is_deleted():
                    del self.tracks[tid]
                    
        # Register new tracks (simplification: usually core tracker manages IDs)
        # This manager is useful if we want to layer additional logic (e.g. smoothing)
        for tid in active_track_ids:
            if tid not in self.tracks:
                self.tracks[tid] = ComponentTrack(tid, None, self.max_age, self.min_hits)

    def prune_tracks(self):
        """Force cleanup"""
        for tid in list(self.tracks.keys()):
            if self.tracks[tid].is_deleted():
                del self.tracks[tid]
