"""
Test Tracking Module
Unit tests for tracking components
"""
import pytest
import numpy as np
from src.tracking.track_manager import TrackManager, ComponentTrack, TrackState

def test_track_creation():
    manager = TrackManager(max_age=5, min_hits=2)
    manager.update_lifecycle([1, 2])
    
    assert 1 in manager.tracks
    assert 2 in manager.tracks
    assert manager.tracks[1].state == TrackState.TENTATIVE

def test_track_confirmation():
    manager = TrackManager(max_age=5, min_hits=2)
    
    # Frame 1: Hit (hits=1)
    manager.update_lifecycle([1]) 
    assert manager.tracks[1].state == TrackState.TENTATIVE
    
    # Frame 2: Hit (hits=2) -> Confirm
    manager.update_lifecycle([1])
    assert manager.tracks[1].state == TrackState.CONFIRMED

def test_track_deletion():
    manager = TrackManager(max_age=1, min_hits=1)
    
    # Create and confirm
    manager.update_lifecycle([1])
    assert manager.tracks[1].state == TrackState.CONFIRMED
    
    # Miss 1 (age=1, time=1)
    manager.update_lifecycle([])
    assert 1 in manager.tracks
    assert not manager.tracks[1].is_deleted()
    
    # Miss 2 (age=2, time=2) > max_age=1 -> Deleted
    manager.update_lifecycle([]) # This triggers predict -> mark_missed
    
    # Logic in manager update: 
    # 1. Check existing: 1 is NOT in present_ids -> predict() -> mark_missed()
    
    assert 1 not in manager.tracks # Should be deleted
