"""
ByteTrack Tracker
High-performance tracker using two-stage association (High + Low confidence)
"""

import numpy as np
from .kalman_tracker import KalmanBoxTracker, KalmanTracker
from scipy.optimize import linear_sum_assignment

class ByteTracker(KalmanTracker):
    """
    ByteTrack: Multi-Object Tracking by Associating Every Detection Box
    """
    
    def __init__(
        self,
        max_age: int = 30,
        min_hits: int = 3,
        iou_threshold: float = 0.3,
        high_thresh: float = 0.6,
        low_thresh: float = 0.1
    ):
        super().__init__(max_age, min_hits, iou_threshold)
        self.high_thresh = high_thresh
        self.low_thresh = low_thresh
        
    def update(
        self,
        detections: np.ndarray,
        classes: np.ndarray = None,
        scores: np.ndarray = None
    ) -> np.ndarray:
        """
        Update tracker with ByteTrack logic (Two-stage matching)
        """
        self.frame_count += 1
        
        # Predict
        for track in self.trackers:
            track.predict()
            
        # Split detections
        high_inds = []
        low_inds = []
        
        if scores is not None:
            for i, score in enumerate(scores):
                if score >= self.high_thresh:
                    high_inds.append(i)
                elif score >= self.low_thresh:
                    low_inds.append(i)
        else:
            high_inds = list(range(len(detections)))
            
        high_dets = detections[high_inds]
        low_dets = detections[low_inds]
        
        # 1. Match High Confidence Detections
        # Only to tracked objects
        track_candidates = [t for t in self.trackers if t.time_since_update <= 1]
        track_indices = [self.trackers.index(t) for t in track_candidates]
        
        # Logic to associate subset of tracks with subset of detections
        # (Simplified: we run association on full lists in base class, here we customize)
        
        # Reuse base class association for high confidence
        matched_h, unmatched_dets_h, unmatched_tracks_h_indices = self._associate_subset(
            track_indices, high_dets
        )
        
        # Update matched high
        for t_real_idx, d_rel_idx in matched_h:
            d_real_idx = high_inds[d_rel_idx]
            score_val = float(scores[d_real_idx]) if scores is not None else 1.0
            self.trackers[t_real_idx].update(detections[d_real_idx], score_val)

        # 2. Match Low Confidence Detections to Unmatched Tracks
        # Get unmatched tracks from first stage
        unmatched_tracks_h = [self.trackers[i] for i in unmatched_tracks_h_indices]
        unmatched_tracks_h_real_indices = unmatched_tracks_h_indices
        
        # Associate
        matched_l, unmatched_dets_l, unmatched_tracks_l_indices = self._associate_subset(
            unmatched_tracks_h_real_indices, low_dets
        )
        
        # Update matched low
        for t_real_idx, d_rel_idx in matched_l:
            d_real_idx = low_inds[d_rel_idx]
            score_val = float(scores[d_real_idx]) if scores is not None else 1.0
            self.trackers[t_real_idx].update(detections[d_real_idx], score_val)
            
        # 3. Create new tracks from Unmatched High Detections
        for d_rel_idx in unmatched_dets_h:
            d_real_idx = high_inds[d_rel_idx]
            cls_val = int(classes[d_real_idx]) if classes is not None else 0
            score_val = float(scores[d_real_idx]) if scores is not None else 1.0
            self._init_track(detections[d_real_idx], cls_val, score_val)
            
        # Cleanup tracks
        self.trackers = [t for t in self.trackers if t.time_since_update <= self.max_age]
        
        # Return outputs
        outputs = []
        for t in self.trackers:
            if (t.time_since_update < 1) and (t.hit_streak >= self.min_hits or self.frame_count <= self.min_hits):
                state = t.get_state()
                # Ensure state is flat
                if hasattr(state, 'flatten'):
                    state = state.flatten()
                # Get scalar values
                cls_id = int(t.class_id) if hasattr(t.class_id, '__len__') == False else int(t.class_id.item() if hasattr(t.class_id, 'item') else t.class_id)
                score_val = float(t.score) if hasattr(t.score, '__len__') == False else float(t.score.item() if hasattr(t.score, 'item') else t.score)
                outputs.append([float(state[0]), float(state[1]), float(state[2]), float(state[3]), t.id, cls_id, score_val])
                
        if len(outputs) > 0:
            return np.array(outputs, dtype=np.float64)
        return np.empty((0, 7))

    def _associate_subset(self, track_indices, detections):
        """Helper to associate subset of tracks and detections"""
        if len(track_indices) == 0:
            return [], list(range(len(detections))), list(range(len(track_indices)))
        if len(detections) == 0:
            return [], [], track_indices
            
        # Compute IoU
        iou_matrix = np.zeros((len(track_indices), len(detections)))
        for i, t_idx in enumerate(track_indices):
            iou_matrix[i, :] = self.compute_iou(self.trackers[t_idx].get_state(), detections)
            
        # Hungarian
        cost_matrix = 1.0 - iou_matrix
        row_inds, col_inds = linear_sum_assignment(cost_matrix)
        
        matches = []
        unmatched_tracks = list(range(len(track_indices)))
        unmatched_dets = list(range(len(detections)))
        
        for r, c in zip(row_inds, col_inds):
            if iou_matrix[r, c] >= self.iou_threshold:
                matches.append((track_indices[r], c)) # Return real track index, relative det index
                if r in unmatched_tracks: unmatched_tracks.remove(r)
                if c in unmatched_dets: unmatched_dets.remove(c)
                
        # Remap unmatched_tracks to real indices
        unmatched_tracks_real = [track_indices[i] for i in unmatched_tracks]
        
        return matches, unmatched_dets, unmatched_tracks_real

    def _init_track(self, box, class_id, score):
        # Helper to init track since base class might not expose it directly or we need custom logic
        new_track = KalmanBoxTracker(
             box, class_id, score
        )
        self.trackers.append(new_track)
