"""
Video Stream Handler
Multi-threaded video capture for RTSP, USB, and File sources.
"""

import cv2
import time
from threading import Thread
from queue import Queue, Empty
from typing import Tuple, Optional
import numpy as np

class VideoStream:
    """Multi-threaded video stream handler"""
    
    def __init__(self, source: str = "0", buffer_size: int = 128):
        """
        Args:
            source: Video source (file path, RTSP URL, or webcam index)
            buffer_size: Maximum frame buffer size
        """
        self.source = source
        self.buffer_size = buffer_size
        
        # Initialize video capture
        if source.isdigit():
            self.stream = cv2.VideoCapture(int(source))
        else:
            self.stream = cv2.VideoCapture(source)
        
        if not self.stream.isOpened():
            raise ValueError(f"Failed to open video source: {source}")
        
        # Get video properties
        self.fps = self.stream.get(cv2.CAP_PROP_FPS) or 30.0  # Default to 30 if 0
        self.width = int(self.stream.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.stream.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Frame buffer
        self.queue = Queue(maxsize=buffer_size)
        self.stopped = False
        
        # Start reading thread
        self.thread = Thread(target=self._update, daemon=True)
        self.thread.start()
        
        # Wait for first frame to be available
        self._wait_for_first_frame()
    
    def _wait_for_first_frame(self, timeout: float = 5.0):
        """Wait for the first frame to be captured"""
        start_time = time.time()
        while self.queue.empty() and not self.stopped:
            if time.time() - start_time > timeout:
                self.stop()
                raise ValueError(f"Timeout waiting for first frame from: {self.source}")
            time.sleep(0.1)
    
    def _update(self):
        """Update thread - continuously reads frames"""
        while not self.stopped:
            if not self.queue.full():
                ret, frame = self.stream.read()
                if not ret:
                    self.stopped = True
                    return
                self.queue.put(frame)
            else:
                time.sleep(0.01)
    
    def read(self, timeout: float = 1.0) -> Tuple[bool, Optional[np.ndarray]]:
        """Read next frame from buffer with timeout
        
        Args:
            timeout: Seconds to wait for a frame
            
        Returns:
            (success, frame) tuple
        """
        if self.stopped:
            return False, None
            
        try:
            frame = self.queue.get(timeout=timeout)
            return True, frame
        except Empty:
            # Check if stream ended
            if self.stopped:
                return False, None
            # Queue temporarily empty, still running
            return False, None
    
    def stop(self):
        """Stop stream"""
        self.stopped = True
        if hasattr(self, 'thread') and self.thread.is_alive():
            self.thread.join(timeout=2.0)
        if hasattr(self, 'stream'):
            self.stream.release()
    
    def __del__(self):
        try:
            self.stop()
        except Exception:
            pass  # Ignore errors during cleanup
