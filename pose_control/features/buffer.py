"""Circular Temporal Windowing Buffer.

Maintains a sliding FIFO queue of 30 frames of 138-dimensional feature vectors
for causal LSTM model sequence inference.
"""

from collections import deque
import numpy as np

from pose_control.config import SEQUENCE_LENGTH, TOTAL_FEATURE_DIM


class FeatureWindowBuffer:
    """Sliding temporal buffer holding the most recent T feature frames."""

    def __init__(self, window_size: int = SEQUENCE_LENGTH, feature_dim: int = TOTAL_FEATURE_DIM):
        self.window_size = window_size
        self.feature_dim = feature_dim
        self.buffer = deque(maxlen=window_size)
        self.reset()

    def reset(self):
        """Clears buffer and prefills with neutral zero-padded feature vectors."""
        self.buffer.clear()
        for _ in range(self.window_size):
            self.buffer.append(np.zeros(self.feature_dim, dtype=np.float32))

    def append(self, feature_vector: np.ndarray):
        """Appends a new feature vector to the rolling temporal window."""
        assert feature_vector.shape == (self.feature_dim,), (
            f"Expected shape ({self.feature_dim},), got {feature_vector.shape}"
        )
        self.buffer.append(feature_vector.astype(np.float32))

    def is_filled(self) -> bool:
        """Returns True if the buffer has accumulated at least window_size non-empty frames."""
        return len(self.buffer) == self.window_size

    def get_tensor(self) -> np.ndarray:
        """Returns contiguous float32 tensor of shape (1, window_size, feature_dim)."""
        array_2d = np.array(self.buffer, dtype=np.float32)
        return np.expand_dims(array_2d, axis=0)

    def get_raw_sequence(self) -> np.ndarray:
        """Returns 2D array of shape (window_size, feature_dim)."""
        return np.array(self.buffer, dtype=np.float32)
