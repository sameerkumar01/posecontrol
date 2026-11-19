"""Kinematic Feature Extraction Subsystem.

Transforms normalized 3D body coordinates into a 138-dimensional multi-modal
feature vector per frame:
1. 99 Normalized 3D Coordinates (All 33 landmarks)
2. 33 First-Order Temporal Velocities (11 Key Strike Landmarks)
3. 6 Biomechanical Joint Flexion Angles
"""

import math
from typing import List, Optional
import numpy as np

from pose_control.config import (
    KEY_VELOCITY_LANDMARK_INDICES,
    POSITION_FEATURE_DIM,
    VELOCITY_FEATURE_DIM,
    JOINT_ANGLE_FEATURE_DIM,
    TOTAL_FEATURE_DIM,
)


class FeatureExtractor:
    """Extracts 138-dimensional spatio-temporal kinematic feature vectors."""

    # Landmark indices for angle calculation:
    NOSE = 0
    L_SHOULDER = 11
    R_SHOULDER = 12
    L_ELBOW = 13
    R_ELBOW = 14
    L_WRIST = 15
    R_WRIST = 16
    L_HIP = 23
    R_HIP = 24
    L_KNEE = 25
    R_KNEE = 26

    def __init__(self, fps: float = 30.0):
        self.dt = 1.0 / max(fps, 1.0)
        self._prev_landmarks: Optional[np.ndarray] = None

    def reset(self):
        """Reset temporal velocity tracking state."""
        self._prev_landmarks = None

    @staticmethod
    def _calculate_angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
        """Calculate interior angle between vectors BA and BC in radians [0, pi]."""
        v1 = a - b
        v2 = c - b
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        if norm1 < 1e-6 or norm2 < 1e-6:
            return 0.0
        cosine = np.dot(v1, v2) / (norm1 * norm2)
        cosine = np.clip(cosine, -1.0, 1.0)
        return float(np.arccos(cosine))

    def extract(self, normalized_landmarks: np.ndarray) -> np.ndarray:
        """Extracts 138-dimensional feature vector from normalized landmarks.

        Args:
            normalized_landmarks: Array of shape (33, 3).

        Returns:
            feature_vector: 1D array of shape (138,), float32.
        """
        assert normalized_landmarks.shape == (33, 3), (
            f"Expected shape (33, 3), got {normalized_landmarks.shape}"
        )

        # -------------------------------------------------------------------
        # 1. Positional Features (99 dimensions)
        # -------------------------------------------------------------------
        pos_features = normalized_landmarks.flatten().astype(np.float32)

        # -------------------------------------------------------------------
        # 2. First-Order Temporal Velocity (33 dimensions)
        # -------------------------------------------------------------------
        if self._prev_landmarks is not None:
            full_velocity = (normalized_landmarks - self._prev_landmarks) / self.dt
        else:
            full_velocity = np.zeros_like(normalized_landmarks)

        key_velocities = full_velocity[KEY_VELOCITY_LANDMARK_INDICES].flatten().astype(np.float32)

        # -------------------------------------------------------------------
        # 3. Biomechanical Joint Flexion Angles (6 dimensions)
        # -------------------------------------------------------------------
        # 1. Left Elbow: Wrist(15) - Elbow(13) - Shoulder(11)
        angle_le = self._calculate_angle(
            normalized_landmarks[self.L_WRIST],
            normalized_landmarks[self.L_ELBOW],
            normalized_landmarks[self.L_SHOULDER],
        )

        # 2. Right Elbow: Wrist(16) - Elbow(14) - Shoulder(12)
        angle_re = self._calculate_angle(
            normalized_landmarks[self.R_WRIST],
            normalized_landmarks[self.R_ELBOW],
            normalized_landmarks[self.R_SHOULDER],
        )

        # 3. Left Shoulder: Elbow(13) - Shoulder(11) - Hip(23)
        angle_ls = self._calculate_angle(
            normalized_landmarks[self.L_ELBOW],
            normalized_landmarks[self.L_SHOULDER],
            normalized_landmarks[self.L_HIP],
        )

        # 4. Right Shoulder: Elbow(14) - Shoulder(12) - Hip(24)
        angle_rs = self._calculate_angle(
            normalized_landmarks[self.R_ELBOW],
            normalized_landmarks[self.R_SHOULDER],
            normalized_landmarks[self.R_HIP],
        )

        # 5. Left Hip: Shoulder(11) - Hip(23) - Knee(25)
        angle_lh = self._calculate_angle(
            normalized_landmarks[self.L_SHOULDER],
            normalized_landmarks[self.L_HIP],
            normalized_landmarks[self.L_KNEE],
        )

        # 6. Right Hip: Shoulder(12) - Hip(24) - Knee(26)
        angle_rh = self._calculate_angle(
            normalized_landmarks[self.R_SHOULDER],
            normalized_landmarks[self.R_HIP],
            normalized_landmarks[self.R_KNEE],
        )

        angle_features = np.array(
            [angle_le, angle_re, angle_ls, angle_rs, angle_lh, angle_rh],
            dtype=np.float32,
        )

        # Update previous landmarks
        self._prev_landmarks = normalized_landmarks.copy()

        # Concatenate: 99 + 33 + 6 = 138 dims
        feature_vector = np.concatenate([pos_features, key_velocities, angle_features])
        assert feature_vector.shape == (TOTAL_FEATURE_DIM,), (
            f"Feature vector shape mismatch: expected ({TOTAL_FEATURE_DIM},), got {feature_vector.shape}"
        )
        return feature_vector
