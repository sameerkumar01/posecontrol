"""Biomechanical Landmark Normalizer.

Transforms raw 3D MediaPipe BlazePose landmarks into an invariant coordinate space:
1. Mid-hip root centering (Translation invariance)
2. Torso-length Euclidean scaling (Scale / height / camera distance invariance)
3. Coronal plane yaw rotation (Orientation invariance)
4. Low-confidence visibility filtering and kinematic forward imputation
"""

import math
from typing import Optional, Tuple
import numpy as np


class PoseNormalizer:
    """Performs 3-stage invariant normalization on 33 3D body landmarks."""

    # Landmark indices based on BlazePose topology
    NOSE = 0
    LEFT_SHOULDER = 11
    RIGHT_SHOULDER = 12
    LEFT_HIP = 23
    RIGHT_HIP = 24

    def __init__(
        self,
        visibility_threshold: float = 0.50,
        max_impute_frames: int = 3,
        epsilon: float = 1e-6,
    ):
        self.visibility_threshold = visibility_threshold
        self.max_impute_frames = max_impute_frames
        self.epsilon = epsilon

        # Imputation history buffers
        self._prev_landmarks: Optional[np.ndarray] = None
        self._prev_velocity: Optional[np.ndarray] = None
        self._missing_counter = np.zeros(33, dtype=np.int32)

    def reset(self):
        """Reset internal temporal imputation state."""
        self._prev_landmarks = None
        self._prev_velocity = None
        self._missing_counter.fill(0)

    def normalize(self, raw_landmarks: np.ndarray) -> Tuple[np.ndarray, float]:
        """Normalize 33 3D body landmarks into invariant coordinates.

        Args:
            raw_landmarks: Array of shape (33, 4) with (x, y, z, visibility) or
                           shape (33, 3) with (x, y, z).

        Returns:
            normalized_coords: Array of shape (33, 3) invariant to translation,
                              scale, and coronal yaw.
            torso_scale: Euclidean distance of torso prior to normalization.
        """
        assert raw_landmarks.shape[0] == 33, f"Expected 33 landmarks, got {raw_landmarks.shape[0]}"

        if raw_landmarks.shape[1] == 4:
            coords = raw_landmarks[:, :3].copy()
            visibilities = raw_landmarks[:, 3]
        else:
            coords = raw_landmarks.copy()
            visibilities = np.ones(33, dtype=np.float32)

        # -------------------------------------------------------------------
        # Step 0: Kinematic Forward Imputation for Low-Visibility Points
        # -------------------------------------------------------------------
        if self._prev_landmarks is not None and self._prev_velocity is not None:
            low_vis_mask = visibilities < self.visibility_threshold
            for idx in range(33):
                if low_vis_mask[idx]:
                    self._missing_counter[idx] += 1
                    if self._missing_counter[idx] <= self.max_impute_frames:
                        # Extrapolate using last known velocity with damping
                        decay = 0.85 ** self._missing_counter[idx]
                        coords[idx] = self._prev_landmarks[idx] + self._prev_velocity[idx] * decay
                else:
                    self._missing_counter[idx] = 0

        # -------------------------------------------------------------------
        # Step 1: Root Centering (Translation Invariance)
        # -------------------------------------------------------------------
        left_hip = coords[self.LEFT_HIP]
        right_hip = coords[self.RIGHT_HIP]
        mid_hip_root = 0.5 * (left_hip + right_hip)

        # Translate all points so mid-hip is at (0, 0, 0)
        coords_translated = coords - mid_hip_root

        # -------------------------------------------------------------------
        # Step 2: Scale Invariance (Torso-Length Normalization)
        # -------------------------------------------------------------------
        left_shoulder = coords[self.LEFT_SHOULDER]
        right_shoulder = coords[self.RIGHT_SHOULDER]
        mid_shoulder = 0.5 * (left_shoulder + right_shoulder)

        torso_vector = mid_shoulder - mid_hip_root
        torso_scale = float(np.linalg.norm(torso_vector))
        safe_scale = max(torso_scale, self.epsilon)

        coords_scaled = coords_translated / safe_scale

        # -------------------------------------------------------------------
        # Step 3: Coronal Plane & Yaw Normalization (Orientation Invariance)
        # -------------------------------------------------------------------
        # Shoulder vector in scaled space
        sh_left_scaled = coords_scaled[self.LEFT_SHOULDER]
        sh_right_scaled = coords_scaled[self.RIGHT_SHOULDER]
        delta_x = sh_right_scaled[0] - sh_left_scaled[0]
        delta_z = sh_right_scaled[2] - sh_left_scaled[2]

        theta_yaw = math.atan2(delta_z, delta_x)

        # Rotation matrix around Y axis: R_y(-theta_yaw)
        cos_y = math.cos(-theta_yaw)
        sin_y = math.sin(-theta_yaw)
        r_yaw = np.array([
            [cos_y, 0.0, sin_y],
            [0.0, 1.0, 0.0],
            [-sin_y, 0.0, cos_y]
        ], dtype=np.float32)

        # Apply rotation: L''' = L'' @ R_y^T
        coords_normalized = np.dot(coords_scaled, r_yaw.T)

        # Update temporal state for next frame
        if self._prev_landmarks is not None:
            self._prev_velocity = coords_normalized - self._prev_landmarks
        else:
            self._prev_velocity = np.zeros_like(coords_normalized)
        self._prev_landmarks = coords_normalized.copy()

        return coords_normalized, safe_scale
