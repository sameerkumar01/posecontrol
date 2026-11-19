"""Kinematic Sequence Data Augmentation Engine.

Implements spatial, temporal, and sensor noise augmentations for 30-frame
martial arts gesture sequences according to the technical architecture.
"""

import math
import random
from typing import List, Tuple
import numpy as np

from pose_control.config import (
    TOTAL_FEATURE_DIM,
    SEQUENCE_LENGTH,
    POSITION_FEATURE_DIM,
    KEY_VELOCITY_LANDMARK_INDICES,
)


class SequenceAugmentor:
    """Stochastic kinematic sequence augmentor for training generalization."""

    # Bilateral pairs for left-right mirroring (BlazePose indices)
    BILATERAL_SWAP_PAIRS = [
        (1, 4),   # Left eye inner <-> Right eye inner
        (2, 5),   # Left eye <-> Right eye
        (3, 6),   # Left eye outer <-> Right eye outer
        (7, 8),   # Left ear <-> Right ear
        (9, 10),  # Mouth left <-> Mouth right
        (11, 12), # Left shoulder <-> Right shoulder
        (13, 14), # Left elbow <-> Right elbow
        (15, 16), # Left wrist <-> Right wrist
        (17, 18), # Left pinky <-> Right pinky
        (19, 20), # Left index <-> Right index
        (21, 22), # Left thumb <-> Right thumb
        (23, 24), # Left hip <-> Right hip
        (25, 26), # Left knee <-> Right knee
        (27, 28), # Left ankle <-> Right ankle
        (29, 30), # Left heel <-> Right heel
        (31, 32), # Left foot index <-> Right foot index
    ]

    def __init__(
        self,
        yaw_range_deg: float = 15.0,
        roll_range_deg: float = 8.0,
        scale_range: Tuple[float, float] = (0.85, 1.15),
        time_warp_range: Tuple[float, float] = (0.80, 1.25),
        gaussian_noise_std: float = 0.015,
        dropout_prob: float = 0.05,
    ):
        self.yaw_range_deg = yaw_range_deg
        self.roll_range_deg = roll_range_deg
        self.scale_range = scale_range
        self.time_warp_range = time_warp_range
        self.gaussian_noise_std = gaussian_noise_std
        self.dropout_prob = dropout_prob

    def augment_raw_landmarks(
        self,
        landmarks_seq: np.ndarray,
        enable_mirror: bool = True,
        enable_rotation: bool = True,
        enable_scaling: bool = True,
        enable_time_warp: bool = True,
        enable_noise: bool = True,
    ) -> np.ndarray:
        """Augments a 3D landmark sequence of shape (T, 33, 3)."""
        T, num_pts, coords = landmarks_seq.shape
        assert num_pts == 33 and coords == 3
        seq = landmarks_seq.copy()

        # 1. 3D Rotation (Yaw & Roll)
        if enable_rotation and random.random() > 0.3:
            yaw = math.radians(random.uniform(-self.yaw_range_deg, self.yaw_range_deg))
            roll = math.radians(random.uniform(-self.roll_range_deg, self.roll_range_deg))

            cos_y, sin_y = math.cos(yaw), math.sin(yaw)
            r_yaw = np.array([
                [cos_y, 0.0, sin_y],
                [0.0, 1.0, 0.0],
                [-sin_y, 0.0, cos_y]
            ], dtype=np.float32)

            cos_z, sin_z = math.cos(roll), math.sin(roll)
            r_roll = np.array([
                [cos_z, -sin_z, 0.0],
                [sin_z, cos_z, 0.0],
                [0.0, 0.0, 1.0]
            ], dtype=np.float32)

            r_matrix = np.dot(r_yaw, r_roll)
            for t in range(T):
                seq[t] = np.dot(seq[t], r_matrix.T)

        # 2. Anatomical Scale Jitter
        if enable_scaling and random.random() > 0.3:
            scale = random.uniform(self.scale_range[0], self.scale_range[1])
            seq = seq * scale

        # 3. Horizontal Mirroring (Left-Right Swap)
        if enable_mirror and random.random() > 0.5:
            # Invert X axis
            seq[:, :, 0] *= -1.0
            # Swap bilateral anatomical indices
            swapped = seq.copy()
            for left_idx, right_idx in self.BILATERAL_SWAP_PAIRS:
                swapped[:, left_idx] = seq[:, right_idx]
                swapped[:, right_idx] = seq[:, left_idx]
            seq = swapped

        # 4. Temporal Time-Warping (Resampling)
        if enable_time_warp and random.random() > 0.4:
            warp_factor = random.uniform(self.time_warp_range[0], self.time_warp_range[1])
            new_len = int(round(T * warp_factor))
            new_len = max(10, min(new_len, 60))

            old_indices = np.linspace(0, T - 1, num=new_len)
            resampled = np.zeros((new_len, num_pts, coords), dtype=np.float32)
            for i in range(num_pts):
                for c in range(coords):
                    resampled[:, i, c] = np.interp(old_indices, np.arange(T), seq[:, i, c])

            # Resample or pad/crop back to original T frames
            final_indices = np.linspace(0, new_len - 1, num=T)
            warped = np.zeros((T, num_pts, coords), dtype=np.float32)
            for i in range(num_pts):
                for c in range(coords):
                    warped[:, i, c] = np.interp(final_indices, np.arange(new_len), resampled[:, i, c])
            seq = warped

        # 5. Gaussian Sensor Noise
        if enable_noise and random.random() > 0.3:
            noise = np.random.normal(0.0, self.gaussian_noise_std, size=seq.shape).astype(np.float32)
            seq += noise

        # 6. Synthetic Landmark Occlusion (Dropout)
        if enable_noise and random.random() > 0.4:
            for pt_idx in range(num_pts):
                if random.random() < self.dropout_prob:
                    # Drop this landmark across a consecutive chunk of frames
                    drop_start = random.randint(0, T - 5)
                    drop_len = random.randint(2, 6)
                    seq[drop_start:drop_start + drop_len, pt_idx] = 0.0

        return seq
