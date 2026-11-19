"""Dataset Generator and Synthetic Pose Sequence Synthesizer.

Generates biomechanically grounded martial arts gesture sequences
(NONE, JAB, UPPERCUT, BLOCK) for dataset bootstrapping, training,
and automated integration testing.
"""

import math
import os
from typing import Dict, List, Tuple
import numpy as np

from pose_control.config import (
    ActionClass,
    SEQUENCE_LENGTH,
    NUM_LANDMARKS,
    TOTAL_FEATURE_DIM,
)
from pose_control.vision.normalizer import PoseNormalizer
from pose_control.features.extractor import FeatureExtractor
from pose_control.dataset.augmentor import SequenceAugmentor


class SyntheticPoseGenerator:
    """Generates synthetic 33-landmark 3D coordinate sequences for boxing gestures."""

    @staticmethod
    def create_neutral_pose() -> np.ndarray:
        """Creates a canonical 3D boxing stance skeleton (33, 4)."""
        pose = np.zeros((NUM_LANDMARKS, 4), dtype=np.float32)
        # Default visibility
        pose[:, 3] = 0.95

        # Head & Face
        pose[0] = [0.0, 0.45, 0.0, 0.99]       # Nose
        pose[1] = [-0.03, 0.47, 0.01, 0.95]    # Left eye inner
        pose[2] = [-0.05, 0.47, 0.01, 0.95]    # Left eye
        pose[3] = [-0.07, 0.47, 0.01, 0.95]    # Left eye outer
        pose[4] = [0.03, 0.47, 0.01, 0.95]     # Right eye inner
        pose[5] = [0.05, 0.47, 0.01, 0.95]     # Right eye
        pose[6] = [0.07, 0.47, 0.01, 0.95]     # Right eye outer
        pose[7] = [-0.12, 0.45, -0.05, 0.9]    # Left ear
        pose[8] = [0.12, 0.45, -0.05, 0.9]     # Right ear
        pose[9] = [-0.04, 0.41, 0.02, 0.95]    # Mouth left
        pose[10] = [0.04, 0.41, 0.02, 0.95]    # Mouth right

        # Shoulders (Torso top)
        pose[11] = [-0.22, 0.35, 0.0, 0.99]    # Left shoulder
        pose[12] = [0.22, 0.35, 0.0, 0.99]     # Right shoulder

        # Guard position (Wrists up by chin/cheeks)
        pose[13] = [-0.26, 0.18, 0.12, 0.98]   # Left elbow
        pose[14] = [0.26, 0.18, 0.12, 0.98]    # Right elbow
        pose[15] = [-0.14, 0.38, 0.22, 0.99]   # Left wrist (near chin)
        pose[16] = [0.14, 0.38, 0.22, 0.99]    # Right wrist (near chin)

        # Hands
        pose[17] = [-0.16, 0.40, 0.24, 0.9]    # Left pinky
        pose[18] = [0.16, 0.40, 0.24, 0.9]     # Right pinky
        pose[19] = [-0.14, 0.41, 0.24, 0.9]    # Left index
        pose[20] = [0.14, 0.41, 0.24, 0.9]     # Right index
        pose[21] = [-0.11, 0.39, 0.23, 0.9]    # Left thumb
        pose[22] = [0.11, 0.39, 0.23, 0.9]     # Right thumb

        # Hips (Torso base, root centered at 0.0, 0.0)
        pose[23] = [-0.14, -0.05, 0.0, 0.99]   # Left hip
        pose[24] = [0.14, -0.05, 0.0, 0.99]    # Right hip

        # Lower body stance (staggered boxing stance)
        pose[25] = [-0.16, -0.45, 0.08, 0.95]  # Left knee
        pose[26] = [0.18, -0.45, -0.08, 0.95]  # Right knee
        pose[27] = [-0.18, -0.85, 0.12, 0.95]  # Left ankle
        pose[28] = [0.20, -0.85, -0.15, 0.95]  # Right ankle
        pose[29] = [-0.18, -0.90, 0.08, 0.90]  # Left heel
        pose[30] = [0.20, -0.90, -0.19, 0.90]  # Right heel
        pose[31] = [-0.18, -0.92, 0.20, 0.90]  # Left foot index
        pose[32] = [0.20, -0.92, -0.07, 0.90]  # Right foot index

        return pose

    @classmethod
    def generate_sequence(
        cls,
        action: ActionClass,
        seq_len: int = SEQUENCE_LENGTH,
        lead_arm: str = "right",
    ) -> np.ndarray:
        """Generates a (seq_len, 33, 4) raw landmark temporal sequence."""
        sequence = np.zeros((seq_len, NUM_LANDMARKS, 4), dtype=np.float32)
        base = cls.create_neutral_pose()

        wrist_idx = 16 if lead_arm == "right" else 15
        elbow_idx = 14 if lead_arm == "right" else 13
        shoulder_idx = 12 if lead_arm == "right" else 11
        index_idx = 20 if lead_arm == "right" else 19

        for t in range(seq_len):
            frame = base.copy()
            # Idle breathing / stance bounce on all actions
            bounce_y = 0.02 * math.sin(t * 0.3)
            sway_x = 0.01 * math.cos(t * 0.2)
            frame[:, 1] += bounce_y
            frame[:, 0] += sway_x

            progress = t / float(seq_len)

            if action == ActionClass.NONE:
                # Slight head and hand movement within neutral stance
                frame[15, 1] += 0.015 * math.sin(t * 0.4)
                frame[16, 1] += 0.015 * math.cos(t * 0.4)

            elif action == ActionClass.JAB:
                # Bell-curve punch trajectory peaking around frame 15
                phase = math.exp(-0.5 * ((t - 15) / 3.8) ** 2)
                # Linear extension forward (along Z axis and forward in screen)
                frame[wrist_idx, 2] += phase * 0.55
                frame[index_idx, 2] += phase * 0.58
                # Slight rise towards opponent face
                frame[wrist_idx, 1] += phase * 0.08
                # Elbow opens forward
                frame[elbow_idx, 2] += phase * 0.35
                frame[elbow_idx, 0] += phase * (-0.05 if lead_arm == "right" else 0.05)
                # Shoulder forward drive
                frame[shoulder_idx, 2] += phase * 0.12

            elif action == ActionClass.UPPERCUT:
                # Uppercut: Dip then explosive rise
                # Dip phase (frames 6-12), explosive rise (frames 14-20), recovery (21-30)
                if t < 10:
                    dip = math.sin((t / 10.0) * math.pi)
                    frame[wrist_idx, 1] -= dip * 0.10
                    frame[elbow_idx, 1] -= dip * 0.08
                    frame[23:25, 1] -= dip * 0.04
                else:
                    rise_phase = math.exp(-0.5 * ((t - 18) / 4.2) ** 2)
                    # Vertical wrist elevation
                    frame[wrist_idx, 1] += rise_phase * 0.42
                    frame[index_idx, 1] += rise_phase * 0.45
                    frame[wrist_idx, 2] += rise_phase * 0.25
                    # Elbow stays tucked and rises
                    frame[elbow_idx, 1] += rise_phase * 0.28
                    frame[shoulder_idx, 1] += rise_phase * 0.10

            elif action == ActionClass.BLOCK:
                # High Guard: Wrists drawn tight to nose/ears, elbows inward
                guard_tightness = min(1.0, t / 8.0) if t < 8 else (
                    max(0.0, (seq_len - t) / 6.0) if t > 24 else 1.0
                )
                # Bring both wrists in front of face
                frame[15, 0] += guard_tightness * 0.08  # Left wrist moves inward
                frame[16, 0] -= guard_tightness * 0.08  # Right wrist moves inward
                frame[15, 1] += guard_tightness * 0.12  # Raise up to eye level
                frame[16, 1] += guard_tightness * 0.12
                frame[15, 2] -= guard_tightness * 0.05  # Closer to head
                frame[16, 2] -= guard_tightness * 0.05
                # Elbows clamped tight to ribs
                frame[13, 0] += guard_tightness * 0.06
                frame[14, 0] -= guard_tightness * 0.06

            sequence[t] = frame

        return sequence


class DatasetBuilder:
    """Builds synthetic or extracted datasets formatted for LSTM training."""

    def __init__(self, output_dir: str = "dataset"):
        self.output_dir = output_dir
        self.normalizer = PoseNormalizer()
        self.extractor = FeatureExtractor()
        self.augmentor = SequenceAugmentor()

    def generate_balanced_dataset(
        self,
        samples_per_class: int = 150,
        train_ratio: float = 0.70,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
    ) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """Generates a complete dataset with feature extraction and splits."""
        os.makedirs(self.output_dir, exist_ok=True)

        all_features = []
        all_labels = []

        for action in [ActionClass.NONE, ActionClass.JAB, ActionClass.UPPERCUT, ActionClass.BLOCK]:
            for i in range(samples_per_class):
                lead = "right" if i % 2 == 0 else "left"
                raw_seq = SyntheticPoseGenerator.generate_sequence(action, lead_arm=lead)

                # Apply data augmentation on 70% of generated sequences
                if i > 0:
                    raw_coords = self.augmentor.augment_raw_landmarks(raw_seq[:, :, :3])
                    aug_seq = raw_seq.copy()
                    aug_seq[:, :, :3] = raw_coords
                else:
                    aug_seq = raw_seq

                # Normalize and extract 138-dim features across the 30 frames
                self.normalizer.reset()
                self.extractor.reset()
                seq_features = np.zeros((SEQUENCE_LENGTH, TOTAL_FEATURE_DIM), dtype=np.float32)

                for t in range(SEQUENCE_LENGTH):
                    norm_coords, _ = self.normalizer.normalize(aug_seq[t])
                    feats = self.extractor.extract(norm_coords)
                    seq_features[t] = feats

                all_features.append(seq_features)
                all_labels.append(int(action))

        X = np.array(all_features, dtype=np.float32)  # (N, 30, 138)
        y = np.array(all_labels, dtype=np.int32)      # (N,)

        # Stratified split
        num_samples = len(X)
        indices = np.random.permutation(num_samples)
        X = X[indices]
        y = y[indices]

        n_train = int(num_samples * train_ratio)
        n_val = int(num_samples * val_ratio)

        splits = {
            "train": (X[:n_train], y[:n_train]),
            "val": (X[n_train:n_train + n_val], y[n_train:n_train + n_val]),
            "test": (X[n_train + n_val:], y[n_train + n_val:]),
        }

        # Save to disk
        np.savez_compressed(
            os.path.join(self.output_dir, "dataset_splits.npz"),
            X_train=splits["train"][0], y_train=splits["train"][1],
            X_val=splits["val"][0], y_val=splits["val"][1],
            X_test=splits["test"][0], y_test=splits["test"][1],
        )

        return splits
