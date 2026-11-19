"""Interactive Dataset Collector for Custom Gesture Data.

Provides a guided webcam recording interface for capturing temporal pose sequences:
1. Class selection (NONE, JAB, UPPERCUT, BLOCK)
2. 3-second preparation countdown
3. 30-frame capture at 30 FPS
4. Extraction to standardized .npz archives with subject metadata
"""

import argparse
import os
import time
from typing import Optional
import cv2
import numpy as np

from pose_control.config import (
    ActionClass,
    CLASS_NAMES,
    SEQUENCE_LENGTH,
    WINDOW_WIDTH,
    WINDOW_HEIGHT,
)
from pose_control.vision.pose_detector import BlazePoseDetector
from pose_control.vision.normalizer import PoseNormalizer
from pose_control.features.extractor import FeatureExtractor


class DatasetCollector:
    """Guided recording utility for creating labeled gesture datasets."""

    def __init__(self, output_dir: str = "dataset/raw"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.detector = BlazePoseDetector()
        self.normalizer = PoseNormalizer()
        self.extractor = FeatureExtractor()

    def record_sample(
        self,
        cap: cv2.VideoCapture,
        action: ActionClass,
        subject_id: str = "subject_01",
        sample_idx: int = 1,
    ) -> bool:
        """Records a single 30-frame sequence for a specified action class."""
        self.normalizer.reset()
        self.extractor.reset()

        raw_landmarks_list = []
        features_list = []

        # 1. Preparation Countdown (3 seconds)
        start_time = time.time()
        while time.time() - start_time < 3.0:
            ret, frame = cap.read()
            if not ret:
                return False
            frame = cv2.flip(frame, 1)
            countdown = 3 - int(time.time() - start_time)

            cv2.putText(
                frame,
                f"RECORDING: {action.name} (Sample #{sample_idx})",
                (50, 60),
                cv2.FONT_HERSHEY_DUPLEX,
                1.0,
                (0, 255, 255),
                2,
            )
            cv2.putText(
                frame,
                f"GET READY: {countdown}",
                (WINDOW_WIDTH // 2 - 120, WINDOW_HEIGHT // 2),
                cv2.FONT_HERSHEY_TRIPLEX,
                2.0,
                (0, 200, 255),
                4,
            )
            cv2.imshow("Gesture Data Collector", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                return False

        # 2. Capture 30 Frames
        print(f"--> Capturing 30 frames for {action.name}...")
        for frame_idx in range(SEQUENCE_LENGTH):
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)

            landmarks = self.detector.process_frame(frame)
            if landmarks is None:
                landmarks = np.zeros((33, 4), dtype=np.float32)

            norm_coords, _ = self.normalizer.normalize(landmarks)
            feats = self.extractor.extract(norm_coords)

            raw_landmarks_list.append(landmarks)
            features_list.append(feats)

            # Flash visual recording cue
            cv2.circle(frame, (50, 50), 20, (0, 0, 255), -1)
            cv2.putText(
                frame,
                f"RECORDING! Frame {frame_idx + 1}/{SEQUENCE_LENGTH}",
                (80, 58),
                cv2.FONT_HERSHEY_DUPLEX,
                0.8,
                (0, 0, 255),
                2,
            )
            cv2.imshow("Gesture Data Collector", frame)
            cv2.waitKey(1)

        if len(features_list) == SEQUENCE_LENGTH:
            filename = f"{action.name.lower()}_{subject_id}_{int(time.time())}_{sample_idx:03d}.npz"
            filepath = os.path.join(self.output_dir, filename)
            np.savez_compressed(
                filepath,
                features=np.array(features_list, dtype=np.float32),
                raw_keypoints=np.array(raw_landmarks_list, dtype=np.float32),
                label=int(action),
                label_name=action.name,
                subject_id=subject_id,
                fps=30,
            )
            print(f"Saved: {filepath}")
            return True
        return False


def main():
    parser = argparse.ArgumentParser(description="Record custom gesture datasets.")
    parser.add_argument("--action", type=str, default="JAB", choices=["NONE", "JAB", "UPPERCUT", "BLOCK"])
    parser.add_argument("--samples", type=int, default=10, help="Number of samples to record.")
    parser.add_argument("--subject", type=str, default="subject_01", help="Subject identifier.")
    args = parser.parse_args()

    action = ActionClass[args.action]
    collector = DatasetCollector()
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, WINDOW_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, WINDOW_HEIGHT)

    try:
        for i in range(args.samples):
            print(f"Sample {i + 1}/{args.samples}...")
            ok = collector.record_sample(cap, action, subject_id=args.subject, sample_idx=i + 1)
            if not ok:
                break
            time.sleep(1.0)
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
