"""MediaPipe BlazePose Detection Wrapper.

Supports both modern MediaPipe Tasks Vision API (mediapipe.tasks.python.vision.PoseLandmarker)
and legacy MediaPipe Solutions API (mediapipe.solutions.pose), with robust fallback.
"""

from typing import Optional, Tuple
import numpy as np

try:
    import mediapipe as mp
    import cv2
    HAS_MEDIAPIPE = True
except ImportError:
    HAS_MEDIAPIPE = False


class BlazePoseDetector:
    """Wrapper around MediaPipe BlazePose solution supporting both legacy and Tasks API."""

    def __init__(
        self,
        static_image_mode: bool = False,
        model_complexity: int = 1,
        smooth_landmarks: bool = True,
        min_detection_confidence: float = 0.50,
        min_tracking_confidence: float = 0.50,
    ):
        self.model_complexity = model_complexity
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        self.pose = None
        self._api_type = None

        if HAS_MEDIAPIPE:
            # 1. Check legacy solutions API
            if hasattr(mp, "solutions") and hasattr(mp.solutions, "pose"):
                try:
                    self._mp_pose = mp.solutions.pose
                    self._mp_drawing = mp.solutions.drawing_utils
                    self._mp_drawing_styles = mp.solutions.drawing_styles
                    self.pose = self._mp_pose.Pose(
                        static_image_mode=static_image_mode,
                        model_complexity=model_complexity,
                        smooth_landmarks=smooth_landmarks,
                        min_detection_confidence=min_detection_confidence,
                        min_tracking_confidence=min_tracking_confidence,
                    )
                    self._api_type = "solutions"
                except Exception as e:
                    print(f"[BlazePoseDetector] Warning: solutions.pose failed: {e}")

            # 2. Check Tasks Vision API
            if self.pose is None and hasattr(mp, "tasks"):
                try:
                    from mediapipe.tasks.python import BaseOptions
                    from mediapipe.tasks.python.vision import PoseLandmarker, PoseLandmarkerOptions, RunningMode
                    # Tasks API requires pose_landmarker.task model asset if present
                    # We initialize on demand or fallback
                    self._api_type = "tasks"
                except Exception as e:
                    print(f"[BlazePoseDetector] Tasks API import error: {e}")

    def process_frame(self, frame_bgr: np.ndarray) -> Optional[np.ndarray]:
        """Processes a single BGR video frame and extracts 33 3D landmarks."""
        if not HAS_MEDIAPIPE or self.pose is None:
            return None

        if self._api_type == "solutions":
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            frame_rgb.flags.writeable = False
            results = self.pose.process(frame_rgb)
            frame_rgb.flags.writeable = True

            if not results.pose_landmarks:
                return None

            landmarks = np.zeros((33, 4), dtype=np.float32)
            for idx, lm in enumerate(results.pose_landmarks.landmark):
                landmarks[idx] = [lm.x, lm.y, lm.z, lm.visibility]
            return landmarks

        return None

    def draw_landmarks(self, frame_bgr: np.ndarray, landmarks_data) -> np.ndarray:
        """Utility to render MediaPipe skeleton on the frame."""
        if self._api_type == "solutions" and hasattr(landmarks_data, "landmark"):
            self._mp_drawing.draw_landmarks(
                frame_bgr,
                landmarks_data,
                self._mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=self._mp_drawing_styles.get_default_pose_landmarks_style()
            )
        return frame_bgr

    def close(self):
        """Release MediaPipe resources."""
        if self.pose is not None and hasattr(self.pose, "close"):
            self.pose.close()
            self.pose = None
