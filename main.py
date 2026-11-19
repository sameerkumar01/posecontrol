"""Pose-Controlled Fighting Game - Main Application Entrypoint.

Integrates:
- Real-time video ingestion (Webcam or Synthetic Pose Stream)
- MediaPipe BlazePose 3D human landmark extraction
- 3-stage invariant biomechanical normalizer (root-centering, torso scale, coronal yaw)
- 138-dimensional kinematic feature extraction
- 30-frame circular sliding window buffer
- Causal 2-layer LSTM sequence classifier with dual-rate EMA smoothing & debouncing
- Deterministic 60 Hz combat engine with AI opponent across 3 difficulty tiers
- Arcade OpenCV HUD overlay with neon wireframes, glowing hitboxes, and particle sparks
"""

import argparse
import sys
import time
from typing import Optional, Tuple
import cv2
import numpy as np

from pose_control.config import (
    ActionClass,
    PlayerState,
    WINDOW_WIDTH,
    WINDOW_HEIGHT,
    WINDOW_TITLE,
    CAMERA_FPS,
    COMBAT_TICK_RATE,
    FIXED_TIMESTEP,
)
from pose_control.vision.pose_detector import BlazePoseDetector
from pose_control.vision.normalizer import PoseNormalizer
from pose_control.features.extractor import FeatureExtractor
from pose_control.inference.engine import GestureInferenceEngine
from pose_control.game.combat_engine import CombatEngine
from pose_control.ui.hud import ArcadeHUDRenderer
from pose_control.dataset.generator import SyntheticPoseGenerator


class PoseCombatApplication:
    """Master application coordinator running the decoupled pipeline."""

    def __init__(
        self,
        camera_id: int = 0,
        model_path: Optional[str] = None,
        ai_difficulty: str = "Medium",
        demo_mode: bool = False,
        headless: bool = False,
    ):
        self.camera_id = camera_id
        self.demo_mode = demo_mode
        self.headless = headless

        # Core Subsystems
        self.detector = BlazePoseDetector()
        self.normalizer = PoseNormalizer()
        self.extractor = FeatureExtractor(fps=CAMERA_FPS)
        self.inference_engine = GestureInferenceEngine(model_path=model_path)
        self.combat_engine = CombatEngine(ai_difficulty=ai_difficulty)
        self.renderer = ArcadeHUDRenderer(width=WINDOW_WIDTH, height=WINDOW_HEIGHT)

        # Video Capture
        self.cap: Optional[cv2.VideoCapture] = None
        self.synthetic_frame_counter = 0
        self.synthetic_current_action = ActionClass.NONE
        self.synthetic_action_timer = 0

        # Performance Monitoring
        self.fps = 30.0
        self.latency_ms = 35.0
        self.last_frame_time = time.monotonic()
        self.accumulator = 0.0

    def start_camera(self) -> bool:
        """Initializes OpenCV video capture with low-latency buffer settings."""
        if self.demo_mode:
            print("[Video] Operating in Synthetic / Demo Pose Stream Mode.")
            return True

        print(f"[Video] Initializing webcam sensor index: {self.camera_id}...")
        self.cap = cv2.VideoCapture(self.camera_id)
        if not self.cap.isOpened():
            print(f"[Warning] Failed to open webcam #{self.camera_id}. Falling back to Synthetic Demo Mode!")
            self.demo_mode = True
            return True

        # Request 720p @ 30 FPS and set buffer size to 1 to minimize latency
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, WINDOW_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, WINDOW_HEIGHT)
        self.cap.set(cv2.CAP_PROP_FPS, CAMERA_FPS)
        try:
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass
        return True

    def _get_next_frame(self) -> Tuple[bool, np.ndarray, Optional[np.ndarray]]:
        """Retrieves next video frame and landmarks from live webcam or synthetic generator."""
        if not self.demo_mode and self.cap is not None and self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                return False, np.zeros((WINDOW_HEIGHT, WINDOW_WIDTH, 3), dtype=np.uint8), None
            # Mirror webcam horizontally for intuitive fighting game control
            frame = cv2.flip(frame, 1)
            landmarks = self.detector.process_frame(frame)
            return True, frame, landmarks
        else:
            # Synthetic / Demo Stream Generator
            self.synthetic_frame_counter += 1
            if self.synthetic_action_timer <= 0:
                # Pick a periodic combat action (NONE, JAB, UPPERCUT, BLOCK)
                actions = [ActionClass.NONE, ActionClass.NONE, ActionClass.JAB, ActionClass.UPPERCUT, ActionClass.BLOCK]
                self.synthetic_current_action = np.random.choice(actions)
                self.synthetic_action_timer = np.random.randint(45, 90)
            else:
                self.synthetic_action_timer -= 1

            t_idx = self.synthetic_frame_counter % 30
            raw_seq = SyntheticPoseGenerator.generate_sequence(self.synthetic_current_action, seq_len=30)
            synthetic_pose = raw_seq[t_idx]

            # Generate synthetic dark arena canvas
            frame = np.zeros((WINDOW_HEIGHT, WINDOW_WIDTH, 3), dtype=np.uint8)
            # Subtle gradient background
            for y in range(WINDOW_HEIGHT):
                intensity = int(18 + 22 * (y / WINDOW_HEIGHT))
                frame[y, :] = (intensity, intensity + 4, intensity + 8)

            # Floor grid line
            cv2.line(frame, (0, 560), (WINDOW_WIDTH, 560), (45, 45, 60), 2)
            cv2.putText(
                frame,
                "SYNTHETIC DEMO STREAM (Press 'D' to toggle webcam, 'J'/'U'/'B' to punch)",
                (WINDOW_WIDTH // 2 - 280, 590),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (90, 90, 110),
                1,
            )

            return True, frame, synthetic_pose

    def run(self, max_frames: Optional[int] = None):
        """Main game loop combining vision, LSTM inference, combat tick, and rendering."""
        self.start_camera()
        print("\n=======================================================")
        print(" Pose-Controlled Fighting Game (MediaPipe + LSTM 60Hz)")
        print(" Controls:")
        print("   [SPACE]    : Reset Round")
        print("   [1, 2, 3]  : Switch AI Difficulty (Easy, Med, Hard)")
        print("   [J, U, B]  : Manual gesture triggers (Jab, Uppercut, Block)")
        print("   [D]        : Toggle Demo / Synthetic Stream")
        print("   [Q / ESC]  : Exit Game")
        print("=======================================================\n")

        self.combat_engine.reset_round()
        frame_idx = 0

        try:
            while True:
                t_start = time.monotonic()
                dt = t_start - self.last_frame_time
                self.last_frame_time = t_start
                self.fps = 0.9 * self.fps + 0.1 * (1.0 / max(dt, 1e-4))

                # 1. Video Ingestion & Pose Extraction
                success, frame, raw_landmarks = self._get_next_frame()
                if not success:
                    print("[Warning] Video stream ended or camera dropped.")
                    break

                # 2. Invariant Normalization & Kinematic Features
                if raw_landmarks is not None:
                    norm_coords, _ = self.normalizer.normalize(raw_landmarks)
                    features = self.extractor.extract(norm_coords)
                else:
                    norm_coords = None
                    features = np.zeros(138, dtype=np.float32)

                # 3. Gesture Classification & Conditioning
                detected_action, conf, smoothed_probs = self.inference_engine.step(features, timestamp=t_start)

                # 4. Deterministic 60 Hz Combat Engine Updates
                self.accumulator += dt
                while self.accumulator >= FIXED_TIMESTEP:
                    if detected_action != ActionClass.NONE:
                        self.combat_engine.process_player_input(detected_action)
                    self.combat_engine.update(FIXED_TIMESTEP)
                    self.accumulator -= FIXED_TIMESTEP

                # 5. Pipeline Latency Measurement
                t_render = time.monotonic()
                self.latency_ms = (t_render - t_start) * 1000.0

                # 6. Render Composite Arcade HUD
                display_landmarks = norm_coords if self.demo_mode else raw_landmarks
                composite = self.renderer.render(
                    canvas=frame,
                    combat_engine=self.combat_engine,
                    player_landmarks=display_landmarks,
                    detected_action=detected_action,
                    confidence=conf,
                    smoothed_probs=smoothed_probs,
                    fps=self.fps,
                    latency_ms=self.latency_ms,
                )

                if not self.headless:
                    cv2.imshow(WINDOW_TITLE, composite)
                    key = cv2.waitKey(1) & 0xFF
                    if key in (ord("q"), 27):  # 'q' or ESC
                        break
                    elif key == ord(" "):
                        self.combat_engine.reset_round()
                    elif key == ord("1"):
                        self.combat_engine.ai.reset("Easy")
                        self.combat_engine.active_banner_text = "AI: EASY DIFFICULTY"
                        self.combat_engine.banner_timer = 40
                    elif key == ord("2"):
                        self.combat_engine.ai.reset("Medium")
                        self.combat_engine.active_banner_text = "AI: MEDIUM DIFFICULTY"
                        self.combat_engine.banner_timer = 40
                    elif key == ord("3"):
                        self.combat_engine.ai.reset("Hard")
                        self.combat_engine.active_banner_text = "AI: HARD DIFFICULTY"
                        self.combat_engine.banner_timer = 40
                    elif key in (ord("j"), ord("J")):
                        self.combat_engine.process_player_input(ActionClass.JAB)
                    elif key in (ord("u"), ord("U")):
                        self.combat_engine.process_player_input(ActionClass.UPPERCUT)
                    elif key in (ord("b"), ord("B")):
                        self.combat_engine.process_player_input(ActionClass.BLOCK)
                    elif key in (ord("d"), ord("D")):
                        self.demo_mode = not self.demo_mode
                        print(f"[Mode] Toggled Demo Mode: {self.demo_mode}")

                frame_idx += 1
                if max_frames is not None and frame_idx >= max_frames:
                    break

        finally:
            self.close()

    def close(self):
        """Cleans up video and system resources."""
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        self.detector.close()
        cv2.destroyAllWindows()
        print("[System] Application shut down cleanly.")


def main():
    parser = argparse.ArgumentParser(description="Pose-Controlled Fighting Game")
    parser.add_argument("--camera", type=int, default=0, help="Camera index.")
    parser.add_argument("--model", type=str, default=None, help="Path to trained Keras model file.")
    parser.add_argument("--difficulty", type=str, default="Medium", choices=["Easy", "Medium", "Hard"])
    parser.add_argument("--demo", action="store_true", help="Run with synthetic demo pose stream.")
    parser.add_argument("--headless", action="store_true", help="Run in headless benchmark mode without GUI window.")
    parser.add_argument("--frames", type=int, default=None, help="Stop after N frames (for benchmarking).")
    args = parser.parse_args()

    app = PoseCombatApplication(
        camera_id=args.camera,
        model_path=args.model,
        ai_difficulty=args.difficulty,
        demo_mode=args.demo,
        headless=args.headless,
    )
    app.run(max_frames=args.frames)


if __name__ == "__main__":
    main()
