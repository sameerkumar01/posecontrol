"""Arcade HUD and Visual Effects Overlay Renderer.

Draws high-energy fighting game overlays directly onto the video canvas using OpenCV:
- Health bars, Guard bars, Round Timer, and Round Counters
- Dynamic skeletal wireframe with state-colored fist hitboxes
- AI Opponent Avatar with attack telegraphing cues
- Impact particle sparks, micro-screen shake, and combo counters
- Real-time gesture probability telemetry and frame data advantage
"""

import math
from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np

from pose_control.config import (
    ActionClass,
    PlayerState,
    WINDOW_WIDTH,
    WINDOW_HEIGHT,
    COLOR_NEON_CYAN,
    COLOR_NEON_MAGENTA,
    COLOR_NEON_YELLOW,
    COLOR_HEALTH_GREEN,
    COLOR_HEALTH_RED,
    COLOR_GUARD_BLUE,
    COLOR_HIT_GOLD,
    COLOR_WHITE,
    COLOR_BLACK,
    NUM_CLASSES,
    CLASS_NAMES,
)
from pose_control.game.combat_engine import CombatEngine


class ArcadeHUDRenderer:
    """Renders combat HUD, character avatars, wireframes, and particle effects."""

    # Skeleton connections based on MediaPipe BlazePose topology
    POSE_BONES = [
        (11, 12),  # Shoulders
        (11, 13), (13, 15),  # Left arm
        (12, 14), (14, 16),  # Right arm
        (11, 23), (12, 24),  # Torso sides
        (23, 24),  # Hips
        (23, 25), (25, 27),  # Left leg
        (24, 26), (26, 28),  # Right leg
    ]

    def __init__(self, width: int = WINDOW_WIDTH, height: int = WINDOW_HEIGHT):
        self.width = width
        self.height = height

    def render(
        self,
        canvas: np.ndarray,
        combat_engine: CombatEngine,
        player_landmarks: Optional[np.ndarray],
        detected_action: ActionClass,
        confidence: float,
        smoothed_probs: np.ndarray,
        fps: float,
        latency_ms: float,
    ) -> np.ndarray:
        """Renders the full composite fighting game frame.

        Args:
            canvas: Base image (H, W, 3) BGR (camera feed or synthetic background).
            combat_engine: Active combat simulation state.
            player_landmarks: Array (33, 3) or (33, 4) in screen coordinates or normalized.
            detected_action: Currently acknowledged action.
            confidence: Confidence score of detected action.
            smoothed_probs: 4-element class probability array.
            fps: Current measured frames per second.
            latency_ms: End-to-end pipeline latency in milliseconds.

        Returns:
            composite_frame: Rendered canvas with HUD overlays.
        """
        output = canvas.copy()
        if output.shape[:2] != (self.height, self.width):
            output = cv2.resize(output, (self.width, self.height))

        # -------------------------------------------------------------------
        # 1. Screen Shake
        # -------------------------------------------------------------------
        if combat_engine.screen_shake_frames > 0:
            intensity = combat_engine.screen_shake_intensity
            dx = int(np.random.uniform(-intensity, intensity))
            dy = int(np.random.uniform(-intensity, intensity))
            m = np.float32([[1, 0, dx], [0, 1, dy]])
            output = cv2.warpAffine(output, m, (self.width, self.height), borderMode=cv2.BORDER_REFLECT)

        # -------------------------------------------------------------------
        # 2. Draw Player Skeletal Wireframe (Mirrored)
        # -------------------------------------------------------------------
        if player_landmarks is not None:
            self._draw_player_skeleton(output, player_landmarks, combat_engine.player_fsm.state)

        # -------------------------------------------------------------------
        # 3. Draw AI Opponent Avatar
        # -------------------------------------------------------------------
        self._draw_ai_avatar(output, combat_engine)

        # -------------------------------------------------------------------
        # 4. Render Visual Hit Sparks
        # -------------------------------------------------------------------
        self._draw_hit_sparks(output, combat_engine.hit_sparks)

        # -------------------------------------------------------------------
        # 5. Top Bar: Health, Guard, Round Timer, Scores
        # -------------------------------------------------------------------
        self._draw_top_bar(output, combat_engine)

        # -------------------------------------------------------------------
        # 6. Combo Counters & Match Banners
        # -------------------------------------------------------------------
        self._draw_combat_status(output, combat_engine)

        # -------------------------------------------------------------------
        # 7. Bottom Bar: Telemetry & Probability Gauges
        # -------------------------------------------------------------------
        self._draw_bottom_telemetry(
            output,
            detected_action,
            confidence,
            smoothed_probs,
            fps,
            latency_ms,
            combat_engine,
        )

        return output

    def _draw_top_bar(self, img: np.ndarray, combat: CombatEngine):
        """Draws health meters, guard meters, timer, and rounds."""
        bar_w = 420
        bar_h = 24
        bar_y = 35

        # Player Health Bar (Left)
        p_health_ratio = max(0.0, combat.player_health / combat.player_max_health)
        cv2.rectangle(img, (50, bar_y), (50 + bar_w, bar_y + bar_h), COLOR_BLACK, -1)
        p_fill = int(bar_w * p_health_ratio)
        p_col = COLOR_HEALTH_GREEN if p_health_ratio > 0.3 else COLOR_HEALTH_RED
        if p_fill > 0:
            cv2.rectangle(img, (50 + (bar_w - p_fill), bar_y), (50 + bar_w, bar_y + bar_h), p_col, -1)
        cv2.rectangle(img, (50, bar_y), (50 + bar_w, bar_y + bar_h), COLOR_WHITE, 2)
        cv2.putText(img, f"P1: {int(combat.player_health)} HP", (55, bar_y - 8), cv2.FONT_HERSHEY_DUPLEX, 0.6, COLOR_WHITE, 1)

        # Player Guard Bar
        p_guard_ratio = max(0.0, combat.player_guard / 100.0)
        cv2.rectangle(img, (50, bar_y + bar_h + 4), (50 + int(bar_w * 0.7 * p_guard_ratio), bar_y + bar_h + 10), COLOR_GUARD_BLUE, -1)
        cv2.rectangle(img, (50, bar_y + bar_h + 4), (50 + int(bar_w * 0.7), bar_y + bar_h + 10), COLOR_WHITE, 1)

        # AI Health Bar (Right)
        ai_health_ratio = max(0.0, combat.ai.health / combat.ai.settings.max_health)
        ai_x = self.width - 50 - bar_w
        cv2.rectangle(img, (ai_x, bar_y), (ai_x + bar_w, bar_y + bar_h), COLOR_BLACK, -1)
        ai_fill = int(bar_w * ai_health_ratio)
        ai_col = COLOR_HEALTH_GREEN if ai_health_ratio > 0.3 else COLOR_HEALTH_RED
        if ai_fill > 0:
            cv2.rectangle(img, (ai_x, bar_y), (ai_x + ai_fill, bar_y + bar_h), ai_col, -1)
        cv2.rectangle(img, (ai_x, bar_y), (ai_x + bar_w, bar_y + bar_h), COLOR_WHITE, 2)
        cv2.putText(img, f"AI [{combat.ai.settings.name}]: {int(combat.ai.health)} HP", (ai_x, bar_y - 8), cv2.FONT_HERSHEY_DUPLEX, 0.6, COLOR_WHITE, 1)

        # AI Guard Bar
        ai_guard_ratio = max(0.0, combat.ai.guard / 100.0)
        cv2.rectangle(img, (ai_x + int(bar_w * 0.3), bar_y + bar_h + 4), (ai_x + int(bar_w * 0.3) + int(bar_w * 0.7 * ai_guard_ratio), bar_y + bar_h + 10), COLOR_GUARD_BLUE, -1)
        cv2.rectangle(img, (ai_x + int(bar_w * 0.3), bar_y + bar_h + 4), (ai_x + bar_w, bar_y + bar_h + 10), COLOR_WHITE, 1)

        # Center Round Timer (99s)
        timer_box_w = 90
        timer_box_x = (self.width - timer_box_w) // 2
        cv2.rectangle(img, (timer_box_x, 20), (timer_box_x + timer_box_w, 75), (20, 20, 30), -1)
        cv2.rectangle(img, (timer_box_x, 20), (timer_box_x + timer_box_w, 75), COLOR_NEON_YELLOW, 2)
        time_text = f"{int(combat.round_time):02d}"
        cv2.putText(img, time_text, (timer_box_x + 16, 62), cv2.FONT_HERSHEY_TRIPLEX, 1.2, COLOR_NEON_YELLOW, 2)

        # Rounds Won Indicators
        for r in range(2):
            p_won = r < combat.player_rounds_won
            ai_won = r < combat.ai_rounds_won
            # Player circles
            cv2.circle(img, (timer_box_x - 30 - r * 25, 48), 8, COLOR_HIT_GOLD if p_won else (60, 60, 60), -1)
            cv2.circle(img, (timer_box_x - 30 - r * 25, 48), 8, COLOR_WHITE, 1)
            # AI circles
            cv2.circle(img, (timer_box_x + timer_box_w + 30 + r * 25, 48), 8, COLOR_HIT_GOLD if ai_won else (60, 60, 60), -1)
            cv2.circle(img, (timer_box_x + timer_box_w + 30 + r * 25, 48), 8, COLOR_WHITE, 1)

    def _draw_player_skeleton(self, img: np.ndarray, landmarks: np.ndarray, state: PlayerState):
        """Draws normalized or screen-space skeletal wireframe on player feed."""
        coords_2d = {}
        for idx in range(min(33, len(landmarks))):
            x_raw, y_raw = landmarks[idx, 0], landmarks[idx, 1]
            # Convert normalized 0-1 to pixel coords in player quadrant (left half)
            if 0.0 <= x_raw <= 1.0 and 0.0 <= y_raw <= 1.0:
                px = int(x_raw * (self.width // 2))
                py = int(y_raw * self.height)
            else:
                # In normalized invariant space [-1, 1], map around center (320, 380)
                px = int(320 + x_raw * 240)
                py = int(380 - y_raw * 240)
            coords_2d[idx] = (px, py)

        # Draw bones
        for u, v in self.POSE_BONES:
            if u in coords_2d and v in coords_2d:
                cv2.line(img, coords_2d[u], coords_2d[v], COLOR_NEON_CYAN, 3, cv2.LINE_AA)

        # Highlight Head Hurtbox
        if 0 in coords_2d:
            cv2.circle(img, coords_2d[0], 22, (200, 100, 100), 2, cv2.LINE_AA)

        # Fist Hitbox Coloring based on Player FSM State
        fist_color = COLOR_HEALTH_GREEN  # Idle
        if state in (PlayerState.WINDUP, PlayerState.ACTIVE):
            fist_color = COLOR_HEALTH_RED  # Attacking
        elif state == PlayerState.BLOCKING:
            fist_color = COLOR_GUARD_BLUE  # Blocking
        elif state in (PlayerState.HIT_STUN, PlayerState.KNOCKDOWN):
            fist_color = COLOR_HIT_GOLD    # In hitstun

        for wrist_idx in (15, 16):
            if wrist_idx in coords_2d:
                radius = 20 if state == PlayerState.ACTIVE else 14
                cv2.circle(img, coords_2d[wrist_idx], radius, fist_color, -1, cv2.LINE_AA)
                cv2.circle(img, coords_2d[wrist_idx], radius + 4, COLOR_WHITE, 2, cv2.LINE_AA)

    def _draw_ai_avatar(self, img: np.ndarray, combat: CombatEngine):
        """Draws arcade-style opponent fighter avatar on right side of screen."""
        center_x = 940
        center_y = 380
        ai_state = combat.ai.fsm.state

        # Color based on state
        body_color = (80, 80, 180)
        if ai_state == PlayerState.BLOCKING:
            body_color = COLOR_GUARD_BLUE
        elif ai_state in (PlayerState.WINDUP, PlayerState.ACTIVE):
            body_color = COLOR_HEALTH_RED
        elif ai_state in (PlayerState.HIT_STUN, PlayerState.KNOCKDOWN):
            body_color = (60, 180, 240)

        # Telegraph Warning Flash (AI preparing strike)
        if combat.ai.is_telegraphing:
            pulse = int(128 + 127 * math.sin(combat.ai.telegraph_timer * 0.4))
            warning_col = (0, pulse, 255)
            cv2.circle(img, (center_x, center_y - 120), 48, warning_col, 5, cv2.LINE_AA)
            cv2.putText(img, "! INCOMING STRIKE !", (center_x - 100, center_y - 180), cv2.FONT_HERSHEY_DUPLEX, 0.65, (0, 220, 255), 2)

        # Head Hurtbox
        cv2.circle(img, (center_x, center_y - 120), 38, body_color, -1, cv2.LINE_AA)
        cv2.circle(img, (center_x, center_y - 120), 38, COLOR_WHITE, 2, cv2.LINE_AA)

        # Torso
        cv2.line(img, (center_x, center_y - 82), (center_x, center_y + 90), body_color, 24, cv2.LINE_AA)

        # Arms based on AI action
        if ai_state == PlayerState.BLOCKING:
            # Crossed guard arms
            cv2.line(img, (center_x - 30, center_y - 40), (center_x - 10, center_y - 110), COLOR_GUARD_BLUE, 14, cv2.LINE_AA)
            cv2.line(img, (center_x + 30, center_y - 40), (center_x + 10, center_y - 110), COLOR_GUARD_BLUE, 14, cv2.LINE_AA)
        elif ai_state == PlayerState.ACTIVE:
            # Punch extension forward towards player
            cv2.line(img, (center_x, center_y - 40), (center_x - 130, center_y - 60), COLOR_HEALTH_RED, 16, cv2.LINE_AA)
            cv2.circle(img, (center_x - 130, center_y - 60), 22, (50, 50, 255), -1)
            cv2.circle(img, (center_x - 130, center_y - 60), 26, COLOR_WHITE, 2)
        else:
            # Normal boxing stance arms
            cv2.line(img, (center_x, center_y - 40), (center_x - 50, center_y - 20), body_color, 12, cv2.LINE_AA)
            cv2.line(img, (center_x - 50, center_y - 20), (center_x - 70, center_y - 70), body_color, 12, cv2.LINE_AA)
            cv2.line(img, (center_x, center_y - 40), (center_x + 40, center_y - 20), body_color, 12, cv2.LINE_AA)
            cv2.line(img, (center_x + 40, center_y - 20), (center_x + 30, center_y - 70), body_color, 12, cv2.LINE_AA)

        # Legs
        cv2.line(img, (center_x, center_y + 90), (center_x - 45, center_y + 220), body_color, 14, cv2.LINE_AA)
        cv2.line(img, (center_x, center_y + 90), (center_x + 45, center_y + 220), body_color, 14, cv2.LINE_AA)

        # State text tag
        state_str = ai_state.name
        cv2.putText(img, f"AI: {state_str}", (center_x - 60, center_y + 250), cv2.FONT_HERSHEY_DUPLEX, 0.6, COLOR_WHITE, 1)

    def _draw_hit_sparks(self, img: np.ndarray, sparks: List):
        """Draws animated particle sparks."""
        for spark in sparks:
            for px, py, _, _ in spark.particles:
                ix, iy = int(px), int(py)
                if 0 <= ix < self.width and 0 <= iy < self.height:
                    cv2.circle(img, (ix, iy), 3, spark.color, -1)

    def _draw_combat_status(self, img: np.ndarray, combat: CombatEngine):
        """Renders banner announcements and combo hit counters."""
        # Active Banner Text
        if combat.banner_timer > 0 and combat.active_banner_text:
            text = combat.active_banner_text
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_TRIPLEX, 1.1, 2)[0]
            bx = (self.width - text_size[0]) // 2
            by = 140
            cv2.rectangle(img, (bx - 20, by - 36), (bx + text_size[0] + 20, by + 12), (10, 10, 10), -1)
            cv2.rectangle(img, (bx - 20, by - 36), (bx + text_size[0] + 20, by + 12), COLOR_NEON_YELLOW, 2)
            cv2.putText(img, text, (bx, by), cv2.FONT_HERSHEY_TRIPLEX, 1.1, COLOR_NEON_YELLOW, 2)

        # Player Combo Counter
        if combat.player_combo > 1:
            combo_str = f"{combat.player_combo} HITS!"
            cv2.putText(img, combo_str, (60, 130), cv2.FONT_HERSHEY_TRIPLEX, 1.3, (50, 150, 255), 3)
            cv2.putText(img, "COMBO", (60, 160), cv2.FONT_HERSHEY_DUPLEX, 0.7, COLOR_WHITE, 1)

    def _draw_bottom_telemetry(
        self,
        img: np.ndarray,
        action: ActionClass,
        conf: float,
        smoothed_probs: np.ndarray,
        fps: float,
        latency_ms: float,
        combat: CombatEngine,
    ):
        """Renders real-time telemetry bar at bottom of canvas."""
        bar_y = self.height - 85
        cv2.rectangle(img, (0, bar_y), (self.width, self.height), (20, 20, 25), -1)
        cv2.line(img, (0, bar_y), (self.width, bar_y), COLOR_NEON_CYAN, 2)

        # Action Detection Label
        act_col = COLOR_HEALTH_GREEN if action != ActionClass.NONE else (180, 180, 180)
        cv2.putText(img, f"ACTION: {action.name}", (30, bar_y + 35), cv2.FONT_HERSHEY_DUPLEX, 0.85, act_col, 2)
        cv2.putText(img, f"CONF: {conf:.2f}", (30, bar_y + 65), cv2.FONT_HERSHEY_DUPLEX, 0.6, COLOR_WHITE, 1)

        # 4-Class Probability Mini-Bars
        gauge_x = 360
        gauge_w = 90
        gauge_h = 14
        for idx, cls_name in enumerate(["NONE", "JAB", "UPPER", "BLOCK"]):
            prob = float(smoothed_probs[idx])
            gy = bar_y + 15 + idx * 16
            cv2.putText(img, f"{cls_name[:4]}:", (gauge_x, gy + 11), cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOR_WHITE, 1)
            # Fill bar
            cv2.rectangle(img, (gauge_x + 55, gy), (gauge_x + 55 + gauge_w, gy + gauge_h), (50, 50, 50), -1)
            fill_w = int(gauge_w * prob)
            bar_c = COLOR_NEON_CYAN if idx != 0 else (120, 120, 120)
            if fill_w > 0:
                cv2.rectangle(img, (gauge_x + 55, gy), (gauge_x + 55 + fill_w, gy + gauge_h), bar_c, -1)
            cv2.rectangle(img, (gauge_x + 55, gy), (gauge_x + 55 + gauge_w, gy + gauge_h), COLOR_WHITE, 1)

        # Performance & Frame Data Advantage
        perf_x = 760
        cv2.putText(img, f"FPS: {fps:.1f}", (perf_x, bar_y + 30), cv2.FONT_HERSHEY_DUPLEX, 0.6, COLOR_NEON_YELLOW, 1)
        cv2.putText(img, f"LATENCY: {latency_ms:.1f} ms", (perf_x, bar_y + 60), cv2.FONT_HERSHEY_DUPLEX, 0.6, COLOR_NEON_YELLOW, 1)

        # Frame Data Advantage
        adv_x = 1000
        p_state = combat.player_fsm.state
        if combat.player_fsm.current_move is not None:
            move_name = combat.player_fsm.current_move.name
            frame_num = combat.player_fsm.frame_in_state
            cv2.putText(img, f"MOVE: {move_name} (F:{frame_num})", (adv_x, bar_y + 30), cv2.FONT_HERSHEY_DUPLEX, 0.6, COLOR_WHITE, 1)
            cv2.putText(img, f"PHASE: {p_state.name}", (adv_x, bar_y + 60), cv2.FONT_HERSHEY_DUPLEX, 0.6, COLOR_HIT_GOLD, 1)
        else:
            cv2.putText(img, f"STATE: {p_state.name}", (adv_x, bar_y + 45), cv2.FONT_HERSHEY_DUPLEX, 0.65, (180, 180, 180), 1)
