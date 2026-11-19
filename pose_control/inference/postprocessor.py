"""Inference Post-Processing, EMA Smoothing, and Debounce Conditioning Engine.

Performs 4-stage conditioning on raw model probabilities:
1. Dual-Rate Exponential Moving Average (EMA) smoothing
2. Class-tailored confidence threshold evaluation
3. Multi-frame temporal persistence hysteresis
4. Refractory mechanical cooldown gating
"""

import time
from typing import Dict, Optional, Tuple
import numpy as np

from pose_control.config import (
    ActionClass,
    CONFIDENCE_THRESHOLDS,
    MIN_ACTIVATION_FRAMES,
    REFRACTORY_PERIODS,
    EMA_ALPHA_STRIKE,
    EMA_ALPHA_STEADY,
    NUM_CLASSES,
)


class ProbabilityPostProcessor:
    """Stabilizes raw model output probabilities and manages trigger events."""

    def __init__(
        self,
        thresholds: Optional[Dict[ActionClass, float]] = None,
        min_frames: Optional[Dict[ActionClass, int]] = None,
        refractory_periods: Optional[Dict[ActionClass, float]] = None,
    ):
        self.thresholds = thresholds or CONFIDENCE_THRESHOLDS
        self.min_frames = min_frames or MIN_ACTIVATION_FRAMES
        self.refractory_periods = refractory_periods or REFRACTORY_PERIODS

        # State tracking
        self.smoothed_probs = np.zeros(NUM_CLASSES, dtype=np.float32)
        self.smoothed_probs[int(ActionClass.NONE)] = 1.0

        self.consecutive_frames = {c: 0 for c in ActionClass}
        self.last_trigger_times = {c: 0.0 for c in ActionClass}

    def reset(self):
        """Reset internal smoothing and debounce state."""
        self.smoothed_probs.fill(0.0)
        self.smoothed_probs[int(ActionClass.NONE)] = 1.0
        for c in ActionClass:
            self.consecutive_frames[c] = 0
            self.last_trigger_times[c] = 0.0

    def process(
        self,
        raw_probs: np.ndarray,
        current_time: Optional[float] = None,
    ) -> Tuple[ActionClass, float, np.ndarray]:
        """Post-processes raw softmax probabilities.

        Args:
            raw_probs: Array of shape (4,) representing [P_none, P_jab, P_uppercut, P_block].
            current_time: Monotonic timestamp (seconds). If None, time.monotonic() is used.

        Returns:
            triggered_action: The acknowledged combat ActionClass (or ActionClass.NONE).
            confidence: The smoothed confidence of the acknowledged action.
            smoothed_probs: The full smoothed probability array.
        """
        if current_time is None:
            current_time = time.monotonic()

        assert raw_probs.shape == (NUM_CLASSES,), f"Expected shape ({NUM_CLASSES},), got {raw_probs.shape}"

        # -------------------------------------------------------------------
        # Stage 1: Dual-Rate Exponential Moving Average (EMA)
        # -------------------------------------------------------------------
        for idx, act in enumerate(ActionClass):
            alpha = EMA_ALPHA_STRIKE if act in (ActionClass.JAB, ActionClass.UPPERCUT) else EMA_ALPHA_STEADY
            self.smoothed_probs[idx] = alpha * raw_probs[idx] + (1.0 - alpha) * self.smoothed_probs[idx]

        # Normalize smoothed probabilities to sum to 1.0
        prob_sum = float(np.sum(self.smoothed_probs))
        if prob_sum > 1e-6:
            self.smoothed_probs /= prob_sum

        # -------------------------------------------------------------------
        # Stage 2 & 3: Thresholding & Multi-Frame Persistence Hysteresis
        # -------------------------------------------------------------------
        candidate_action = ActionClass.NONE
        candidate_conf = float(self.smoothed_probs[int(ActionClass.NONE)])

        for act in [ActionClass.UPPERCUT, ActionClass.JAB, ActionClass.BLOCK]:
            idx = int(act)
            prob = float(self.smoothed_probs[idx])
            thresh = self.thresholds[act]
            release_margin = thresh * 0.85

            if prob >= thresh:
                self.consecutive_frames[act] += 1
            elif prob < release_margin:
                self.consecutive_frames[act] = 0

            # Check if persistence requirement is satisfied
            if self.consecutive_frames[act] >= self.min_frames[act]:
                candidate_action = act
                candidate_conf = prob
                break  # Priority order: Uppercut > Jab > Block

        # If no active gesture qualifies, default to NONE
        if candidate_action == ActionClass.NONE:
            return ActionClass.NONE, candidate_conf, self.smoothed_probs.copy()

        # -------------------------------------------------------------------
        # Stage 4: Refractory Mechanical Cooldown Gate
        # -------------------------------------------------------------------
        cooldown = self.refractory_periods[candidate_action]
        elapsed = current_time - self.last_trigger_times[candidate_action]

        if cooldown > 0.0 and elapsed < cooldown:
            # Action suppressed by refractory debounce gate
            return ActionClass.NONE, candidate_conf, self.smoothed_probs.copy()

        # Action is valid and acknowledged!
        self.last_trigger_times[candidate_action] = current_time

        # For instantaneous strikes (JAB, UPPERCUT), reset streak counter so they require new windup
        if candidate_action in (ActionClass.JAB, ActionClass.UPPERCUT):
            self.consecutive_frames[candidate_action] = 0

        return candidate_action, candidate_conf, self.smoothed_probs.copy()
