"""Real-Time Gesture Inference Engine.

Orchestrates temporal sequence buffering, recurrent neural network forward evaluation,
probability smoothing, and debounced action dispatching.
"""

from typing import Optional, Tuple
import numpy as np

from pose_control.config import ActionClass, TOTAL_FEATURE_DIM
from pose_control.features.buffer import FeatureWindowBuffer
from pose_control.inference.postprocessor import ProbabilityPostProcessor
from pose_control.model.network import LightweightNumpyLSTM, HAS_TENSORFLOW

if HAS_TENSORFLOW:
    import tensorflow as tf


class GestureInferenceEngine:
    """End-to-end real-time sequence inference pipeline."""

    def __init__(
        self,
        model_path: Optional[str] = None,
        use_lightweight_fallback: bool = False,
    ):
        self.buffer = FeatureWindowBuffer()
        self.postprocessor = ProbabilityPostProcessor()
        self.tf_model = None

        if not use_lightweight_fallback and HAS_TENSORFLOW and model_path is not None:
            try:
                from pose_control.model.network import MultiClassFocalLoss
                self.tf_model = tf.keras.models.load_model(
                    model_path,
                    custom_objects={"MultiClassFocalLoss": MultiClassFocalLoss},
                    compile=False,
                )
                print(f"[Model] Successfully loaded Keras LSTM model from {model_path}")
            except Exception as e:
                print(f"[Warning] Failed to load Keras model from {model_path}: {e}")
                self.tf_model = None

        # Always initialize lightweight evaluator for instantaneous fallback or CPU mode
        self.numpy_evaluator = LightweightNumpyLSTM()

    def reset(self):
        """Reset temporal buffer and smoothing states."""
        self.buffer.reset()
        self.postprocessor.reset()

    def step(
        self,
        feature_vector: np.ndarray,
        timestamp: Optional[float] = None,
    ) -> Tuple[ActionClass, float, np.ndarray]:
        """Processes a single new 138-dim feature frame and performs inference.

        Args:
            feature_vector: 1D array of shape (138,).
            timestamp: Monotonic timestamp for debounce timing.

        Returns:
            action: Evaluated ActionClass.
            confidence: Smoothed confidence score.
            probabilities: Smoothed class probability vector [NONE, JAB, UPPERCUT, BLOCK].
        """
        self.buffer.append(feature_vector)
        tensor = self.buffer.get_tensor()  # Shape: (1, 30, 138)

        # Forward pass
        if self.tf_model is not None:
            raw_probs = self.tf_model(tensor, training=False).numpy()[0]
        else:
            raw_probs = self.numpy_evaluator.predict(tensor)[0]

        # Post-process, smooth, threshold, and debounce
        action, conf, smoothed = self.postprocessor.process(raw_probs, current_time=timestamp)
        return action, conf, smoothed
