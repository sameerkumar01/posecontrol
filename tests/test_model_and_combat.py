"""Unit Tests for Model Architecture, Post-Processor, and Combat Engine."""

import numpy as np
import pytest

from pose_control.config import (
    ActionClass,
    PlayerState,
    TOTAL_FEATURE_DIM,
    SEQUENCE_LENGTH,
    NUM_CLASSES,
)
from pose_control.model.network import (
    create_combat_lstm_model,
    LightweightNumpyLSTM,
    HAS_TENSORFLOW,
)
from pose_control.inference.postprocessor import ProbabilityPostProcessor
from pose_control.game.fsm import FighterFSM
from pose_control.game.combat_engine import CombatEngine


def test_numpy_lstm_inference():
    evaluator = LightweightNumpyLSTM()
    dummy_input = np.zeros((2, SEQUENCE_LENGTH, TOTAL_FEATURE_DIM), dtype=np.float32)

    probs = evaluator.predict(dummy_input)
    assert probs.shape == (2, NUM_CLASSES)
    np.testing.assert_allclose(np.sum(probs, axis=-1), [1.0, 1.0], atol=1e-5)


@pytest.mark.skipif(not HAS_TENSORFLOW, reason="TensorFlow not available")
def test_keras_lstm_model():
    model = create_combat_lstm_model()
    dummy_input = np.zeros((1, SEQUENCE_LENGTH, TOTAL_FEATURE_DIM), dtype=np.float32)
    output = model(dummy_input, training=False).numpy()

    assert output.shape == (1, NUM_CLASSES)
    assert np.isclose(np.sum(output), 1.0, atol=1e-5)
    # Validate parameter count is approximately ~190,788
    total_params = model.count_params()
    assert 180000 <= total_params <= 200000, f"Expected ~190k params, got {total_params}"


def test_postprocessor_hysteresis_and_debounce():
    proc = ProbabilityPostProcessor()
    t = 100.0

    # Low probability jab does not trigger
    raw = np.array([0.3, 0.7, 0.0, 0.0], dtype=np.float32)
    act, conf, _ = proc.process(raw, current_time=t)
    assert act == ActionClass.NONE

    # Sustained high raw probability jab (0.95)
    raw_high = np.array([0.02, 0.96, 0.01, 0.01], dtype=np.float32)

    # Frame 1 & 2: EMA climbing from cold start (0.0) up towards threshold (0.80)
    act1, _, _ = proc.process(raw_high, current_time=t + 0.033)
    assert act1 == ActionClass.NONE
    act2, _, _ = proc.process(raw_high, current_time=t + 0.066)
    assert act2 == ActionClass.NONE

    # Frame 3: EMA exceeds 0.80 (first frame above threshold -> count = 1)
    act3, _, _ = proc.process(raw_high, current_time=t + 0.099)
    assert act3 == ActionClass.NONE

    # Frame 4: Second consecutive frame above threshold -> Triggers JAB!
    act4, conf4, _ = proc.process(raw_high, current_time=t + 0.132)
    assert act4 == ActionClass.JAB
    assert conf4 >= 0.80

    # Frame 5: Immediate next frame is suppressed by refractory cooldown (450ms)
    act5, _, _ = proc.process(raw_high, current_time=t + 0.165)
    assert act5 == ActionClass.NONE


def test_fighter_fsm_and_frame_data():
    fsm = FighterFSM()
    assert fsm.state == PlayerState.IDLE

    # Trigger Jab
    ok = fsm.trigger_move(ActionClass.JAB)
    assert ok
    assert fsm.state == PlayerState.WINDUP
    assert fsm.current_move.name == "JAB"

    # Startup frames (6 frames)
    for _ in range(6):
        fsm.tick()
    assert fsm.state == PlayerState.ACTIVE

    # Active frames (3 frames)
    for _ in range(3):
        fsm.tick()
    assert fsm.state == PlayerState.RECOVERY

    # Recovery frames (9 frames)
    for _ in range(9):
        fsm.tick()
    assert fsm.state == PlayerState.IDLE


def test_combat_engine_collision_and_counter_hit():
    combat = CombatEngine()
    combat.reset_round()

    # Force player into ACTIVE Jab
    combat.player_fsm.trigger_move(ActionClass.JAB)
    for _ in range(6):  # startup
        combat.player_fsm.tick()
    assert combat.player_fsm.state == PlayerState.ACTIVE

    # Put AI into WINDUP to test Counter-Hit bonus (+35%)
    combat.ai.fsm.trigger_move(ActionClass.JAB)
    initial_ai_health = combat.ai.health

    combat.update()

    # AI should have taken counter-hit damage: 12 * 1.35 = 16.2 HP
    expected_damage = 12.0 * 1.35
    assert pytest.approx(combat.ai.health, abs=0.1) == initial_ai_health - expected_damage
    assert combat.player_combo == 1
    assert combat.ai.fsm.state == PlayerState.HIT_STUN
