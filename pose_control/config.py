"""Central Configuration Module for Pose-Controlled Fighting Game.

Defines all biomechanical, machine learning, combat engine, and UI parameters
strictly adhering to ISO/IEC/IEEE 29148:2018 and technical architecture specifications.
"""

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Dict, List, Tuple


class ActionClass(IntEnum):
    """Discrete gesture and combat action classes."""
    NONE = 0
    JAB = 1
    UPPERCUT = 2
    BLOCK = 3


CLASS_NAMES: Dict[int, str] = {
    ActionClass.NONE: "NONE",
    ActionClass.JAB: "JAB",
    ActionClass.UPPERCUT: "UPPERCUT",
    ActionClass.BLOCK: "BLOCK",
}

NUM_CLASSES = len(ActionClass)


class PlayerState(IntEnum):
    """Combat finite state machine states."""
    IDLE = 0
    WINDUP = 1
    ACTIVE = 2
    RECOVERY = 3
    BLOCKING = 4
    BLOCK_STUN = 5
    HIT_STUN = 6
    KNOCKDOWN = 7
    KO = 8


# ---------------------------------------------------------------------------
# Biomechanical & Kinematic Dimensions
# ---------------------------------------------------------------------------
NUM_LANDMARKS = 33
COORDS_PER_LANDMARK = 3  # (x, y, z)
POSITION_FEATURE_DIM = NUM_LANDMARKS * COORDS_PER_LANDMARK  # 99

# 11 Key Landmarks for 1st-derivative velocities:
# Nose(0), L_Shoulder(11), R_Shoulder(12), L_Elbow(13), R_Elbow(14),
# L_Wrist(15), R_Wrist(16), L_Pinky(19), R_Pinky(20), L_Hip(23), R_Hip(24)
KEY_VELOCITY_LANDMARK_INDICES: List[int] = [0, 11, 12, 13, 14, 15, 16, 19, 20, 23, 24]
VELOCITY_FEATURE_DIM = len(KEY_VELOCITY_LANDMARK_INDICES) * 3  # 33

# 6 Biomechanical joint angles: L/R elbow, L/R shoulder, L/R hip
JOINT_ANGLE_FEATURE_DIM = 6

TOTAL_FEATURE_DIM = POSITION_FEATURE_DIM + VELOCITY_FEATURE_DIM + JOINT_ANGLE_FEATURE_DIM  # 138

# Temporal Windowing
SEQUENCE_LENGTH = 30  # 30 frames (1.0 second at 30 FPS)
CAMERA_FPS = 30
COMBAT_TICK_RATE = 60  # Deterministic 60 Hz game loop
FIXED_TIMESTEP = 1.0 / COMBAT_TICK_RATE  # ~16.67 ms


# ---------------------------------------------------------------------------
# Real-Time Inference & Conditioning Parameters
# ---------------------------------------------------------------------------
# Class-specific confidence activation thresholds
CONFIDENCE_THRESHOLDS: Dict[ActionClass, float] = {
    ActionClass.NONE: 0.50,
    ActionClass.JAB: 0.80,
    ActionClass.UPPERCUT: 0.85,
    ActionClass.BLOCK: 0.75,
}

# Minimum consecutive sustained frames to trigger action
MIN_ACTIVATION_FRAMES: Dict[ActionClass, int] = {
    ActionClass.NONE: 1,
    ActionClass.JAB: 2,
    ActionClass.UPPERCUT: 3,
    ActionClass.BLOCK: 4,
}

# Refractory period / mechanical cooldown (in seconds)
REFRACTORY_PERIODS: Dict[ActionClass, float] = {
    ActionClass.NONE: 0.0,
    ActionClass.JAB: 0.450,       # 450 ms
    ActionClass.UPPERCUT: 0.650,  # 650 ms
    ActionClass.BLOCK: 0.0,       # continuous guard
}

# Dual-Rate Exponential Moving Average smoothing alpha
EMA_ALPHA_STRIKE = 0.65  # Ballistic responsive tracking for JAB & UPPERCUT
EMA_ALPHA_STEADY = 0.40  # Heavy noise dampening for BLOCK & NONE


# ---------------------------------------------------------------------------
# Combat Frame Data (Based on 60 Hz Ticks)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class MoveData:
    name: str
    action_class: ActionClass
    startup_frames: int
    active_frames: int
    recovery_frames: int
    base_damage: float
    chip_damage: float
    on_hit_adv: int
    on_block_adv: int
    is_knockdown: bool = False

    @property
    def total_frames(self) -> int:
        return self.startup_frames + self.active_frames + self.recovery_frames


MOVE_DATABASE: Dict[ActionClass, MoveData] = {
    ActionClass.JAB: MoveData(
        name="JAB",
        action_class=ActionClass.JAB,
        startup_frames=6,    # 100 ms
        active_frames=3,     # 50 ms
        recovery_frames=9,   # 150 ms
        base_damage=12.0,
        chip_damage=1.0,
        on_hit_adv=4,
        on_block_adv=-2,     # Safe on block
        is_knockdown=False,
    ),
    ActionClass.UPPERCUT: MoveData(
        name="UPPERCUT",
        action_class=ActionClass.UPPERCUT,
        startup_frames=12,   # 200 ms
        active_frames=4,     # 66 ms
        recovery_frames=18,  # 300 ms
        base_damage=28.0,
        chip_damage=3.0,
        on_hit_adv=12,
        on_block_adv=-10,    # Highly unsafe on block
        is_knockdown=True,
    ),
    ActionClass.BLOCK: MoveData(
        name="BLOCK",
        action_class=ActionClass.BLOCK,
        startup_frames=2,    # 33 ms
        active_frames=99999, # Held
        recovery_frames=4,   # 66 ms
        base_damage=0.0,
        chip_damage=0.0,
        on_hit_adv=0,
        on_block_adv=2,
        is_knockdown=False,
    ),
}

BLOCK_DAMAGE_REDUCTION = 0.80  # 80% mitigation while guarding
COUNTER_HIT_MULTIPLIER = 1.35  # +35% damage during opponent windup/active attack
HITSTUN_FRAMES_JAB = 14
HITSTUN_FRAMES_UPPERCUT = 30
BLOCKSTUN_FRAMES = 8


# ---------------------------------------------------------------------------
# AI Opponent Difficulty Presets
# ---------------------------------------------------------------------------
@dataclass
class AIDifficultySettings:
    name: str
    reaction_delay: float         # Perception delay in seconds
    block_prob_jab: float         # 0.0 - 1.0
    block_prob_uppercut: float    # 0.0 - 1.0
    telegraph_duration: float     # Visual flash before strike in seconds
    counter_punish_rate: float    # Probability of frame-punishing on block
    max_health: float = 100.0


AI_PRESETS: Dict[str, AIDifficultySettings] = {
    "Easy": AIDifficultySettings(
        name="Easy",
        reaction_delay=0.450,
        block_prob_jab=0.20,
        block_prob_uppercut=0.15,
        telegraph_duration=0.600,
        counter_punish_rate=0.10,
        max_health=100.0,
    ),
    "Medium": AIDifficultySettings(
        name="Medium",
        reaction_delay=0.250,
        block_prob_jab=0.55,
        block_prob_uppercut=0.65,
        telegraph_duration=0.350,
        counter_punish_rate=0.45,
        max_health=150.0,
    ),
    "Hard": AIDifficultySettings(
        name="Hard",
        reaction_delay=0.120,
        block_prob_jab=0.85,
        block_prob_uppercut=0.90,
        telegraph_duration=0.150,
        counter_punish_rate=0.85,
        max_health=200.0,
    ),
}


# ---------------------------------------------------------------------------
# Display & Canvas Dimensions
# ---------------------------------------------------------------------------
WINDOW_WIDTH = 1280
WINDOW_HEIGHT = 720
WINDOW_TITLE = "Pose-Controlled Fighting Game - MediaPipe & LSTM (Arcade 60Hz)"

# Color definitions in BGR format
COLOR_NEON_CYAN: Tuple[int, int, int] = (255, 255, 0)
COLOR_NEON_MAGENTA: Tuple[int, int, int] = (255, 0, 255)
COLOR_NEON_YELLOW: Tuple[int, int, int] = (0, 230, 255)
COLOR_HEALTH_GREEN: Tuple[int, int, int] = (50, 220, 50)
COLOR_HEALTH_RED: Tuple[int, int, int] = (50, 50, 220)
COLOR_GUARD_BLUE: Tuple[int, int, int] = (255, 150, 30)
COLOR_HIT_GOLD: Tuple[int, int, int] = (0, 215, 255)
COLOR_WHITE: Tuple[int, int, int] = (255, 255, 255)
COLOR_BLACK: Tuple[int, int, int] = (15, 15, 15)
