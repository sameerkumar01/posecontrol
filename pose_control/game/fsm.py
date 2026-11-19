"""Combat Character Finite State Machine (FSM).

Implements standard competitive fighting game frame data logic:
Startup -> Active -> Recovery -> Neutral, plus Hitstun, Blockstun, and Knockdown.
"""

from typing import Optional
from pose_control.config import (
    PlayerState,
    ActionClass,
    MoveData,
    MOVE_DATABASE,
)


class FighterFSM:
    """Finite State Machine managing a combatant's attack, defense, and hit states."""

    def __init__(self, is_player: bool = True):
        self.is_player = is_player
        self.state = PlayerState.IDLE
        self.current_move: Optional[MoveData] = None
        self.frame_in_state = 0
        self.hit_registered_this_attack = False

    def reset(self):
        """Reset state machine to neutral IDLE."""
        self.state = PlayerState.IDLE
        self.current_move = None
        self.frame_in_state = 0
        self.hit_registered_this_attack = False

    def can_act(self) -> bool:
        """Returns True if fighter is in neutral IDLE and free to initiate an action."""
        return self.state in (PlayerState.IDLE, PlayerState.BLOCKING)

    def trigger_move(self, action: ActionClass) -> bool:
        """Attempts to execute a move from neutral IDLE.

        Returns:
            success: True if the move was accepted and transitioned into WINDUP/BLOCKING.
        """
        if not self.can_act():
            return False

        if action == ActionClass.BLOCK:
            self.state = PlayerState.BLOCKING
            self.current_move = MOVE_DATABASE[ActionClass.BLOCK]
            self.frame_in_state = 0
            return True

        if action in (ActionClass.JAB, ActionClass.UPPERCUT):
            move = MOVE_DATABASE[action]
            self.state = PlayerState.WINDUP
            self.current_move = move
            self.frame_in_state = 0
            self.hit_registered_this_attack = False
            return True

        return False

    def release_block(self):
        """Fighter stopped blocking gesture."""
        if self.state == PlayerState.BLOCKING:
            self.state = PlayerState.IDLE
            self.current_move = None
            self.frame_in_state = 0

    def apply_hit(self, damage: float, is_blocked: bool, is_knockdown: bool, hitstun_frames: int):
        """Transitions fighter into HIT_STUN or BLOCK_STUN."""
        self.frame_in_state = 0
        self.current_move = None
        self.hit_registered_this_attack = False

        if is_blocked:
            self.state = PlayerState.BLOCK_STUN
            self._stun_duration = 8  # 8 frames blockstun
        elif is_knockdown:
            self.state = PlayerState.KNOCKDOWN
            self._stun_duration = 36 # 36 frames knockdown and wakeup
        else:
            self.state = PlayerState.HIT_STUN
            self._stun_duration = hitstun_frames

    def set_ko(self):
        """Sets fighter to KO state."""
        self.state = PlayerState.KO
        self.current_move = None
        self.frame_in_state = 0

    def tick(self):
        """Advances state machine by one 60 Hz frame."""
        self.frame_in_state += 1

        if self.state == PlayerState.WINDUP:
            if self.current_move and self.frame_in_state >= self.current_move.startup_frames:
                self.state = PlayerState.ACTIVE
                self.frame_in_state = 0

        elif self.state == PlayerState.ACTIVE:
            if self.current_move and self.frame_in_state >= self.current_move.active_frames:
                self.state = PlayerState.RECOVERY
                self.frame_in_state = 0

        elif self.state == PlayerState.RECOVERY:
            if self.current_move and self.frame_in_state >= self.current_move.recovery_frames:
                self.state = PlayerState.IDLE
                self.current_move = None
                self.frame_in_state = 0

        elif self.state in (PlayerState.HIT_STUN, PlayerState.BLOCK_STUN, PlayerState.KNOCKDOWN):
            if self.frame_in_state >= getattr(self, "_stun_duration", 12):
                self.state = PlayerState.IDLE
                self.frame_in_state = 0
