"""Heuristic AI Opponent Behavior and State Machine.

Simulates an intelligent fighting game adversary across 3 difficulty tiers
(Easy, Medium, Hard) featuring human-like reaction latency, visual attack
telegraphing cues, guard reaction, and frame-advantage whiff punishing.
"""

import random
from typing import Optional, Tuple
from pose_control.config import (
    ActionClass,
    PlayerState,
    AIDifficultySettings,
    AI_PRESETS,
    COMBAT_TICK_RATE,
)
from pose_control.game.fsm import FighterFSM


class AIOpponent:
    """Simulated sparring opponent driven by hierarchical reaction heuristics."""

    def __init__(self, difficulty: str = "Medium"):
        self.settings: AIDifficultySettings = AI_PRESETS.get(difficulty, AI_PRESETS["Medium"])
        self.fsm = FighterFSM(is_player=False)
        self.health = self.settings.max_health
        self.guard = 100.0

        # AI Decision & Timing State
        self.telegraph_timer = 0
        self.is_telegraphing = False
        self.telegraphed_move: Optional[ActionClass] = None
        self.attack_cooldown_timer = 40  # Initial pacing delay
        self.reaction_delay_timer = 0
        self.planned_reaction: Optional[ActionClass] = None

    def reset(self, difficulty: Optional[str] = None):
        """Reset AI match state."""
        if difficulty is not None:
            self.settings = AI_PRESETS.get(difficulty, AI_PRESETS["Medium"])
        self.fsm.reset()
        self.health = self.settings.max_health
        self.guard = 100.0
        self.telegraph_timer = 0
        self.is_telegraphing = False
        self.telegraphed_move = None
        self.attack_cooldown_timer = 40
        self.reaction_delay_timer = 0
        self.planned_reaction = None

    def update(self, player_fsm: FighterFSM):
        """Advances AI logic by one 60 Hz frame.

        Args:
            player_fsm: The state machine of the human player to react against.
        """
        self.fsm.tick()

        if self.fsm.state == PlayerState.KO:
            return

        # -------------------------------------------------------------------
        # 1. Reaction Logic to Player Attacks (Blocking & Countering)
        # -------------------------------------------------------------------
        if player_fsm.state == PlayerState.WINDUP and self.reaction_delay_timer == 0 and self.planned_reaction is None:
            # Player initiated strike; roll AI perception check
            move = player_fsm.current_move
            prob = self.settings.block_prob_jab if move and move.action_class == ActionClass.JAB else self.settings.block_prob_uppercut

            if random.random() < prob:
                # Decide to block after reaction delay
                delay_frames = int(self.settings.reaction_delay * COMBAT_TICK_RATE)
                self.reaction_delay_timer = max(1, delay_frames)
                self.planned_reaction = ActionClass.BLOCK

        # Execute scheduled player reaction
        if self.reaction_delay_timer > 0:
            self.reaction_delay_timer -= 1
            if self.reaction_delay_timer == 0 and self.planned_reaction == ActionClass.BLOCK:
                if self.fsm.can_act():
                    self.fsm.trigger_move(ActionClass.BLOCK)
                self.planned_reaction = None

        # Release block if player is no longer attacking
        if self.fsm.state == PlayerState.BLOCKING and player_fsm.state in (PlayerState.IDLE, PlayerState.RECOVERY):
            self.fsm.release_block()

        # -------------------------------------------------------------------
        # 2. Punish Whiffs / Frame-Advantage Exploitation
        # -------------------------------------------------------------------
        if player_fsm.state == PlayerState.RECOVERY and self.fsm.can_act() and not self.is_telegraphing:
            if random.random() < self.settings.counter_punish_rate:
                # Immediate punish strike
                punish_move = ActionClass.JAB if random.random() > 0.4 else ActionClass.UPPERCUT
                self.fsm.trigger_move(punish_move)
                return

        # -------------------------------------------------------------------
        # 3. Autonomous Offense & Attack Telegraphing
        # -------------------------------------------------------------------
        if self.attack_cooldown_timer > 0:
            self.attack_cooldown_timer -= 1

        if self.is_telegraphing:
            self.telegraph_timer -= 1
            if self.telegraph_timer <= 0:
                self.is_telegraphing = False
                if self.telegraphed_move and self.fsm.can_act():
                    self.fsm.trigger_move(self.telegraphed_move)
                self.telegraphed_move = None
                self.attack_cooldown_timer = random.randint(60, 140)  # Pacing between strikes

        elif self.fsm.can_act() and self.attack_cooldown_timer <= 0 and player_fsm.state == PlayerState.IDLE:
            # Initiate a telegraphed strike
            self.is_telegraphing = True
            telegraph_frames = int(self.settings.telegraph_duration * COMBAT_TICK_RATE)
            self.telegraph_timer = max(4, telegraph_frames)
            self.telegraphed_move = ActionClass.JAB if random.random() > 0.35 else ActionClass.UPPERCUT
