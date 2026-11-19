"""Deterministic 60 Hz Combat Engine and Hitbox Collision Resolver.

Handles fixed-timestep combat simulation, capsule-to-sphere collision resolution,
frame advantage, counter-hits, combo counters, and health/guard management.
"""

from dataclasses import dataclass, field
import math
import random
from typing import List, Optional, Tuple

from pose_control.config import (
    ActionClass,
    PlayerState,
    MoveData,
    MOVE_DATABASE,
    BLOCK_DAMAGE_REDUCTION,
    COUNTER_HIT_MULTIPLIER,
    HITSTUN_FRAMES_JAB,
    HITSTUN_FRAMES_UPPERCUT,
    FIXED_TIMESTEP,
)
from pose_control.game.fsm import FighterFSM
from pose_control.game.ai_opponent import AIOpponent


@dataclass
class VisualHitSpark:
    x: int
    y: int
    color: Tuple[int, int, int]
    radius: float = 6.0
    lifetime_frames: int = 8
    current_frame: int = 0
    particles: List[Tuple[float, float, float, float]] = field(default_factory=list)

    def __post_init__(self):
        if not self.particles:
            # Generate 16 radial sparks (px, py, vx, vy)
            for _ in range(16):
                angle = random.uniform(0, 2 * math.pi)
                speed = random.uniform(3.0, 10.0)
                self.particles.append((
                    float(self.x),
                    float(self.y),
                    math.cos(angle) * speed,
                    math.sin(angle) * speed,
                ))


class CombatEngine:
    """Deterministic fighting game simulation engine running at 60 Hz."""

    def __init__(self, ai_difficulty: str = "Medium"):
        self.player_fsm = FighterFSM(is_player=True)
        self.ai = AIOpponent(difficulty=ai_difficulty)

        # Health & Guard Meters
        self.player_max_health = 100.0
        self.player_health = 100.0
        self.player_guard = 100.0

        # Match State
        self.round_time = 99.0
        self.player_rounds_won = 0
        self.ai_rounds_won = 0
        self.current_round = 1
        self.is_round_over = False
        self.round_winner: Optional[str] = None

        # Combat Feedback
        self.player_combo = 0
        self.ai_combo = 0
        self.screen_shake_frames = 0
        self.screen_shake_intensity = 0.0
        self.hit_sparks: List[VisualHitSpark] = []
        self.active_banner_text = ""
        self.banner_timer = 0

    def reset_round(self):
        """Resets combatants for a new round."""
        self.player_fsm.reset()
        self.ai.reset()
        self.player_health = self.player_max_health
        self.player_guard = 100.0
        self.round_time = 99.0
        self.is_round_over = False
        self.round_winner = None
        self.player_combo = 0
        self.ai_combo = 0
        self.screen_shake_frames = 0
        self.hit_sparks.clear()
        self.active_banner_text = f"ROUND {self.current_round} - FIGHT!"
        self.banner_timer = 90  # 1.5 seconds

    def process_player_input(self, action: ActionClass):
        """Dispatches recognized player gesture into player FSM."""
        if self.is_round_over:
            return

        if action == ActionClass.BLOCK:
            self.player_fsm.trigger_move(ActionClass.BLOCK)
        elif action in (ActionClass.JAB, ActionClass.UPPERCUT):
            if self.player_fsm.trigger_move(action):
                self.active_banner_text = f">> PLAYER {action.name} <<"
                self.banner_timer = 35
        elif action == ActionClass.NONE:
            if self.player_fsm.state == PlayerState.BLOCKING:
                self.player_fsm.release_block()

    def update(self, dt: float = FIXED_TIMESTEP):
        """Deterministic 60 Hz tick."""
        # Update match timer
        if not self.is_round_over and self.round_time > 0:
            self.round_time = max(0.0, self.round_time - dt)
            if self.round_time <= 0:
                self._resolve_time_out()

        # Update Visuals
        if self.banner_timer > 0:
            self.banner_timer -= 1
        if self.screen_shake_frames > 0:
            self.screen_shake_frames -= 1

        # Update particle sparks
        alive_sparks = []
        for spark in self.hit_sparks:
            spark.current_frame += 1
            if spark.current_frame < spark.lifetime_frames:
                # Update particle positions
                updated = []
                for px, py, vx, vy in spark.particles:
                    updated.append((px + vx, py + vy, vx * 0.85, vy * 0.85))
                spark.particles = updated
                alive_sparks.append(spark)
        self.hit_sparks = alive_sparks

        # Update FSMs
        self.player_fsm.tick()
        self.ai.update(self.player_fsm)

        # Guard meter recovery
        if self.player_fsm.state != PlayerState.BLOCKING:
            self.player_guard = min(100.0, self.player_guard + 0.15)
        if self.ai.fsm.state != PlayerState.BLOCKING:
            self.ai.guard = min(100.0, self.ai.guard + 0.15)

        # -------------------------------------------------------------------
        # Collision Resolution: Player Attack -> AI Opponent
        # -------------------------------------------------------------------
        if (
            self.player_fsm.state == PlayerState.ACTIVE
            and not self.player_fsm.hit_registered_this_attack
            and self.player_fsm.current_move is not None
            and self.ai.fsm.state not in (PlayerState.KO, PlayerState.KNOCKDOWN)
        ):
            self._resolve_player_strike_on_ai()

        # -------------------------------------------------------------------
        # Collision Resolution: AI Opponent -> Player
        # -------------------------------------------------------------------
        if (
            self.ai.fsm.state == PlayerState.ACTIVE
            and not self.ai.fsm.hit_registered_this_attack
            and self.ai.fsm.current_move is not None
            and self.player_fsm.state not in (PlayerState.KO, PlayerState.KNOCKDOWN)
        ):
            self._resolve_ai_strike_on_player()

    def _resolve_player_strike_on_ai(self):
        """Resolves Player attack landing on AI Opponent."""
        self.player_fsm.hit_registered_this_attack = True
        move = self.player_fsm.current_move
        is_blocked = self.ai.fsm.state == PlayerState.BLOCKING

        if is_blocked:
            damage = move.chip_damage
            self.ai.guard = max(0.0, self.ai.guard - 15.0)
            self.ai.health = max(0.0, self.ai.health - damage)
            self.ai.fsm.apply_hit(damage, is_blocked=True, is_knockdown=False, hitstun_frames=8)
            self.screen_shake_frames = 3
            self.screen_shake_intensity = 4.0
            # Guard spark in blue/gold
            self.hit_sparks.append(VisualHitSpark(880, 280, color=(255, 200, 50)))
        else:
            # Clean hit
            is_counter = self.ai.fsm.state in (PlayerState.WINDUP, PlayerState.ACTIVE)
            mult = COUNTER_HIT_MULTIPLIER if is_counter else 1.0
            damage = move.base_damage * mult
            self.ai.health = max(0.0, self.ai.health - damage)
            self.player_combo += 1

            hitstun = HITSTUN_FRAMES_UPPERCUT if move.is_knockdown else HITSTUN_FRAMES_JAB
            self.ai.fsm.apply_hit(damage, is_blocked=False, is_knockdown=move.is_knockdown, hitstun_frames=hitstun)

            self.screen_shake_frames = 6 if move.is_knockdown else 4
            self.screen_shake_intensity = 10.0 if move.is_knockdown else 6.0
            self.hit_sparks.append(VisualHitSpark(880, 260, color=(50, 50, 255)))

            if is_counter:
                self.active_banner_text = ">> COUNTER HIT! <<"
                self.banner_timer = 40

        # Check AI KO
        if self.ai.health <= 0:
            self.ai.fsm.set_ko()
            self._end_round(winner="PLAYER")

    def _resolve_ai_strike_on_player(self):
        """Resolves AI attack landing on Player."""
        self.ai.fsm.hit_registered_this_attack = True
        move = self.ai.fsm.current_move
        is_blocked = self.player_fsm.state == PlayerState.BLOCKING

        if is_blocked:
            damage = move.chip_damage
            self.player_guard = max(0.0, self.player_guard - 15.0)
            self.player_health = max(0.0, self.player_health - damage)
            self.player_fsm.apply_hit(damage, is_blocked=True, is_knockdown=False, hitstun_frames=8)
            self.screen_shake_frames = 3
            self.screen_shake_intensity = 4.0
            self.hit_sparks.append(VisualHitSpark(400, 280, color=(255, 200, 50)))
        else:
            is_counter = self.player_fsm.state in (PlayerState.WINDUP, PlayerState.ACTIVE)
            mult = COUNTER_HIT_MULTIPLIER if is_counter else 1.0
            damage = move.base_damage * mult
            self.player_health = max(0.0, self.player_health - damage)
            self.player_combo = 0  # Reset player combo
            self.ai_combo += 1

            hitstun = HITSTUN_FRAMES_UPPERCUT if move.is_knockdown else HITSTUN_FRAMES_JAB
            self.player_fsm.apply_hit(damage, is_blocked=False, is_knockdown=move.is_knockdown, hitstun_frames=hitstun)

            self.screen_shake_frames = 6 if move.is_knockdown else 4
            self.screen_shake_intensity = 10.0 if move.is_knockdown else 6.0
            self.hit_sparks.append(VisualHitSpark(400, 260, color=(50, 50, 255)))

        if self.player_health <= 0:
            self.player_fsm.set_ko()
            self._end_round(winner="AI")

    def _resolve_time_out(self):
        """Resolves match when 99s timer hits 0."""
        if self.player_health > self.ai.health:
            self._end_round(winner="PLAYER")
        elif self.ai.health > self.player_health:
            self._end_round(winner="AI")
        else:
            self._end_round(winner="DRAW")

    def _end_round(self, winner: str):
        """Handles round termination and victory score."""
        self.is_round_over = True
        self.round_winner = winner
        if winner == "PLAYER":
            self.player_rounds_won += 1
            self.active_banner_text = "K.O.! PLAYER WINS ROUND!"
        elif winner == "AI":
            self.ai_rounds_won += 1
            self.active_banner_text = "K.O.! AI OPPONENT WINS ROUND!"
        else:
            self.active_banner_text = "TIME OVER - DRAW!"
        self.banner_timer = 120
