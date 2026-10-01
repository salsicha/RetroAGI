"""
mario_scenario_env.py — Lightweight SMB-style platformer for AI training.

Gym v26 API: reset() → (obs, info)   step() → (obs, reward, terminated, truncated, info)
All coordinates are in world space; camera_x is the left edge of the viewport.
"""

import json
import math
import os
import random
from dataclasses import asdict, dataclass

import numpy as np
import pygame

from retroagi.core.smb_physics import NESPlayerMotion
from retroagi.core.smb_pixel_types import TYPE_ID

from .stomp import stomp_collision_geometry

# ── Gym-compatible space stubs (no gym dependency required) ──────────────────


class _DiscreteSpace:
    def __init__(self, n):
        self.n = n
        self.dtype = np.int64

    def sample(self):
        return random.randrange(self.n)


class _BoxSpace:
    def __init__(self, shape, low=0, high=255, dtype=np.uint8):
        self.shape = shape
        self.low = low
        self.high = high
        self.dtype = dtype


# ── Player motion ─────────────────────────────────────────────────────────────
#
# Block SMB is the transfer source for the real-emulator Full SMB stage, so the
# player moves with the NES's own motion rules (retroagi.core.smb_physics):
# fixed-point acceleration, run/walk limits, braking, air steering, variable
# jump forces and button edges. There is no coyote time and no jump buffer;
# holding jump through a landing does not jump again. The NES small body is
# 10x12 pixels and the Goomba damage body 10x6.
ENEMY_GRAVITY = 0.5

# ── Drawing ───────────────────────────────────────────────────────────────────
#
# The picture is purely visual: nothing below changes layouts, physics or
# rewards. Solid platforms are drawn as one of four kinds, as in real SMB
# (floor and stairs are ground, floating rows are bricks with question blocks,
# pipes are green). A scenario may name each platform's kind in
# "platform_kinds", aligned with "platforms"; an untagged (None) platform is a
# brick row when open space lies directly beneath it and ground otherwise.
# Moving platforms are always lifts. render_labels() repeats render()'s shapes
# with each shape's pixel type in place of its colour, so labels are exact.
PLATFORM_KINDS = ("ground", "brick", "question_block", "pipe")
TILE = 16
SKY = (107, 140, 255)
GROUND, GROUND_EDGE = (139, 69, 19), (84, 38, 8)
BRICK, BRICK_MORTAR = (200, 76, 12), (64, 24, 0)
QUESTION, QUESTION_EDGE, QUESTION_MARK = (252, 160, 68), (136, 72, 0), (96, 40, 0)
PIPE, PIPE_LIGHT, PIPE_DARK = (0, 168, 0), (128, 208, 16), (0, 80, 0)
LIFT, LIFT_EDGE = (216, 216, 216), (120, 120, 120)
COIN, COIN_DARK = (255, 215, 0), (204, 140, 0)
GOOMBA, GOOMBA_FEET, GOOMBA_SQUISHED = (160, 32, 240), (72, 0, 112), (100, 0, 160)
PLANT_HEAD, PLANT_SPOT, PLANT_STEM = (216, 40, 96), (255, 200, 220), (0, 112, 72)
MARIO, MARIO_SKIDDING, EYE = (255, 0, 0), (255, 220, 0), (255, 255, 255)


def question_cell(world_x):
    """Whether the 16-pixel brick cell starting at world x is a question block."""
    return (int(world_x) // TILE) % 3 == 1


class _Canvas:
    """Draws each shape in its colour, or filled with its pixel type id."""

    def __init__(self, surface, labels):
        self.surface = surface
        self.labels = labels

    def _color(self, color, kind):
        return (TYPE_ID[kind],) * 3 if self.labels else color

    def fill(self, color, kind):
        self.surface.fill(self._color(color, kind))

    def rect(self, color, kind, rect, width=0):
        pygame.draw.rect(self.surface, self._color(color, kind), rect, width)

    def ellipse(self, color, kind, rect):
        pygame.draw.ellipse(self.surface, self._color(color, kind), rect)


@dataclass(frozen=True)
class BlockSMBRewardConfig:
    """Tunable scalar rewards owned by the Block SMB environment."""

    progress_per_pixel: float = 0.05
    coin: float = 10.0
    enemy_stomp: float = 5.0
    goal: float = 50.0
    fall_death: float = -10.0
    gap_jump: float = -5.0
    enemy_hit: float = -10.0
    frame_penalty: float = -0.01
    # Potential-based goal-distance shaping: per-step reward proportional to
    # the decrease in normalized goal distance. Off globally by default;
    # scenarios opt in with a "reward_goal_distance_shaping" coefficient so
    # B-level isolation families get a dense vertical-progress gradient.
    goal_distance_shaping: float = 0.0
    # Energy regulator: control-effort cost per frame a jump button is held.
    # Eager but fruitless actions are unrewarded, and among successful
    # strategies the minimal sufficient effort nets the most, so hold duration
    # must condition on obstacle size instead of defaulting to maximum. The
    # accumulated charge is refunded when the episode ends in death, so dying
    # while trying costs no energy and giving up never beats attempting (the
    # first suite run collapsed failing families to 3-frame taps because
    # shorter holds minimized energy on doomed attempts). Off globally;
    # scenarios opt in with a "reward_energy_jump" coefficient.
    energy_jump: float = 0.0
    # Wait-survival shaping: per-frame trickle while alive inside a
    # scenario-declared wait window, so waiting is not paid exclusively
    # through the end-discounted goal. Off globally; scenarios opt in with
    # a "reward_wait_survival" coefficient.
    wait_survival: float = 0.0

    def __post_init__(self) -> None:
        if self.progress_per_pixel < 0:
            raise ValueError("progress_per_pixel must be non-negative")
        for name in ("coin", "enemy_stomp", "goal", "goal_distance_shaping", "wait_survival"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be non-negative")
        for name in ("fall_death", "gap_jump", "enemy_hit", "frame_penalty", "energy_jump"):
            if getattr(self, name) > 0:
                raise ValueError(f"{name} must be non-positive")

    def zero_terms(self) -> dict[str, float]:
        return {
            "progress": 0.0,
            "coin": 0.0,
            "enemy_stomp": 0.0,
            "goal": 0.0,
            "goal_distance": 0.0,
            "energy": 0.0,
            "fall_death": 0.0,
            "gap_jump": 0.0,
            "enemy_hit": 0.0,
            "wait_survival": 0.0,
            "frame_penalty": 0.0,
        }


# ── Main environment ──────────────────────────────────────────────────────────


class MarioScenarioEnv:
    """
    Scriptable 2-D platformer environment.

    Actions
    -------
    0 NOOP | 1 RIGHT | 2 RIGHT+JUMP | 3 LEFT | 4 LEFT+JUMP | 5 JUMP

    Scenario dict keys
    ------------------
    world_width   : int  (default = viewport width)
    mario         : [x, y]
    platforms     : list of [x, y, w, h] or {'x','y','w','h', 'moving':[min_x,max_x,speed]}
    platform_kinds: optional list aligned with platforms: "ground", "brick",
                    "question_block", "pipe" or None (drawing only)
    coins         : list of [x, y, w, h]
    enemies       : list of [x, y, patrol_min, patrol_max] or
                    [x, y, patrol_min, patrol_max, speed] or
                    dict with keys x,y,patrol_min,patrol_max,speed,edge_aware;
                    a piranha_plant uses kind,x,pipe_top and optional cycle durations
    goal          : [x, y, w, h]
    """

    # ── Construction ─────────────────────────────────────────────────────────

    def __init__(
        self,
        width: int = 256,
        height: int = 240,
        world_width: int = None,
        reward_config: BlockSMBRewardConfig = BlockSMBRewardConfig(),
    ):
        self.width = width
        self.height = height
        self.world_width = world_width if world_width is not None else width
        self.reward_config = reward_config
        self.motion = NESPlayerMotion()
        self.max_walk_speed = 2.5
        self.max_fall_speed = 4.5

        # Spaces (Gym-compatible stubs)
        self.action_space = _DiscreteSpace(6)
        self.action_space_n = 6  # legacy alias
        self.observation_space = _BoxSpace((height, width, 3))

        # Offscreen observations only need software surfaces; the interactive
        # demo initializes the display subsystem when it opens a window.
        self.screen = pygame.Surface((self.width, self.height))
        self._label_screen = pygame.Surface((self.width, self.height))

        # RNG (seeded via self.seed())
        self._rng = random.Random()

        # State (populated by reset)
        self.mario = None
        self.platforms = []  # list of platform dicts
        self.platform_kinds = []  # drawn kind of each platform
        self.coins = []
        self.enemies = []
        self.goal = None
        self._goal_on_stomp = False
        self._require_stomp_before_goal = False
        self._stomp_credited = False
        self._require_bridge_before_goal = False
        self._bridge_boarded = False
        self._bridge_crossed = False
        self._bridge_jump_task = None
        self._bridge_then_terrain = False
        self._bridge_jump_launched = False
        self._goal_requires_support = False
        self._single_jump_attempt = False
        self._attempt_failed = False
        self._goal_credited = False
        self.camera_x = 0.0
        self.score = 0
        self.steps = 0
        self.max_steps = 1000
        self._max_x_reached = 0.0

    def seed(self, n: int = None):
        """Set RNG seed for reproducible procedural generation."""
        self._rng.seed(n)
        return [n]

    def close(self):
        """Clean up pygame resources."""
        pygame.quit()

    # ── Gym space properties ──────────────────────────────────────────────────

    # (already set as instance attrs above; properties kept for completeness)

    # ── Reset ─────────────────────────────────────────────────────────────────

    def reset(self, scenario: dict = None, seed: int = None):
        """
        Reset and return (obs, info) — Gym v26 API.
        Passing seed= here is equivalent to calling self.seed(seed) first.
        """
        if seed is not None:
            self.seed(seed)

        self.steps = 0
        self.score = 0
        self.camera_x = 0.0
        self._max_x_reached = 0.0
        self._airborne_started_with_jump = False

        if scenario is None:
            scenario = {
                "mario": [20, 180],
                "platforms": [
                    [0, 220, 256, 20],
                    [120, 170, 40, 10],
                ],
                "coins": [[130, 140, 10, 10]],
            }

        self.world_width = scenario.get("world_width", self.width)
        self.motion = NESPlayerMotion()

        # Mario state
        self.mario = {
            "x": float(scenario["mario"][0]),
            "y": float(scenario["mario"][1]),
            "vx": 0.0,
            "vy": 0.0,
            "w": 10,
            "h": 12,
            "on_ground": False,
            "facing": 1,  # 1 = right, -1 = left
            "skidding": False,
            "jump_held": False,  # was jump action present last frame?
        }
        if "mario_velocity" in scenario:
            vx, vy = scenario["mario_velocity"]
            self.motion.x_speed = int(round(float(vx) * 16))
            self.motion.y_speed = int(float(vy))
            self.motion.y_force = int((float(vy) % 1) * 256)
            self.mario["vx"], self.mario["vy"] = float(vx), float(vy)
            self.motion.moving = 1 if vx > 0 else -1 if vx < 0 else 0
        self._max_x_reached = self.mario["x"]
        self._progress_per_pixel = float(
            scenario.get("reward_progress_per_pixel", self.reward_config.progress_per_pixel)
        )

        # Platforms — accept list [x,y,w,h] or dict
        self.platforms = []
        for p in scenario.get("platforms", []):
            self.platforms.append(self._parse_platform(p))
        self.platform_kinds = self._platform_kinds(scenario.get("platform_kinds"))

        # Settle the spawn: scenarios place Mario a few pixels above his
        # surface, which used to leave him airborne for the first frames.
        # That phantom fall made the adaptive controller read the spawn
        # touchdown as a jump's LANDING and release every episode-opening
        # jump after ~4 frames — invisible while episodes allowed retries,
        # fatal for single-jump scenarios. If a platform top lies within a
        # short drop below the spawn, start standing on it.
        mario_rect = pygame.Rect(
            int(self.mario["x"]), int(self.mario["y"]), self.mario["w"], self.mario["h"]
        )
        for platform in self.platforms:
            rect = platform["rect"]
            horizontally_over = mario_rect.right > rect.left and mario_rect.left < rect.right
            drop = rect.top - mario_rect.bottom
            if horizontally_over and 0 <= drop <= 8:
                self.mario["y"] = float(rect.top - self.mario["h"])
                self.mario["on_ground"] = True
                break

        # Coins
        self.coins = [
            {"rect": pygame.Rect(c[0], c[1], c[2], c[3]), "collected": False}
            for c in scenario.get("coins", [])
        ]

        # Enemies — accept list or dict; optional 5th element = speed
        self.enemies = []
        for e in scenario.get("enemies", []):
            enemy = self._parse_enemy(e)
            if enemy.get("kind") != "piranha_plant":
                # NES Goomba damage body is 10x6, four pixels above its
                # physical feet. Background support is a separate probe.
                enemy.update(w=10, h=6, foot_offset=4, y=enemy["y"] + 4)
            self.enemies.append(enemy)

        self.goal = pygame.Rect(*scenario["goal"]) if "goal" in scenario else None
        # Keep terrain coordinates stable through a small finish overshoot.
        self._terrain_left = self.goal is not None and self.goal.centerx < self.mario["x"]
        # B-level stomp teaching: the goal rides the (possibly patrolling)
        # enemy, so goal-distance shaping and the observation's goal slots
        # track the moving target, and goal credit is granted by the stomp
        # itself rather than by touching the goal rect.
        self._goal_on_stomp = bool(scenario.get("goal_on_stomp", False))
        from .tasks import scenario_family

        family = scenario_family(scenario)
        self._require_stomp_before_goal = bool(
            scenario.get("require_stomp_before_goal", False) or family == "enemy_stomp"
        )
        self._stomp_credited = False
        self._require_bridge_before_goal = bool(
            scenario.get("require_bridge_before_goal", False)
            or family in ("bridge_wait", "wait_timing", "moving_bridge")
        )
        self._bridge_boarded = False
        self._bridge_crossed = False
        self._bridge_jump_task = None
        self._bridge_jump_launched = False
        self._bridge_jump_task = scenario.get("bridge_jump_task")
        self._bridge_then_terrain = bool(scenario.get("bridge_then_terrain", False))
        if self._bridge_jump_task not in (None, "mount", "dismount"):
            raise ValueError("Invalid bridge jump task")
        if self._bridge_jump_task == "dismount":
            support = next(p for p in self.platforms if p.get("moving"))
            self.mario["_platform"] = support
            self._bridge_boarded = True
        self._goal_credited = False
        self._goal_requires_support = bool(
            scenario.get("goal_requires_support") or family in ("pit_leap", "pipe_mount")
        )
        self._single_jump_attempt = bool(
            scenario.get("single_jump_attempt") or family in ("pit_leap", "pipe_mount")
        )
        self._attempt_failed = False
        self._track_goal_to_enemy()
        # Per-scenario opt-in overrides the config default coefficient.
        self._goal_distance_shaping = float(
            scenario.get("reward_goal_distance_shaping", self.reward_config.goal_distance_shaping)
            or 0.0
        )
        self._prev_goal_distance = self._normalized_goal_distance()
        self._energy_jump = float(
            scenario.get("reward_energy_jump", self.reward_config.energy_jump) or 0.0
        )
        self._episode_energy = 0.0
        # Wait-survival shaping: a small per-frame trickle while alive inside
        # a scenario-declared wait window, so waiting is not exclusively paid
        # through the end-discounted goal reward. Opt-in per scenario.
        self._wait_survival = float(scenario.get("reward_wait_survival", 0.0) or 0.0)
        window = scenario.get("wait_window")
        self._wait_window = (
            (int(window[0]), int(window[1]))
            if isinstance(window, (list, tuple)) and len(window) == 2
            else None
        )

        obs = self.render()
        _, reward_terms = self._finalize_reward_terms(self.reward_config.zero_terms())
        info = self._build_info(
            reward_terms=reward_terms,
            death=False,
            terminated=False,
            truncated=False,
        )
        return obs, info

    # ── Step ──────────────────────────────────────────────────────────────────

    def _track_goal_to_enemy(self) -> None:
        """Snap the goal rect onto the first live enemy for goal_on_stomp.

        The goal rect becomes a moving proxy for the stomp target: shaping
        and the goal observation slots follow the patrolling enemy, while
        goal credit itself is granted only by the stomp collision.
        """

        if not self._goal_on_stomp or self.goal is None:
            return
        for enemy in self.enemies:
            if not enemy["dead"]:
                self.goal.midbottom = (
                    int(round(enemy["x"] + enemy["w"] / 2)),
                    int(round(enemy["y"])),
                )
                return

    def _normalized_goal_distance(self) -> float | None:
        """Normalized Mario-center to goal-center distance (matches state_vec)."""

        if self.goal is None:
            return None
        ww = float(self.world_width)
        wh = float(self.height)
        mx = self.mario["x"] + self.mario["w"] / 2
        my = self.mario["y"] + self.mario["h"] / 2
        goal_dx = (self.goal.centerx - mx) / ww
        goal_dy = (self.goal.centery - my) / wh
        return min(math.hypot(goal_dx, goal_dy), 1.0)

    def step(self, action: int):
        """
        Advance one frame.
        Returns (obs, reward, terminated, truncated, info) — Gym v26 API.
        terminated = game-ending event (death / goal)
        truncated  = timeout
        """
        previous_max_x = self._max_x_reached
        grounded_before = self.mario["on_ground"]
        support_before = self.mario.get("_platform")
        carry_dx = 0.0
        self.steps += 1
        reward_terms = self.reward_config.zero_terms()
        terminated = False
        truncated = False
        death = False
        info = {}
        stomp_geometry = None

        useful_bridge_wait = False
        if self._require_bridge_before_goal and action == 0 and self._wait_survival > 0:
            from .bridge_traversal import bridge_safe_wait_frames

            safe_waits = bridge_safe_wait_frames(self)
            useful_bridge_wait = bool(safe_waits) and 1 not in safe_waits
        jump_pressed = action in [2, 4, 5]
        move_x = 1 if action in [1, 2] else (-1 if action in [3, 4] else 0)
        # Energy regulator: every frame of held jump costs effort, so eager
        # but fruitless exertion nets negative and minimal sufficient holds
        # beat maximal ones among successful strategies.
        if jump_pressed and self._energy_jump < 0.0:
            reward_terms["energy"] += self._energy_jump
            self._episode_energy += self._energy_jump

        dx, dy, jumped = self.motion.advance(
            direction=move_x,
            jump=jump_pressed,
            grounded=self.mario["on_ground"],
            y=self.mario["y"],
            run=move_x > 0,
        )
        self.mario["vx"] = self.motion.x_speed / 16
        self.mario["vy"] = self.motion.y_speed + self.motion.y_force / 256
        self.mario["facing"] = self.motion.facing
        self.mario["skidding"] = bool(move_x and move_x * self.motion.x_speed < 0)
        self.mario["jump_held"] = jump_pressed
        if jumped:
            self.mario["on_ground"] = False
            self._airborne_started_with_jump = True
        # Inclusive foot contact preserves support at zero displacement;
        # a fractional position probe would alter the next ledge departure.
        if self.mario["on_ground"]:
            dy = 0.0

        # ── 5. Update moving platforms ────────────────────────────────────────
        for plat in self.platforms:
            if not plat["moving"]:
                continue
            old_px = plat["rect"].x
            # Track a float position so fractional speeds accumulate instead of
            # truncating to zero motion every frame.
            plat["move_x"] += plat["move_speed"] * plat["move_dir"]
            if plat["move_x"] <= plat["move_min"]:
                plat["move_x"] = float(plat["move_min"])
                plat["move_dir"] = 1
            elif plat["move_x"] >= plat["move_max"]:
                plat["move_x"] = float(plat["move_max"])
                plat["move_dir"] = -1
            plat["rect"].x = int(round(plat["move_x"]))
            plat["delta_x"] = plat["rect"].x - old_px

        # ── 6. Resolve X collisions ───────────────────────────────────────────
        self.mario["x"] += dx
        mario_rect = pygame.Rect(self.mario["x"], self.mario["y"], self.mario["w"], self.mario["h"])

        for plat in self.platforms:
            r = plat["rect"]
            if mario_rect.colliderect(r):
                if self.mario["vx"] > 0:
                    mario_rect.right = r.left
                elif self.mario["vx"] < 0:
                    mario_rect.left = r.right
                elif plat.get("delta_x", 0) > 0:
                    # A platform sliding into a stationary Mario pushes him
                    # instead of leaving the overlap for the Y pass to resolve
                    # (which would teleport him on top).
                    mario_rect.left = r.right
                elif plat.get("delta_x", 0) < 0:
                    mario_rect.right = r.left
                self.mario["x"] = mario_rect.x
                self.mario["vx"] = 0
                self.motion.wall_contact()

        # ── 7. Resolve Y collisions ───────────────────────────────────────────
        previous_y = self.mario["y"]
        previous_bottom = previous_y + self.mario["h"]
        self.mario["y"] += dy
        mario_rect.y = self.mario["y"]
        prev_on_ground = self.mario["on_ground"]
        self.mario["on_ground"] = False
        self.mario["_platform"] = None  # track which platform Mario stands on

        for plat in self.platforms:
            r = plat["rect"]
            horizontal_overlap = mario_rect.right > r.left and mario_rect.left < r.right
            current_bottom = self.mario["y"] + self.mario["h"]
            landed_on_top = (
                self.mario["vy"] >= 0
                and horizontal_overlap
                and previous_bottom <= r.top <= current_bottom
            )
            hit_ceiling = (
                self.mario["vy"] < 0
                and horizontal_overlap
                and previous_y >= r.bottom >= self.mario["y"]
            )
            if mario_rect.colliderect(r) or landed_on_top or hit_ceiling:
                if self.mario["vy"] >= 0:  # falling / level
                    mario_rect.bottom = r.top
                    self.mario["on_ground"] = True
                    self.mario["_platform"] = plat
                elif self.mario["vy"] < 0:  # hitting ceiling
                    mario_rect.top = r.bottom
                self.mario["y"] = mario_rect.y
                self.mario["vy"] = 0
                self.motion.vertical_contact()

        if (
            self._bridge_jump_task
            and grounded_before
            and self._airborne_started_with_jump
            and not self.mario["on_ground"]
        ):
            from_bridge = bool(support_before and support_before.get("moving"))
            self._bridge_jump_launched = (
                not from_bridge if self._bridge_jump_task == "mount" else from_bridge
            )

        # ── 8. Carry Mario on moving platform ────────────────────────────────
        if (
            self.mario["on_ground"]
            and self.mario["_platform"]
            and self.mario["_platform"]["moving"]
        ):
            carry_dx = self.mario["_platform"]["delta_x"]
            self.mario["x"] += carry_dx
            mario_rect.x = self.mario["x"]

        if self.mario["on_ground"]:
            self._airborne_started_with_jump = False

        # ── 10. Camera (right-only scroll, clamped) ───────────────────────────
        target_cam = self.mario["x"] - self.width // 3
        if target_cam > self.camera_x:
            self.camera_x = target_cam
        self.camera_x = max(0.0, min(self.camera_x, self.world_width - self.width))

        # ── 11. World boundaries ──────────────────────────────────────────────
        if self.mario["x"] < self.camera_x:
            self.mario["x"] = self.camera_x
        if self.mario["x"] > self.world_width - self.mario["w"]:
            self.mario["x"] = self.world_width - self.mario["w"]
        mario_rect.x = self.mario["x"]

        # ── 12. Rightward-progress reward ─────────────────────────────────────
        if self.mario["x"] > self._max_x_reached:
            reward_terms["progress"] += (
                self.mario["x"] - self._max_x_reached
            ) * self._progress_per_pixel
            self._max_x_reached = self.mario["x"]

        # ── 12b. Goal-distance shaping (potential-based, opt-in) ─────────────
        if self._goal_distance_shaping > 0.0 and self.goal is not None:
            current_goal_distance = self._normalized_goal_distance()
            if self._prev_goal_distance is not None and current_goal_distance is not None:
                reward_terms["goal_distance"] += self._goal_distance_shaping * (
                    self._prev_goal_distance - current_goal_distance
                )
            self._prev_goal_distance = current_goal_distance

        # ── 13. Fall death ────────────────────────────────────────────────────
        if self.mario["y"] > self.height:
            terminated = True
            death = True
            if self._airborne_started_with_jump and not self._has_horizontal_platform_support():
                reward_terms["gap_jump"] += self.reward_config.gap_jump
            reward_terms["fall_death"] += self.reward_config.fall_death

        # ── 14. Update enemies ────────────────────────────────────────────────
        rects = [p["rect"] for p in self.platforms]
        for enemy in self.enemies:
            if enemy["dead"]:
                continue
            self._update_enemy(enemy, rects)
        self._track_goal_to_enemy()

        # ── 15. Enemy collision ───────────────────────────────────────────────
        for enemy in self.enemies:
            if enemy["dead"]:
                continue
            er = pygame.Rect(enemy["x"], enemy["y"], enemy["w"], enemy["h"])
            geometry = stomp_collision_geometry(mario_rect, er, self.mario["vy"])
            if (self._goal_on_stomp or self._require_stomp_before_goal) and stomp_geometry is None:
                stomp_geometry = geometry
            if not mario_rect.colliderect(er):
                continue
            if geometry["stomp"] and enemy.get("stompable", True):
                # Stomp!
                enemy["dead"] = True
                self._stomp_credited = True
                reward_terms["enemy_stomp"] += self.reward_config.enemy_stomp
                self.score += 5
                self.motion.bounce()
                self.mario["vy"] = self.motion.y_speed
                self.mario["on_ground"] = False
                if self._goal_on_stomp:
                    # Landing on the enemy IS the goal: grant goal credit so
                    # success flows through the same channel as rect goals.
                    terminated = True
                    reward_terms["goal"] += self.reward_config.goal
                    self._goal_credited = True
            else:
                terminated = True
                death = True
                reward_terms["enemy_hit"] += self.reward_config.enemy_hit

        # ── 16. Coin collection ───────────────────────────────────────────────
        for coin in self.coins:
            if not coin["collected"] and mario_rect.colliderect(coin["rect"]):
                coin["collected"] = True
                reward_terms["coin"] += self.reward_config.coin
                self.score += 10

        # ── 16b. Wait-survival shaping ────────────────────────────────────────
        if (
            self._wait_survival > 0.0
            and action == 0
            and not death
            and (
                useful_bridge_wait
                if self._require_bridge_before_goal
                else (
                    self._wait_window is not None
                    and self._wait_window[0] <= self.steps <= self._wait_window[1]
                )
            )
        ):
            reward_terms["wait_survival"] += self._wait_survival

        if self._require_bridge_before_goal and self.mario["on_ground"]:
            support = self.mario.get("_platform")
            if support and support.get("moving"):
                self._bridge_boarded = True
            elif support and self._bridge_boarded:
                bridge = next((p for p in self.platforms if p.get("moving")), None)
                if bridge and support["rect"].left > bridge["move_min"]:
                    self._bridge_crossed = True

        if self._bridge_jump_task and self._bridge_jump_launched and not death:
            support = self.mario.get("_platform")
            landed = not prev_on_ground and self.mario["on_ground"] and support is not None
            bridge = next(p for p in self.platforms if p.get("moving"))
            # A physical landing is authoritative, including a valid edge
            # contact. Requiring full-body containment on its first frame would
            # reject a successful mount that settles farther onto the platform.
            mounted = landed and support is bridge
            dismounted = (
                landed and not support.get("moving") and support["rect"].left > bridge["move_min"]
            )
            if (self._bridge_jump_task == "mount" and mounted) or (
                self._bridge_jump_task == "dismount" and dismounted
            ):
                self._goal_credited = True
                terminated = True
                reward_terms["goal"] += self.reward_config.goal
            if landed:
                self._bridge_jump_launched = False

        # ── 17. Goal ──────────────────────────────────────────────────────────
        # Under goal_on_stomp the rect is only a tracking proxy for shaping
        # and observations; brushing it mid-air must not count as success.
        if (
            self.goal
            and not self._bridge_jump_task
            and not self._goal_on_stomp
            and not death
            and (not self._require_stomp_before_goal or self._stomp_credited)
            and (not self._require_bridge_before_goal or self._bridge_crossed)
            and (
                not self._goal_requires_support
                or (
                    self.mario["on_ground"]
                    and self.mario.get("_platform") is not None
                    and self.mario["_platform"]["rect"].top == self.goal.bottom
                )
            )
            and mario_rect.colliderect(self.goal)
        ):
            terminated = True
            reward_terms["goal"] += self.reward_config.goal
            self._goal_credited = True

        if (
            self._single_jump_attempt
            and not prev_on_ground
            and self.mario["on_ground"]
            and not self._goal_credited
        ):
            self._attempt_failed = True
            terminated = True

        # ── 18. Timeout ───────────────────────────────────────────────────────
        if self.steps >= self.max_steps:
            truncated = True

        # ── 18b. Success-conditioned energy: refund on death ─────────────────
        # Dying while trying costs no energy, so a doomed attempt is never
        # cheaper than a real one and failing families cannot collapse into
        # minimal-effort taps. Wasteful holds still cost on surviving paths.
        if death and self._episode_energy < 0.0:
            reward_terms["energy"] -= self._episode_energy
            self._episode_energy = 0.0

        # Small per-step penalty to encourage speed.
        reward_terms["frame_penalty"] += self.reward_config.frame_penalty
        reward, reward_terms = self._finalize_reward_terms(reward_terms)

        obs = self.render()
        info = self._build_info(
            reward_terms=reward_terms,
            death=death,
            terminated=terminated,
            truncated=truncated,
        )
        # Attribute the already-paid high-water progress reward to passive carry.
        # It cannot be farmed by riding back and forth over previously reached x.
        info["bridge_carry_progress"] = (
            max(0.0, min(carry_dx, self.mario["x"] - previous_max_x)) * self._progress_per_pixel
            if not death and self._require_bridge_before_goal
            else 0.0
        )
        if stomp_geometry is not None:
            info["stomp_geometry"] = stomp_geometry
        return obs, reward, terminated, truncated, info

    # ── Render ────────────────────────────────────────────────────────────────

    def render(self) -> np.ndarray:
        """Returns an (H, W, 3) uint8 RGB array of the current viewport.

        The finish marker is simulator truth and is never drawn.
        """
        self._draw(_Canvas(self.screen, labels=False))
        return np.transpose(pygame.surfarray.array3d(self.screen), (1, 0, 2))

    def render_labels(self) -> np.ndarray:
        """Each pixel's type (smb_pixel_types.TYPE_ID) as an (H, W) uint8 array.

        The same shapes as render(), in the same order, so every pixel carries
        the type of the shape drawn on top of it. Eyes belong to their owner.
        """
        self._draw(_Canvas(self._label_screen, labels=True))
        return np.ascontiguousarray(pygame.surfarray.array_red(self._label_screen).T)

    def enemy_screen_rects(self) -> list[pygame.Rect]:
        """The screen rectangle each drawn enemy is painted inside, in draw order."""
        cam = int(self.camera_x)
        rects = []
        for enemy in self.enemies:
            if enemy["h"] <= 0:
                continue
            foot = enemy.get("foot_offset", 0)
            rects.append(
                pygame.Rect(int(enemy["x"]) - cam, int(enemy["y"]), enemy["w"], enemy["h"] + foot)
            )
        return rects

    def _draw(self, canvas: _Canvas) -> None:
        cam = int(self.camera_x)
        canvas.fill(SKY, "background")
        if len(self.platform_kinds) != len(self.platforms):
            raise ValueError("platform kinds are out of step with the platforms")
        for plat, kind in zip(self.platforms, self.platform_kinds):
            r = plat["rect"]
            screen_rect = pygame.Rect(r.x - cam, r.y, r.w, r.h)
            if screen_rect.right <= 0 or screen_rect.left >= self.width:
                continue
            if kind == "moving_platform":
                self._draw_lift(canvas, screen_rect)
            elif kind == "pipe":
                self._draw_pipe(canvas, screen_rect)
            elif kind == "question_block":
                for cell in self._cells(screen_rect):
                    self._draw_question_block(canvas, cell)
            elif kind in ("brick", "brick_row"):
                self._draw_bricks(canvas, screen_rect)
                if kind == "brick_row":
                    for cell in self._cells(screen_rect):
                        if question_cell(cell.x + cam):
                            self._draw_question_block(canvas, cell)
            else:
                self._draw_ground(canvas, screen_rect)

        for coin in self.coins:
            if not coin["collected"]:
                screen_rect = coin["rect"].move(-cam, 0)
                canvas.ellipse(COIN, "coin", screen_rect)
                inner = screen_rect.inflate(-6, -4)
                if inner.w > 0 and inner.h > 0:
                    canvas.ellipse(COIN_DARK, "coin", inner)

        for enemy, rect in zip(
            (e for e in self.enemies if e["h"] > 0), self.enemy_screen_rects()
        ):
            if enemy.get("kind") == "piranha_plant":
                self._draw_plant(canvas, rect)
            else:
                self._draw_goomba(canvas, enemy, rect)

        # Mario: yellow while skidding, red otherwise; the eye shows facing.
        m = self.mario
        body = pygame.Rect(int(m["x"]) - cam, int(m["y"]), m["w"], m["h"])
        canvas.rect(MARIO_SKIDDING if m["skidding"] else MARIO, "mario", body)
        eye_x = body.right - 4 if m["facing"] > 0 else body.left + 2
        canvas.rect(EYE, "mario", pygame.Rect(eye_x, body.top + 2, 2, 2))

    def _cells(self, rect):
        """16-pixel-wide cells from the platform's left edge, visible ones only."""
        start = rect.left + max(0, -rect.left) // TILE * TILE
        for x in range(start, min(rect.right, self.width), TILE):
            yield pygame.Rect(x, rect.top, min(TILE, rect.right - x), rect.h)

    def _draw_ground(self, canvas, rect):
        canvas.rect(GROUND, "ground", rect)
        for cell in self._cells(rect):
            canvas.rect(GROUND_EDGE, "ground", pygame.Rect(cell.left, rect.top, 1, rect.h))
        for y in range(rect.top, rect.bottom, TILE):
            canvas.rect(GROUND_EDGE, "ground", pygame.Rect(rect.left, y, rect.w, 1))

    def _draw_bricks(self, canvas, rect):
        canvas.rect(BRICK, "brick", rect)
        for course, y in enumerate(range(rect.top, rect.bottom, TILE // 2)):
            height = min(TILE // 2, rect.bottom - y)
            canvas.rect(BRICK_MORTAR, "brick", pygame.Rect(rect.left, y, rect.w, 1))
            for cell in self._cells(rect):
                x = cell.left + (TILE // 2 if course % 2 else 0)
                if x < cell.right:
                    canvas.rect(BRICK_MORTAR, "brick", pygame.Rect(x, y, 1, height))

    def _draw_question_block(self, canvas, cell):
        canvas.rect(QUESTION, "question_block", cell)
        canvas.rect(QUESTION_EDGE, "question_block", cell, 1)
        if cell.w >= 8 and cell.h >= 8:
            mark = pygame.Rect(0, 0, 4, 4)
            mark.center = cell.center
            canvas.rect(QUESTION_MARK, "question_block", mark)

    def _draw_pipe(self, canvas, rect):
        canvas.rect(PIPE, "pipe", rect)
        lip = pygame.Rect(rect.left, rect.top, rect.w, min(8, rect.h))
        body = pygame.Rect(rect.left, lip.bottom, rect.w, rect.bottom - lip.bottom)
        canvas.rect(PIPE_LIGHT, "pipe", pygame.Rect(rect.left + 3, rect.top, 3, rect.h))
        canvas.rect(PIPE_DARK, "pipe", lip, 1)
        if body.h:
            canvas.rect(PIPE_DARK, "pipe", pygame.Rect(body.left, body.top, 1, body.h))
            canvas.rect(PIPE_DARK, "pipe", pygame.Rect(body.right - 1, body.top, 1, body.h))

    def _draw_lift(self, canvas, rect):
        canvas.rect(LIFT, "moving_platform", rect)
        canvas.rect(LIFT_EDGE, "moving_platform", rect, 1)
        for x in range(rect.left + 4, rect.right - 3, 8):
            canvas.rect(LIFT_EDGE, "moving_platform", pygame.Rect(x, rect.centery - 1, 2, 2))

    def _draw_goomba(self, canvas, enemy, rect):
        body_h = enemy["h"]
        if enemy["dead"]:
            # Flattened where it was stomped, resting on its feet line.
            canvas.rect(
                GOOMBA_SQUISHED, "enemy", pygame.Rect(rect.left, rect.bottom - 4, rect.w, 4)
            )
            return
        canvas.rect(GOOMBA, "enemy", pygame.Rect(rect.left, rect.top, rect.w, body_h))
        feet = rect.h - body_h
        if feet > 0:
            for x in (rect.left, rect.right - 4):
                canvas.rect(GOOMBA_FEET, "enemy", pygame.Rect(x, rect.top + body_h, 4, feet))
        eye_x = rect.right - 4 if enemy["direction"] > 0 else rect.left + 2
        canvas.rect(EYE, "enemy", pygame.Rect(eye_x, rect.top + 1, 2, 2))

    def _draw_plant(self, canvas, rect):
        head = pygame.Rect(rect.left, rect.top, rect.w, min(8, rect.h))
        if rect.h > head.h:
            stem = pygame.Rect(0, head.bottom, 4, rect.bottom - head.bottom)
            stem.centerx = rect.centerx
            canvas.rect(PLANT_STEM, "enemy", stem)
        canvas.rect(PLANT_HEAD, "enemy", head)
        if head.h >= 4:
            for x in (head.left + 2, head.right - 4):
                canvas.rect(PLANT_SPOT, "enemy", pygame.Rect(x, head.top + 1, 2, 2))

    # ── Structured state / info ───────────────────────────────────────────────

    @staticmethod
    def _finalize_reward_terms(reward_terms: dict[str, float]) -> tuple[float, dict[str, float]]:
        terms = {name: float(value) for name, value in reward_terms.items() if name != "total"}
        total = float(sum(terms.values()))
        terms["total"] = total
        return total, terms

    def _has_horizontal_platform_support(self) -> bool:
        """Return whether Mario horizontally overlaps any platform surface."""

        left = self.mario["x"]
        right = self.mario["x"] + self.mario["w"]
        return any(
            right > platform["rect"].left and left < platform["rect"].right
            for platform in self.platforms
        )

    def _build_info(
        self,
        reward_terms: dict[str, float] = None,
        *,
        death: bool | None = None,
        terminated: bool = False,
        truncated: bool = False,
    ) -> dict:
        """
        Returns a rich info dict containing:
        - mario   : core kinematic state
        - camera_x, max_x_reached
        - nearest_coin, nearest_enemy : {dx, dy, dist}  (normalised 0-1)
        - platform_below_dist          : normalised 0-1
        - state_vec                    : geometry features in world coordinates (FEATURE_NAMES)
        - reward_terms                 : transition reward breakdown
        - reward_total                 : scalar transition reward
        - reward_config                : resolved reward configuration
        """
        reward_total, reward_terms = self._finalize_reward_terms(
            reward_terms or self.reward_config.zero_terms()
        )
        if death is None:
            death = bool(
                reward_terms.get("fall_death", 0.0) < 0.0
                or reward_terms.get("enemy_hit", 0.0) < 0.0
            )

        from retroagi.core.smb_geometry import geometry_features

        features = geometry_features(
            self,
            death=death,
            terminated=terminated,
            truncated=truncated,
        )
        m = self.mario

        return {
            "mario": {
                k: m[k]
                for k in (
                    "x",
                    "y",
                    "vx",
                    "vy",
                    "on_ground",
                    "facing",
                    "skidding",
                )
            },
            "camera_x": self.camera_x,
            "max_x_reached": self._max_x_reached,
            # The stage-wide symbolic state; the policy's observation is smb_scene's.
            "state_vec": features.pop("state"),
            **features,
            "stomp_completed": self._stomp_credited,
            "bridge_boarded": self._bridge_boarded,
            "bridge_crossed": self._bridge_crossed,
            "attempt_failed": self._attempt_failed,
            "reward_terms": dict(reward_terms),
            "reward_total": reward_total,
            "reward_config": asdict(self.reward_config),
            "death": death,
            "terminated": bool(terminated),
            "truncated": bool(truncated),
        }

    # ── Enemy helpers ─────────────────────────────────────────────────────────

    def _update_enemy(self, enemy: dict, platform_rects: list):
        """Move enemy, apply gravity, resolve platform collisions, patrol logic."""
        if enemy.get("kind") == "piranha_plant":
            from .piranha import position_plant

            enemy["plant_tick"] += 1
            position_plant(enemy)
            return
        # Gravity
        enemy["vy"] += ENEMY_GRAVITY
        if enemy["vy"] > self.max_fall_speed:
            enemy["vy"] = self.max_fall_speed

        # Horizontal move — check edge-awareness before stepping
        step_x = enemy["speed"] * enemy["direction"]
        if enemy["edge_aware"] and enemy["on_ground"]:
            # Peek one pixel ahead at feet level; turn if no platform below
            peek_x = enemy["x"] + step_x + (enemy["w"] if enemy["direction"] > 0 else -1)
            feet_y = enemy["y"] + enemy["h"] + enemy.get("foot_offset", 0) + 1
            supported = any(
                r.left <= peek_x <= r.right and r.top <= feet_y <= r.bottom for r in platform_rects
            )
            if not supported:
                enemy["direction"] *= -1
                step_x = enemy["speed"] * enemy["direction"]

        enemy["x"] += step_x

        # Clamp to explicit patrol bounds (still respected when edge_aware=True)
        if enemy["x"] <= enemy["patrol_min"]:
            enemy["x"] = enemy["patrol_min"]
            enemy["direction"] = 1
        elif enemy["x"] >= enemy["patrol_max"]:
            enemy["x"] = enemy["patrol_max"]
            enemy["direction"] = -1

        # Y — apply and resolve
        enemy["y"] += enemy["vy"]
        enemy["on_ground"] = False
        er = pygame.Rect(
            enemy["x"], enemy["y"], enemy["w"], enemy["h"] + enemy.get("foot_offset", 0)
        )
        for r in platform_rects:
            if er.colliderect(r):
                if enemy["vy"] >= 0:
                    er.bottom = r.top
                    enemy["on_ground"] = True
                elif enemy["vy"] < 0:
                    er.top = r.bottom
                enemy["y"] = er.y
                enemy["vy"] = 0

    # ── Parsing helpers ───────────────────────────────────────────────────────

    def _platform_kinds(self, kinds) -> list[str]:
        """How each platform is drawn; never consulted by physics or rewards.

        Lifts are "moving_platform". A tagged platform keeps its tag. An
        untagged one is a "brick_row" (bricks with question blocks at fixed
        cells) when open space lies directly beneath any part of it, and
        ground otherwise (floor, stairs, raised ground).
        """
        if kinds is None:
            kinds = [None] * len(self.platforms)
        if len(kinds) != len(self.platforms):
            raise ValueError("platform_kinds must name one kind per platform")
        resolved = []
        for plat, kind in zip(self.platforms, kinds):
            if kind is not None and kind not in PLATFORM_KINDS:
                raise ValueError(f"platform kind must be one of {PLATFORM_KINDS} or None")
            if plat["moving"]:
                resolved.append("moving_platform")
            elif kind is not None:
                resolved.append(kind)
            else:
                resolved.append("brick_row" if self._floating(plat) else "ground")
        return resolved

    def _floating(self, plat) -> bool:
        rect = plat["rect"]
        if rect.bottom >= self.height:
            return False
        covered = np.zeros(rect.w, dtype=bool)
        for other in self.platforms:
            r = other["rect"]
            if other is plat or other["moving"] or not r.top <= rect.bottom < r.bottom:
                continue
            covered[max(r.left - rect.left, 0) : max(r.right - rect.left, 0)] = True
        return not covered.all()

    @staticmethod
    def _parse_platform(p) -> dict:
        """Accept [x,y,w,h] list or {'x','y','w','h','moving':[…]} dict."""
        if isinstance(p, dict):
            x, y, w, h = p["x"], p["y"], p["w"], p["h"]
            mv = p.get("moving")
        else:
            x, y, w, h = p[0], p[1], p[2], p[3]
            mv = None

        plat = {
            "rect": pygame.Rect(x, y, w, h),
            "moving": mv is not None,
            "move_min": int(mv[0]) if mv else 0,
            "move_max": int(mv[1]) if mv else 0,
            "move_speed": float(mv[2]) if mv else 0.0,
            "move_dir": int(p.get("direction", 1)) if isinstance(p, dict) else 1,
            "delta_x": 0,
        }
        plat["move_x"] = float(plat["rect"].x)
        return plat

    @staticmethod
    def _parse_enemy(e) -> dict:
        """Accept [x,y,pmin,pmax[,speed[,direction]]] or a dictionary."""
        if isinstance(e, dict) and e.get("kind") == "piranha_plant":
            from .piranha import parse_plant

            return parse_plant(e)
        if isinstance(e, dict):
            x, y = float(e["x"]), float(e["y"])
            pmin, pmax = float(e["patrol_min"]), float(e["patrol_max"])
            speed = float(e.get("speed", 1.5))
            edge_aware = bool(e.get("edge_aware", False))
            direction = int(e.get("direction", 1))
        else:
            x, y = float(e[0]), float(e[1])
            pmin, pmax = float(e[2]), float(e[3])
            speed = float(e[4]) if len(e) > 4 else 1.5
            edge_aware = False
            direction = int(e[5]) if len(e) > 5 else 1

        if direction not in (-1, 1):
            raise ValueError("enemy direction must be -1 or 1")
        return {
            "x": x,
            "y": y,
            "vx": 0.0,
            "vy": 0.0,
            "w": 12,
            "h": 14,
            "speed": speed,
            "direction": direction,
            "patrol_min": pmin,
            "patrol_max": pmax,
            "edge_aware": edge_aware,
            "on_ground": False,
            "dead": False,
        }

    # ── Procedural level generator ────────────────────────────────────────────

    @classmethod
    def generate_scenario(
        cls,
        num_screens: int = 3,
        gap_range: tuple = (24, 48),
        platform_height_range: tuple = (140, 200),
        platform_width_range: tuple = (40, 80),
        enemy_density: float = 0.5,
        moving_platform_chance: float = 0.2,
        seed: int = None,
    ) -> dict:
        """
        Generate a random scrolling level.

        Parameters
        ----------
        num_screens            : how many viewport-widths wide the level is
        gap_range              : (min, max) horizontal gap between platforms in px
        platform_height_range  : (min, max) y coordinate for platforms (lower y = higher up)
        platform_width_range   : (min, max) platform width in px
        enemy_density          : probability [0,1] that a platform gets an enemy
        moving_platform_chance : probability [0,1] that a platform moves horizontally
        seed                   : RNG seed for reproducibility
        """
        rng = random.Random(seed)
        VIEW_W = 256
        world_width = VIEW_W * num_screens
        floor_y = 220
        floor_h = 20

        platforms = [[0, floor_y, world_width, floor_h]]  # continuous floor
        coins = []
        enemies = []

        x = 60  # starting x for first gap after spawn area
        while x < world_width - VIEW_W // 2:
            pw = rng.randint(*platform_width_range)
            py = rng.randint(*platform_height_range)
            gap = rng.randint(*gap_range)

            # Decide if moving
            moving = rng.random() < moving_platform_chance
            if moving:
                move_dist = rng.randint(20, 50)
                mv = [max(0, x - move_dist), x + move_dist, rng.uniform(0.5, 1.5)]
                platforms.append({"x": x, "y": py, "w": pw, "h": 10, "moving": mv})
            else:
                platforms.append([x, py, pw, 10])

            # Coin above platform center
            cx = x + pw // 2 - 5
            coins.append([cx, py - 18, 10, 10])

            # Enemy on this platform
            if rng.random() < enemy_density:
                ey = py - 14  # stand on top of platform
                enemies.append(
                    {
                        "x": float(x + 2),
                        "y": float(ey),
                        "patrol_min": float(x),
                        "patrol_max": float(x + pw - 14),
                        "speed": rng.uniform(0.8, 2.2),
                        "edge_aware": True,
                    }
                )

            x += pw + gap

        # Goal at end
        goal_x = world_width - 36
        goal = [goal_x, floor_y - 40, 16, 40]
        platforms.append([goal_x - 10, floor_y - 40, 36, 10])  # goal platform

        return {
            "world_width": world_width,
            "mario": [20, floor_y - 12],  # Standing on the floor (Mario is 12 px tall).
            "platforms": platforms,
            "coins": coins,
            "enemies": enemies,
            "goal": goal,
        }

    # ── Misc helpers ──────────────────────────────────────────────────────────

    @staticmethod
    def load_scenario_from_json(filepath: str) -> dict:
        with open(filepath) as f:
            return json.load(f)


# ── Interactive demo ──────────────────────────────────────────────────────────


def main():
    """Run the interactive pygame scenario simulator."""
    pygame.init()
    env = MarioScenarioEnv()

    # Hand-crafted scrolling level with moving platforms and varied enemies
    custom_scenario = {
        "world_width": 768,
        "mario": [20, 180],
        "platforms": [
            # Screen 1
            [0, 220, 100, 20],
            [130, 180, 50, 10],
            {"x": 205, "y": 145, "w": 50, "h": 10, "moving": [180, 240, 1.0]},
            # Screen 2
            [270, 220, 120, 20],
            [320, 170, 40, 10],
            {"x": 395, "y": 130, "w": 55, "h": 10, "moving": [370, 440, 1.4]},
            [460, 190, 40, 10],
            # Screen 3
            [520, 220, 248, 20],
            [560, 170, 40, 10],
            [640, 130, 40, 10],
            [720, 100, 48, 120],
        ],
        "coins": [
            [145, 150, 10, 10],
            [220, 115, 10, 10],
            [335, 140, 10, 10],
            [410, 100, 10, 10],
            [575, 140, 10, 10],
            [655, 100, 10, 10],
        ],
        # Enemies: mix of fixed-speed list form and edge-aware dict form
        "enemies": [
            [10, 200, 2, 90],  # slow default
            [132, 160, 130, 178, 1.0],  # explicit speed
            {
                "x": 272.0,
                "y": 200.0,
                "patrol_min": 270.0,
                "patrol_max": 388.0,
                "speed": 2.0,
                "edge_aware": True,
            },
            [325, 150, 322, 438, 1.2],
            {
                "x": 522.0,
                "y": 200.0,
                "patrol_min": 520.0,
                "patrol_max": 660.0,
                "speed": 1.8,
                "edge_aware": True,
            },
            [562, 150, 560, 598, 0.9],
        ],
        "goal": [730, 80, 16, 20],
    }

    config_path = os.path.join(os.path.dirname(__file__), "scenarios", "level_1.json")
    if os.path.exists(config_path):
        scenario_config = MarioScenarioEnv.load_scenario_from_json(config_path)
    else:
        scenario_config = custom_scenario

    obs, info = env.reset(scenario=scenario_config)

    display = pygame.display.set_mode((env.width, env.height))
    pygame.display.set_caption("Mario AI Scenario Simulator")
    clock = pygame.time.Clock()

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        action = np.random.choice([0, 1, 1, 1, 2, 2, 5])
        obs, reward, terminated, truncated, info = env.step(action)

        surface = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        display.blit(surface, (0, 0))
        pygame.display.flip()
        clock.tick(30)

        if terminated or truncated:
            reason = "terminated" if terminated else "truncated (timeout)"
            print(f"Episode ended ({reason}) | score={env.score} | max_x={env._max_x_reached:.0f}")
            obs, info = env.reset(scenario=custom_scenario)

    env.close()


if __name__ == "__main__":
    main()
