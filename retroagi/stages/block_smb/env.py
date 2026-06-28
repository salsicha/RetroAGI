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


# ── Tuning constants ──────────────────────────────────────────────────────────

COYOTE_FRAMES = 5  # frames after leaving a ledge where jumping is still allowed
JUMP_BUFFER_FRAMES = 6  # frames before landing where a queued jump fires on contact
JUMP_CUT_FACTOR = 0.45  # vy multiplier when jump is released early (variable height)


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

    def __post_init__(self) -> None:
        if self.progress_per_pixel < 0:
            raise ValueError("progress_per_pixel must be non-negative")
        for name in ("coin", "enemy_stomp", "goal"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be non-negative")
        for name in ("fall_death", "gap_jump", "enemy_hit", "frame_penalty"):
            if getattr(self, name) > 0:
                raise ValueError(f"{name} must be non-positive")

    def zero_terms(self) -> dict[str, float]:
        return {
            "progress": 0.0,
            "coin": 0.0,
            "enemy_stomp": 0.0,
            "goal": 0.0,
            "fall_death": 0.0,
            "gap_jump": 0.0,
            "enemy_hit": 0.0,
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
    coins         : list of [x, y, w, h]
    enemies       : list of [x, y, patrol_min, patrol_max] or
                    [x, y, patrol_min, patrol_max, speed] or
                    dict with keys x,y,patrol_min,patrol_max,speed,edge_aware
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

        # Physics
        self.gravity = 0.5
        self.max_fall_speed = 8.0
        self.jump_power = -8.5

        # Horizontal momentum
        self.max_walk_speed = 3.0
        self.accel = 0.3
        self.decel = 0.2
        self.skid_decel = 0.5

        # Spaces (Gym-compatible stubs)
        self.action_space = _DiscreteSpace(6)
        self.action_space_n = 6  # legacy alias
        self.observation_space = _BoxSpace((height, width, 3))

        # Offscreen observations only need software surfaces; the interactive
        # demo initializes the display subsystem when it opens a window.
        self.screen = pygame.Surface((self.width, self.height))

        # RNG (seeded via self.seed())
        self._rng = random.Random()

        # State (populated by reset)
        self.mario = None
        self.platforms = []  # list of platform dicts
        self.coins = []
        self.enemies = []
        self.goal = None
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

        # Mario state
        self.mario = {
            "x": float(scenario["mario"][0]),
            "y": float(scenario["mario"][1]),
            "vx": 0.0,
            "vy": 0.0,
            "w": 14,
            "h": 16,
            "on_ground": False,
            "facing": 1,  # 1 = right, -1 = left
            "skidding": False,
            # Jump feel
            "coyote_frames": 0,  # counts down after leaving ground
            "jump_buffer": 0,  # counts down after jump pressed in air
            "jump_held": False,  # was jump action present last frame?
        }
        self._max_x_reached = self.mario["x"]

        # Platforms — accept list [x,y,w,h] or dict
        self.platforms = []
        for p in scenario.get("platforms", []):
            self.platforms.append(self._parse_platform(p))

        # Coins
        self.coins = [
            {"rect": pygame.Rect(c[0], c[1], c[2], c[3]), "collected": False}
            for c in scenario.get("coins", [])
        ]

        # Enemies — accept list or dict; optional 5th element = speed
        self.enemies = []
        for e in scenario.get("enemies", []):
            self.enemies.append(self._parse_enemy(e))

        self.goal = pygame.Rect(*scenario["goal"]) if "goal" in scenario else None

        obs = self.render()
        _, reward_terms = self._finalize_reward_terms(self.reward_config.zero_terms())
        info = self._build_info(reward_terms=reward_terms)
        return obs, info

    # ── Step ──────────────────────────────────────────────────────────────────

    def step(self, action: int):
        """
        Advance one frame.
        Returns (obs, reward, terminated, truncated, info) — Gym v26 API.
        terminated = game-ending event (death / goal)
        truncated  = timeout
        """
        self.steps += 1
        reward_terms = self.reward_config.zero_terms()
        terminated = False
        truncated = False
        info = {}

        jump_pressed = action in [2, 4, 5]
        move_x = 1 if action in [1, 2] else (-1 if action in [3, 4] else 0)

        # ── 1. Horizontal momentum ────────────────────────────────────────────
        vx = self.mario["vx"]
        self.mario["skidding"] = False

        if move_x != 0:
            self.mario["facing"] = move_x
            if (move_x > 0 and vx < 0) or (move_x < 0 and vx > 0):
                self.mario["skidding"] = True
                vx += move_x * self.skid_decel
            else:
                vx += move_x * self.accel
            vx = max(-self.max_walk_speed, min(self.max_walk_speed, vx))
        else:
            if vx > 0:
                vx = max(0.0, vx - self.decel)
            elif vx < 0:
                vx = min(0.0, vx + self.decel)

        self.mario["vx"] = vx

        # ── 2. Variable jump height (cut on release) ──────────────────────────
        was_jump_held = self.mario["jump_held"]
        self.mario["jump_held"] = jump_pressed

        if was_jump_held and not jump_pressed and self.mario["vy"] < 0:
            # Jump released early — cut upward velocity
            self.mario["vy"] *= JUMP_CUT_FACTOR

        # ── 3. Jump with coyote time + jump buffer ────────────────────────────
        if jump_pressed:
            if not was_jump_held:
                # Fresh press — register in buffer regardless of ground state
                self.mario["jump_buffer"] = JUMP_BUFFER_FRAMES
        else:
            self.mario["jump_buffer"] = max(0, self.mario["jump_buffer"] - 1)

        can_jump = self.mario["on_ground"] or self.mario["coyote_frames"] > 0
        wants_jump = self.mario["jump_buffer"] > 0

        if can_jump and wants_jump:
            self.mario["vy"] = self.jump_power
            self.mario["on_ground"] = False
            self.mario["coyote_frames"] = 0
            self.mario["jump_buffer"] = 0
            self._airborne_started_with_jump = True

        # ── 4. Gravity ────────────────────────────────────────────────────────
        self.mario["vy"] += self.gravity
        if self.mario["vy"] > self.max_fall_speed:
            self.mario["vy"] = self.max_fall_speed

        # ── 5. Update moving platforms ────────────────────────────────────────
        for plat in self.platforms:
            if not plat["moving"]:
                continue
            old_px = plat["rect"].x
            plat["rect"].x += int(plat["move_speed"] * plat["move_dir"])
            if plat["rect"].x <= plat["move_min"]:
                plat["rect"].x = plat["move_min"]
                plat["move_dir"] = 1
            elif plat["rect"].x >= plat["move_max"]:
                plat["rect"].x = plat["move_max"]
                plat["move_dir"] = -1
            plat["delta_x"] = plat["rect"].x - old_px

        # ── 6. Resolve X collisions ───────────────────────────────────────────
        self.mario["x"] += self.mario["vx"]
        mario_rect = pygame.Rect(self.mario["x"], self.mario["y"], self.mario["w"], self.mario["h"])

        for plat in self.platforms:
            r = plat["rect"]
            if mario_rect.colliderect(r):
                if self.mario["vx"] > 0:
                    mario_rect.right = r.left
                elif self.mario["vx"] < 0:
                    mario_rect.left = r.right
                self.mario["x"] = mario_rect.x
                self.mario["vx"] = 0

        # ── 7. Resolve Y collisions ───────────────────────────────────────────
        previous_y = self.mario["y"]
        previous_bottom = previous_y + self.mario["h"]
        self.mario["y"] += self.mario["vy"]
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

        # ── 8. Carry Mario on moving platform ────────────────────────────────
        if (
            self.mario["on_ground"]
            and self.mario["_platform"]
            and self.mario["_platform"]["moving"]
        ):
            self.mario["x"] += self.mario["_platform"]["delta_x"]
            mario_rect.x = self.mario["x"]

        # ── 9. Coyote time bookkeeping ────────────────────────────────────────
        if self.mario["on_ground"]:
            self.mario["coyote_frames"] = COYOTE_FRAMES
            self._airborne_started_with_jump = False
        else:
            self.mario["coyote_frames"] = max(0, self.mario["coyote_frames"] - 1)

        # Jump buffer ticks down every frame (already done above but also here for landing)
        if not prev_on_ground and self.mario["on_ground"] and self.mario["jump_buffer"] > 0:
            # Landed with a buffered jump — fire it next frame naturally (buffer still > 0)
            pass

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

        # ── 12. Rightward-progress reward ─────────────────────────────────────
        if self.mario["x"] > self._max_x_reached:
            reward_terms["progress"] += (
                self.mario["x"] - self._max_x_reached
            ) * self.reward_config.progress_per_pixel
            self._max_x_reached = self.mario["x"]

        # ── 13. Fall death ────────────────────────────────────────────────────
        if self.mario["y"] > self.height:
            terminated = True
            if self._airborne_started_with_jump and not self._has_horizontal_platform_support():
                reward_terms["gap_jump"] += self.reward_config.gap_jump
            reward_terms["fall_death"] += self.reward_config.fall_death

        # ── 14. Update enemies ────────────────────────────────────────────────
        rects = [p["rect"] for p in self.platforms]
        for enemy in self.enemies:
            if enemy["dead"]:
                continue
            self._update_enemy(enemy, rects)

        # ── 15. Enemy collision ───────────────────────────────────────────────
        for enemy in self.enemies:
            if enemy["dead"]:
                continue
            er = pygame.Rect(enemy["x"], enemy["y"], enemy["w"], enemy["h"])
            if not mario_rect.colliderect(er):
                continue
            prev_bottom = mario_rect.bottom - self.mario["vy"]
            if self.mario["vy"] > 0 and prev_bottom <= er.centery:
                # Stomp!
                enemy["dead"] = True
                reward_terms["enemy_stomp"] += self.reward_config.enemy_stomp
                self.score += 5
                self.mario["vy"] = self.jump_power * 0.55
                self.mario["on_ground"] = False
            else:
                terminated = True
                reward_terms["enemy_hit"] += self.reward_config.enemy_hit

        # ── 16. Coin collection ───────────────────────────────────────────────
        for coin in self.coins:
            if not coin["collected"] and mario_rect.colliderect(coin["rect"]):
                coin["collected"] = True
                reward_terms["coin"] += self.reward_config.coin
                self.score += 10

        # ── 17. Goal ──────────────────────────────────────────────────────────
        if self.goal and mario_rect.colliderect(self.goal):
            terminated = True
            reward_terms["goal"] += self.reward_config.goal

        # ── 18. Timeout ───────────────────────────────────────────────────────
        if self.steps >= self.max_steps:
            truncated = True

        # Small per-step penalty to encourage speed.
        reward_terms["frame_penalty"] += self.reward_config.frame_penalty
        reward, reward_terms = self._finalize_reward_terms(reward_terms)

        obs = self.render()
        info = self._build_info(reward_terms=reward_terms)
        return obs, reward, terminated, truncated, info

    # ── Render ────────────────────────────────────────────────────────────────

    def render(self) -> np.ndarray:
        """Returns an (H, W, 3) uint8 RGB array of the current viewport."""
        cam = int(self.camera_x)
        self.screen.fill((107, 140, 255))  # sky blue

        # Platforms — green tint for moving, brown for static
        for plat in self.platforms:
            r = plat["rect"]
            sr = pygame.Rect(r.x - cam, r.y, r.w, r.h)
            color = (80, 160, 40) if plat["moving"] else (139, 69, 19)
            pygame.draw.rect(self.screen, color, sr)

        # Coins (gold)
        for coin in self.coins:
            if not coin["collected"]:
                r = coin["rect"]
                sr = pygame.Rect(r.x - cam, r.y, r.w, r.h)
                pygame.draw.ellipse(self.screen, (255, 215, 0), sr)

        # Goal (bright green)
        if self.goal:
            sr = pygame.Rect(self.goal.x - cam, self.goal.y, self.goal.w, self.goal.h)
            pygame.draw.rect(self.screen, (0, 255, 0), sr)

        # Enemies
        for enemy in self.enemies:
            sx = int(enemy["x"]) - cam
            sy = int(enemy["y"])
            if enemy["dead"]:
                squish = pygame.Rect(sx, sy + enemy["h"] - 4, enemy["w"], 4)
                pygame.draw.rect(self.screen, (100, 0, 160), squish)
            else:
                pygame.draw.rect(
                    self.screen, (160, 32, 240), pygame.Rect(sx, sy, enemy["w"], enemy["h"])
                )
                eye_x = sx + (8 if enemy["direction"] > 0 else 2)
                pygame.draw.circle(self.screen, (255, 255, 255), (eye_x, sy + 4), 2)

        # Mario — yellow when skidding, red otherwise; eye shows facing
        msx = int(self.mario["x"]) - cam
        msy = int(self.mario["y"])
        color = (255, 220, 0) if self.mario["skidding"] else (255, 0, 0)
        pygame.draw.rect(
            self.screen, color, pygame.Rect(msx, msy, self.mario["w"], self.mario["h"])
        )
        eye_x = msx + (10 if self.mario["facing"] > 0 else 2)
        pygame.draw.circle(self.screen, (255, 255, 255), (eye_x, msy + 4), 2)

        return np.transpose(pygame.surfarray.array3d(self.screen), (1, 0, 2))

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

    def _build_info(self, reward_terms: dict[str, float] = None) -> dict:
        """
        Returns a rich info dict containing:
        - mario   : core kinematic state
        - camera_x, max_x_reached
        - nearest_coin, nearest_enemy : {dx, dy, dist}  (normalised 0-1)
        - platform_below_dist          : normalised 0-1
        - state_vec                    : flat float32 array ready for RL (24 dims)
        - reward_terms                 : transition reward breakdown
        - reward_total                 : scalar transition reward
        - reward_config                : resolved reward configuration
        """
        reward_total, reward_terms = self._finalize_reward_terms(
            reward_terms or self.reward_config.zero_terms()
        )

        m = self.mario
        ww = self.world_width
        wh = self.height

        def _nearest(items, get_rect):
            best_dist = float("inf")
            best_dx = best_dy = 1.0
            mx, my = m["x"] + m["w"] / 2, m["y"] + m["h"] / 2
            for item in items:
                r = get_rect(item)
                dx = (r.centerx - mx) / ww
                dy = (r.centery - my) / wh
                d = math.hypot(dx, dy)
                if d < best_dist:
                    best_dist, best_dx, best_dy = d, dx, dy
            return {"dx": best_dx, "dy": best_dy, "dist": min(best_dist, 1.0)}

        active_coins = [c for c in self.coins if not c["collected"]]
        active_enemies = [e for e in self.enemies if not e["dead"]]

        nc = (
            _nearest(active_coins, lambda c: c["rect"])
            if active_coins
            else {"dx": 1.0, "dy": 1.0, "dist": 1.0}
        )
        ne = (
            _nearest(active_enemies, lambda e: pygame.Rect(e["x"], e["y"], e["w"], e["h"]))
            if active_enemies
            else {"dx": 1.0, "dy": 1.0, "dist": 1.0}
        )

        # Distance to nearest platform below Mario's feet
        mario_bottom = m["y"] + m["h"]
        plat_below = 1.0
        for p in self.platforms:
            r = p["rect"]
            if r.left <= m["x"] + m["w"] and r.right >= m["x"]:
                if r.top >= mario_bottom:
                    dist = (r.top - mario_bottom) / wh
                    plat_below = min(plat_below, dist)

        mx, my = m["x"] + m["w"] / 2, m["y"] + m["h"] / 2
        mario_right = m["x"] + m["w"]
        if self.goal is None:
            goal_dx = goal_dy = goal_dist = 1.0
        else:
            goal_dx = (self.goal.centerx - mx) / ww
            goal_dy = (self.goal.centery - my) / wh
            goal_dist = min(math.hypot(goal_dx, goal_dy), 1.0)

        support_candidates = []
        for p in self.platforms:
            r = p["rect"]
            if r.left <= mx <= r.right and (mario_bottom - 2.0) <= r.top <= (mario_bottom + 8.0):
                support_candidates.append(r)
        if support_candidates:
            support = min(support_candidates, key=lambda r: abs(r.top - mario_bottom))
            support_right_dx = (support.right - mario_right) / ww
        else:
            support_right_dx = 1.0

        next_platform_dx = 1.0
        next_platform_dy = 1.0
        ahead_platforms = [p["rect"] for p in self.platforms if p["rect"].left > mario_right]
        if ahead_platforms:
            next_platform = min(ahead_platforms, key=lambda r: r.left - mario_right)
            next_platform_dx = (next_platform.left - mario_right) / ww
            next_platform_dy = (next_platform.top - mario_bottom) / wh

        def _ground_ahead(offset: float) -> float:
            probe_x = mario_right + offset
            best: float | None = None
            for p in self.platforms:
                r = p["rect"]
                if r.left <= probe_x <= r.right:
                    dy = (r.top - mario_bottom) / wh
                    if best is None or abs(dy) < abs(best):
                        best = dy
            return 1.0 if best is None else best

        state_vec = np.array(
            [
                m["x"] / ww,
                m["y"] / wh,
                m["vx"] / self.max_walk_speed,
                m["vy"] / self.max_fall_speed,
                float(m["on_ground"]),
                float(m["facing"]),
                float(m["skidding"]),
                float(m["coyote_frames"]) / COYOTE_FRAMES,
                float(m["jump_buffer"]) / JUMP_BUFFER_FRAMES,
                nc["dx"],
                nc["dy"],
                nc["dist"],
                ne["dx"],
                ne["dist"],
                min(float(self.steps) / 200.0, 1.0),
                goal_dx,
                goal_dy,
                goal_dist,
                support_right_dx,
                next_platform_dx,
                next_platform_dy,
                _ground_ahead(24.0),
                _ground_ahead(48.0),
                _ground_ahead(72.0),
            ],
            dtype=np.float32,
        )

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
                    "coyote_frames",
                    "jump_buffer",
                )
            },
            "camera_x": self.camera_x,
            "max_x_reached": self._max_x_reached,
            "nearest_coin": nc,
            "nearest_enemy": ne,
            "platform_below_dist": plat_below,
            "goal_delta": {"dx": goal_dx, "dy": goal_dy, "dist": goal_dist},
            "support_right_dx": support_right_dx,
            "next_platform_delta": {"dx": next_platform_dx, "dy": next_platform_dy},
            "ground_ahead": {
                "24": _ground_ahead(24.0),
                "48": _ground_ahead(48.0),
                "72": _ground_ahead(72.0),
            },
            "reward_terms": dict(reward_terms),
            "reward_total": reward_total,
            "reward_config": asdict(self.reward_config),
            "state_vec": state_vec,
        }

    # ── Enemy helpers ─────────────────────────────────────────────────────────

    def _update_enemy(self, enemy: dict, platform_rects: list):
        """Move enemy, apply gravity, resolve platform collisions, patrol logic."""
        # Gravity
        enemy["vy"] += self.gravity
        if enemy["vy"] > self.max_fall_speed:
            enemy["vy"] = self.max_fall_speed

        # Horizontal move — check edge-awareness before stepping
        step_x = enemy["speed"] * enemy["direction"]
        if enemy["edge_aware"] and enemy["on_ground"]:
            # Peek one pixel ahead at feet level; turn if no platform below
            peek_x = enemy["x"] + step_x + (enemy["w"] if enemy["direction"] > 0 else -1)
            feet_y = enemy["y"] + enemy["h"] + 1
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
        er = pygame.Rect(enemy["x"], enemy["y"], enemy["w"], enemy["h"])
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
            "move_dir": 1,
            "delta_x": 0,
        }
        return plat

    @staticmethod
    def _parse_enemy(e) -> dict:
        """Accept list [x,y,pmin,pmax] or [x,y,pmin,pmax,speed] or dict."""
        if isinstance(e, dict):
            x, y = float(e["x"]), float(e["y"])
            pmin, pmax = float(e["patrol_min"]), float(e["patrol_max"])
            speed = float(e.get("speed", 1.5))
            edge_aware = bool(e.get("edge_aware", False))
        else:
            x, y = float(e[0]), float(e[1])
            pmin, pmax = float(e[2]), float(e[3])
            speed = float(e[4]) if len(e) > 4 else 1.5
            edge_aware = False

        return {
            "x": x,
            "y": y,
            "vx": 0.0,
            "vy": 0.0,
            "w": 12,
            "h": 14,
            "speed": speed,
            "direction": 1,
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
            "mario": [20, floor_y - 16],
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
