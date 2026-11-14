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
from functools import partial

import numpy as np
import pygame

from retroagi.domains.smb.perception.pixel_types import TYPE_ID
from retroagi.domains.smb.physics import (SPRING_FORCE, SPRING_LAUNCH, SPRING_SQUASH, SPRING_STRONG, SPRING_THROW,
                                         NESPlayerMotion)

from . import tactic_schedule
from .piranha import plant_sprite
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
# player moves with the NES's own motion rules (retroagi.domains.smb.physics):
# fixed-point acceleration, run/walk limits, braking, air steering, variable
# jump forces and button edges. There is no coyote time and no jump buffer;
# holding jump through a landing does not jump again. The NES small body is
# 10x12 pixels and the Goomba damage body 10x6.
# An enemy or a mushroom falls off a ledge as the game's do: this much
# faster each frame, up to ENEMY_FALL (measured in the game's memory, Goombas
# and Koopas falling in worlds 1-1 to 5-1: within 1.6 pixels of them on
# average; the 0.5 up to 4.5 written first was 11 off). The planner's rules
# take the same (planner.ENEMY_GRAVITY).
ENEMY_GRAVITY, ENEMY_FALL = 0.2, 3.4

# ── Drawing ───────────────────────────────────────────────────────────────────
#
# The picture is purely visual: nothing below changes layouts, physics or
# rewards. Solid platforms are drawn as one of four kinds, as in real SMB
# (floor and stairs are ground, floating rows are bricks with question blocks,
# pipes are green). A scenario may name each platform's kind in
# "platform_kinds", aligned with "platforms"; an untagged (None) platform is a
# brick row when open space lies directly beneath it and ground otherwise.
# Moving platforms are always lifts. Power-ups (mushrooms) only reward Mario
# for touching them; he does not grow. render_labels() repeats render()'s
# shapes with each shape's pixel type in place of its colour, and
# render_instances() with the number of the object that drew it, so both are
# exact.
PLATFORM_KINDS = ("ground", "brick", "question_block", "pipe", "cannon", "spring")
TILE = 16
SKY = (107, 140, 255)
GROUND, GROUND_EDGE = (139, 69, 19), (84, 38, 8)
BRICK, BRICK_MORTAR = (200, 76, 12), (64, 24, 0)
QUESTION, QUESTION_EDGE, QUESTION_MARK = (252, 160, 68), (136, 72, 0), (96, 40, 0)
PIPE, PIPE_LIGHT, PIPE_DARK = (0, 168, 0), (128, 208, 16), (0, 80, 0)
LIFT, LIFT_EDGE = (216, 216, 216), (120, 120, 120)
SPRING = (200, 60, 60)
COIN, COIN_DARK = (255, 215, 0), (204, 140, 0)
GOOMBA, GOOMBA_FEET, GOOMBA_SQUISHED = (160, 32, 240), (72, 0, 112), (100, 0, 160)
PLANT_HEAD, PLANT_SPOT, PLANT_STEM = (216, 40, 96), (255, 200, 220), (0, 112, 72)
MONSTER, MONSTER_SPIKE, MONSTER_BELLY = (24, 104, 40), (240, 240, 208), (248, 168, 56)
# The monster: an enemy that cannot be stomped (touching it from any side
# kills Mario), larger than a Goomba. Drawn as its own sprite, it is the
# vision's "other" enemy kind, like Full SMB's Spiny, Hammer Bro and Bullet Bill.
MONSTER_W, MONSTER_H = 16, 20
MARIO, MARIO_SKIDDING, EYE = (255, 0, 0), (255, 220, 0), (255, 255, 255)
POWER_UP_CAP, POWER_UP_SPOT, POWER_UP_STEM = (232, 48, 24), (252, 252, 252), (252, 216, 168)
POWER_UP_SIZE = 16
# Question blocks, mushrooms and big Mario (levels that give block contents,
# domains/smb/sim/levels.py), as the game has them: points in its
# proportions (a stomp 100, a coin 200, a mushroom 1000, here 5, 10, 50), a
# mushroom moving MUSHROOM_SPEED pixels a frame, big Mario's body BIG_W by
# BIG_H, and the frames he cannot be hurt after shrinking from a hit.
COIN_POINTS, MUSHROOM_POINTS = 10, 50
MUSHROOM_SPEED = 1.0
SMALL_W, SMALL_H, BIG_W, BIG_H = 10, 12, 12, 24
HURT_FRAMES = 120
# Koopa Troopas: a 12 by 12 body 3 pixels above its feet (1-1's, game
# memory); a stomp leaves a still shell, touched it is kicked away at
# SHELL_SPEED pixels a frame (Mario unhurt by it for KICK_GRACE frames), a
# moving shell kills the enemies it meets and hurts Mario unless stomped
# (which stops it).
KOOPA_W, KOOPA_H, KOOPA_FEET = 12, 12, 3
SHELL_SPEED = 3.0
KICK_GRACE = 8
# Koopa Paratroopas (5-1's green ones, game memory): a Koopa with wings that
# hops, HOP_SPEED pixels a frame up each time it lands, under HOP_GRAVITY
# (44 pixels high, a hop every 55 frames); a stomp takes its wings: an
# ordinary Koopa.
HOP_SPEED, HOP_GRAVITY = -3.25, 0.12
# Bullet Bills (5-1's, game memory): fired from a cannon (a "cannon"
# platform, a column the game draws as ground) toward Mario, a BULLET_W by
# BULLET_H box BULLET_BELOW under the cannon's top, flying BULLET_SPEED
# pixels a frame through everything; a stomp drops it. A cannon on screen
# fires after each of CANNON_FRAMES in turn (the game's gaps, 31 to 392
# frames, looked random; a fixed turn keeps replays from saved states alike).
BULLET_W, BULLET_H, BULLET_BELOW, BULLET_SPEED = 10, 6, 6, 1.5
CANNON_FRAMES = (46, 200, 98, 264, 170, 31, 240, 120, 214, 88, 180, 152)
# Hammer Bros (3-1's, game memory): an 8 by 24 box (drawn 16 by 24, 4 in
# from each side) whose bottom is 4 below its feet; it walks HAMMER_BRO_WALK
# pixels either way of where it started, HAMMER_BRO_SPEED a frame; after
# each of HAMMER_BRO_JUMPS frames in turn it jumps, up to the floor
# HAMMER_BRO_REACH above it (starting HAMMER_BRO_UP pixels a frame upward,
# through the row over it) or down through what it stands on to what is
# below (starting HAMMER_BRO_DOWN upward), under HAMMER_BRO_GRAVITY, falling
# at most ENEMY_FALL; after each of HAMMER_FRAMES in turn it throws a hammer
# (the game's gaps: 49 or 50 frames, now and then a throw left out). It
# throws only while on screen; a stomp kills it.
HAMMER_BRO_W, HAMMER_BRO_H, HAMMER_BRO_FEET = 8, 24, -4
HAMMER_BRO_WALK, HAMMER_BRO_SPEED = 8, 0.25
HAMMER_BRO_REACH = 64
HAMMER_BRO_UP, HAMMER_BRO_DOWN, HAMMER_BRO_GRAVITY = -6.25, -3.25, 0.25
HAMMER_BRO_JUMPS = (215, 241)
HAMMER_FRAMES = (50, 49, 49, 49, 50, 49, 49, 97)
# A hammer (game memory, 116 throws): held HAMMER_HOLD frames over the
# Hammer Bro's head (its middle HAMMER_OVER above the Bro's top), then
# thrown toward Mario, HAMMER_SPEED pixels a frame sideways and HAMMER_THROW
# upward, falling under HAMMER_GRAVITY to at most HAMMER_FALL, through
# everything. Drawn spinning in two 8-pixel pieces, it hurts Mario by the
# HAMMER_W by HAMMER_H box at its middle (not measured), held or thrown.
HAMMER_W, HAMMER_H = 8, 8
HAMMER_HOLD, HAMMER_OVER = 14, 5
HAMMER_SPEED, HAMMER_THROW, HAMMER_GRAVITY, HAMMER_FALL = 1.0, -2.0, 0.0625, 4.5
# Pre-episode frames ("watch_frames") end with Mario standing: up to this many
# more are watched while one who starts above the floor drops onto it.
LANDING_WATCH = 64
# Lakitu and his Spinies (8-2's, game memory, 2026-11-12). Lakitu flies with
# his top LAKITU_Y from the screen's top; his speed (sixteenths of a pixel a
# frame) is a base less a quarter of how far he is from Mario (counted to
# LAKITU_REACH): the base LAKITU_BASE[0] while the screen does not scroll,
# [1] while Mario scrolls it, [2] while he runs at LAKITU_FAST or more and it
# scrolls 2 pixels or more. He keeps on past Mario until LAKITU_REACH beyond
# him, then turns back (going right he first slows by one each frame to a
# stop; going left he turns at once). His collision box is the game's for a
# Koopa (LAKITU_BOX: inside his drawn 16 by 24, not read from memory). He
# throws an egg when he comes on the screen and every LAKITU_THROW frames
# after while fewer than LAKITU_COMPANY other enemies are on it; at the end
# of his stretch he flies off left at LAKITU_LEAVE. An egg starts 8 pixels
# above him, EGG_UP a frame upward, under EGG_GRAVITY to EGG_FALL, never
# sideways; landed, it is a Spiny walking SPINY_SPEED toward Mario (four
# landings in five), which cannot be stomped (neither can an egg) and has a
# Goomba's collision box.
LAKITU_Y = 40
LAKITU_BASE = (20, 47, 63)
LAKITU_REACH = 60
LAKITU_FAST = 25
LAKITU_BOX = (2, 9, 12, 12)  # x, y from his place; width, height
LAKITU_THROW = 128
LAKITU_COMPANY = 4
LAKITU_LEAVE = 1.0
EGG_ABOVE, EGG_UP, EGG_GRAVITY, EGG_FALL = 8, -3.06, 0.125, 3.25
SPINY_SPEED = 0.5
# Flying Koopa Paratroopas (game memory): a red one (1-3's) swings straight
# down RED_FLIGHT[0] pixels from where it is placed and back each
# RED_FLIGHT[1] frames, a green one (7-3's) GREEN_FLIGHT[0] to the left and
# back each GREEN_FLIGHT[1] frames, bobbing GREEN_BOB[0] up and down each
# GREEN_BOB[1] frames; each a sine fitted to what was read (on average 2.7,
# 1.4 and 0.5 pixels off: the game's slows less near its ends, and the red
# one dips 5 above where it starts there). Stomped, either loses its wings
# and walks as a Koopa of its colour.
RED_FLIGHT = (104.0, 402.6)
GREEN_FLIGHT = (95.0, 319.8)
GREEN_BOB = (8.0, 128.0)
# Frenzies (game memory): over a stretch of level (scenario "frenzies": its
# kind and the x Mario is at from and until), things come on their own. A
# Bullet Bill frenzy (5-3's) sends a Bill from BILL_FROM pixels right of the
# screen's left edge every BILL_GAP frames, flying left at BULLET_SPEED, its
# box BILL_HEIGHTS from the top in turn. A leaping Cheep Cheep frenzy (2-3's)
# sends one up from below the screen within CHEEP_REACH of Mario after each
# of CHEEP_GAPS (frames, the game's, in a pseudo-random turn), CHEEP_UP a
# frame upward under CHEEP_GRAVITY, across at one of CHEEP_SPEEDS, toward
# Mario seven times in eight; it can be stomped, and its box is 10 by 6 (the
# game's, 2 in from where it is placed and 14 down).
BILL_FROM, BILL_GAP, BILL_HEIGHTS = 290, 240, (70, 118, 166)
CHEEP_REACH, CHEEP_FROM = 128, 262
CHEEP_GAPS = (16, 16, 16, 32, 64, 72, 72, 96, 96, 112, 120, 160, 168)
CHEEP_UP, CHEEP_GRAVITY = -4.97, 0.0524
CHEEP_SPEEDS = (5 / 16, 6 / 16, 12 / 16, 14 / 16)
# Lifts (game memory, Mario riding them: 1-3's and 3-3's): LIFT_W wide (the
# game's large lifts, six 8-pixel pieces), LIFT_H thick; one going "across"
# swings LIFT_ACROSS[0] pixels to the left of where it is placed and back
# each LIFT_ACROSS[1] frames, one going "updown" LIFT_UPDOWN[0] down and
# back each LIFT_UPDOWN[1] (each a sine, as the flying Paratroopas'), and
# one that "falls" drops LIFT_FALL pixels a frame while Mario stands on it.
# A lift carries Mario standing on it, up and down as well as across.
LIFT_W, LIFT_H = 48, 8
LIFT_ACROSS = (53.0, 240.0)
LIFT_UPDOWN = (137.0, 340.0)
LIFT_FALL = 2.25
# Springboards: SPRING_W wide (game memory: 2-1's), a solid down to the
# floor (the game's stands on a half block that stops Mario walking into it
# and that it is pressed down to); how one throws Mario is his motion's
# (physics.SPRING_SQUASH).
SPRING_W = 16
# Fire bars (game memory, 1-4's and 5-4's; their fireballs are only drawn
# sprites): FIREBAR_BALLS fireballs (FIREBAR_LONG on a long one),
# FIREBAR_GAP apart out from the middle of the block it turns on, each
# FIREBAR_BALL square (as drawn: the game's own test of a touch is not
# measured); it turns FIREBAR_TURN of a FIREBAR_STEPS-step circle a frame
# (once round in 205 frames; a "fast" one FIREBAR_FAST, once round in 146:
# the game's turns, 40/256 and 56/256 of a step, read from its program),
# the fireballs at the whole step it has reached (step 0 straight up, a
# quarter round to the right; "clockwise" counts up).
FIREBAR_BALLS, FIREBAR_LONG, FIREBAR_GAP, FIREBAR_BALL = 6, 12, 8, 8
FIREBAR_TURN, FIREBAR_FAST, FIREBAR_STEPS = 40 / 256, 56 / 256, 32
# Lava fireballs (game memory, 2-4's): one jumps up out of its pit from
# LAVA_FROM (the game's y: under the screen) at a speed from LAVA_UP (by
# the fraction the game draws at random; the game's two leaps measured
# peaked at 73 and 90), under LAVA_GRAVITY, falling at
# most LAVA_FALL, and jumps again LAVA_WAIT frames times one of LAVA_WAITS
# later; its box is the game's (LAVA_BOX: in from its place, and size).
LAVA_FROM, LAVA_UP, LAVA_GRAVITY, LAVA_FALL = 258.0, (-5.95, -6.35), 28 / 256, 3.25
LAVA_WAIT, LAVA_WAITS = 21, tuple(range(6, 16))
LAVA_BOX = (3, 14, 10, 6)
# Bowser (game memory, 1-4's and 6-4's, and his routine in the game's
# program). He stands on his bridge, his place (the game's x and y of his
# front half) BOWSER_FEET over his feet; his body is two halves of the
# game's, side by side, here one box (BOWSER_BOX: in from his place, and
# size, facing left; facing right it starts BOWSER_BACK to the left of his
# place, his rear half behind him). Once in BOWSER_EVERY frames he moves
# his step: BOWSER_CHASE at first, then -1 or 1 (a quarter of a pixel a
# frame), turning back once he is a range (one of BOWSER_RANGES, drawn each
# time he passes where he started) either way of where he started. Once in
# BOWSER_TURN frames he faces left; while Mario is to the right of his place
# and he stands, he faces right, his step is BOWSER_CHASE, he neither jumps
# nor opens his mouth, and he stops BOWSER_CHASE_LIMIT right of where he
# started. He jumps one of BOWSER_RANGES frames after he lands (first after
# BOWSER_FIRST_JUMP), BOWSER_JUMP a frame up, under BOWSER_GRAVITY, falling
# at most BOWSER_FALL. His mouth opens BOWSER_FIRST_FLAME frames after he
# comes on screen, stays open BOWSER_MOUTH frames (he does not walk), and
# as it shuts he breathes a flame; it opens again after the next of
# FLAME_WAITS, in turn (FLAME_HARD fewer in the hard castles, worlds 5 to
# 8). He cannot be stomped. A flame (and a free flame, coming before he is
# on screen from FLAME_FROM right of the screen's left edge, the next of
# FLAME_WAITS plus BOWSER_MOUTH frames after the one before) flies left at
# FLAME_SPEED (hard: the second), from his place plus FLAME_MOUTH, a pixel a
# frame up or down until its place is level with one of FLAME_LEVELS (the
# game's 144, 128 and 112 over castle floors 160 down; Block SMB's floor is
# 208 down), then level; its box is the game's (FLAME_BOX), and it cannot
# be stomped. In worlds 6 to 8 ("hammers") he throws hammers: in the air,
# once in BOWSER_EVERY frames, one of the game's HAMMER_PLACES loose-item
# places is drawn (the first and last half as often as each other one), and
# if no hammer of his is in it and its place is on screen a hammer appears
# there, its place (its hurt box's corner: the game's box for a hammer is
# HAMMER_W by HAMMER_H from it) BOWSER_HAMMER_AT from his; it is held
# there HAMMER_HOLD frames, then thrown the way he faces (6-4's memory: 6
# to 9 a jump). Not here: the axe.
BOWSER_FEET, BOWSER_BOX, BOWSER_BACK = 32, (0, 2, 32, 27), 16
BOWSER_EVERY, BOWSER_TURN, BOWSER_CHASE, BOWSER_CHASE_LIMIT = 4, 16, 2, 88
BOWSER_RANGES = (33, 65, 17, 49)
BOWSER_FIRST_JUMP, BOWSER_JUMP, BOWSER_GRAVITY, BOWSER_FALL = 32, -2.0, 15 / 256, 2.0
BOWSER_FIRST_FLAME, BOWSER_MOUTH = 223, 32
FLAME_WAITS = (191, 64, 191, 191, 191, 64, 64, 191)
FLAME_HARD = 16
FLAME_FROM, FLAME_SPEED, FLAME_MOUTH = 287, (1.25, 1.375), (-14, 8)
FLAME_LEVELS = (192, 176, 160, 192)
FLAME_BOX = (6, 4, 4, 4)
HAMMER_PLACES, BOWSER_HAMMER_AT = 9, (2, -10)

# What each strategy is paid for, as multipliers of the reward terms (training
# only; a layout names its strategy in "strategy", see strategy_families).
# Speed run is paid for time alone: five times the frame cost, nothing for
# coins or kills, and a time bonus at the goal (the goal reward again, times
# the share of its deadline left unused). Max points is paid for points: coins
# and kills, each worth 25.
STRATEGY_REWARDS = {
    "speed_run": {"frame_penalty": 5.0, "coin": 0.0, "enemy_stomp": 0.0},
    "max_points": {"coin": 2.5, "enemy_stomp": 5.0},
}


def question_cell(world_x):
    """Whether the 16-pixel brick cell starting at world x is a question block."""
    return (int(world_x) // TILE) % 3 == 1


class _Canvas:
    """Draws each shape in its colour, its pixel type id, or its object's number.

    In "instances" mode the number of the object being drawn (``instance``,
    0 for none) is written into the red and green channels (low and high byte).
    In "labels" mode both at once: the type id in red, the object's number in
    green and blue (low and high byte), so one drawing gives both.
    """

    def __init__(self, surface, mode="colour"):
        if mode not in ("colour", "types", "instances", "labels"):
            raise ValueError(f"unknown drawing mode {mode!r}")
        self.surface = surface
        self.mode = mode
        self.instance = 0

    def _color(self, color, kind):
        if self.mode == "types":
            return (TYPE_ID[kind],) * 3
        if self.mode == "instances":
            return (self.instance & 0xFF, self.instance >> 8, 0)
        if self.mode == "labels":
            return (TYPE_ID[kind], self.instance & 0xFF, self.instance >> 8)
        return color

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
    power_up: float = 10.0
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
        for name in (
            "coin",
            "power_up",
            "enemy_stomp",
            "goal",
            "goal_distance_shaping",
            "wait_survival",
        ):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be non-negative")
        for name in ("fall_death", "gap_jump", "enemy_hit", "frame_penalty", "energy_jump"):
            if getattr(self, name) > 0:
                raise ValueError(f"{name} must be non-positive")

    def zero_terms(self) -> dict[str, float]:
        return {
            "progress": 0.0,
            "coin": 0.0,
            "power_up": 0.0,
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
                    a dict with "game_rules": True follows the real game's
                    rules instead (no patrol: it wakes when on screen, walks
                    at 0.5 px a frame, turns at walls, falls off ledges);
                    a piranha_plant uses kind,x,pipe_top and optional cycle durations
                    (it stays down while Mario is near it: piranha.WAITS_NEAR);
                    kind "monster" is a large walker that cannot be stomped
    goal          : [x, y, w, h]
    optional_bridge: a moving platform Mario may ride or not (its crossing is
                    tracked, but the goal does not require it)
    tactics       : the layout's tactic segments, in order (tactic_schedule;
                    training only: teacher tokens, route rules, success)
    watch_frames  : pre-episode frames the world runs while Mario stands; their
                    pictures (watched_screens) are for the agent to watch first
    strategy      : the strategy the layout is played for (STRATEGY_REWARDS),
                    with "strategy_objective": {"deadline": frames} (the goal
                    counts only until then) or {"points": count} (only with at
                    least that many points: coins collected plus enemies
                    killed); missing it ends the episode as a loss
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
        self.power_ups = []
        self.items = []  # mushrooms out of question blocks
        self.cannons = []  # Bullet Bill cannons: x, w, top, timer, shots
        self.frenzies = []  # stretches where things come on their own: kind, from, until, timer, count, turn
        self.stomped = False
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
        self._optional_bridge = False
        self._bridge_jump_launched = False
        self._goal_requires_support = False
        self._single_jump_attempt = False
        self._attempt_failed = False
        self._goal_credited = False
        self._strategy = None  # the strategy a layout is played for (STRATEGY_REWARDS)
        self._strategy_objective = {}
        self._objective_missed = False
        self.watched_screens = []  # pre-episode pictures (reset, "watch_frames")
        self.watched_labels = []  # ... and their true labels, for teachers only
        self.watched_truth = []  # ... and the simulator's state then (true_frame)
        self.camera_x = 0.0
        self.scroll_x = width // 3
        # Whether step and reset draw the picture (sampling outcomes from the
        # simulator's state needs none: drawing was 88% of a step's time).
        self.draws = True
        self.score = 0
        self.steps = 0
        self.max_steps = 1000
        self.frame_budget = 320  # frames a teacher's route may take (reset reads the layout's)
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
        # Where on screen the camera keeps Mario's left side as it scrolls
        # (the real game's is about 115: domains/smb/sim/levels.py).
        self.scroll_x = int(scenario.get("scroll_x", self.width // 3))
        # Long composed layouts state how many frames their route may take.
        self.frame_budget = int(scenario.get("frame_budget", 320))
        self.max_steps = max(1000, self.frame_budget)
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
            "big": False,  # grown by a mushroom
            "safe": 0,  # frames left in which enemies cannot hurt him
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
        for plat in self.platforms:
            if plat.get("lift"):  # (at its point of the swing from the start: no move counted)
                plat["tick"] -= 1
                self._move_lift(plat, False)
                plat.update(vx=0.0, vy=0.0, delta_x=0, delta_y=0)
        self.cannons = [
            {"x": p["rect"].x, "w": p["rect"].w, "top": p["rect"].y, "timer": CANNON_FRAMES[0], "shots": 0}
            for p, kind in zip(self.platforms, self.platform_kinds) if kind == "cannon"
        ]
        self.frenzies = [{"kind": f["kind"], "from": float(f["from"]), "until": float(f["until"]), "timer": 1,
                          "count": int(f.get("count", 0)), "turn": int(f.get("turn", 1)), "hard": bool(f.get("hard"))}
                         for f in scenario.get("frenzies", [])]
        if scenario.get("mario_big"):
            self._resize(big=True)  # (standing where small Mario would)
        # What each question block holds ({platform index: "coin" or
        # "mushroom"}); the mushrooms out and moving (items).
        for index, content in (scenario.get("block_contents") or {}).items():
            self.platforms[int(index)]["contents"] = content
        self.items = []

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
                self.mario["_platform"] = platform
                break

        # Coins
        self.coins = [
            {"rect": pygame.Rect(c[0], c[1], c[2], c[3]), "collected": False}
            for c in scenario.get("coins", [])
        ]

        # Power-ups: [x, y] (a 16x16 mushroom) or [x, y, w, h]; reward only.
        self.power_ups = []
        self.stomped = False
        for p in scenario.get("power_ups", []):
            w, h = (p[2], p[3]) if len(p) >= 4 else (POWER_UP_SIZE, POWER_UP_SIZE)
            self.power_ups.append({"rect": pygame.Rect(p[0], p[1], w, h), "collected": False})

        # Enemies — accept list or dict; optional 5th element = speed
        self.enemies = []
        for e in scenario.get("enemies", []):
            if isinstance(e, dict) and e.get("kind") == "firebar":
                self.enemies.extend(self._firebar(e))  # (each fireball its own enemy)
                continue
            if isinstance(e, dict) and e.get("kind") == "lava":
                self.enemies.append(self._lava(e))
                continue
            if isinstance(e, dict) and e.get("kind") == "bowser":
                self.enemies.append(self._bowser(e))
                continue
            enemy = self._parse_enemy(e)
            if enemy.get("kind") == "monster":
                enemy.update(w=MONSTER_W, h=MONSTER_H, foot_offset=0, stompable=False)
            elif enemy.get("kind") == "koopa":
                # (its feet where a Goomba's would be: 14 below the y given)
                enemy.update(w=KOOPA_W, h=KOOPA_H, foot_offset=KOOPA_FEET, y=enemy["y"] + 14 - KOOPA_FEET - KOOPA_H)
                if enemy.get("flight"):
                    enemy["home"] = (enemy["x"], enemy["y"])  # (its swing's start: where it is placed)
            elif enemy.get("kind") == "hammer_bro":
                # (standing, its feet where a Goomba's would be; its first
                # throw and jump after the frames given, or the game's first)
                enemy.update(w=HAMMER_BRO_W, h=HAMMER_BRO_H, foot_offset=HAMMER_BRO_FEET, speed=HAMMER_BRO_SPEED,
                             y=enemy["y"] + 14 - HAMMER_BRO_FEET - HAMMER_BRO_H, on_ground=True,
                             patrol_min=enemy["x"] - HAMMER_BRO_WALK, patrol_max=enemy["x"] + HAMMER_BRO_WALK,
                             throw_timer=int(e.get("throw_wait", 11)), throws=0,
                             jump_timer=int(e.get("jump_wait", HAMMER_BRO_JUMPS[0])), jumps=0, through=None)
            elif enemy.get("kind") == "lakitu":
                # (x where the game would place him; he flies at LAKITU_Y;
                # he leaves once Mario is "until" into the level)
                bx, by, bw, bh = LAKITU_BOX
                enemy.update(x=enemy["x"] + bx, y=float(LAKITU_Y + by), w=bw, h=bh, foot_offset=0, speed=0.0,
                             direction=-1, lakitu_speed=0, until=float(e.get("until", 1e9)), throw_timer=1)
            elif enemy.get("kind") != "piranha_plant":
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
        self._action_jump_direction = int(scenario.get("action_jump_direction", 0))
        self._prefer_enemy_bypass = bool(scenario.get("prefer_enemy_bypass", False))
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
        # A moving platform Mario may ride or not: its crossing is tracked for
        # the layout's tactics, but the goal does not require it.
        self._optional_bridge = bool(scenario.get("optional_bridge", False))
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
        tactic_schedule.start(self, scenario.get("tactics"))
        self._strategy = scenario.get("strategy")
        if self._strategy is not None and self._strategy not in STRATEGY_REWARDS:
            raise ValueError(f"unknown strategy {self._strategy!r}")
        self._strategy_objective = dict(scenario.get("strategy_objective") or {})
        self._objective_missed = False

        # The initial observation must use the same camera placement as step().
        # Otherwise the first stationary frame appears to move Mario sideways.
        self.camera_x = max(
            0.0, min(self.mario["x"] - self.scroll_x, self.world_width - self.width)
        )
        self.scrolled = 0.0
        obs = self.render() if self.draws else None
        # Pre-episode frames (scenario "watch_frames"): the world runs while
        # Mario stands, and their pictures are kept for the agent to watch
        # (prefilled history), so its first decision has seen how things move.
        # The episode, its frame count and its routes start after them.
        # Mario stands while they are watched: one who starts above the floor
        # is watched dropping onto it too (up to LANDING_WATCH more frames),
        # so the first decision is made standing, not in the air.
        self.watched_screens, self.watched_labels, self.watched_truth = [], [], []
        watch = int(scenario.get("watch_frames", 0) or 0)
        while len(self.watched_screens) < watch or (
            watch and not self.mario["on_ground"] and len(self.watched_screens) < watch + LANDING_WATCH
        ):
            self.watched_screens.append(obs)
            self.watched_labels.append(self.scene_labels())  # training-only truth
            self.watched_truth.append(self.true_frame())
            obs, _, done, truncated, _ = self.step(0)
            if done or truncated:
                raise ValueError("the pre-episode frames ended the episode")
        if watch and not self.mario["on_ground"]:
            raise ValueError("Mario has not landed by the end of the pre-episode frames")
        if self.watched_screens:
            self.steps = 0
            tactic_schedule._begin(self)
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
        self.stomped = False  # set when this step resolves a stomp (scene_labels)
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

        on_spring = bool(self.mario["on_ground"] and support_before is not None
                         and support_before.get("lift") == "spring")
        dx, dy, jumped = self.motion.advance(
            direction=move_x,
            jump=jump_pressed and not on_spring,
            grounded=self.mario["on_ground"] and not on_spring,  # (held there: his speed as in the air)
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
        if on_spring:
            dx = 0  # (held where he is while it presses him down)
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
            if plat.get("lift"):
                self._move_lift(plat, plat is support_before)
                continue
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
            if plat is support_before and grounded_before:
                continue  # (what he stands on, risen into his feet: carried up below, not a wall)
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
        if self.mario["on_ground"] and support_before is not None and support_before.get("delta_y"):
            self.mario["y"] += support_before["delta_y"]  # carried up or down by the lift he stands on
            mario_rect.y = self.mario["y"]
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
            # A head reaching a block's bottom edge has not hit it (the block
            # fills the rows above it, as in the game and the planner).
            hit_ceiling = (
                self.mario["vy"] < 0
                and horizontal_overlap
                and previous_y >= r.bottom > self.mario["y"]
            )
            if mario_rect.colliderect(r) or landed_on_top or hit_ceiling:
                if self.mario["vy"] >= 0:  # falling / level
                    mario_rect.bottom = r.top
                    self.mario["on_ground"] = True
                    standing = self.mario["_platform"]
                    if standing is None or standing.get("lift") != "spring":  # (pressed down, level with its block)
                        self.mario["_platform"] = plat
                elif self.mario["vy"] < 0:  # hitting ceiling
                    mario_rect.top = r.bottom
                    self._bump(plat, reward_terms)
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

        spring = self.mario["_platform"] if self.mario["on_ground"] else None
        if spring is not None and spring.get("lift") == "spring" and spring["squash"] == SPRING_THROW:
            self._spring_throw(spring)

        if self.mario["on_ground"]:
            self._airborne_started_with_jump = False

        # ── 10. Camera (right-only scroll, clamped) ───────────────────────────
        target_cam = self.mario["x"] - self.scroll_x
        before = self.camera_x
        if target_cam > self.camera_x:
            self.camera_x = target_cam
        self.camera_x = max(0.0, min(self.camera_x, self.world_width - self.width))
        self.scrolled = self.camera_x - before

        # ── 11. World boundaries ──────────────────────────────────────────────
        # The screen's left edge (the camera never scrolls back) and the
        # world's right end stop Mario as a wall does: his speed is lost. In
        # the game, pinned at the left edge his speed is 0 (Full SMB, 1-1);
        # kept here, a run back from the edge first spent 13 frames undoing
        # it, and the executor gave up on it (max_points_descend_forward,
        # 2026-10-16).
        if self.mario["x"] <= self.camera_x:
            self.mario["x"] = self.camera_x
            if self.motion.x_speed < 0:
                self.mario["vx"] = 0
                self.motion.wall_contact()
        if self.mario["x"] >= self.world_width - self.mario["w"]:
            self.mario["x"] = self.world_width - self.mario["w"]
            if self.motion.x_speed > 0:
                self.mario["vx"] = 0
                self.motion.wall_contact()
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
        self._shells_hit(reward_terms)
        self._fire_cannons()
        self._frenzies()
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
            if enemy.get("kind") == "koopa" and self._touch_koopa(enemy, geometry, mario_rect, reward_terms):
                continue  # stomped into a shell, a shell stopped or kicked: no harm
            if geometry["stomp"] and enemy.get("stompable", True):
                # Stomp!
                self.stomped = True
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
            elif self.mario["safe"] > 0:
                pass  # just hurt: enemies pass through him
            elif self.mario["big"]:
                self._resize(big=False)  # hurt: he shrinks
                self.mario["safe"] = HURT_FRAMES
                mario_rect = pygame.Rect(self.mario["x"], self.mario["y"], self.mario["w"], self.mario["h"])
            else:
                terminated = True
                death = True
                reward_terms["enemy_hit"] += self.reward_config.enemy_hit
        self.mario["safe"] = max(0, self.mario["safe"] - 1)

        # ── 15b. Mushrooms out of question blocks ────────────────────────────
        for item in list(self.items):
            self._update_item(item, rects)
            if item["y"] > self.height:
                self.items.remove(item)
            elif mario_rect.colliderect(pygame.Rect(item["x"], item["y"], item["w"], item["h"])):
                self.items.remove(item)
                self.score += MUSHROOM_POINTS
                reward_terms["power_up"] += self.reward_config.power_up
                if not self.mario["big"]:
                    self._resize(big=True)
                    mario_rect = pygame.Rect(self.mario["x"], self.mario["y"], self.mario["w"], self.mario["h"])

        # ── 16. Coin collection ───────────────────────────────────────────────
        for coin in self.coins:
            if not coin["collected"] and mario_rect.colliderect(coin["rect"]):
                coin["collected"] = True
                reward_terms["coin"] += self.reward_config.coin
                self.score += 10
        for power_up in self.power_ups:
            if not power_up["collected"] and mario_rect.colliderect(power_up["rect"]):
                power_up["collected"] = True
                reward_terms["power_up"] += self.reward_config.power_up

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

        if (self._require_bridge_before_goal or self._optional_bridge) and self.mario["on_ground"]:
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

        # ── 16c. The layout's tactics (training only) ─────────────────────────
        # Standing on a platform the current segment forbids, or leaving a
        # hold area early, ends the episode as a loss: Mario left the route.
        if tactic_schedule.track(self) and not terminated:
            terminated = True

        # ── 16d. The strategy's objective (training only) ────────────────────
        deadline = self._strategy_objective.get("deadline")
        if deadline is not None and self.steps > deadline and not terminated:
            terminated = True
            self._objective_missed = True

        # ── 17. Goal ──────────────────────────────────────────────────────────
        # Under goal_on_stomp the rect is only a tracking proxy for shaping
        # and observations; brushing it mid-air must not count as success.
        if (
            self.goal
            and not self._bridge_jump_task
            and not self._goal_on_stomp
            and not death
            and not self._off_route
            and tactic_schedule.goal_allowed(self)
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
            if self.points() < self._strategy_objective.get("points", 0):
                self._objective_missed = True  # reached the goal without the points
            else:
                reward_terms["goal"] += self.reward_config.goal
                self._goal_credited = True
                if deadline is not None and self._strategy == "speed_run":
                    reward_terms["goal"] += self.reward_config.goal * max(
                        0.0, 1.0 - self.steps / deadline
                    )

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
        for term, scale in STRATEGY_REWARDS.get(self._strategy, {}).items():
            reward_terms[term] *= scale
        reward, reward_terms = self._finalize_reward_terms(reward_terms)

        obs = self.render() if self.draws else None
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
        self._draw(_Canvas(self.screen))
        return np.transpose(pygame.surfarray.array3d(self.screen), (1, 0, 2))

    def render_labels(self) -> np.ndarray:
        """Each pixel's type (smb_pixel_types.TYPE_ID) as an (H, W) uint8 array.

        The same shapes as render(), in the same order, so every pixel carries
        the type of the shape drawn on top of it. Eyes belong to their owner.
        """
        return self._render_types_and_instances()[0]

    def render_instances(self) -> np.ndarray:
        """Which object drew each pixel, as an (H, W) int32 array (-1 for none).

        The same shapes as render(), in the same order, so a pixel belongs to
        the object drawn on top of it. Object numbers are those of
        instance_meta(); terrain other than moving platforms belongs to none.
        """
        return self._render_types_and_instances()[1]

    def _render_types_and_instances(self) -> tuple:
        """(render_labels(), render_instances()) from one drawing ("labels":
        the type in red, the object's number in green and blue)."""
        self._draw(_Canvas(self._label_screen, "labels"))
        pixels = pygame.surfarray.pixels3d(self._label_screen)
        types = np.ascontiguousarray(pixels[:, :, 0].T)
        instances = pixels[:, :, 1].T.astype(np.int32) + (pixels[:, :, 2].T.astype(np.int32) << 8) - 1
        del pixels  # the surface stays locked while its pixels are viewed
        return types, instances

    def instance_meta(self) -> tuple[dict[int, str], dict[int, str]]:
        """(category, enemy kind) of every object number render_instances() can show."""
        categories, kinds = {}, {}
        for number, (category, kind, _draw) in enumerate(self._objects()):
            if category is not None:
                categories[number] = category
                if kind is not None:
                    kinds[number] = kind
        return categories, kinds

    def scene_labels(self):
        """The frame's exact SceneLabels: types, objects, Mario's standing and facing.

        Like the picture a policy sees (smb_pixel_types.visible_window), the
        labels show only the window the NES shows and repeat its edge outside.
        """
        from retroagi.domains.smb.perception.pixel_types import visible_window
        from retroagi.domains.smb.perception.scene_labels import SceneLabels

        categories, kinds = self.instance_meta()
        types, instances = self._render_types_and_instances()
        return SceneLabels(
            types=visible_window(types),
            instances=visible_window(instances),
            categories=categories,
            kinds=kinds,
            standing=bool(self.mario["on_ground"]),
            facing_right=self.mario["facing"] > 0,
            stomping=self.stomped,
        )

    def true_frame(self) -> dict:
        """Training only: the simulator's state at this frame, as recordings
        keep it (legacy/block_smb/truth_recording.py), in world pixels:
        Mario's collision box, every enemy's collision box and the box it is
        drawn in (its feet included), whether each enemy is absent (a plant
        down in its pipe), alive or dead and whether it can be stomped, every
        moving platform's rectangle, and the camera's left edge."""
        m = self.mario
        enemies, drawn, state = [], [], []
        for e in self.enemies:
            foot = e.get("foot_offset", 0)
            enemies.append((e["x"], e["y"], e["x"] + e["w"], e["y"] + e["h"]))
            if e.get("kind") == "piranha_plant":
                x, y, w, h = plant_sprite(e)
                drawn.append((x, y, x + w, y + h))
                state.append(0 if h <= 0 else 1)
                continue
            drawn.append((e["x"], e["y"], e["x"] + e["w"], e["y"] + e["h"] + foot))
            state.append(2 if e["dead"] else 0 if e["h"] <= 0 else 1)
        return {
            "mario": (m["x"], m["y"], m["x"] + m["w"], m["y"] + m["h"]),
            "enemies": enemies,
            "enemies_drawn": drawn,
            "enemy_state": state,
            "stompable": [bool(e.get("stompable", True)) for e in self.enemies],
            "platforms": [
                (p["rect"].left, p["rect"].top, p["rect"].right, p["rect"].bottom)
                for p in self.platforms
                if p.get("moving")
            ],
            "camera": float(self.camera_x),
        }

    def enemy_screen_rects(self) -> list[pygame.Rect]:
        """The screen rectangle each drawn enemy is painted inside, in draw order."""
        cam = int(self.camera_x)
        rects = []
        for enemy in self.enemies:
            if enemy.get("kind") == "piranha_plant":
                x, y, w, h = plant_sprite(enemy)
                if h > 0:
                    rects.append(pygame.Rect(int(x) - cam, int(y), w, h))
                continue
            if enemy["h"] <= 0:
                continue
            foot = enemy.get("foot_offset", 0)
            rects.append(
                pygame.Rect(int(enemy["x"]) - cam, int(enemy["y"]), enemy["w"], enemy["h"] + foot)
            )
        return rects

    def _draw(self, canvas: _Canvas) -> None:
        canvas.fill(SKY, "background")
        for number, (category, _kind, draw) in enumerate(self._objects()):
            # Object numbers start at 1 on the canvas; 0 means no object.
            canvas.instance = number + 1 if category is not None else 0
            draw(canvas)

    def _objects(self):
        """Every drawn shape group in drawing order: (category, enemy kind, draw).

        category is None for terrain that is not a separate object (ground,
        bricks, question blocks, pipes); moving platforms, coins, power-ups,
        enemies and Mario are objects.
        """
        cam = int(self.camera_x)
        if len(self.platform_kinds) != len(self.platforms):
            raise ValueError("platform kinds are out of step with the platforms")
        for plat, kind in zip(self.platforms, self.platform_kinds):
            r = plat["rect"]
            screen_rect = pygame.Rect(r.x - cam, r.y, r.w, r.h)
            if screen_rect.right <= 0 or screen_rect.left >= self.width:
                continue
            if kind == "moving_platform":
                yield "moving_platform", None, partial(self._draw_lift, rect=screen_rect)
            elif kind == "spring":
                yield "moving_platform", None, partial(self._draw_spring, rect=screen_rect)
            elif kind == "pipe":
                yield None, None, partial(self._draw_pipe, rect=screen_rect)
            elif kind == "cannon":
                yield None, None, partial(self._draw_ground, rect=screen_rect)
            elif kind == "question_block":
                yield None, None, partial(self._draw_question_row, rect=screen_rect)
            elif kind in ("brick", "brick_row"):
                yield (
                    None,
                    None,
                    partial(self._draw_brick_row, rect=screen_rect, questions=kind == "brick_row"),
                )
            else:
                yield None, None, partial(self._draw_ground, rect=screen_rect)
        for coin in self.coins:
            if not coin["collected"]:
                yield "coin", None, partial(self._draw_coin, rect=coin["rect"].move(-cam, 0))
        for power_up in self.power_ups:
            if not power_up["collected"]:
                yield (
                    "power_up",
                    None,
                    partial(self._draw_power_up, rect=power_up["rect"].move(-cam, 0)),
                )
        drawn = (e for e in self.enemies if (plant_sprite(e)[3] if e.get("kind") == "piranha_plant" else e["h"]) > 0)
        for enemy, rect in zip(drawn, self.enemy_screen_rects()):
            if enemy.get("kind") == "piranha_plant":
                yield "enemy", "plant", partial(self._draw_plant, rect=rect)
            elif enemy.get("kind") == "monster":
                yield "enemy", "other", partial(self._draw_monster, enemy=enemy, rect=rect)
            elif enemy.get("kind") in ("bullet", "hammer", "hammer_bro", "lakitu", "spiny", "cheep", "firebar", "lava",
                                       "bowser", "flame"):
                if not enemy.get("gone") and not (enemy.get("kind") in ("hammer_bro", "lakitu") and enemy["dead"]):
                    yield "enemy", "other", partial(self._draw_monster, enemy=enemy, rect=rect)
            else:
                kind = "defeated" if enemy["dead"] else "walker"
                yield "enemy", kind, partial(self._draw_goomba, enemy=enemy, rect=rect)
        yield "mario", None, self._draw_mario

    def _draw_mario(self, canvas):
        # Yellow while skidding, red otherwise; the eye shows facing.
        cam = int(self.camera_x)
        m = self.mario
        body = pygame.Rect(int(m["x"]) - cam, int(m["y"]), m["w"], m["h"])
        canvas.rect(MARIO_SKIDDING if m["skidding"] else MARIO, "mario", body)
        eye_x = body.right - 4 if m["facing"] > 0 else body.left + 2
        canvas.rect(EYE, "mario", pygame.Rect(eye_x, body.top + 2, 2, 2))

    def _draw_question_row(self, canvas, rect):
        for cell in self._cells(rect):
            self._draw_question_block(canvas, cell)

    def _draw_brick_row(self, canvas, rect, questions):
        cam = int(self.camera_x)
        self._draw_bricks(canvas, rect)
        if questions:
            for cell in self._cells(rect):
                if question_cell(cell.x + cam):
                    self._draw_question_block(canvas, cell)

    def _draw_coin(self, canvas, rect):
        canvas.ellipse(COIN, "coin", rect)
        inner = rect.inflate(-6, -4)
        if inner.w > 0 and inner.h > 0:
            canvas.ellipse(COIN_DARK, "coin", inner)

    def _draw_power_up(self, canvas, rect):
        # A mushroom: a spotted cap over a pale stem.
        cap = pygame.Rect(rect.left, rect.top, rect.w, max(1, rect.h * 5 // 8))
        stem = pygame.Rect(0, 0, max(1, rect.w * 5 // 8), rect.bottom - cap.centery)
        stem.midbottom = (rect.centerx, rect.bottom)
        canvas.rect(POWER_UP_STEM, "power_up", stem)
        canvas.ellipse(POWER_UP_CAP, "power_up", cap)
        for x in (cap.left + cap.w // 4 - 1, cap.right - cap.w // 4 - 2):
            canvas.rect(POWER_UP_SPOT, "power_up", pygame.Rect(x, cap.top + 2, 3, 3))

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

    def _draw_spring(self, canvas, rect):
        canvas.rect(SPRING, "moving_platform", rect)
        canvas.rect(LIFT_EDGE, "moving_platform", rect, 1)

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

    def _draw_monster(self, canvas, enemy, rect):
        # A spiked shell over a pale belly, with an eye on the side it walks to.
        canvas.rect(MONSTER, "enemy", rect)
        belly = pygame.Rect(rect.left + 3, rect.bottom - 6, rect.w - 6, 4)
        canvas.rect(MONSTER_BELLY, "enemy", belly)
        for x in range(rect.left + 1, rect.right - 2, 5):
            canvas.rect(MONSTER_SPIKE, "enemy", pygame.Rect(x, rect.top, 3, 3))
        eye_x = rect.right - 5 if enemy["direction"] > 0 else rect.left + 3
        canvas.rect(EYE, "enemy", pygame.Rect(eye_x, rect.top + 6, 2, 3))

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

    def points(self) -> int:
        """Max points' score so far: coins collected plus enemies killed (stomped)."""
        coins = sum(coin["collected"] for coin in self.coins)
        return coins + sum(bool(enemy["dead"]) for enemy in self.enemies)

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
            "stomp_completed": self._stomp_credited,
            "bridge_boarded": self._bridge_boarded,
            "bridge_crossed": self._bridge_crossed,
            "attempt_failed": self._attempt_failed,
            "off_route": bool(getattr(self, "_off_route", False)),
            "objective_missed": bool(getattr(self, "_objective_missed", False)),
            "reward_terms": dict(reward_terms),
            "reward_total": reward_total,
            "reward_config": asdict(self.reward_config),
            "death": death,
            "terminated": bool(terminated),
            "truncated": bool(truncated),
        }

    # ── Enemy helpers ─────────────────────────────────────────────────────────

    def _touch_koopa(self, enemy: dict, geometry: dict, mario_rect, reward_terms: dict) -> bool:
        """Mario touching a Koopa: stomped, it becomes a still shell (a
        moving shell stops); a still shell is kicked away from him; a shell
        just kicked does not hurt. False when the touch hurts him."""
        shell = enemy.get("shell")
        if geometry["stomp"] and enemy.get("wings"):
            enemy.update(wings=False, vy=0.0)  # its wings lost: an ordinary Koopa
            self.score += 5
            reward_terms["enemy_stomp"] += self.reward_config.enemy_stomp
            self.stomped = True
        elif geometry["stomp"] and shell != "still":
            if shell is None:
                self.score += 5
                reward_terms["enemy_stomp"] += self.reward_config.enemy_stomp
                self.stomped = True
            enemy.update(shell="still", speed=0.0)
        elif shell == "still":
            middle = enemy["x"] + enemy["w"] / 2
            away = 1 if middle >= mario_rect.centerx else -1
            enemy.update(shell="moving", speed=SHELL_SPEED, direction=away, kick_grace=KICK_GRACE)
        elif not enemy.get("kick_grace"):
            return False
        if geometry["stomp"]:
            self.motion.bounce()
            self.mario["vy"] = self.motion.y_speed
            self.mario["on_ground"] = False
        return True

    def _fire_cannons(self) -> None:
        """Each cannon on screen fires a Bullet Bill toward Mario when its
        time comes (CANNON_FRAMES in turn)."""
        middle = self.mario["x"] + self.mario["w"] / 2
        for cannon in self.cannons:
            cannon["timer"] -= 1
            if cannon["timer"] > 0:
                continue
            cannon["shots"] += 1
            cannon["timer"] = CANNON_FRAMES[cannon["shots"] % len(CANNON_FRAMES)]
            if not self.camera_x <= cannon["x"] < self.camera_x + self.width:
                continue
            direction = -1 if middle < cannon["x"] + cannon["w"] / 2 else 1
            x = cannon["x"] + (cannon["w"] - BULLET_W) / 2
            self.enemies.append({
                "kind": "bullet", "x": x, "y": cannon["top"] + BULLET_BELOW, "vx": 0.0, "vy": 0.0,
                "w": BULLET_W, "h": BULLET_H, "speed": BULLET_SPEED, "direction": direction,
                "patrol_min": -1e9, "patrol_max": 1e9, "edge_aware": False, "on_ground": False,
                "dead": False, "stompable": True,
            })

    def _move_lift(self, lift: dict, stood_on: bool) -> None:
        """A lift's move this frame (LIFT_ACROSS, LIFT_UPDOWN, LIFT_FALL):
        its place (whole pixels, as Mario's collisions take it), its move
        (delta_x, delta_y) and its speed (vx, vy)."""
        lift["tick"] += 1
        home_x, home_y = lift["home"]
        x, y = lift["lift_x"], lift["lift_y"]
        if lift["lift"] == "across":
            reach, period = LIFT_ACROSS
            x = home_x - reach / 2 * (1 - math.cos(2 * math.pi * lift["tick"] / period))
        elif lift["lift"] == "updown":
            reach, period = LIFT_UPDOWN
            y = home_y + reach / 2 * (1 - math.cos(2 * math.pi * lift["tick"] / period))
        elif lift["lift"] == "spring":
            squash = lift["squash"]
            if squash is not None:
                squash = squash + 1 if squash + 1 < len(SPRING_SQUASH) else None
            elif stood_on:
                squash = 0
            if squash is None:
                lift["strong"] = False
            elif stood_on and SPRING_STRONG[0] <= squash <= SPRING_STRONG[1] and self.mario["jump_held"]:
                lift["strong"] = True
            lift["squash"] = squash
            y = home_y + (SPRING_SQUASH[squash] if squash is not None else 0)
        elif stood_on and y < self.height + 32:
            y += LIFT_FALL
        rect = lift["rect"]
        old = (rect.x, rect.y)
        rect.x, rect.y = int(round(x)), int(round(y))
        lift.update(lift_x=x, lift_y=y, vx=x - lift["lift_x"], vy=y - lift["lift_y"],
                    delta_x=rect.x - old[0], delta_y=rect.y - old[1])

    @staticmethod
    def _firebar(bar: dict) -> list:
        """A fire bar's fireballs (FIREBAR_BALLS) on the block at ``x``,
        ``y``, at step ``angle`` of its turn, turning "clockwise" or not,
        "fast" or not."""
        hub = (bar["x"] + TILE / 2, bar["y"] + TILE / 2)
        turn = FIREBAR_FAST if bar.get("fast") else FIREBAR_TURN
        turn = turn if bar.get("clockwise") else -turn
        balls = []
        for k in range(int(bar.get("balls", FIREBAR_BALLS))):
            ball = {"kind": "firebar", "hub": hub, "radius": float(k * FIREBAR_GAP),
                    "angle": float(bar.get("angle", 0.0)) % FIREBAR_STEPS, "turn": turn, "x": 0.0, "y": 0.0,
                    "w": FIREBAR_BALL, "h": FIREBAR_BALL, "vx": 0.0, "vy": 0.0, "speed": 0.0, "direction": 1,
                    "patrol_min": -1e9, "patrol_max": 1e9, "edge_aware": False, "on_ground": False,
                    "foot_offset": 0, "dead": False, "stompable": False}
            MarioScenarioEnv._place_fireball(ball)
            balls.append(ball)
        return balls

    @staticmethod
    def _lava(spec: dict) -> dict:
        """A lava fireball in its pit at ``x``, first jumping after ``wait``
        frames (its later jumps and speeds from its ``turn``)."""
        bx, by, bw, bh = LAVA_BOX
        return {"kind": "lava", "x": float(spec["x"]) + bx, "y": LAVA_FROM + by, "vx": 0.0, "vy": 0.0,
                "w": bw, "h": bh, "speed": 0.0, "direction": 1, "patrol_min": -1e9, "patrol_max": 1e9,
                "edge_aware": False, "on_ground": False, "foot_offset": 0, "dead": False, "stompable": False,
                "timer": int(spec.get("wait", 1)), "turn": int(spec.get("turn", 1))}

    def _leap(self, lava: dict) -> None:
        """A lava fireball's frame: jumping when its time comes (LAVA_UP,
        LAVA_WAITS), else rising and falling (LAVA_GRAVITY, LAVA_FALL)."""
        lava["timer"] -= 1
        if lava["timer"] <= 0:
            lava["turn"] = (lava["turn"] * 1103515245 + 12345) % 2**31
            low, high = LAVA_UP
            lava["vy"] = low + (high - low) * (lava["turn"] % 1024) / 1023
            lava["y"] = LAVA_FROM + LAVA_BOX[1]
            lava["timer"] = LAVA_WAIT * LAVA_WAITS[(lava["turn"] // 1024) % len(LAVA_WAITS)]
        elif lava["y"] > self.height + 64:
            return  # (far under the screen it waits for its next jump)
        lava["y"] += lava["vy"]
        lava["vy"] = min(lava["vy"] + LAVA_GRAVITY, LAVA_FALL)

    @staticmethod
    def _bowser(spec: dict) -> dict:
        """Bowser, his place at ``x``, ``y`` (BOWSER_* notes): he wakes as
        he comes on screen; ``turn`` sets his random draws, ``clock`` the
        game's frame count, ``count`` where his flames start in FLAME_WAITS,
        ``hard`` a hard castle's flames, ``breathes`` whether he breathes
        them, ``hammers`` whether he throws hammers."""
        bowser = {"kind": "bowser", "px": float(spec["x"]), "py": float(spec["y"]), "origin": float(spec["x"]),
                  "ground": float(spec["y"]), "step": BOWSER_CHASE, "range": 0, "facing": -1, "vx": 0.0, "vy": 0.0,
                  "w": BOWSER_BOX[2], "h": BOWSER_BOX[3], "speed": 0.0, "direction": -1, "patrol_min": -1e9,
                  "patrol_max": 1e9, "edge_aware": False, "on_ground": True,
                  "foot_offset": BOWSER_FEET - BOWSER_BOX[1] - BOWSER_BOX[3], "dead": False, "stompable": False,
                  "game_rules": True, "awake": False, "jump_timer": BOWSER_FIRST_JUMP,
                  "flame_timer": BOWSER_FIRST_FLAME, "mouth": False, "breathes": bool(spec.get("breathes", True)),
                  "flames": int(spec.get("count", 0)),
                  "clock": int(spec.get("clock", 0)) % 256, "turn": int(spec.get("turn", 1)),
                  "hard": bool(spec.get("hard")), "hammers": bool(spec.get("hammers")),
                  # (the enemy each loose-item place holds, -1: none; a tuple, as states are copied shallow)
                  "places": (-1,) * HAMMER_PLACES}
        MarioScenarioEnv._place_bowser(bowser)
        return bowser

    @staticmethod
    def _place_bowser(bowser: dict) -> None:
        """Bowser's box from his place and the way he faces (BOWSER_BOX)."""
        bx, by, _, _ = BOWSER_BOX
        bowser["x"] = bowser["px"] + bx - (BOWSER_BACK if bowser["facing"] > 0 else 0)
        bowser["y"] = bowser["py"] + by

    @staticmethod
    def _bowser_draw(bowser: dict) -> int:
        """One of BOWSER_RANGES, by Bowser's turn (the game's draw)."""
        bowser["turn"] = (bowser["turn"] * 1103515245 + 12345) % 2**31
        return BOWSER_RANGES[(bowser["turn"] >> 16) % len(BOWSER_RANGES)]

    def _update_bowser(self, bowser: dict) -> None:
        """Bowser's frame, as the game's routine runs it (BOWSER_* notes):
        his timers count down; he walks (not while his mouth is open),
        jumps and breathes flames."""
        bowser["clock"] = (bowser["clock"] + 1) % 256
        bowser["jump_timer"] = max(0, bowser["jump_timer"] - 1)
        bowser["flame_timer"] = max(0, bowser["flame_timer"] - 1)
        if not bowser["mouth"]:
            if bowser["clock"] % BOWSER_TURN == 0:
                bowser["facing"] = -1
            walk = True
            if bowser["jump_timer"] and self.mario["x"] > bowser["px"]:  # Mario behind him: he turns and follows
                bowser.update(facing=1, step=BOWSER_CHASE, jump_timer=BOWSER_FIRST_JUMP, flame_timer=BOWSER_MOUTH)
                walk = bowser["px"] - bowser["origin"] < BOWSER_CHASE_LIMIT
            if walk and bowser["clock"] % BOWSER_EVERY == 0:
                if bowser["px"] == bowser["origin"]:
                    bowser["range"] = self._bowser_draw(bowser)
                bowser["px"] += bowser["step"]
                away = bowser["px"] - bowser["origin"]
                if bowser["facing"] < 0 and abs(away) >= bowser["range"]:
                    bowser["step"] = -1 if away >= 0 else 1
        if bowser["jump_timer"] == 0:  # in the air
            bowser["py"] += bowser["vy"]
            bowser["vy"] = min(bowser["vy"] + BOWSER_GRAVITY, BOWSER_FALL)
            if bowser["hammers"] and bowser["clock"] % BOWSER_EVERY == 0:
                self._bowser_hammer(bowser)
            if bowser["py"] >= bowser["ground"]:
                bowser.update(py=bowser["ground"], vy=0.0, jump_timer=self._bowser_draw(bowser))
        elif bowser["jump_timer"] == 1:
            bowser["py"] -= 1
            bowser["vy"] = BOWSER_JUMP
        if bowser["breathes"] and bowser["flame_timer"] == 0:
            bowser["flame_timer"] = BOWSER_MOUTH
            bowser["mouth"] = not bowser["mouth"]
            if not bowser["mouth"]:  # (shutting: a flame)
                bowser["flames"] += 1
                bowser["flame_timer"] = FLAME_WAITS[(bowser["flames"] - 1) % len(FLAME_WAITS)] - (
                    FLAME_HARD if bowser["hard"] else 0)
                bowser["turn"] = (bowser["turn"] * 1103515245 + 12345) % 2**31
                level = FLAME_LEVELS[(bowser["turn"] >> 16) % len(FLAME_LEVELS)]
                mx, my = FLAME_MOUTH
                # (from his place in whole pixels, as the game's y is kept)
                self.enemies.append(self._flame(bowser["px"] + mx, math.floor(bowser["py"]) + my, level, bowser["hard"]))
        # (his speed as the state reads it: his step's share of each frame)
        bowser["speed"] = 0.0 if bowser["mouth"] else abs(bowser["step"]) / BOWSER_EVERY
        bowser["direction"] = 1 if bowser["step"] > 0 else -1
        bowser["on_ground"] = bowser["jump_timer"] > 0
        self._place_bowser(bowser)

    def _bowser_hammer(self, bowser: dict) -> None:
        """Bowser's try at a hammer (BOWSER_* notes): a loose-item place is
        drawn; free, and the hammer's place on screen, a hammer appears in it."""
        bowser["turn"] = (bowser["turn"] * 1103515245 + 12345) % 2**31
        bits = bowser["turn"] >> 16
        place = bits % 8 or (bits >> 3) % 2 * 8  # (the game's: three bits, or if none is set the fourth)
        held = bowser["places"][place]
        if held >= 0 and not self.enemies[held].get("gone"):
            return
        x = bowser["px"] + BOWSER_HAMMER_AT[0]
        if not self.camera_x <= x < self.camera_x + self.width:
            return
        bowser["places"] = bowser["places"][:place] + (len(self.enemies),) + bowser["places"][place + 1:]
        # (moved the frame it appears, as the game's is; it is held through the frame it is thrown)
        self._throw_hammer(bowser, held=HAMMER_HOLD + 1)

    @staticmethod
    def _flame(x: float, y: float, level: float, hard: bool) -> dict:
        """A flame, its place at ``x``, ``y``, going to ``level`` (FLAME_*)."""
        bx, by, bw, bh = FLAME_BOX
        return {"kind": "flame", "fx": float(x), "fy": float(y), "level": float(level), "x": float(x) + bx,
                "y": float(y) + by, "w": bw, "h": bh, "vx": 0.0, "vy": float((level > y) - (level < y)),
                "speed": FLAME_SPEED[1 if hard else 0], "direction": -1, "patrol_min": -1e9, "patrol_max": 1e9,
                "edge_aware": False, "on_ground": False, "foot_offset": 0, "dead": False, "stompable": False}

    def _update_flame(self, flame: dict) -> None:
        """A flame's frame: left at its speed, up or down a pixel until level
        with where it goes (FLAME_* notes); gone once off the screen."""
        flame["fx"] -= flame["speed"]
        if flame["fy"] != flame["level"]:
            flame["fy"] += flame["vy"]
        else:
            flame["vy"] = 0.0
        flame["x"], flame["y"] = flame["fx"] + FLAME_BOX[0], flame["fy"] + FLAME_BOX[1]
        if flame["x"] < self.camera_x - 64:
            flame.update(dead=True, gone=True)  # (kept, so the others keep their places)

    @staticmethod
    def _place_fireball(ball: dict) -> None:
        """A fire bar's fireball where the whole step its bar has reached
        puts it."""
        step = math.floor(ball["angle"]) % FIREBAR_STEPS
        theta = 2 * math.pi * step / FIREBAR_STEPS
        hx, hy = ball["hub"]
        ball["x"] = hx + ball["radius"] * math.sin(theta) - FIREBAR_BALL / 2
        ball["y"] = hy - ball["radius"] * math.cos(theta) - FIREBAR_BALL / 2

    def _spring_throw(self, spring: dict) -> None:
        """Mario thrown up by the springboard he stands on (SPRING_LAUNCH)."""
        motion = self.motion
        motion.y_speed = SPRING_LAUNCH[1 if spring["strong"] else 0]
        motion.y_force = motion.y_fraction = 0
        motion.force = motion.force_up = motion.force_down = SPRING_FORCE
        motion.origin_y = int(self.mario["y"])
        self.mario.update(on_ground=False, _platform=None, vy=float(motion.y_speed))

    def _frenzies(self) -> None:
        """Bullet Bills, leaping Cheep Cheeps and Bowser's free flames of the
        frenzies Mario is in (BILL_GAP, CHEEP_GAPS, FLAME_WAITS)."""
        for frenzy in self.frenzies:
            if not frenzy["from"] <= self.mario["x"] < frenzy["until"]:
                continue
            if frenzy["kind"] == "flames" and any(e.get("kind") == "bowser" and e.get("awake") and not e["dead"]
                                                  for e in self.enemies):
                continue  # (none while Bowser is on: he breathes his own)
            frenzy["timer"] -= 1
            if frenzy["timer"] > 0:
                continue
            frenzy["count"] += 1
            # (the game's turns are pseudo-random: a fixed sequence from the scenario's turn)
            frenzy["turn"] = (frenzy["turn"] * 1103515245 + 12345) % 2**31
            turn = frenzy["turn"] >> 8
            base = {"vx": 0.0, "patrol_min": -1e9, "patrol_max": 1e9, "edge_aware": False, "on_ground": False,
                    "dead": False, "stompable": True}
            if frenzy["kind"] == "flames":
                frenzy["timer"] = FLAME_WAITS[(frenzy["count"] - 1) % len(FLAME_WAITS)] + BOWSER_MOUTH - (
                    FLAME_HARD if frenzy["hard"] else 0)
                level = FLAME_LEVELS[turn % len(FLAME_LEVELS)]
                self.enemies.append(self._flame(self.camera_x + FLAME_FROM, level, level, frenzy["hard"]))
                continue
            if frenzy["kind"] == "bills":
                frenzy["timer"] = BILL_GAP
                self.enemies.append({**base, "kind": "bullet", "x": self.camera_x + BILL_FROM, "vy": 0.0,
                                     "y": float(BILL_HEIGHTS[frenzy["count"] % len(BILL_HEIGHTS)]),
                                     "w": BULLET_W, "h": BULLET_H, "speed": BULLET_SPEED, "direction": -1})
                continue
            frenzy["timer"] = CHEEP_GAPS[turn % len(CHEEP_GAPS)]
            middle = self.mario["x"] + self.mario["w"] / 2
            x = middle + (turn // 16) % (2 * CHEEP_REACH + 1) - CHEEP_REACH
            toward = 1 if x < middle else -1
            self.enemies.append({**base, "kind": "cheep", "x": x, "y": float(CHEEP_FROM), "vy": CHEEP_UP, "w": 10,
                                 "h": 6, "speed": CHEEP_SPEEDS[(turn // 8192) % len(CHEEP_SPEEDS)],
                                 "direction": toward if (turn // 65536) % 8 else -toward})

    def _shells_hit(self, reward_terms: dict) -> None:
        """Moving shells kill the enemies they meet."""
        for shell in self.enemies:
            if shell.get("shell") != "moving" or shell["dead"]:
                continue
            shell["kick_grace"] = max(0, shell.get("kick_grace", 0) - 1)
            body = pygame.Rect(shell["x"], shell["y"], shell["w"], shell["h"])
            for other in self.enemies:
                if other is shell or other["dead"] or other.get("kind") in ("piranha_plant", "hammer", "firebar", "lava", "bowser",
                                                                                       "flame"):
                    continue
                if body.colliderect(pygame.Rect(other["x"], other["y"], other["w"], other["h"])):
                    other["dead"] = True
                    self.score += 5
                    reward_terms["enemy_stomp"] += self.reward_config.enemy_stomp

    def _bump(self, plat: dict, reward_terms: dict) -> None:
        """Mario's head hit ``plat`` from below: a question block gives its
        coin or lets out its mushroom (once); big Mario breaks a brick."""
        if plat.get("contents") and not plat.get("used"):
            plat["used"] = True
            if plat["contents"] == "coin":
                self.score += COIN_POINTS
                reward_terms["coin"] += self.reward_config.coin
            else:
                r = plat["rect"]
                self.items.append({"kind": plat["contents"], "x": float(r.x), "y": float(r.top - POWER_UP_SIZE),
                                   "w": POWER_UP_SIZE, "h": POWER_UP_SIZE, "vx": MUSHROOM_SPEED, "vy": 0.0})
            return
        index = self.platforms.index(plat)
        if self.mario["big"] and self.platform_kinds[index] == "brick" and not plat.get("contents"):
            plat["rect"] = pygame.Rect(-10_000, -10_000, 0, 0)  # broken: gone, its index kept

    def _resize(self, big: bool) -> None:
        """Mario grown by a mushroom or shrunk by a hit, his feet and middle
        where they were."""
        m = self.mario
        w, h = (BIG_W, BIG_H) if big else (SMALL_W, SMALL_H)
        m["x"] += (m["w"] - w) / 2
        m["y"] += m["h"] - h
        m["w"], m["h"], m["big"] = w, h, big

    def _update_item(self, item: dict, platform_rects: list) -> None:
        """A mushroom moves on as a walker does (turned by walls, falling
        off ledges), at its own speed."""
        item["vy"] = min(item["vy"] + ENEMY_GRAVITY, ENEMY_FALL)
        item["x"] += item["vx"]
        body = pygame.Rect(item["x"], item["y"], item["w"], item["h"])
        if any(body.colliderect(r) and r.top < body.bottom - 2 for r in platform_rects):
            item["x"] -= item["vx"]
            item["vx"] = -item["vx"]
        item["y"] += item["vy"]
        body = pygame.Rect(item["x"], item["y"], item["w"], item["h"])
        for r in platform_rects:
            if body.colliderect(r) and item["vy"] >= 0:
                item["y"] = float(r.top - item["h"])
                item["vy"] = 0.0

    def _update_hammer_bro(self, bro: dict, platform_rects: list) -> None:
        """A Hammer Bro's frame (HAMMER_BRO_* notes): its walk, its throw when
        due, its jump when due, and its flight."""
        bro["x"] += bro["speed"] * bro["direction"]
        if not bro["patrol_min"] < bro["x"] < bro["patrol_max"]:
            bro["x"] = min(max(bro["x"], bro["patrol_min"]), bro["patrol_max"])
            bro["direction"] *= -1
        bro["throw_timer"] -= 1
        if bro["throw_timer"] <= 0:
            bro["throws"] += 1
            bro["throw_timer"] = HAMMER_FRAMES[bro["throws"] % len(HAMMER_FRAMES)]
            if self.camera_x <= bro["x"] < self.camera_x + self.width:
                self._throw_hammer(bro)
        x0, x1 = bro["x"], bro["x"] + bro["w"]
        feet = bro["y"] + bro["h"] + bro["foot_offset"]
        if bro["on_ground"]:
            bro["jump_timer"] -= 1
            if bro["jump_timer"] > 0:
                return
            bro["jumps"] += 1
            bro["jump_timer"] = HAMMER_BRO_JUMPS[bro["jumps"] % len(HAMMER_BRO_JUMPS)]
            over = lambda r: r.left < x1 and x0 < r.right  # noqa: E731
            above = any(over(r) and abs(feet - r.top - HAMMER_BRO_REACH) <= 8 for r in platform_rects)
            below = any(over(r) and r.top > feet + 1 for r in platform_rects)
            if not (above or below):
                return
            # Up when it can and either cannot go down or it is its turn.
            up = above and not (below and bro["jumps"] % 2)
            bro.update(vy=HAMMER_BRO_UP if up else HAMMER_BRO_DOWN, through=None if up else feet, on_ground=False)
        bro["vy"] = min(bro["vy"] + HAMMER_BRO_GRAVITY, ENEMY_FALL)
        if bro["vy"] > 0:  # falling, it lands on the first top it meets (not the one it dropped through)
            for r in platform_rects:
                if r.left < x1 and x0 < r.right and feet <= r.top <= feet + bro["vy"] and r.top != bro["through"]:
                    bro.update(y=r.top - bro["h"] - bro["foot_offset"], vy=0.0, on_ground=True, through=None)
                    return
        bro["y"] += bro["vy"]
        if bro["y"] > self.height:
            bro.update(dead=True, gone=True)

    @staticmethod
    def _hammer_held(hammer: dict, bro: dict) -> None:
        """A held hammer's place: over a Hammer Bro's head (HAMMER_* notes),
        or BOWSER_HAMMER_AT from Bowser's place; it moves as he does."""
        if bro.get("kind") == "bowser":
            hammer.update(x=bro["px"] + BOWSER_HAMMER_AT[0], y=math.floor(bro["py"]) + BOWSER_HAMMER_AT[1])
        else:
            hammer.update(x=bro["x"] + (bro["w"] - HAMMER_W) / 2, y=bro["y"] - HAMMER_OVER - HAMMER_H / 2)
        hammer.update(speed=bro["speed"], direction=bro["direction"])

    def _throw_hammer(self, bro: dict, held: int = HAMMER_HOLD) -> None:
        """A hammer appears over ``bro``'s head, held ``held`` frames
        (HAMMER_* notes)."""
        index = next(k for k, e in enumerate(self.enemies) if e is bro)
        hammer = {
            "kind": "hammer", "bro": index, "held": held,
            "vx": 0.0, "vy": 0.0, "w": HAMMER_W, "h": HAMMER_H,
            "patrol_min": -1e9, "patrol_max": 1e9, "edge_aware": False, "on_ground": False,
            "dead": False, "stompable": False,
        }
        self._hammer_held(hammer, bro)
        self.enemies.append(hammer)

    def _update_hammer(self, hammer: dict) -> None:
        """A hammer's frame: held over its Hammer Bro, it moves with it until
        thrown toward Mario; thrown, it flies under its own gravity through
        everything until it is off the screen."""
        if hammer["held"] > 0:
            bro = self.enemies[hammer["bro"]]
            hammer["held"] -= 1
            self._hammer_held(hammer, bro)
            if hammer["held"] == 0 or bro["dead"]:
                hammer["held"] = 0
                middle = self.mario["x"] + self.mario["w"] / 2
                # (a Hammer Bro faces Mario; Bowser throws the way he faces)
                way = bro["facing"] if bro.get("kind") == "bowser" else 1 if middle > hammer["x"] + HAMMER_W / 2 else -1
                hammer.update(speed=HAMMER_SPEED, direction=way, vy=HAMMER_THROW)
            return
        hammer["vy"] = min(hammer["vy"] + HAMMER_GRAVITY, HAMMER_FALL)
        hammer["x"] += hammer["speed"] * hammer["direction"]
        hammer["y"] += hammer["vy"]
        if hammer["y"] > self.height or not self.camera_x - 64 <= hammer["x"] <= self.camera_x + self.width + 64:
            hammer.update(dead=True, gone=True)  # (kept, so the others keep their places)

    def _fly(self, koopa: dict) -> None:
        """A flying Paratroopa's swing (RED_FLIGHT, GREEN_FLIGHT, GREEN_BOB)
        from where it was placed; its speed and direction as it moves."""
        koopa["flight_tick"] += 1
        tick, (home_x, home_y) = koopa["flight_tick"], koopa["home"]
        if koopa["flight"] == "down":
            reach, period = RED_FLIGHT
            x, y = home_x, home_y + reach / 2 * (1 - math.cos(2 * math.pi * tick / period))
        else:
            reach, period = GREEN_FLIGHT
            bob, bob_period = GREEN_BOB
            x = home_x - reach / 2 * (1 - math.cos(2 * math.pi * tick / period))
            phase = (tick % bob_period) / bob_period
            y = home_y - bob + 2 * bob * abs(2 * phase - 1)
        dx, dy = x - koopa["x"], y - koopa["y"]
        koopa.update(x=x, y=y, vy=dy, speed=abs(dx), direction=(1 if dx > 0 else -1) if dx else koopa["direction"])

    def _update_lakitu(self, lakitu: dict) -> None:
        """Lakitu's flight after Mario (LAKITU_BASE), his eggs, and his
        leaving at the end of his stretch."""
        if self.mario["x"] >= lakitu["until"]:
            lakitu.update(speed=LAKITU_LEAVE, direction=-1)
            lakitu["x"] -= LAKITU_LEAVE
            if lakitu["x"] + lakitu["w"] < self.camera_x - 64:
                lakitu.update(dead=True, gone=True)  # flown off: kept, so the others keep their places
            return
        apart = (lakitu["x"] + lakitu["w"] / 2) - (self.mario["x"] + self.mario["w"] / 2)
        toward = 1 if apart < 0 else -1  # the way to Mario
        if abs(apart) >= LAKITU_REACH and toward != lakitu["direction"] and lakitu["direction"] > 0:
            lakitu["lakitu_speed"] -= 1  # going right, past him: slowed to a stop first
            if lakitu["lakitu_speed"] > 0:
                lakitu["x"] += lakitu["lakitu_speed"] / 16
                lakitu["speed"] = lakitu["lakitu_speed"] / 16
                return
        if abs(apart) >= LAKITU_REACH:
            lakitu["direction"] = toward
        fast = abs(self.motion.x_speed) >= LAKITU_FAST and self.scrolled >= 2
        base = LAKITU_BASE[0] if self.scrolled <= 0 else LAKITU_BASE[2 if fast else 1]
        lakitu["lakitu_speed"] = max(1, base - int(min(abs(apart), LAKITU_REACH)) // 4)
        lakitu["speed"] = lakitu["lakitu_speed"] / 16
        lakitu["x"] += lakitu["direction"] * lakitu["speed"]
        lakitu["throw_timer"] -= 1
        if lakitu["throw_timer"] > 0:
            return
        lakitu["throw_timer"] = LAKITU_THROW
        left, right = self.camera_x, self.camera_x + self.width
        company = sum(1 for e in self.enemies if e is not lakitu and not e["dead"] and left <= e["x"] < right)
        if company >= LAKITU_COMPANY:
            return
        bx, by, _, _ = LAKITU_BOX
        middle = self.mario["x"] + self.mario["w"] / 2
        # (an egg has a Goomba's body: 10 by 6, 14 below where the game places it)
        x, y = lakitu["x"] - bx + 3, LAKITU_Y - EGG_ABOVE + 14
        self.enemies.append({
            "kind": "spiny", "egg": True, "x": x, "y": float(y), "vx": 0.0, "vy": EGG_UP, "w": 10, "h": 6,
            "foot_offset": 4, "speed": 0.0, "direction": 1 if middle > x + 5 else -1,
            "patrol_min": -1e9, "patrol_max": 1e9, "edge_aware": False, "on_ground": False,
            "dead": False, "stompable": False, "game_rules": True, "awake": True,
        })

    def _update_egg(self, egg: dict, platform_rects: list) -> None:
        """A Spiny's egg falling (EGG_GRAVITY); landed, a Spiny walking
        toward Mario."""
        feet = egg["y"] + egg["h"] + egg["foot_offset"]
        egg["vy"] = min(egg["vy"] + EGG_GRAVITY, EGG_FALL)
        for r in platform_rects:
            if (egg["vy"] > 0 and r.left < egg["x"] + egg["w"] and egg["x"] < r.right
                    and feet <= r.top <= feet + egg["vy"]):
                middle = self.mario["x"] + self.mario["w"] / 2
                egg.update(y=r.top - egg["h"] - egg["foot_offset"], vy=0.0, egg=False, on_ground=True,
                           speed=SPINY_SPEED, direction=1 if middle > egg["x"] + egg["w"] / 2 else -1)
                return
        egg["y"] += egg["vy"]
        if egg["y"] > self.height:
            egg.update(dead=True, gone=True)

    def _update_enemy(self, enemy: dict, platform_rects: list):
        """Move enemy, apply gravity, resolve platform collisions, patrol logic."""
        if enemy.get("kind") == "piranha_plant":
            from .piranha import position_plant, stays_down_near

            if not stays_down_near(enemy, self.mario["x"] + self.mario["w"] / 2):
                enemy["plant_tick"] += 1
            position_plant(enemy)
            return
        if enemy.get("kind") == "bullet":
            enemy["x"] += enemy["speed"] * enemy["direction"]
            if not self.camera_x - 64 <= enemy["x"] <= self.camera_x + self.width + 64:
                enemy.update(dead=True, gone=True)  # flown off: kept, so the others keep their places
            return
        if enemy.get("kind") == "hammer":
            self._update_hammer(enemy)
            return
        if enemy.get("kind") == "firebar":
            enemy["angle"] = (enemy["angle"] + enemy["turn"]) % FIREBAR_STEPS
            self._place_fireball(enemy)
            return
        if enemy.get("kind") == "lava":
            self._leap(enemy)
            return
        if enemy.get("kind") == "flame":
            self._update_flame(enemy)
            return
        if enemy.get("kind") == "cheep":
            enemy["vy"] += CHEEP_GRAVITY
            enemy["x"] += enemy["speed"] * enemy["direction"]
            enemy["y"] += enemy["vy"]
            if (enemy["vy"] > 0 and enemy["y"] > CHEEP_FROM) or not (
                    self.camera_x - 64 <= enemy["x"] <= self.camera_x + self.width + 64):
                enemy.update(dead=True, gone=True)  # back down or off the screen: kept, so the others keep their places
            return
        if (enemy.get("kind") == "monster" or enemy.get("game_rules")) and not enemy.get("awake"):
            # Like NES enemies, a monster (and any enemy under the game's
            # rules) starts moving once it is on screen.
            if enemy["x"] >= self.camera_x + self.width:
                return
            enemy["awake"] = True
        if enemy.get("kind") == "hammer_bro":
            self._update_hammer_bro(enemy, platform_rects)
            return
        if enemy.get("kind") == "bowser":
            self._update_bowser(enemy)
            return
        if enemy.get("kind") == "lakitu":
            self._update_lakitu(enemy)
            return
        if enemy.get("egg"):
            self._update_egg(enemy, platform_rects)
            return
        if enemy.get("flight") and enemy.get("wings"):
            self._fly(enemy)
            return
        # Gravity (a winged Koopa hops each time it lands)
        if enemy.get("wings"):
            if enemy["on_ground"]:
                enemy["vy"] = HOP_SPEED
            enemy["vy"] += HOP_GRAVITY
        else:
            enemy["vy"] = min(enemy["vy"] + ENEMY_GRAVITY, ENEMY_FALL)
        if enemy["vy"] > self.max_fall_speed:
            enemy["vy"] = self.max_fall_speed

        # Horizontal move — check edge-awareness before stepping
        step_x = enemy["speed"] * enemy["direction"]
        if enemy["edge_aware"] and enemy["on_ground"] and not enemy.get("shell"):
            # As the game's red Koopa does (game memory, 1-3 and 3-3): turned
            # back where the floor under its middle would end
            peek_x = enemy["x"] + enemy["w"] / 2 + step_x
            feet_y = enemy["y"] + enemy["h"] + enemy.get("foot_offset", 0) + 1
            supported = any(
                r.left <= peek_x < r.right and r.top <= feet_y <= r.bottom for r in platform_rects
            )
            if not supported:
                enemy["direction"] *= -1
                step_x = enemy["speed"] * enemy["direction"]

        enemy["x"] += step_x
        if enemy.get("game_rules") and step_x:
            # The game's walkers turn back when they walk into a wall (a pipe,
            # a step): its side, above their feet.
            body = pygame.Rect(enemy["x"], enemy["y"], enemy["w"], enemy["h"])
            if any(body.colliderect(r) and r.top < body.bottom - 2 for r in platform_rects):
                enemy["x"] -= step_x
                enemy["direction"] *= -1

        # Clamp to explicit patrol bounds (still respected when edge_aware=True)
        if enemy["x"] <= enemy["patrol_min"]:
            enemy["x"] = enemy["patrol_min"]
            enemy["direction"] = 1
        elif enemy["x"] >= enemy["patrol_max"]:
            enemy["x"] = enemy["patrol_max"]
            enemy["direction"] = -1

        # Y — apply and resolve. Its feet reaching a top exactly (not by
        # whole pixels) land it: standing, it stays where it stands (by
        # whole pixels a standing enemy sank a fraction of a pixel for two
        # frames in three before it was set back, and stood only the third).
        feet = enemy["y"] + enemy["h"] + enemy.get("foot_offset", 0)
        enemy["y"] += enemy["vy"]
        enemy["on_ground"] = False
        er = pygame.Rect(
            enemy["x"], enemy["y"], enemy["w"], enemy["h"] + enemy.get("foot_offset", 0)
        )
        for r in platform_rects:
            lands = (enemy["vy"] >= 0 and r.left < enemy["x"] + enemy["w"] and enemy["x"] < r.right
                     and feet <= r.top <= feet + enemy["vy"])
            if lands or er.colliderect(r):
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
            if plat.get("lift") == "spring":
                resolved.append("spring")
            elif plat["moving"]:
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
        if isinstance(p, dict) and p.get("lift"):
            # (a lift: "across", "updown" or "falls", from where it is placed, at a point of its swing)
            plat.update(moving=True, lift=p["lift"], home=(float(x), float(y)), lift_x=float(x), lift_y=float(y),
                        tick=int(p.get("phase", 0)), vx=0.0, vy=0.0, delta_y=0)
            if p["lift"] == "spring":  # (pressed down this frame of SPRING_SQUASH; None: at rest)
                plat.update(squash=None, strong=False)
        return plat

    @staticmethod
    def _parse_enemy(e) -> dict:
        """Accept [x,y,pmin,pmax[,speed[,direction]]] or a dictionary."""
        if isinstance(e, dict) and e.get("kind") == "piranha_plant":
            from .piranha import parse_plant

            return parse_plant(e)
        kind = e.get("kind") if isinstance(e, dict) else None
        if kind not in (None, "monster", "koopa", "red_koopa", "paratroopa", "red_paratroopa", "green_flyer",
                        "hammer_bro", "lakitu"):
            raise ValueError(f"unknown enemy kind {kind!r}")
        wings = kind in ("paratroopa", "red_paratroopa", "green_flyer")
        # A red Koopa: a Koopa that turns back at ledges (edge_aware)
        red = kind in ("red_koopa", "red_paratroopa")
        # A flying Paratroopa: its swing, from where it is placed (FLIGHTS)
        flight = {"red_paratroopa": "down", "green_flyer": "across"}.get(kind)
        if wings or red:
            kind = "koopa"
        game_rules = isinstance(e, dict) and bool(e.get("game_rules", False))
        if isinstance(e, dict):
            x, y = float(e["x"]), float(e["y"])
            # Under the game's rules there is no patrol: walls turn the
            # walker and ledges let it fall.
            pmin = float(e.get("patrol_min", -1e9 if game_rules else x))
            pmax = float(e.get("patrol_max", 1e9 if game_rules else x))
            speed = float(e.get("speed", 0.5 if game_rules else 1.5))
            edge_aware = bool(e.get("edge_aware", red))
            direction = int(e.get("direction", -1 if game_rules else 1))
        else:
            x, y = float(e[0]), float(e[1])
            pmin, pmax = float(e[2]), float(e[3])
            speed = float(e[4]) if len(e) > 4 else 1.5
            edge_aware = False
            direction = int(e[5]) if len(e) > 5 else 1

        if direction not in (-1, 1):
            raise ValueError("enemy direction must be -1 or 1")
        return {
            **({"kind": kind} if kind else {}),
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
            **({"game_rules": True} if game_rules else {}),
            **({"wings": True} if wings else {}),
            **({"red": True} if red else {}),
            **({"flight": flight, "flight_tick": int(e.get("phase", 0))} if flight else {}),
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
