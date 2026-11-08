"""Random levels made of the real game's pieces (docs/choose-and-control-plan.md
phase 2; user, 2026-11-07: "The levels could be randomly generated and I
could still pass them").

A level is a run of World 1-1's pieces, in pixels as the game draws them
(16-pixel tiles, the floor's top at 208): pits 2 to 4 tiles wide, pipes 2 to
4 tiles tall, staircases up and down with a valley or a pit between, rows of
bricks at the game's two heights (4 and 8 tiles above the floor, Goombas
sometimes walking on the upper one), single Goombas and pairs, and a
staircase before the flag, each separated by a few tiles of floor
(random_level). The floor is cut where a pipe or a step stands, as the game
draws it. Goombas follow the game's rules (env.py "game_rules": they wake as
they come on screen, walk left at half a pixel a frame, turn at walls, fall
off ledges), and the camera keeps Mario where the real game does
(SCROLL_X).

BlockTruthState gives the controller its state from the simulator as
TruthState does from the real game's memory: what is on screen, the screen's
left edge a wall. play() runs a chooser (stages/full_smb/scripted_chooser.py)
and the controller (core/smb_controller.py) through a level.
"""

import random
from copy import copy
from typing import Optional

from retroagi.core.smb_controller import Controller, State
from retroagi.core.smb_planner import Enemy, Solid, screen_edge

TILE = 16
FLOOR = 208
BOTTOM = 240
SCREEN = 256
# The real game keeps Mario's left side about here on screen as it scrolls
# (Full SMB, 1-1: 115 in nine scrolling frames of ten).
SCROLL_X = 115
# Mario starts standing on the floor this far in.
START_X = 40
# Floor before the first piece, between pieces, and after the last (tiles).
START_FLAT = 10
FLAT = (3, 8)
END_FLAT = 10
# The pieces and how often each comes.
PIECES = {"pipe": 3, "pit": 2, "goombas": 3, "bricks": 2, "stairs": 1}
PIPE_TILES = (2, 3, 4)
PIT_TILES = (2, 2, 3, 3, 4)
STAIR_TILES = (3, 4)
# Brick rows: their tops (pixels) and lengths (tiles).
LOW_ROW, HIGH_ROW = FLOOR - 4 * TILE, FLOOR - 8 * TILE
LOW_TILES, HIGH_TILES = (1, 5), (3, 8)
# Two Goombas walk this far apart (pixels), as in 1-1.
PAIR_GAP = 24
# A Goomba's body as the simulator places it: its feet this far below the
# y it is given.
GOOMBA_FEET = 14
# Frames a level may take per pixel of its length.
FRAMES_PER_PIXEL = 3


def _goomba(x: float, top: float) -> dict:
    return {"x": float(x), "y": float(top - GOOMBA_FEET), "game_rules": True}


def random_level(seed: int, pieces: int = 12) -> dict:
    """A level of ``pieces`` random pieces then the closing staircase and
    flag, the same for the same ``seed`` (a Block SMB scenario)."""
    rng = random.Random(seed)
    x = START_FLAT * TILE
    pits, solids, enemies = [], [], []  # pits (x0, x1); solids [x, y, w, h, kind]

    def column(x0, tiles):
        top = FLOOR - tiles * TILE
        solids.append([x0, top, TILE, BOTTOM - top, "ground"])

    for _ in range(pieces):
        kind = rng.choices(list(PIECES), weights=list(PIECES.values()))[0]
        if kind == "pipe":
            top = FLOOR - rng.choice(PIPE_TILES) * TILE
            solids.append([x, top, 2 * TILE, BOTTOM - top, "pipe"])
            x += 2 * TILE
        elif kind == "pit":
            width = rng.choice(PIT_TILES) * TILE
            pits.append((x, x + width))
            x += width
        elif kind == "goombas":
            enemies.append(_goomba(x, FLOOR))
            if rng.random() < 0.5:
                enemies.append(_goomba(x + PAIR_GAP, FLOOR))
            x += 3 * TILE
        elif kind == "bricks":
            low = rng.randint(*LOW_TILES)
            solids.append([x, LOW_ROW, low * TILE, TILE, "brick"])
            end = x + low * TILE
            if rng.random() < 0.5:
                start = x + rng.randint(1, 2) * TILE
                high = rng.randint(*HIGH_TILES)
                solids.append([start, HIGH_ROW, high * TILE, TILE, "brick"])
                if rng.random() < 0.5:
                    enemies.append(_goomba(start + (high - 1) * TILE, HIGH_ROW))
                end = max(end, start + high * TILE)
            x = end
        else:  # stairs: up, a valley of floor or a pit, down
            height = rng.choice(STAIR_TILES)
            for k in range(height):
                column(x, k + 1)
                x += TILE
            gap = 2 * TILE
            if rng.random() < 0.5:
                pits.append((x, x + gap))
            x += gap
            for k in range(height):
                column(x, height - k)
                x += TILE
        x += rng.randint(*FLAT) * TILE
    # The closing staircase, then the flag on the floor beyond.
    height = rng.randint(4, 8)
    for k in range(height):
        column(x, k + 1)
        x += TILE
    column(x, height)
    x += TILE + 4 * TILE
    flag = x
    width = x + END_FLAT * TILE

    # The floor, cut at pits and where pipes and steps stand.
    cuts = sorted(pits + [(s[0], s[0] + s[2]) for s in solids if s[4] in ("pipe", "ground")])
    floor, start = [], 0
    for x0, x1 in cuts:
        if x0 > start:
            floor.append([start, FLOOR, x0 - start, BOTTOM - FLOOR, "ground"])
        start = max(start, x1)
    floor.append([start, FLOOR, width - start, BOTTOM - FLOOR, "ground"])
    platforms = floor + solids
    return {
        "task": {"family": "game_level"},
        "world_width": width,
        "scroll_x": SCROLL_X,
        "frame_budget": width * FRAMES_PER_PIXEL,
        "mario": [START_X, FLOOR - 12],
        "platforms": [p[:4] for p in platforms],
        "platform_kinds": [p[4] for p in platforms],
        "enemies": enemies,
        "goal": [flag, 0, TILE, BOTTOM],
        "metadata": {"game_level": {"seed": seed, "pieces": pieces}},
    }


_KINDS = {"ground": "ground", "pipe": "pipe", "brick": "block", "question_block": "block", "moving_platform": "moving"}


class BlockTruthState:
    """The controller's state from the simulator: what is on the screen,
    in world pixels, as TruthState reads the real game's memory."""

    def read(self, env) -> State:
        left, right = env.camera_x, env.camera_x + env.width
        solids = []
        for k, (p, kind) in enumerate(zip(env.platforms, env.platform_kinds)):
            r = p["rect"]
            if r.right <= left or r.left >= right:
                continue
            solids.append(Solid(max(r.left, left), r.top, min(r.right, right), r.bottom,
                                _KINDS.get(kind, "block"), ("platforms", k)))
        enemies = []
        for k, e in enumerate(env.enemies):
            awake = e.get("awake", not e.get("game_rules"))
            box = (e["x"], e["y"], e["x"] + e["w"], e["y"] + e["h"])
            if e["dead"] or e["h"] <= 0 or not awake or box[2] <= left or box[0] >= right:
                continue
            # (standing, its fall speed is gravity's step the floor stopped)
            vy = 0.0 if e["on_ground"] else e["vy"]
            enemies.append(Enemy(box, e["speed"] * e["direction"], vy, ("enemies", k),
                                 stompable=bool(e.get("stompable", True))))
        m = env.mario
        return State(
            body=(m["x"], m["y"], m["x"] + m["w"], m["y"] + m["h"]),
            motion=copy(env.motion),
            grounded=bool(m["on_ground"]),
            solids=solids + [screen_edge(left)],
            enemies=enemies,
        )


def play(scenario: dict, chooser, frames: Optional[int] = None) -> dict:
    """``chooser`` (a ScriptedChooser) and the controller through
    ``scenario``: how it ended ("finished", "died" or "timeout"), the frames
    taken, the furthest x reached and the score."""
    from .env import MarioScenarioEnv

    env = MarioScenarioEnv(world_width=scenario["world_width"])
    env.reset(scenario=scenario)
    source, controller = BlockTruthState(), Controller()
    last_event, last_choice, end = None, None, "timeout"
    limit = frames or scenario.get("frame_budget", 6000)
    for frame in range(limit):
        state = source.read(env)
        button = 0
        if controller.choice is None or controller.event is not None:
            if state.grounded:
                last_event = controller.event
                last_choice = chooser.decide(state, last_event, last_choice)
                controller.start(last_choice, state)
        if controller.event is None:
            button = controller.press(state)
        _, _, terminated, truncated, info = env.step(button)
        if terminated or truncated:
            end = "died" if info["death"] else "finished" if env._goal_credited else "timeout"
            break
    return {"end": end, "frames": frame + 1, "x": env._max_x_reached, "score": env.score}


def run_levels(seeds, modes=("speed_run", "max_points"), pieces: int = 12) -> list:
    """play() the scripted chooser through random_level(seed) for each seed
    and mode: one row per run (seed, mode and play's result)."""
    from retroagi.stages.full_smb.scripted_chooser import ScriptedChooser

    rows = []
    for seed in seeds:
        level = random_level(seed, pieces)
        for mode in modes:
            rows.append({"seed": seed, "mode": mode, **play(level, ScriptedChooser(mode))})
    return rows


def main(argv=None) -> int:
    import argparse
    import json
    from collections import Counter

    parser = argparse.ArgumentParser(description="The scripted chooser on random levels of the game's pieces")
    parser.add_argument("--levels", type=int, default=100)
    parser.add_argument("--first-seed", type=int, default=0)
    parser.add_argument("--pieces", type=int, default=12)
    parser.add_argument("--output", default=None)
    args = parser.parse_args(argv)
    rows = run_levels(range(args.first_seed, args.first_seed + args.levels), pieces=args.pieces)
    for mode in sorted({r["mode"] for r in rows}):
        runs = [r for r in rows if r["mode"] == mode]
        ends = Counter(r["end"] for r in runs)
        print(f"{mode:12s} finished {ends['finished']}/{len(runs)}  died {ends['died']}  timeout {ends['timeout']}"
              f"  failed seeds {[r['seed'] for r in runs if r['end'] != 'finished']}")
    if args.output:
        with open(args.output, "w") as f:
            json.dump(rows, f, indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
