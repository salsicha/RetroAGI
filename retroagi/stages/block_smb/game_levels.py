"""Random levels made of the real game's pieces (docs/choose-and-control-plan.md
phase 2; user, 2026-11-07: "The levels could be randomly generated and I
could still pass them").

A level is a run of World 1-1's pieces, in pixels as the game draws them
(16-pixel tiles, the floor's top at 208): pits 2 to 4 tiles wide, pipes 2 to
4 tiles tall, staircases up and down with a valley or a pit between (each
top one or two columns wide), rows of bricks and question blocks at the
game's two heights (4 and 8 tiles above the floor, Goombas sometimes walking
on the upper one; a question block holds a coin or a mushroom, big Mario
breaks bricks), question blocks over the floor, single Goombas and pairs,
Koopa Troopas (a stomp leaves a shell that a touch kicks), and from World
5-1 piranha plants in pipes, hopping Koopa Paratroopas and Bullet Bill
cannons, and a staircase before the flag, each separated by a few tiles of floor
(random_level). The floor is cut where a pipe or a step stands, as the game
draws it. Goombas follow the game's rules (env.py "game_rules": they wake as
they come on screen, walk left at half a pixel a frame, turn at walls, fall
off ledges), and the camera keeps Mario where the real game does
(SCROLL_X).

BlockTruthState gives the controller its state from the simulator as
TruthState does from the real game's memory: what is on screen (question
blocks not yet hit as their own kind, mushrooms as items), the screen's left
edge a wall. play() runs a chooser (stages/full_smb/scripted_chooser.py)
and the controller (core/smb_controller.py) through a level.
"""

import random
from copy import copy
from typing import Optional

from retroagi.core.smb_controller import Controller, State
from retroagi.core.smb_planner import Enemy, Solid, screen_edge

from .piranha import EXPOSED_FRAMES, HIDDEN_FRAMES, RISE_FRAMES

PLANT_CYCLE = 2 * RISE_FRAMES + EXPOSED_FRAMES + HIDDEN_FRAMES

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
PIECES = {"pipe": 3, "pit": 2, "goombas": 3, "bricks": 2, "stairs": 1, "questions": 1, "cannon": 1}
PIPE_TILES = (2, 3, 4)
PIT_TILES = (2, 2, 3, 3, 4)
STAIR_TILES = (3, 4)
STAIR_TOPS = (1, 2)
# Brick rows: their tops (pixels) and lengths (tiles).
LOW_ROW, HIGH_ROW = FLOOR - 4 * TILE, FLOOR - 8 * TILE
LOW_TILES, HIGH_TILES = (1, 5), (3, 8)
# In a brick row this share of tiles are question blocks; a question block
# holds a mushroom MUSHROOM_SHARE of the time, else a coin. The "questions"
# piece is 1 to 3 question blocks over the floor (1-1's first).
QUESTION_SHARE = 0.3
MUSHROOM_SHARE = 0.35
# The share of levels Mario starts big (as after 1-1's first mushroom). Both
# shares are above the game's so the outcome network sees mushrooms and big
# Mario often enough (at the game's, sampled states almost never had one).
BIG_START = 0.3
# Two Goombas walk this far apart (pixels), as in 1-1; an enemy piece is a
# Koopa Troopa this share of the time.
PAIR_GAP = 24
KOOPA_SHARE = 0.25
# World 5-1's things: a pipe holds a piranha plant PLANT_SHARE of the time
# (at a random point of its cycle); an enemy piece is a Koopa Paratroopa
# PARATROOPA_SHARE of the time; a cannon is CANNON_TILES tall.
PLANT_SHARE = 0.4
PARATROOPA_SHARE = 0.2
CANNON_TILES = (1, 2)
# A Goomba's body as the simulator places it: its feet this far below the
# y it is given.
GOOMBA_FEET = 14
# Frames a level may take per pixel of its length.
FRAMES_PER_PIXEL = 3


def _goomba(x: float, top: float, kind: Optional[str] = None) -> dict:
    out = {"x": float(x), "y": float(top - GOOMBA_FEET), "game_rules": True}
    if kind:
        out["kind"] = kind
    return out


def random_level(seed: int, pieces: int = 12) -> dict:
    """A level of ``pieces`` random pieces then the closing staircase and
    flag, the same for the same ``seed`` (a Block SMB scenario)."""
    rng = random.Random(seed)
    x = START_FLAT * TILE
    pits, solids, enemies = [], [], []  # pits (x0, x1); solids [x, y, w, h, kind]
    contents = {}  # index in solids -> "coin" or "mushroom"

    def column(x0, tiles):
        top = FLOOR - tiles * TILE
        solids.append([x0, top, TILE, BOTTOM - top, "ground"])

    def block(x0, top, question):
        if question:
            contents[len(solids)] = "mushroom" if rng.random() < MUSHROOM_SHARE else "coin"
        solids.append([x0, top, TILE, TILE, "question_block" if question else "brick"])

    def row(x0, top, tiles):
        for k in range(tiles):
            block(x0 + k * TILE, top, rng.random() < QUESTION_SHARE)

    for _ in range(pieces):
        kind = rng.choices(list(PIECES), weights=list(PIECES.values()))[0]
        if kind == "pipe":
            top = FLOOR - rng.choice(PIPE_TILES) * TILE
            solids.append([x, top, 2 * TILE, BOTTOM - top, "pipe"])
            if rng.random() < PLANT_SHARE:
                enemies.append({"kind": "piranha_plant", "x": float(x + TILE // 2), "pipe_top": float(top),
                                "phase": rng.randrange(PLANT_CYCLE)})
            x += 2 * TILE
        elif kind == "cannon":
            top = FLOOR - rng.choice(CANNON_TILES) * TILE
            solids.append([x, top, TILE, BOTTOM - top, "cannon"])
            x += TILE
        elif kind == "pit":
            width = rng.choice(PIT_TILES) * TILE
            pits.append((x, x + width))
            x += width
        elif kind == "goombas":
            roll = rng.random()
            if roll < PARATROOPA_SHARE:
                enemies.append(_goomba(x, FLOOR, "paratroopa"))
            elif roll < PARATROOPA_SHARE + KOOPA_SHARE:
                enemies.append(_goomba(x, FLOOR, "koopa"))
            else:
                enemies.append(_goomba(x, FLOOR))
                if rng.random() < 0.5:
                    enemies.append(_goomba(x + PAIR_GAP, FLOOR))
            x += 3 * TILE
        elif kind == "questions":
            for k in range(rng.randint(1, 3)):
                block(x + 2 * k * TILE, LOW_ROW, True)
            x += 5 * TILE
        elif kind == "bricks":
            low = rng.randint(*LOW_TILES)
            row(x, LOW_ROW, low)
            end = x + low * TILE
            if rng.random() < 0.5:
                start = x + rng.randint(1, 2) * TILE
                high = rng.randint(*HIGH_TILES)
                row(start, HIGH_ROW, high)
                if rng.random() < 0.5:
                    enemies.append(_goomba(start + (high - 1) * TILE, HIGH_ROW))
                end = max(end, start + high * TILE)
            x = end
        else:  # stairs: up, a valley of floor or a pit, down
            height = rng.choice(STAIR_TILES)
            # Each side's top one or two columns wide (1-1's second pair:
            # two up, one down).
            up_top, down_top = rng.choice(STAIR_TOPS), rng.choice(STAIR_TOPS)
            for k in range(height + up_top - 1):
                column(x, min(k + 1, height))
                x += TILE
            gap = 2 * TILE
            if rng.random() < 0.5:
                pits.append((x, x + gap))
            x += gap
            for k in range(height + down_top - 1):
                column(x, min(height, height + down_top - 1 - k))
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
    cuts = sorted(pits + [(s[0], s[0] + s[2]) for s in solids if s[4] in ("pipe", "ground", "cannon")])
    floor, start = [], 0
    for x0, x1 in cuts:
        if x0 > start:
            floor.append([start, FLOOR, x0 - start, BOTTOM - FLOOR, "ground"])
        start = max(start, x1)
    floor.append([start, FLOOR, width - start, BOTTOM - FLOOR, "ground"])
    platforms = floor + solids
    big = rng.random() < BIG_START
    return {
        "mario_big": big,
        "block_contents": {len(floor) + k: c for k, c in contents.items()},
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


_KINDS = {"ground": "ground", "pipe": "pipe", "brick": "block", "question_block": "block", "moving_platform": "moving",
          "cannon": "ground"}  # (the game's labels call a cannon ground)


def enemy_kind(enemy: dict) -> str:
    """The planner's kind of a simulator enemy (smb_planner.ENEMY_KINDS)."""
    if enemy.get("kind") == "piranha_plant":
        return "plant"
    if enemy.get("kind") == "bullet":
        return "flier"
    return "hopper" if enemy.get("wings") else "walker"


def plant_velocity(plant: dict) -> float:
    """A plant's vertical speed by its cycle (pixels a frame; up negative)."""
    rise, exposed = plant["rise_frames"], plant["exposed_frames"]
    phase = plant["plant_tick"] % (2 * rise + exposed + plant["hidden_frames"])
    speed = plant["plant_height"] / rise
    return -speed if phase < rise else speed if rise + exposed <= phase < 2 * rise + exposed else 0.0
# A question block not yet hit (the game draws it with its question mark).
QUESTION = "question"


class BlockTruthState:
    """The controller's state from the simulator: what is on the screen,
    in world pixels, as TruthState reads the real game's memory."""

    def read(self, env) -> State:
        left, right = env.camera_x, env.camera_x + env.width
        solids = []
        for k, (p, kind) in enumerate(zip(env.platforms, env.platform_kinds)):
            r = p["rect"]
            if r.width == 0 or r.right <= left or r.left >= right:
                continue
            seen = QUESTION if p.get("contents") and not p.get("used") else _KINDS.get(kind, "block")
            solids.append(Solid(max(r.left, left), r.top, min(r.right, right), r.bottom, seen, ("platforms", k)))
        enemies = []
        for k, e in enumerate(env.enemies):
            awake = e.get("awake", not e.get("game_rules"))
            box = (e["x"], e["y"], e["x"] + e["w"], e["y"] + e["h"])
            if e["dead"] or e["h"] <= 0 or not awake or box[2] <= left or box[0] >= right:
                continue
            kind = enemy_kind(e)
            if kind == "plant":
                if e["exposed"] <= 0:
                    continue  # down in its pipe: not seen
                vy = plant_velocity(e)
            else:
                # (standing, its fall speed is gravity's step the floor stopped)
                vy = 0.0 if e["on_ground"] else e["vy"]
            enemies.append(Enemy(box, e["speed"] * e["direction"], vy, ("enemies", k),
                                 stompable=bool(e.get("stompable", True)), kind=kind))
        items = [Enemy((i["x"], i["y"], i["x"] + i["w"], i["y"] + i["h"]), i["vx"], i["vy"], ("items", k),
                       stompable=False)
                 for k, i in enumerate(env.items) if i["x"] + i["w"] > left and i["x"] < right]
        m = env.mario
        return State(
            body=(m["x"], m["y"], m["x"] + m["w"], m["y"] + m["h"]),
            motion=copy(env.motion),
            grounded=bool(m["on_ground"]),
            solids=solids + [screen_edge(left)],
            enemies=enemies,
            items=items,
        )


def play(scenario: dict, chooser, frames: Optional[int] = None) -> dict:
    """``chooser`` (a ScriptedChooser) and the controller through
    ``scenario``: how it ended ("finished", "died" or "timeout"), the frames
    taken, the furthest x reached and the score."""
    from .env import MarioScenarioEnv

    env = MarioScenarioEnv(world_width=scenario["world_width"])
    env.draws = False
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


def run_levels(seeds, modes=("speed_run", "max_points"), pieces: int = 12, make_chooser=None) -> list:
    """play() a chooser (``make_chooser(mode)``; the scripted chooser by
    default) through random_level(seed) for each seed and mode: one row per
    run (seed, mode and play's result)."""
    if make_chooser is None:
        from retroagi.stages.full_smb.scripted_chooser import ScriptedChooser as make_chooser

    rows = []
    for seed in seeds:
        level = random_level(seed, pieces)
        for mode in modes:
            rows.append({"seed": seed, "mode": mode, **play(level, make_chooser(mode))})
    return rows


def main(argv=None) -> int:
    import argparse
    import json
    from collections import Counter

    parser = argparse.ArgumentParser(description="A chooser on random levels of the game's pieces")
    parser.add_argument("--model", default=None,
                        help="an outcome network (outcome_train.py) to choose by the three rules; "
                             "the scripted chooser if not given")
    parser.add_argument("--levels", type=int, default=100)
    parser.add_argument("--first-seed", type=int, default=0)
    parser.add_argument("--pieces", type=int, default=12)
    parser.add_argument("--output", default=None)
    args = parser.parse_args(argv)
    make_chooser = None
    if args.model:
        from retroagi.core.devices import select_device
        from retroagi.core.outcome_chooser import OutcomeChooser

        from .outcome_train import load_model

        device = select_device("auto")
        model = load_model(args.model, device)

        def make_chooser(mode):
            return OutcomeChooser(model, mode, device)

    rows = run_levels(range(args.first_seed, args.first_seed + args.levels), pieces=args.pieces,
                      make_chooser=make_chooser)
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
