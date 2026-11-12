"""A real level copied into Block SMB from the game's memory: its layout
(floor, pits, pipes, steps, bricks and question blocks) and where each
enemy starts, read while a random player sweeps the level (as
obstacles.sweep does), then written as a Block SMB level (the format of
sim/levels.random_level) the simulator plays under the game's rules.

Copying the look and behaviour of the game is how Block SMB is built; the
policy never learns in the game itself. Only levels set aside for building
Block SMB are copied (the others are kept to score the finished agent).

What the copy leaves out is said in its "left_out": enemies of kinds Block
SMB does not have yet (a Hammer Bro), and what a question block held.

    python -m retroagi.domains.smb.nes.clone --level Level5-1 [--output data/block_smb_levels] [--seeds 1 2 3]

``--check`` tells how true a copy is (fidelity): the level is played in the
game, and every few frames Block SMB's Mario is put where the game's is,
moving as he moves, in the copy without its enemies; the same buttons are
pressed for AHEAD frames, and where he ends is compared with where the
game's Mario did. Where they part, the copy's layout or the simulator's
motion differs from the game's there.

    python -m retroagi.domains.smb.nes.clone --level Level5-1 --check
"""

import argparse
import json
from collections import Counter
from copy import copy
from pathlib import Path

from retroagi.domains.smb.perception.scene_labels import scene_from_labels
from retroagi.domains.smb.sim.levels import (
    BOTTOM,
    END_FLAT,
    FLOOR,
    FRAMES_PER_PIXEL,
    GOOMBA_FEET,
    SCROLL_X,
    START_X,
    TILE,
)

from .obstacles import EDGE, RAISED, SCREEN_HEIGHT, SCREEN_WIDTH, _on_the_ground
from .pixel_labels import Unexplained, label_frame, state_blocks
from .truth import KIND_NAMES, true_frame
from .vision_frames import played_frames

# The game's enemy codes Block SMB has, as the simulator's enemy kinds
# (None: a Goomba, its plain walker).
KINDS = {0x06: None, 0x00: "koopa", 0x0E: "paratroopa", 0x0D: "piranha_plant"}
BULLET = 0x33
# A tile with air under it is part of the level when it was seen in at
# least this many frames at a top on the game's grid: a lift passing is
# seen a few frames at each height, and is not copied (Block SMB's copies
# have no lifts yet).
FLOATING_SEEN = 30


def _tile(x: float) -> int:
    """``x`` at the game's 16-pixel grid."""
    return int(round(x / TILE)) * TILE


def capture(level: str, seeds=(1, 2, 3), frames: int = 60_000, every: int = 4) -> dict:
    """What ``level`` holds, in level pixels, from sweeps by the random
    player: {"pipes": [(x0, x1, top)], "pits": [(x0, x1)], "steps": [(x0,
    x1, top)] (columns standing on the floor), "floating": [(x0, x1, top)]
    (solid tiles with air under them that are neither bricks nor question
    blocks), "blocks": [(x0, top, kind)], "enemies": [(x, bottom, kind
    code)] where each was first seen, "finish": the flag's x or None,
    "seen_to": the furthest x seen}."""
    pipes, pits, steps, floating, blocks, enemies = Counter(), Counter(), Counter(), Counter(), {}, {}
    finish, furthest = None, 0
    for seed in seeds:
        start, awake = None, [False] * 6
        for frame, before, after, earlier in played_frames(level, frames=frames, seed=seed, every=every):
            if earlier is None:
                awake = [False] * 6  # (rewound: the slots awake now count as just woken)
            ram = state_blocks(after)["RAM"]
            if start is None:
                start = (int(ram[0x75F]), int(ram[0x75C]))
            truth = true_frame(ram, start)
            if truth["finished"]:
                finish = finish or truth["level_x"]
                continue
            furthest = max(furthest, truth["level_x"])
            camera = truth["camera"]
            for k, slot in enumerate(truth["slots"]):
                woke, awake[k] = slot["active"] and not awake[k], slot["active"]
                box = slot["box"]
                seen = box is not None and box[2] > box[0] and box[3] <= SCREEN_HEIGHT
                if not (seen and slot["active"] and slot["enemy"] and not slot["defeated"]):
                    awake[k] = awake[k] and seen  # (not yet drawn: it has not been seen waking)
                    continue
                if woke and box[0] > truth["mario"][2]:
                    x = round(box[0])
                    if not any(abs(x - other) <= 16 and kind == slot["kind"] for other, (_, kind) in enemies.items()):
                        enemies[x] = (round(box[3]), slot["kind"])
            try:
                labelled = label_frame(frame, before, after, family=level)
            except Unexplained:
                continue
            scene = scene_from_labels(labelled.scene)

            def whole(x0, x1):
                return EDGE < x0 and x1 < SCREEN_WIDTH - EDGE

            for p in scene.pipes:
                if whole(p[0], p[2]):
                    pipes[_tile(p[0] + camera), _tile(p[2] + camera), int(p[1])] += 1
            for g in scene.gaps:
                if whole(g.x0, g.x1):
                    pits[_tile(g.x0 + camera), _tile(g.x1 + camera)] += 1
            for s in scene.surfaces:
                on_pipe = any(s.x0 < p[2] and p[0] < s.x1 for p in scene.pipes)
                if not (whole(s.x0, s.x1) and not s.moving and s.top <= FLOOR - RAISED and not on_pipe):
                    continue
                grounded = _on_the_ground(labelled.labels, s)
                for x in range(_tile(s.x0 + camera), _tile(s.x1 + camera), TILE):
                    # (a brick's or a question block's top is that block's own)
                    on_block = any(b.box[0] + camera - 4 <= x <= b.box[2] + camera - 12 and abs(b.box[1] - s.top) <= 2
                                   for b in scene.blocks)
                    if grounded:
                        steps[x, x + TILE, int(s.top)] += 1
                    elif not on_block:
                        floating[x, x + TILE, int(s.top)] += 1
            for b in scene.blocks:
                # (blocks side by side are labelled as one box: each tile of it that is whole in the picture)
                for x in range(_tile(b.box[0] + camera), _tile(b.box[2] + camera), TILE):
                    if not whole(x - camera, x + TILE - camera):
                        continue
                    key = (x, int(b.box[1]))
                    # (a question block hit becomes a plain one: what it was first is what it is)
                    if blocks.get(key) != "question_block":
                        blocks[key] = b.kind
    def often(counts):
        # (seen in at least a few frames: one frame's mislabel is not a thing)
        return sorted(key for key, n in counts.items() if n >= 3)

    return {
        "level": level,
        "pipes": often(pipes), "pits": often(pits), "steps": often(steps),
        "floating": sorted(key for key, n in floating.items() if n >= FLOATING_SEEN and key[2] % TILE == 0),
        "blocks": sorted((x, top, kind) for (x, top), kind in blocks.items()),
        "enemies": sorted((x, bottom, kind) for x, (bottom, kind) in enemies.items()),
        "finish": finish, "seen_to": furthest,
    }


def scenario(captured: dict) -> dict:
    """The Block SMB level of a captured one (sim/levels.random_level's
    format): the floor cut at its pits and where pipes and steps stand,
    its pipes, steps, bricks and question blocks, and its enemies of the
    kinds Block SMB has, under the game's rules. "left_out" in its metadata
    counts what could not be copied."""
    end = captured["finish"] or captured["seen_to"]
    width = end + END_FLAT * TILE
    solids, left_out = [], Counter()
    for x0, x1, top in captured["pipes"]:
        solids.append([x0, top, x1 - x0, BOTTOM - top, "pipe"])
    tops = {}
    for x0, x1, top in captured["steps"]:
        tops[x0] = min(top, tops.get(x0, FLOOR))  # (the column's highest top)
    for x0, top in sorted(tops.items()):
        if not any(p[0] <= x0 < p[0] + p[2] for p in solids):
            solids.append([x0, top, TILE, BOTTOM - top, "ground"])
    cuts = sorted([tuple(p) for p in captured["pits"]] + [(s[0], s[0] + s[2]) for s in solids])
    floor, start = [], 0
    for x0, x1 in cuts:
        if x0 > start:
            floor.append([start, FLOOR, x0 - start, BOTTOM - FLOOR, "ground"])
        start = max(start, x1)
    floor.append([start, FLOOR, width - start, BOTTOM - FLOOR, "ground"])
    blocks = [[x, top, TILE, TILE, kind] for x, top, kind in captured["blocks"]]
    taken = {(b[0], b[1]) for b in blocks}
    blocks += [[x0, top, TILE, TILE, "ground"] for x0, _, top in captured.get("floating", ())
               if (x0, top) not in taken]
    left_out["what question blocks hold (each given a coin)"] = sum(b[4] == "question_block" for b in blocks)
    platforms = floor + solids + blocks
    contents = {k: "coin" for k, p in enumerate(platforms) if p[4] == "question_block"}
    enemies = []
    for x, bottom, kind in captured["enemies"]:
        if kind == BULLET:
            # A Bullet Bill comes from a cannon: the step it was first seen at is one.
            cannon = min((p for p in platforms if p[4] == "ground" and p[1] < FLOOR),
                         key=lambda p: abs(p[0] - x), default=None)
            if cannon is not None and abs(cannon[0] - x) <= 2 * TILE:
                cannon[4] = "cannon"
            else:
                left_out["Bullet Bill with no cannon found"] += 1
        elif kind not in KINDS:
            left_out[KIND_NAMES.get(kind, f"object {kind:02X}")] += 1
        elif KINDS[kind] == "piranha_plant":
            pipe = min((p for p in solids if p[4] == "pipe"), key=lambda p: abs(p[0] + p[2] / 2 - x), default=None)
            if pipe is None or abs(pipe[0] + pipe[2] / 2 - x) > 2 * TILE:
                left_out["Piranha Plant with no pipe found"] += 1
            else:
                enemies.append({"kind": "piranha_plant", "x": float(pipe[0] + TILE // 2), "pipe_top": float(pipe[1]),
                                "phase": 0})
        else:
            # On what it stood when first seen: the top nearest under its box.
            under = [p[1] for p in platforms if p[0] <= x + 6 <= p[0] + p[2] and p[1] >= bottom - 2]
            top = min(under, default=FLOOR)
            enemies.append({"x": float(x), "y": float(top - GOOMBA_FEET), "game_rules": True,
                            **({"kind": KINDS[kind]} if KINDS[kind] else {})})
    return {
        "mario_big": False,
        "block_contents": contents,
        "task": {"family": "game_level"},
        "world_width": width,
        "scroll_x": SCROLL_X,
        "frame_budget": width * FRAMES_PER_PIXEL,
        "mario": [START_X, FLOOR - 12],
        "platforms": [p[:4] for p in platforms],
        "platform_kinds": [p[4] for p in platforms],
        "enemies": enemies,
        "goal": [end, 0, TILE, BOTTOM],
        "metadata": {"game_level": {"copied_from": captured["level"], "left_out": dict(left_out)}},
    }


# The fidelity check: frames the copy is run on from each start, how often a
# start is taken, and how near its Mario must end to the game's (pixels).
AHEAD = 16
EVERY = 8
NEAR = 2


def fidelity(level: str, copied: dict, frames: int = 900, states=(None,)) -> dict:
    """How the copy ``copied`` (scenario's answer) follows the game on
    ``level`` (module notes), over a play of ``frames`` frames from each of
    the game's saved ``states`` (None: the level's start): {"starts", "near"
    (the share ending within NEAR pixels of the game's Mario), "off"
    [(level x, pixels off)] for the starts that did not}. Starts with an
    enemy near the game's Mario are left out (the copy is run without
    enemies), and those where he is big, dying or done."""
    from retroagi.domains.smb.physics import NESPlayerMotion
    from retroagi.domains.smb.sim.world import SimWorld
    from retroagi.framework import registry
    from retroagi.framework.agent import Agent

    from .pixel_labels import PLAYER_FLOAT_STATE

    domain = registry.load("smb")
    agent = Agent(domain.source("memory", level), domain.behaviour, domain.controller, domain.idle)
    sim = SimWorld({**copied, "enemies": [], "mario_big": False})
    sim.reset()
    env = sim.env
    starts, off = 0, []
    for state in states:
        world = domain.real_world(level, state)
        world.reset()
        agent.reset(0, domain.modes[0])
        played = []  # per frame: the button, Mario's box, his motion, whether he stands, the camera, left out
        for _ in range(frames):
            ram = world.game.env.get_ram()
            truth = true_frame(ram, world.game.start)
            box = truth["mario"]
            near = any(s["active"] and s["box"] and not s["defeated"] and abs(s["box"][0] - box[0]) < 48
                       for s in truth["slots"])
            action = agent.act(world)
            played.append((action, box, NESPlayerMotion.from_ram(ram, top=box[1]),
                           int(ram[PLAYER_FLOAT_STATE]) == 0, truth["camera"],
                           near or truth["dying"] or truth["finished"] or box[3] - box[1] > 14))
            if world.step(action).over:
                break
        world.close()
        for t in range(0, len(played) - AHEAD, EVERY):
            if any(frame[5] for frame in played[t:t + AHEAD + 1]):
                continue
            _, box, motion, stands, camera, _ = played[t]
            env.mario.update(x=float(box[0]), y=float(box[1]), on_ground=stands)
            env.motion = copy(motion)
            env.camera_x = float(camera)
            over = False
            for k in range(AHEAD):
                over = over or sim.step(played[t + k][0]).over
            there = played[t + AHEAD][1]
            error = 99.0 if over else max(abs(env.mario["x"] - there[0]), abs(env.mario["y"] - there[1]))
            if over:
                sim.reset()
            starts += 1
            if error > NEAR:
                off.append((int(box[0]), round(float(error), 1)))
    return {"starts": starts, "near": 1 - len(off) / max(1, starts), "off": off}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--level", required=True, help="a saved level start, e.g. Level5-1")
    parser.add_argument("--output", type=Path, default=Path("data/block_smb_levels"))
    parser.add_argument("--seeds", type=int, nargs="*", default=(1, 2, 3), help="one sweep of the level per seed")
    parser.add_argument("--check", action="store_true", help="tell how true the copy already made is")
    args = parser.parse_args(argv)
    if args.check:
        # From every saved state of the level's obstacle bench, if it has one: the whole level's length.
        bench = Path("data/full_smb_obstacles") / args.level
        states = sorted({p.read_bytes() for p in bench.glob("*.state")}) or (None,)
        found = fidelity(args.level, json.loads((args.output / f"{args.level}.json").read_text()), states=states)
        print(f"{args.level}: of {found['starts']} starts the copy's Mario ends within {NEAR} pixels of the game's "
              f"in {found['near']:.3f}")
        places = Counter(x // 64 * 64 for x, _ in found["off"])
        if places:
            print("  off where (level x, starts): " + ", ".join(f"{x} ({n})" for x, n in sorted(places.items())))
        return 0
    captured = capture(args.level, tuple(args.seeds))
    copy_ = scenario(captured)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / f"{args.level}.captured.json").write_text(json.dumps(captured, indent=1) + "\n")
    (args.output / f"{args.level}.json").write_text(json.dumps(copy_, indent=1) + "\n")
    kinds = Counter(copy_["platform_kinds"])
    print(f"{args.level}: seen to x {captured['seen_to']}, flag at {captured['finish']}; {kinds['pipe']} pipes, "
          f"{len(captured['pits'])} pits, {kinds['ground']} floor pieces and steps, {kinds['brick']} bricks, "
          f"{kinds['question_block']} question blocks, {kinds['cannon']} cannons, {len(copy_['enemies'])} enemies; "
          f"left out: {copy_['metadata']['game_level']['left_out'] or 'nothing'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
