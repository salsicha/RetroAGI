"""A level's obstacles and a saved game state just before each (the obstacle
test bench, docs/choose-and-control-plan.md phase 0; user, 2026-11-07).

One long sweep of the level by the vision frames' random player
(vision_frames.played_frames: it rewinds after deaths, so it reaches the
whole level) both maps the level and keeps states:

- the map, from the exact labels of game memory (pixel_labels) and the
  game's own enemy slots (truth.true_frame), in level pixels: every pipe,
  every pit, every step (a raised standable column on the ground, not a
  floating block), and where each enemy is first seen when its slot wakes
  (it wakes when the screen reaches it: the game's wake-up rule) with where
  Mario was then; the x where the level is finished;
- a state every BUCKET pixels of Mario's progress, the first where he
  stands still on something (no speed: a player that sees only the screen
  cannot know a speed it has not watched) and is not dying.

The obstacles (obstacles_of): each pipe, each pit, each group of enemies
woken together, each run of steps, and the flag. Each starts from the kept
state at its approach point: APPROACH pixels before a pipe, a pit, steps or
the flag; for enemies, just before Mario's scroll wakes the first of them.
A start is usable (usable) when Mario stands through STILL_FRAMES frames of
standing still and no enemy is within CLEAR pixels of him; otherwise the
nearest usable kept state before it is taken (memory can read "standing" a
frame before a fall, and the enemies of one group are still about at the
next one's start).
It is passed when Mario stands PASSED pixels beyond its end (obstacle_bench).
"""

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

from retroagi.core.smb_pixel_types import TYPE_ID
from retroagi.core.smb_scene_labels import scene_from_labels

from .pixel_labels import PLAYER_FLOAT_STATE, Unexplained, label_frame, state_blocks

# [Player_X_Speed]
PLAYER_X_SPEED = 0x57
from .truth import true_frame
from .vision_frames import _dying, played_frames

# Pixels of Mario's progress between kept states.
BUCKET = 16
# How far before a pipe, a pit, steps or the flag an obstacle's state is.
APPROACH = 96
# How far before the scroll wakes an enemy group its state is.
WAKE_LEAD = 16
# Enemies first seen this close together are one obstacle.
ENEMY_GROUP = 48
# One enemy woken again after a rewind: first seen within this of before.
SAME_ENEMY = 16
# Pieces of one pipe, pit or step seen in different frames: within this.
SAME = 3
# A surface this far above the floor's top or more is raised (pixels).
RAISED = 8
FLOOR_TOP = 208
# Objects cut by the picture's edge are not measured (their ends are unseen).
EDGE = 8
SCREEN_WIDTH, SCREEN_HEIGHT = 256, 240
# A start's check: frames of standing still, and how near an enemy may be.
STILL_FRAMES = 8
CLEAR = 64


@dataclass
class Obstacle:
    """One obstacle: what it is, its span in level pixels, and the level x
    where Mario's saved state starts."""

    name: str
    kind: str
    x0: int
    x1: int
    approach_x: int


@dataclass
class LevelMap:
    pipes: list
    pits: list
    steps: list
    enemies: list  # (first seen level x, Mario's level x then, kind)
    finish_x: Optional[int]
    furthest_x: int


def _merged(spans) -> list:
    """Spans (x0, x1, ...) seen in many frames, each once."""
    out = []
    for span in sorted(set(spans)):
        if out and abs(span[0] - out[-1][0]) <= SAME and abs(span[1] - out[-1][1]) <= SAME:
            continue
        out.append(span)
    return out


def _inside(x0, x1) -> bool:
    return EDGE < x0 and x1 < SCREEN_WIDTH - EDGE


def sweep(level: str, *, seed: int = 1, frames: int = 60_000, every: int = 4):
    """(LevelMap, kept states {bucket: state bytes}) from one long play of
    ``level`` by the random player; frames memory cannot explain are skipped
    for the map."""
    pipes, pits, steps, enemies = [], [], [], {}
    # Per enemy slot: awake last time, woken but not yet seen on screen, and
    # where its enemy was last seen (a slot handed straight to another enemy
    # between sampled frames shows as a jump).
    awake, waking, last_x = [False] * 6, [False] * 6, [None] * 6
    states: dict = {}
    furthest, finish = 0, None
    start = None
    for frame, before, after, earlier in played_frames(level, frames=frames, seed=seed, every=every):
        if earlier is None:
            # Rewound after a death (or the start): slots awake in the
            # restored state count as just woken.
            awake, last_x = [False] * 6, [None] * 6
        ram = state_blocks(after)["RAM"]
        if start is None:
            start = (int(ram[0x75F]), int(ram[0x75C]))
        truth = true_frame(ram, start)
        if truth["finished"]:
            finish = finish or truth["level_x"]
            continue
        furthest = max(furthest, truth["level_x"])
        bucket = truth["level_x"] // BUCKET
        resting = int(ram[PLAYER_FLOAT_STATE]) == 0 and int(ram[PLAYER_X_SPEED]) == 0
        if bucket not in states and resting and not _dying(ram):
            states[bucket] = before
        camera = truth["camera"]
        for k, slot in enumerate(truth["slots"]):
            if slot["active"] and not awake[k]:
                waking[k] = True
            awake[k] = slot["active"]
            if not slot["active"]:
                last_x[k] = None
            box = slot["box"]
            # A slot just woken can report a placeholder box (no width, below
            # the screen) before its enemy is drawn: not a sighting yet.
            seen_now = box is not None and box[2] > box[0] and box[3] <= SCREEN_HEIGHT
            if not (seen_now and slot["active"] and slot["enemy"] and not slot["defeated"]):
                continue
            x = round(box[0])
            jumped = last_x[k] is not None and abs(x - last_x[k]) > ENEMY_GROUP
            last_x[k] = x
            if waking[k] or jumped:
                waking[k] = False
                ahead = x > truth["mario"][2]
                if ahead and not any(abs(x - seen) <= SAME_ENEMY for seen in enemies):
                    enemies[x] = (x, truth["level_x"], slot["kind"])
        try:
            labelled = label_frame(frame, before, after, family=level)
        except Unexplained:
            continue
        scene = scene_from_labels(labelled.scene)
        for p in scene.pipes:
            if _inside(p[0], p[2]):
                pipes.append((round(p[0] + camera), round(p[2] + camera), int(p[1])))
        for g in scene.gaps:
            if _inside(g.x0, g.x1):
                pits.append((round(g.x0 + camera), round(g.x1 + camera)))
        for s in scene.surfaces:
            on_pipe = any(s.x0 < p[2] and p[0] < s.x1 for p in scene.pipes)
            if (
                _inside(s.x0, s.x1)
                and not s.moving
                and s.top <= FLOOR_TOP - RAISED
                and not on_pipe
                and _on_the_ground(labelled.labels, s)
            ):
                steps.append((round(s.x0 + camera), round(s.x1 + camera), int(s.top)))
    return (
        LevelMap(_merged(pipes), _merged(pits), _merged(steps), sorted(enemies.values()), finish, furthest),
        states,
    )


def _on_the_ground(labels, surface) -> bool:
    """Whether ``surface`` tops a column standing on the floor (a step): no
    background between its top and the floor's top in its middle column; a
    floating block has sky under it."""
    middle = (surface.x0 + surface.x1) // 2
    column = labels[surface.top + 1 : FLOOR_TOP, middle]
    return bool(len(column)) and not (column == TYPE_ID["background"]).any()


def _runs(spans, gap: int) -> list:
    """Spans joined where they lie within ``gap`` pixels of each other."""
    out = []
    for x0, x1, *_ in sorted(spans):
        if out and x0 <= out[-1][1] + gap:
            out[-1][1] = max(out[-1][1], x1)
        else:
            out.append([x0, x1])
    return out


def sweeps(level: str, seeds=(1, 2, 3), **kwargs):
    """sweep with each of ``seeds``: the first's map, and per bucket the first
    seed's kept state that has one (resting states are fewer than standing
    ones: one sweep can leave stretches of the level without any)."""
    level_map, states = None, {}
    for seed in seeds:
        found, kept = sweep(level, seed=seed, **kwargs)
        level_map = level_map or found
        for bucket, state in kept.items():
            states.setdefault(bucket, state)
    return level_map, states


def obstacles_of(level_map: LevelMap) -> list:
    """The level's obstacles in order of level x."""
    found = [Obstacle("pipe", "pipe", x0, x1, x0 - APPROACH) for x0, x1, _ in level_map.pipes]
    found += [Obstacle("pit", "pit", x0, x1, x0 - APPROACH) for x0, x1 in _runs(level_map.pits, SAME)]
    found += [Obstacle("steps", "steps", x0, x1, x0 - APPROACH) for x0, x1 in _runs(level_map.steps, BUCKET)]
    groups: list = []
    for x, mario_x, kind in level_map.enemies:
        if groups and x - groups[-1][-1][0] <= ENEMY_GROUP:
            groups[-1].append((x, mario_x, kind))
        else:
            groups.append([(x, mario_x, kind)])
    for group in groups:
        xs = [x for x, _, _ in group]
        found.append(Obstacle("enemies", "enemies", min(xs), max(xs), min(m for _, m, _ in group) - WAKE_LEAD))
    if level_map.finish_x is not None:
        found = [o for o in found if o.x0 < level_map.finish_x]
        found.append(Obstacle("flag", "flag", level_map.finish_x, level_map.finish_x, level_map.finish_x - APPROACH))
    found.sort(key=lambda o: (o.approach_x, o.x0))
    for k, o in enumerate(found):
        o.name = f"{k:02d}_{o.kind}_{o.x0}"
    return found


def usable(level: str, state: bytes) -> bool:
    """Whether Mario, started from ``state``, stands still for STILL_FRAMES
    frames alive and standing, with no enemy within CLEAR pixels."""
    from .play import FullSMBGame

    game = FullSMBGame(level, state)
    try:
        game.reset()
        for frame in range(STILL_FRAMES + 1):
            ram = game.env.get_ram()
            truth = true_frame(ram, game.start)
            mario = truth["mario"]
            if _dying(ram) or int(ram[PLAYER_FLOAT_STATE]) != 0 or mario[3] > FLOOR_TOP:
                return False
            for slot in truth["slots"]:
                box = slot["box"]
                if slot["active"] and slot["enemy"] and box and not slot["defeated"] and box[2] > box[0]:
                    if box[0] < mario[2] + CLEAR and mario[0] < box[2] + CLEAR:
                        return False
            if frame < STILL_FRAMES:
                game.step(0)
        return True
    finally:
        game.close()


def state_for(states: dict, approach_x: int, check=None) -> Optional[tuple]:
    """(bucket, state) kept nearest before ``approach_x`` that ``check``
    (a state's test, e.g. usable) passes; None if none does."""
    for bucket in sorted((b for b in states if b * BUCKET <= approach_x), reverse=True):
        if check is None or check(states[bucket]):
            return bucket, states[bucket]
    return None


def save_bench(level: str, level_map: LevelMap, states: dict, folder: Path) -> list:
    """Write each obstacle's state (<name>.state) and index.json in
    ``folder``; returns the obstacles saved (those with a state before them)."""
    folder.mkdir(parents=True, exist_ok=True)
    saved = []
    for o in obstacles_of(level_map):
        kept = state_for(states, max(0, o.approach_x), check=lambda state: usable(level, state))
        if kept is None:
            continue
        bucket, state = kept
        (folder / f"{o.name}.state").write_bytes(state)
        saved.append({**asdict(o), "start_x": bucket * BUCKET})
    index = {"level": level, "obstacles": saved, "map": asdict(level_map)}
    (folder / "index.json").write_text(json.dumps(index, indent=1) + "\n")
    return saved
