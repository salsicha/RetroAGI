"""Pictures of each family's layouts, to check the teaching by eye.

A family's layouts are made with every drawn parameter at the same few fixed
points of its range (QuantileDraws: by default its lowest, middle and highest
value), at each difficulty; one picture a layout. Each picture shows:

- the layout's ground, pipes, blocks, coins and goal;
- every place each enemy and moving platform occupies while the episode can
  last, by the simulator's own update from the start of the watched frames (a
  blur: the longer it stays somewhere, the darker);
- the path of the teacher's Mario, played through the training pipeline
  (vision, executor; teacher_share 1, labelled);
- a heat spot where Mario's feet are at each decision of the skill: orange on
  the ground, teal on an enemy he has just stomped, magenta anywhere else in
  the air, with a black cross where the teacher's choice names no action and
  an arrow where it is a run-up that keeps its speed into the jump after it;
- each decision's number and the action taught there (the same action
  decided again where Mario stands still, as holds are, numbered once).

A lesson won by stomping its enemy shows no goal box. Nor does a hold with
walkers coming (a ``dodge`` hold: anywhere on the floor at the end wins);
there each jump over a walker shows the walker's trigger box instead: a
buffer centred on Mario, as wide as the gap to the walker when he jumped
(its trigger distance at its speed; a colour for each walker).

atlas.json also lists, per layout, the moves the teacher passed over for
failing when started a little later (controller_teacher._tolerant). The
pictures are also tiled one sheet a family (sheets/), a row a difficulty and a
column a quantile. The networks run on the GPU where there is one.

With ``--policy CHECKPOINT`` the atlas shows a trained skill instead: its own
play on its run's validation layouts (the teacher carrying out its actions,
as in training), each decision green where it chose the teacher's action at
that moment and red where it chose another (both listed), and a cross where a
lost episode ended.

Training-only diagnostic: nothing learns from it.

    python -m retroagi.stages.block_smb.family_atlas --out artifacts/block_smb/family_atlas
"""

import argparse
import json
import math
from pathlib import Path
from typing import Optional

import numpy as np

from .monte_carlo import (
    BLOCK_SMB_MC_FAMILIES,
    BLOCK_SMB_MC_ID,
    CombinationDraws,
    _frame_budget,
    _generate_family_scenario,
    _verified_route,
    _with_sample_metadata,
    block_smb_monte_carlo_family_specs,
)

QUANTILES = (0.0, 0.5, 1.0)
DIFFICULTIES = ("easy", "medium", "hard")
SCALE = 2
# Frames of enemy and moving-platform motion blurred (at least; or the
# layout's whole frame budget).
BLUR_FRAMES = 600

BACKGROUND = (247, 247, 244)
GROUND = (138, 143, 152)
PIPE = (60, 154, 74)
BLOCK = (176, 122, 60)
COIN = (230, 194, 41)
GOAL = (46, 139, 87)
LIFT = (51, 102, 204)
ENEMY = {"walker": (211, 51, 51), "piranha_plant": (150, 60, 190), "monster": (120, 20, 20)}
PATH = (34, 34, 34)
ON_GROUND = (255, 140, 0)
ON_ENEMY = (0, 160, 150)
IN_AIR = (220, 0, 220)
# A dodge hold's trigger box, one colour for each walker.
TRIGGER = ((46, 139, 87), (40, 90, 200), (190, 110, 0))
# A trained skill's decision (policy_atlas): the teacher's action at that
# moment, or another.
AGREES = (34, 160, 70)
DIFFERS = (220, 30, 30)


class QuantileDraws(CombinationDraws):
    """Stands in for random.Random in a family's layout generator: every
    draw takes the value at the same fixed point ``q`` of its values (0 the
    first, 1 the last; an option or an order by its place in the list)."""

    def __init__(self, q: float):
        super().__init__()
        self.q = q

    def _pick(self, values):
        index = round(self.q * (len(values) - 1))
        self.taken.append(index)
        self.counts.append(len(values))
        return values[index]


def atlas_layout(family: str, difficulty: str, q: float):
    """(the layout with every draw at ``q``, None) when its route is verified
    as the sampler verifies one; else (None, why it was rejected)."""
    draws = QuantileDraws(q)
    scenario, parameters, actions = _generate_family_scenario(
        family, draws, split="train", difficulty=difficulty
    )
    route, source, reachability = _verified_route(family, scenario, actions)
    if route is None:
        return None, str(source)
    oracle = {
        "kind": "scripted_action_sequence",
        "actions": list(route[: _frame_budget(scenario)]),
        "action_source": f"{family}:{source}",
    }
    return (
        _with_sample_metadata(
            scenario,
            family=family,
            split="train",
            seed=0,
            sample_seed=0,
            sample_index=0,
            scenario_id=f"{BLOCK_SMB_MC_ID}.atlas.{family}.{difficulty}.q{q:.2f}",
            parameters={**dict(parameters), "difficulty_bin": difficulty, "quantile": q},
            constraints=block_smb_monte_carlo_family_specs()[family].constraints,
            oracle=oracle,
            reachability=reachability,
        ),
        None,
    )


def occupancy(scenario: dict, frames: int):
    """How many of ``frames`` frames each world pixel is covered by an enemy
    (by kind) and by a moving platform, from the start of the watched frames,
    by the simulator's own update with Mario standing where he starts:
    ({kind: [H, W] counts}, [H, W] counts), and the world's size."""
    from .env import MarioScenarioEnv

    env = MarioScenarioEnv()
    env.reset(scenario={**scenario, "watch_frames": 0})
    width, height = int(env.world_width), int(env.height)
    enemies: dict[str, np.ndarray] = {}
    lifts = np.zeros((height, width), np.float32)

    def cover(grid, x0, y0, x1, y1):
        x0, x1 = max(0, int(math.floor(x0))), min(width, int(math.ceil(x1)))
        y0, y1 = max(0, int(math.floor(y0))), min(height, int(math.ceil(y1)))
        if x1 > x0 and y1 > y0:
            grid[y0:y1, x0:x1] += 1

    rects = [p["rect"] for p in env.platforms]
    try:
        for _ in range(frames):
            for e in env.enemies:
                if e["dead"] or e["h"] <= 0:
                    continue
                grid = enemies.setdefault(
                    e.get("kind") or "walker", np.zeros((height, width), np.float32)
                )
                # As drawn (env.enemy_screen_rects): the body and its feet.
                foot = e.get("foot_offset", 0)
                cover(grid, e["x"], e["y"], e["x"] + e["w"], e["y"] + e["h"] + foot)
                env._update_enemy(e, rects)
            for plat in env.platforms:
                if not plat.get("moving"):
                    continue
                r = plat["rect"]
                cover(lifts, r.left, r.top, r.right, r.bottom)
                plat["move_x"] += plat["move_speed"] * plat["move_dir"]
                if plat["move_x"] <= plat["move_min"]:
                    plat["move_x"], plat["move_dir"] = float(plat["move_min"]), 1
                elif plat["move_x"] >= plat["move_max"]:
                    plat["move_x"], plat["move_dir"] = float(plat["move_max"]), -1
                r.x = int(round(plat["move_x"]))
        statics = [
            (kind, (p["rect"].left, p["rect"].top, p["rect"].right, p["rect"].bottom))
            for p, kind in zip(env.platforms, env.platform_kinds)
            if not p.get("moving")
        ]
        coins = [(c["rect"].left, c["rect"].top, c["rect"].right, c["rect"].bottom) for c in env.coins]
        goal = (env.goal.left, env.goal.top, env.goal.right, env.goal.bottom) if env.goal else None
        if any(seg.get("dodge") for seg in scenario.get("tactics") or ()):
            goal = None  # anywhere on the floor; the trigger boxes are drawn
    finally:
        env.close()
    return {
        "enemies": enemies,
        "lifts": lifts,
        "statics": statics,
        "coins": coins,
        "goal": goal,
        "size": (width, height),
    }


def trigger(env, command):
    """For a jump while holding with ``dodge`` (tactic_schedule): (gap,
    enemy) for the walker it jumps over (the nearest coming at Mario): how
    far its box is from his, his trigger distance for it; else None."""
    from . import tactic_schedule

    if command is None or command.mode != "jump" or not tactic_schedule.current(env).get("dodge"):
        return None
    m, found = env.mario, []
    for i, e in enumerate(env.enemies):
        if e["dead"] or e["h"] <= 0:
            continue
        if e["x"] >= m["x"] + m["w"]:
            gap, coming = e["x"] - (m["x"] + m["w"]), e["direction"] < 0
        elif e["x"] + e["w"] <= m["x"]:
            gap, coming = m["x"] - (e["x"] + e["w"]), e["direction"] > 0
        else:
            gap, coming = 0.0, True
        if coming:
            found.append((round(gap, 1), i))
    return min(found) if found else None


def on_enemy(env) -> bool:
    """Whether Mario has just stomped an enemy: the simulator resolved it this
    frame, or his feet are on an enemy's top (vision sees the landing a frame
    or two before the simulator kills it, and the action ends there)."""
    if env.stomped:
        return True
    m = env.mario
    feet = m["y"] + m["h"]
    return any(
        not e["dead"]
        and e["h"] > 0
        and -2 <= feet - e["y"] <= 6
        and m["x"] < e["x"] + e["w"]
        and e["x"] < m["x"] + m["w"]
        for e in env.enemies
    )


def play_teacher(scenario: dict, family: str, difficulty: str):
    """The teacher's episode on ``scenario`` through the training pipeline:
    Mario's path (world middle x, feet y per frame), and at each decision of
    the skill where Mario is, whether he stands, has just stomped or is in the
    air, and the action taught there (after run-ups are named)."""
    import torch

    from retroagi.core.action_tokens import VERBS, pointer_target
    from retroagi.core.layered_policy import POINTER_START, LayeredSMBPolicy

    from . import layered_train as lt
    from .teacher_replay import reference_vision

    from . import controller_teacher as ct

    path, decisions, fragile = [], [], []
    original_potential, original_note = lt._potential, lt._Lane.note_decision
    original_tolerant, original_init = ct._tolerant, lt._Lane.__init__

    def init(self, task, copy):
        # Where Mario stands when the episode starts, before its first frame
        # (after any watched frames).
        original_init(self, task, copy)
        m = self.env.mario
        path.append((m["x"] + m["w"] / 2, m["y"] + m["h"]))

    def tolerant(env, state, target, command):
        # Count the moves the teacher passes over for failing a later start.
        ok = original_tolerant(env, state, target, command)
        if not ok:
            fragile.append(str(command))
        return ok

    def potential(env):
        # Called once after every frame the episode plays (the teacher's
        # trials step the same simulator, but not through here).
        m = env.mario
        path.append((m["x"] + m["w"] / 2, m["y"] + m["h"]))
        return original_potential(env)

    def note(self, learner, decision):
        env, m = self.env, self.env.mario
        decisions.append(
            {
                "frame": len(self.frames["button"]),
                "x": m["x"] + m["w"] / 2,
                "feet": m["y"] + m["h"],
                "place": "ground" if m["on_ground"] else "enemy" if on_enemy(env) else "air",
                "command": str(self.asked.get("skill")),
                "trigger": trigger(env, self.asked.get("skill")),
                "half": m["w"] / 2,
                "hold_frames": getattr(self.asked.get("skill"), "frames", 0),
            }
        )
        return original_note(self, learner, decision)

    lt._potential, lt._Lane.note_decision, ct._tolerant = potential, note, tolerant
    lt._Lane.__init__ = init
    try:
        task = lt.EpisodeTask(
            index=0,
            family=family,
            split="train",
            seed=0,
            sample_index=0,
            difficulty=difficulty,
            teacher_share=1.0,
            label=True,
            scenario=scenario,
        )
        # The networks on the GPU where there is one; the simulator on the CPU.
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if "vision" not in _OBSERVER:
            observer = reference_vision()[0]
            observer.vision.to(device)
            _OBSERVER["vision"] = observer
        record = lt.play_episodes(
            _OBSERVER["vision"],
            LayeredSMBPolicy().to(device).eval(),
            "skill",
            [task],
            device,
            1,
        )[0]
    finally:
        lt._potential, lt._Lane.note_decision = original_potential, original_note
        ct._tolerant, lt._Lane.__init__ = original_tolerant, original_init
    labels = record.labels
    for i, d in enumerate(decisions[: len(labels["valid"])]):
        named = bool(labels["valid"][i]) and bool(labels.get("action_valid", labels["valid"])[i])
        if not labels["valid"][i]:
            d["action"] = None
        elif named:
            d["action"] = (
                VERBS[int(labels["verb"][i])],
                list(pointer_target(int(labels["pointer"][i]) + POINTER_START)),
            )
        else:
            d["action"] = "unnamed"
    return {
        "path": path,
        "decisions": decisions,
        "won": bool(record.won),
        "end": record.end,
        "fragile_moves": fragile,
    }


_OBSERVER: dict = {}
_POLICIES: dict = {}


def _named(action):
    """An Action as (verb, (kind, slot)) for action_words (None: none)."""
    if action is None:
        return None
    if action.verb == "hold" or action.target is None:
        return ("hold", ("surfaces", 0))
    return (action.verb, list(action.target))


def play_policy(scenario: dict, family: str, difficulty: str, checkpoint: str):
    """A trained skill's own episode on ``scenario`` through the training
    pipeline: its own choices (teacher share 0), carried out by the teacher as
    in training. Mario's path, and at each decision where Mario is, the skill's
    action and the teacher's at the same moment (whether they agree)."""
    import torch

    from . import layered_train as lt
    from .teacher_replay import reference_vision

    path, decisions = [], []
    original_potential, original_note = lt._potential, lt._Lane.note_decision
    original_init = lt._Lane.__init__

    def init(self, task, copy):
        original_init(self, task, copy)
        m = self.env.mario
        path.append((m["x"] + m["w"] / 2, m["y"] + m["h"]))

    def potential(env):
        m = env.mario
        path.append((m["x"] + m["w"] / 2, m["y"] + m["h"]))
        return original_potential(env)

    def note(self, learner, decision):
        env, m = self.env, self.env.mario
        example = self.asked.get("action_label")
        teacher = example["action"] if example is not None else None
        own = decision.chosen.get(learner)
        decisions.append(
            {
                "frame": len(self.frames["button"]),
                "x": m["x"] + m["w"] / 2,
                "feet": m["y"] + m["h"],
                "place": "ground" if m["on_ground"] else "enemy" if on_enemy(env) else "air",
                "command": str(decision.skill),
                "half": m["w"] / 2,
                "action": _named(own),
                "teacher": _named(teacher),
                "agrees": teacher is not None and own == teacher,
            }
        )
        return original_note(self, learner, decision)

    lt._potential, lt._Lane.note_decision, lt._Lane.__init__ = potential, note, init
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if "vision" not in _OBSERVER:
            observer = reference_vision()[0]
            observer.vision.to(device)
            _OBSERVER["vision"] = observer
        if checkpoint not in _POLICIES:
            _POLICIES[checkpoint] = lt.load_layered_checkpoint(checkpoint, device)[0].eval()
        task = lt.EpisodeTask(
            index=0,
            family=family,
            split="validation",
            seed=0,
            sample_index=0,
            difficulty=difficulty,
            teacher_share=0.0,
            label=True,
            scenario=scenario,
        )
        record = lt.play_episodes(
            _OBSERVER["vision"], _POLICIES[checkpoint], "skill", [task], device, 1
        )[0]
    finally:
        lt._potential, lt._Lane.note_decision = original_potential, original_note
        lt._Lane.__init__ = original_init
    return {
        "path": path,
        "decisions": decisions,
        "won": bool(record.won),
        "end": record.end,
        "fragile_moves": [],
    }


def action_words(action) -> str:
    """An action as words: "run_to", ("surfaces", 0) -> "run_to surface 0"."""
    if action is None:
        return "no command"
    if action == "unnamed":
        return "no name"
    verb, (kind, slot) = action
    if verb == "hold":
        return "hold"  # aimed at nothing: its pointer slot is unused
    return f"{verb} {SINGULAR.get(kind, kind)} {slot}"


SINGULAR = {
    "enemies": "enemy",
    "surfaces": "surface",
    "blocks": "block",
    "gaps": "gap",
    "pipes": "pipe",
    "moving_platforms": "moving platform",
}


PLACES = {"ground": "", "enemy": " (on the stomped enemy)", "air": " (IN THE AIR)"}


def grouped(decisions) -> list:
    """Consecutive decisions taking the same action where Mario stands still
    (within 4 pixels; a hold decided again and again while a platform
    carries him, say) as one: [(first decision, how many)]."""
    groups = []
    for d in decisions:
        last = groups[-1][0] if groups else None
        if (
            last is not None
            and d.get("action") == last.get("action")
            and d["place"] == last["place"]
            and abs(d["x"] - groups[-1][2]) <= 4
        ):
            groups[-1][1] += 1
            groups[-1][2] = d["x"]
        else:
            groups.append([d, 1, d["x"]])
    return [(d, count) for d, count, _ in groups]


def keeps_speed(decision) -> bool:
    """Whether the decision's command is a run-up that keeps its speed."""
    return "carry=True" in decision.get("command", "")


def draw(world: dict, played: dict, title: str, path: Path) -> None:
    """The layout picture (module docstring) as a PNG at ``path``: the sky
    above everything in it cropped, the decisions numbered on it and listed
    under it."""
    from PIL import Image, ImageDraw, ImageFont

    width, height = world["size"]
    image = np.zeros((height, width, 3), np.float32)
    image[:] = BACKGROUND
    for kind, (x0, y0, x1, y1) in world["statics"]:
        color = (
            PIPE
            if kind == "pipe"
            else BLOCK if kind in ("brick", "brick_row", "question_block") else GROUND
        )
        image[max(0, y0) : min(height, y1), max(0, x0) : min(width, x1)] = color

    def blur(counts, color):
        if counts is None or not counts.any():
            return
        share = counts / counts.max()
        alpha = np.where(counts > 0, 0.2 + 0.8 * np.sqrt(share), 0.0)[..., None]
        image[:] = image * (1 - alpha) + np.array(color, np.float32) * alpha

    blur(world["lifts"], LIFT)
    for kind, counts in world["enemies"].items():
        blur(counts, ENEMY.get(kind, ENEMY["walker"]))
    tops = [y0 for _, (_, y0, _, _) in world["statics"]]
    tops += [y - 14 for _, y in played["path"]] + [y0 for _, y0, _, _ in world["coins"]]
    if world["goal"]:
        tops.append(world["goal"][1])
    for counts in [world["lifts"], *world["enemies"].values()]:
        rows = np.flatnonzero(counts.any(axis=1))
        if len(rows):
            tops.append(int(rows[0]))
    top = int(max(0, min(tops, default=0) - 16))
    font = ImageFont.load_default(size=13)
    small = ImageFont.load_default(size=12)
    head = 26
    lines, line = [], ""
    groups = grouped(played["decisions"])
    for number, (d, count) in enumerate(groups, 1):
        gap = f" (trigger {d['trigger'][0]:.0f} px)" if d.get("trigger") else ""
        if d.get("hold_frames"):
            gap = f" ({d['hold_frames']} frames)"
        teacher = (
            f" (teacher: {action_words(d.get('teacher'))})"
            if "agrees" in d and not d["agrees"]
            else ""
        )
        item = (
            f"{number} {action_words(d.get('action'))}{teacher}"
            f"{' (keeps its speed)' if keeps_speed(d) else ''}"
            f"{PLACES[d['place']]}{f' x{count}' if count > 1 else ''}{gap}"
        )
        if line and font.getlength(line + "   " + item) > width * SCALE - 12:
            lines.append(line)
            line = item
        else:
            line = f"{line}   {item}" if line else item
    if line:
        lines.append(line)
    picture = Image.fromarray(image[top:].clip(0, 255).astype(np.uint8)).resize(
        (width * SCALE, (height - top) * SCALE), Image.NEAREST
    )
    canvas = Image.new(
        "RGB", (width * SCALE, head + (height - top) * SCALE + 8 + 17 * len(lines)), BACKGROUND
    )
    canvas.paste(picture, (0, head))
    pen = ImageDraw.Draw(canvas, "RGBA")

    def at(x, y):
        return x * SCALE, (y - top) * SCALE + head

    for x0, y0, x1, y1 in world["coins"]:
        pen.ellipse([*at(x0, y0), *at(x1, y1)], fill=COIN)
    if world["goal"]:
        pen.rectangle([*at(*world["goal"][:2]), *at(*world["goal"][2:])], outline=GOAL, width=3)
    if played["path"]:
        pen.line([at(x, y) for x, y in played["path"]], fill=PATH + (170,), width=1)
        x, y = played["path"][0]
        pen.rectangle([*at(x - 5, y - 12), *at(x + 5, y)], outline=PATH, width=1)
    for d, _ in groups:
        if d.get("trigger"):
            # The walker's trigger box: centred on Mario, its edge where the
            # walker was when he jumped.
            gap, enemy = d["trigger"]
            reach = d.get("half", 5) + gap
            pen.rectangle(
                [*at(d["x"] - reach, d["feet"] - 22), *at(d["x"] + reach, d["feet"])],
                outline=TRIGGER[enemy % len(TRIGGER)],
                width=2,
            )
    for number, (d, _) in enumerate(groups, 1):
        if "agrees" in d:
            color = AGREES if d["agrees"] else DIFFERS
        else:
            color = {"ground": ON_GROUND, "enemy": ON_ENEMY, "air": IN_AIR}[d["place"]]
        cx, cy = at(d["x"], d["feet"])
        for radius, alpha in ((14, 50), (9, 90), (5, 210)):
            pen.ellipse([cx - radius, cy - radius, cx + radius, cy + radius], fill=color + (alpha,))
        if keeps_speed(d):
            # A run-up keeping its speed: an arrow along it.
            side = -1 if "x=-" in d.get("command", "") else 1
            pen.polygon(
                [(cx + side * 22, cy + 8), (cx + side * 10, cy + 3), (cx + side * 10, cy + 13)],
                fill=(20, 20, 20),
            )
            pen.line([cx + side * 2, cy + 8, cx + side * 12, cy + 8], fill=(20, 20, 20), width=2)
        if d.get("action") in (None, "unnamed"):
            pen.line([cx - 7, cy - 7, cx + 7, cy + 7], fill=(0, 0, 0), width=2)
            pen.line([cx - 7, cy + 7, cx + 7, cy - 7], fill=(0, 0, 0), width=2)
        pen.text((cx - 4, cy - 30 - 12 * (number % 2)), str(number), fill=(20, 20, 20), font=small)
    if not played.get("won", True) and played["path"]:
        # Where a lost episode ended.
        cx, cy = at(*played["path"][-1])
        pen.line([cx - 11, cy - 22, cx + 11, cy], fill=(0, 0, 0), width=4)
        pen.line([cx - 11, cy, cx + 11, cy - 22], fill=(0, 0, 0), width=4)
    pen.text((6, 5), title, fill=(20, 20, 20), font=font)
    y = head + (height - top) * SCALE + 6
    for text in lines:
        pen.text((6, y), text, fill=(20, 20, 20), font=font)
        y += 17
    canvas.save(path, optimize=True)


def _one_thread():
    import torch

    torch.set_num_threads(1)


def _job(job):
    family, difficulty, q, out = job
    scenario, why = atlas_layout(family, difficulty, q)
    entry = {"family": family, "difficulty": difficulty, "quantile": q}
    if scenario is None:
        return {**entry, "rejected": why}
    fingerprint = json.dumps(
        {k: v for k, v in scenario.items() if k != "metadata"}, sort_keys=True, default=str
    )
    frames = max(BLUR_FRAMES, int(scenario.get("frame_budget", 0) or 0)) + int(
        scenario.get("watch_frames", 0) or 0
    )
    world = occupancy(scenario, frames)
    if scenario.get("goal_on_stomp"):
        # Won by the stomp itself: the goal box only marks where the enemy
        # starts, which it walks away from.
        world["goal"] = None
    played = play_teacher(scenario, family, difficulty)
    name = f"{family}/{difficulty}_q{round(q * 100):03d}.png"
    (out / family).mkdir(parents=True, exist_ok=True)
    title = (
        f"{family} · {difficulty} · draws at {q:.0%} of their ranges · teacher "
        f"{'won' if played['won'] else 'LOST (' + played['end'] + ')'}"
        f"{' by the stomp' if scenario.get('goal_on_stomp') and played['won'] else ''}"
    )
    draw(world, played, title, out / name)
    return {
        **entry,
        "image": name,
        "fingerprint": fingerprint,
        "won": played["won"],
        "end": played["end"],
        "fragile_moves": played["fragile_moves"],
        "decisions": [
            {k: (list(v) if isinstance(v, tuple) else v) for k, v in d.items()}
            for d in played["decisions"]
        ],
    }


def make_atlas(out: Path, families, quantiles=QUANTILES, difficulties=DIFFICULTIES, workers=8):
    """Every family's pictures under ``out`` and atlas.json listing them (a
    layout made twice is listed once)."""
    import multiprocessing
    import os
    from concurrent.futures import ProcessPoolExecutor

    out.mkdir(parents=True, exist_ok=True)
    jobs = [(f, d, q, out) for f in families for d in difficulties for q in quantiles]
    # One thread a worker: the workers are the parallelism.
    os.environ.update(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
    with ProcessPoolExecutor(
        workers, mp_context=multiprocessing.get_context("spawn"), initializer=_one_thread
    ) as pool:
        results = list(pool.map(_job, jobs))
    seen, entries = set(), []
    for r in results:
        key = (r["family"], r.get("fingerprint"))
        if r.get("fingerprint") and key in seen:
            (out / r["image"]).unlink(missing_ok=True)
            continue
        seen.add(key)
        r.pop("fingerprint", None)
        entries.append(r)
    (out / "atlas.json").write_text(json.dumps(entries, indent=1) + "\n")
    family_sheets(out, entries)
    return entries


def family_sheets(out: Path, entries: list, columns: Optional[str] = None) -> list:
    """One sheet a family under ``out``/sheets: its pictures tiled with a
    row a difficulty and a column for each entry's "column" (its quantile
    when it has none), the layouts the generator does not draw (their reason)
    and those made twice left blank. ``columns`` says what the columns are in
    the heading. Returns the sheets' paths."""
    from PIL import Image, ImageDraw, ImageFont

    def column(e):
        return e.get("column", e.get("quantile"))

    def label(e):
        return e.get("column_label") or f"{e['quantile']:.0%}"

    font = ImageFont.load_default(size=16)
    (out / "sheets").mkdir(parents=True, exist_ok=True)
    by_family: dict = {}
    for e in entries:
        by_family.setdefault(e["family"], []).append(e)
    paths = []
    for family, rows in by_family.items():
        played = [e for e in rows if "image" in e]
        if not played:
            continue
        images = {
            (e["difficulty"], column(e)): Image.open(out / e["image"]).convert("RGB") for e in played
        }
        blank = {(e["difficulty"], column(e)): e for e in rows if "image" not in e}
        cols = sorted({column(e) for e in rows})
        difficulties = [d for d in DIFFICULTIES if any(e["difficulty"] == d for e in rows)]
        width = max(i.width for i in images.values())
        height = max(i.height for i in images.values())
        pad, head = 8, 32
        sheet = Image.new(
            "RGB",
            (len(cols) * (width + pad) + pad, head + len(difficulties) * (height + pad) + pad),
            BACKGROUND,
        )
        pen = ImageDraw.Draw(sheet)
        won = sum(bool(e.get("won")) for e in played)
        who = "the skill" if any("agreement" in e for e in played) else "the teacher"
        rejected = sum(bool(e.get("rejected")) for e in rows)
        heading = columns or "draws at " + ", ".join(
            label(e) for e in sorted({column(e): e for e in rows}.values(), key=column)
        ) + " of their ranges"
        pen.text(
            (pad, 8),
            f"{family}: {who} won {won} of {len(played)} layouts"
            f"{f'; {rejected} not drawn by the generator' if rejected else ''}"
            f" (rows {', '.join(difficulties)}; columns: {heading})",
            fill=(0, 0, 0),
            font=font,
        )
        for r, difficulty in enumerate(difficulties):
            for c, key in enumerate(cols):
                x, y = pad + c * (width + pad), head + pad + r * (height + pad)
                picture = images.get((difficulty, key))
                if picture is not None:
                    sheet.paste(picture, (x, y))
                    continue
                pen.rectangle([x, y, x + width, y + height], outline=(200, 200, 200))
                e = blank.get((difficulty, key))
                why = e.get("rejected") if e else None
                pen.text(
                    (x + 12, y + 12),
                    f"{difficulty}, {label(e)}: not drawn by the generator ({why})"
                    if why
                    else f"{difficulty}: the same layout as another, or none",
                    fill=(110, 110, 110),
                    font=font,
                )
        path = out / "sheets" / f"{family}.png"
        sheet.save(path)
        paths.append(path)
    return paths


def _policy_job(job):
    family, difficulty, column, task, checkpoint, out = job
    from . import layered_train as lt

    scenario = lt.task_scenario(task)
    frames = max(BLUR_FRAMES, int(scenario.get("frame_budget", 0) or 0)) + int(
        scenario.get("watch_frames", 0) or 0
    )
    world = occupancy(scenario, frames)
    if scenario.get("goal_on_stomp"):
        world["goal"] = None
    played = play_policy(scenario, family, difficulty, checkpoint)
    agree = sum(d["agrees"] for d in played["decisions"])
    total = len(played["decisions"])
    name = f"{family}/{difficulty}_{column}.png"
    (out / family).mkdir(parents=True, exist_ok=True)
    title = (
        f"{family} · {difficulty} · validation layout {task.sample_index} · the skill "
        f"{'won' if played['won'] else 'LOST (' + played['end'] + ')'} · the teacher's action "
        f"at {agree} of {total} decisions (green: the same, red: another)"
    )
    draw(world, played, title, out / name)
    return {
        "family": family,
        "difficulty": difficulty,
        "column": column,
        "column_label": f"layout {task.sample_index}",
        "sample_index": task.sample_index,
        "image": name,
        "won": played["won"],
        "end": played["end"],
        "agreement": [agree, total],
        "decisions": [
            {k: (list(v) if isinstance(v, tuple) else v) for k, v in d.items()}
            for d in played["decisions"]
        ],
    }


def make_policy_atlas(out: Path, checkpoint: str, families, workers=8):
    """The trained skill in ``checkpoint`` on its run's validation layouts of
    ``families`` (the run's own seed and count), one picture a layout: its own
    choices, green where it chose the teacher's action at that moment, red
    where it chose another, a cross where a lost episode ended. Sheets and
    policy.json as make_atlas's."""
    import multiprocessing
    import os
    from concurrent.futures import ProcessPoolExecutor

    from . import layered_train as lt

    _, saved = lt.load_layered_checkpoint(checkpoint, "cpu")
    config = lt.LayeredTrainConfig(**{**saved["config"], "families": tuple(families)})
    tasks = lt._tasks(config, "validation", config.validation_layouts_per_difficulty, 0, 0.0, False)
    jobs, count = [], {}
    for task in sorted(tasks, key=lambda t: (t.family, DIFFICULTIES.index(t.difficulty), t.sample_index)):
        key = (task.family, task.difficulty)
        jobs.append((task.family, task.difficulty, count.get(key, 0), task, str(checkpoint), out))
        count[key] = count.get(key, 0) + 1
    out.mkdir(parents=True, exist_ok=True)
    os.environ.update(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
    with ProcessPoolExecutor(
        workers, mp_context=multiprocessing.get_context("spawn"), initializer=_one_thread
    ) as pool:
        entries = list(pool.map(_policy_job, jobs))
    (out / "policy.json").write_text(json.dumps(entries, indent=1) + "\n")
    family_sheets(out, entries, columns="its run's validation layouts")
    return entries


def main(argv: Optional[list] = None) -> None:
    from . import layered_train as lt

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", required=True)
    parser.add_argument("--families", nargs="*", default=None)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--policy",
        default=None,
        help="a trained skill's checkpoint: draw its own play on its run's validation layouts instead",
    )
    args = parser.parse_args(argv)
    families = args.families or lt.learner_families("skill", BLOCK_SMB_MC_FAMILIES)
    if args.policy:
        entries = make_policy_atlas(Path(args.out), args.policy, families, workers=args.workers)
        lost = [f"{e['family']} {e['difficulty']} layout {e['sample_index']} ({e['end']})" for e in entries if not e["won"]]
        agree = sum(e["agreement"][0] for e in entries)
        total = sum(e["agreement"][1] for e in entries)
        print(
            f"{len(entries)} layouts, {len(lost)} lost by the skill{': ' + ', '.join(lost) if lost else ''}; "
            f"the teacher's action at {agree} of {total} decisions: {Path(args.out) / 'policy.json'}; "
            f"one sheet a family in {Path(args.out) / 'sheets'}"
        )
        return
    entries = make_atlas(Path(args.out), families, workers=args.workers)
    air = sum(d["place"] == "air" for e in entries for d in e.get("decisions", ()))
    played = [e for e in entries if "image" in e]
    lost = [f"{e['family']} {e['difficulty']} {e['quantile']:.0%}" for e in played if not e["won"]]
    print(
        f"{len(played)} layouts played ({len(entries) - len(played)} not drawn by the generator), "
        f"{len(lost)} lost by the teacher{': ' + ', '.join(lost) if lost else ''}; "
        f"{air} decisions in the air: {Path(args.out) / 'atlas.json'}; one sheet a family in "
        f"{Path(args.out) / 'sheets'}"
    )


if __name__ == "__main__":
    main()
