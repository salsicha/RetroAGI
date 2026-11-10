"""Outcome sampling (docs/choose-and-control-plan.md phase 3): the outcome
network's training data, measured, not taught.

At a decision point on a random level (game_levels.random_level) the
simulator's state is saved and every candidate choice (candidates) is played
from it by the same executor the agent uses (core/smb_controller.py), then
the state is put back (the choices: core/smb_choices.py). Its outcome
(outcome): how it ended (the controller's event), whether Mario died during
it, or after it, buttons released, until he has stood WINDOW frames, with
no probe choice from where it ended saving him (the chooser's first rule:
not dying soon after either, when nothing can be done about it), how far right it took
him and in how many frames, the points it gained, whether he reached the
flag, and whether it left him stuck: no forward choice from where it ended
(probes, Mario's speed kept) feasible, safe and getting him on (one choice at a time, the chooser
walked into a valley between staircases and waited there).

The states come from play on random levels by the scripted chooser
(stages/full_smb/scripted_chooser.py), which takes a random candidate
(a kind of choice at random, then one of that kind) instead EXPLORE of the
time, and bumps a question block in reach or collects an item on screen
TRY_POWER_UPS of the time so that the states include what a careful
player would not get into. Every decision point is kept: the state as
tokens relative to Mario (state_tokens: Mario; the surfaces and enemies on
screen), each candidate as (kind, the slot of its thing, modifier), each
outcome as numbers (OUTCOME_FIELDS).
"""

import random
from typing import Optional

import numpy as np

from retroagi.core.smb_choices import (
    ENEMY_SLOTS,
    ENEMY_WIDTH,
    ITEM_SLOTS,
    ITEM_WIDTH,
    KINDS,
    MARIO_WIDTH,
    RUN_BACK,
    RUN_LONG,
    SURFACE_SLOTS,
    SURFACE_WIDTH,
    candidates,
    state_tokens,
)
from retroagi.core.smb_controller import Choice, Controller
from retroagi.core.smb_planner import Goal

from .env_state import restore_env_state, snapshot_env_state
from .game_levels import BlockTruthState, random_level

# Frames after a choice ends, Mario standing, in which a death still counts
# against it (a landing an enemy walks into at once was no good landing),
# after up to FALLING frames for him to come down (a run ended over a pit
# arrives with him still falling into it).
WINDOW = 16
FALLING = 64
# A choice lasts at most this many frames.
LONGEST = 240
# The share of decisions the player makes at random; and how often it bumps
# a question block in reach, or collects an item on screen (so that states
# with a mushroom out, and big Mario, are common enough to learn from).
EXPLORE = 0.3
TRY_POWER_UPS = 0.5
# After a choice, Mario is stuck when no forward probe (probes) from where
# it left him is feasible, safe and takes him at least STUCK_PROGRESS pixels
# right; PROBE_SURFACES of the surfaces ahead are probed.
STUCK_PROGRESS = 16
PROBE_SURFACES = 3
ENDS = ("landed", "stomped", "arrived", "waited", "threat", "blocked", "infeasible", "timeout", "died", "finished",
        "edge")
OUTCOME_FIELDS = ("died", "died_after", "frames", "progress", "points", "finished", "stuck", "end")


def probes(state) -> list:
    """The forward choices tried after a choice to see whether Mario can go
    on: a long run, landing on each of the PROBE_SURFACES surfaces ahead
    (the part ahead of him of the one he is on), passing and stomping the
    nearest enemy ahead."""
    body = state.body
    mx = (body[0] + body[2]) / 2
    out = [Choice("run", x=mx + RUN_LONG, keep_speed=True)]
    ahead = sorted((s for s in state.solids if s.kind in ("ground", "pipe", "block", "question", "moving")
                    and s.x1 > mx + STUCK_PROGRESS + 8), key=lambda s: max(s.x0, mx))
    for s in ahead[:PROBE_SURFACES]:
        x0 = max(s.x0, mx + STUCK_PROGRESS)
        out.append(Choice("jump", Goal("land", s.ref, x0, s.x1, top=s.top)))
    coming = [e for e in state.enemies if (e.box[0] + e.box[2]) / 2 > mx]
    if coming:
        e = min(coming, key=lambda e: e.box[0])
        out += [Choice("jump", Goal("pass", e.ref, side=1)), Choice("jump", Goal("stomp", e.ref))]
    return out


def question_refs(state) -> set:
    """The refs of the question blocks not yet hit on screen."""
    return {s.ref for s in state.solids if s.kind == "question"}


def outcome(env, source: BlockTruthState, choice: Choice, probe: bool = True) -> tuple:
    """``choice`` played by the controller from the simulator's present
    state, then WINDOW frames standing: the OUTCOME_FIELDS (stuck only when
    ``probe``; a death counts as stuck). The simulator is left where the
    play ended."""
    state = source.read(env)
    x0, points0 = env.mario["x"], env.score
    controller = Controller()
    controller.start(choice, state)
    frames, died, finished, end = 0, False, False, controller.event
    while end is None:
        if frames >= LONGEST:
            end = "timeout"
            break
        button = controller.press(state)
        # The frame on which the choice ends is played too, with the button it
        # gives, as in play (game_levels.play): a run's "blocked" comes 2
        # pixels short of the wall, and that frame stops Mario against it.
        _, _, terminated, truncated, info = env.step(button)
        if controller.event is None or terminated:
            frames += 1
        if terminated or truncated:
            died, finished = bool(info["death"]), bool(env._goal_credited)
            end = "died" if died else "finished" if finished else "timeout"
            break
        if controller.event is not None:
            end = controller.event
            break
        state = source.read(env)
    progress = env.mario["x"] - x0
    # A choice can end with Mario in the air (a run off a ledge); nothing is
    # chosen until he stands, so the buttons stay released until he lands,
    # as in play. (Probed in mid-air, a "jump" seemed to save him from the
    # Goomba below 1-1's third pipe, and running off it looked safe.)
    falling = 0
    while not (died or finished or env.mario["on_ground"]) and falling < FALLING:
        _, _, terminated, truncated, info = env.step(0)
        falling += 1
        if terminated or truncated:
            died, finished = bool(info["death"]), bool(env._goal_credited)
            end = "died" if died else "finished" if finished else end
    ended = snapshot_env_state(env) if probe else None  # where the probes start, Mario's speed kept
    died_after = False
    if not (died or finished):
        standing = 0
        for _ in range(FALLING + WINDOW):
            _, _, terminated, truncated, info = env.step(0)
            if terminated or truncated:
                died_after = bool(info["death"])
                finished = finished or bool(env._goal_credited)
                break
            standing = standing + 1 if env.mario["on_ground"] else 0
            if standing >= WINDOW:
                break
    stuck = died or died_after
    if probe and not (died or finished):
        # Probed from where the choice ended, not after the window: standing
        # there first, Mario slid to rest, and a landing on a step at speed,
        # a jump across ready, was taken for a dead end. A death in the
        # window counts only when no probe (a brake back among them) from
        # there survives: landing at speed on 1-1's two-column staircase
        # top, he slid off it standing, though jumping on across was safe.
        stuck, saved = True, False
        restore_env_state(env, ended)
        state = source.read(env)
        mx = (state.body[0] + state.body[2]) / 2
        for option in probes(state) + ([Choice("run", x=mx - RUN_BACK)] if died_after else []):
            r = outcome(env, source, option, probe=False)
            restore_env_state(env, ended)
            safe = r[2] > 0 and ENDS[int(r[-1])] != "infeasible" and not (r[0] or r[1])
            saved = saved or safe
            if safe and r[3] >= STUCK_PROGRESS:
                stuck = False
            if saved and not stuck:
                break
        if died_after and saved:
            died_after = False
    return (float(died), float(died_after), float(frames), float(progress), float(env.score - points0),
            float(finished), float(stuck), float(ENDS.index(end)))


def sample_level(seed: int, mode: str, *, explore: float = EXPLORE, rng: Optional[random.Random] = None,
                 pieces: int = 12) -> dict:
    """Play random_level(seed) in ``mode``, sampling every candidate's
    outcome at every decision point: a shard of arrays (see module notes)."""
    from retroagi.stages.full_smb.scripted_chooser import ScriptedChooser

    from .env import MarioScenarioEnv

    rng = rng or random.Random(seed)
    level = random_level(seed, pieces)
    env = MarioScenarioEnv(world_width=level["world_width"])
    env.draws = False
    env.reset(scenario=level)
    source, controller, chooser = BlockTruthState(), Controller(), ScriptedChooser(mode)
    last_event, last_choice = None, None
    marios, surfaces, enemies, items, rows, outcomes = [], [], [], [], [], []
    for _ in range(level["frame_budget"]):
        state = source.read(env)
        if (controller.choice is None or controller.event is not None) and state.grounded:
            mario, surface, enemy, item, slots = state_tokens(state)
            snapshot = snapshot_env_state(env)
            options = candidates(state)
            index = len(marios)
            marios.append(mario)
            surfaces.append(surface)
            enemies.append(enemy)
            items.append(item)
            for choice, kind, ref, modifier in options:
                outcomes.append(outcome(env, source, choice))
                rows.append((index, kind, slots.get(ref, -1) if ref is not None else -1, modifier))
                restore_env_state(env, snapshot)
            last_event = controller.event
            questions = [o for o in options if o[1] == KINDS.index("bump") and o[0].goal.ref in question_refs(state)]
            items_out = [o for o in options if o[1] == KINDS.index("collect")]
            if items_out and rng.random() < TRY_POWER_UPS:
                last_choice = rng.choice(items_out)[0]
            elif questions and rng.random() < TRY_POWER_UPS:
                last_choice = rng.choice(questions)[0]
            elif rng.random() < explore:
                # a kind of choice at random, then one of that kind: bumps and
                # collects, a few among thirty landings, were almost never
                # tried, and no state had a mushroom out or Mario big
                kind = rng.choice(sorted({k for _, k, _, _ in options}))
                last_choice = rng.choice([o for o in options if o[1] == kind])[0]
            else:
                last_choice = chooser.decide(state, last_event, last_choice)
            controller = Controller()
            controller.start(last_choice, state)
        button = controller.press(state) if controller.event is None else 0
        _, _, terminated, truncated, _ = env.step(button)
        if terminated or truncated:
            break
    return {
        "mario": np.array(marios, np.float32).reshape(-1, MARIO_WIDTH),
        "surfaces": np.array(surfaces, np.float32).reshape(-1, SURFACE_SLOTS, SURFACE_WIDTH),
        "enemies": np.array(enemies, np.float32).reshape(-1, ENEMY_SLOTS, ENEMY_WIDTH),
        "items": np.array(items, np.float32).reshape(-1, ITEM_SLOTS, ITEM_WIDTH),
        "choices": np.array(rows, np.int32).reshape(-1, 4),  # state, kind, slot, modifier
        "outcomes": np.array(outcomes, np.float32).reshape(-1, len(OUTCOME_FIELDS)),
        "seed": np.full(len(marios), seed, np.int32),
        "mode": np.full(len(marios), ("speed_run", "max_points").index(mode), np.int8),
    }


def _sample_one(job: tuple) -> tuple:
    seed, mode, folder = job
    shard = sample_level(seed, mode)
    path = f"{folder}/level_{seed:06d}_{mode}.npz"
    np.savez_compressed(path, **shard)
    return seed, mode, len(shard["mario"]), len(shard["choices"])


def collect(seeds, folder: str, workers: int = 8, modes=("speed_run", "max_points")) -> list:
    """sample_level for every seed and mode on ``workers`` processes, one
    shard file each in ``folder``: (seed, mode, states, choices) per shard."""
    import multiprocessing
    from pathlib import Path

    Path(folder).mkdir(parents=True, exist_ok=True)
    jobs = [(seed, mode, folder) for seed in seeds for mode in modes]
    with multiprocessing.get_context("spawn").Pool(workers) as pool:
        return list(pool.imap_unordered(_sample_one, jobs))


def load(folder: str) -> dict:
    """Every shard in ``folder`` joined, each choice's state index made
    global."""
    from pathlib import Path

    shards = [dict(np.load(p)) for p in sorted(Path(folder).glob("*.npz"))]
    offset, choices = 0, []
    for shard in shards:
        c = shard["choices"].copy()
        c[:, 0] += offset
        choices.append(c)
        offset += len(shard["mario"])
    out = {k: np.concatenate([s[k] for s in shards]) for k in shards[0] if k != "choices"}
    out["choices"] = np.concatenate(choices)
    return out


def pack(folder: str, parts: int = 8) -> None:
    """The shard files in ``folder`` joined into ``parts`` larger ones
    (part_NN.npz; fewer files to keep), the shards removed."""
    from pathlib import Path

    files = sorted(Path(folder).glob("level_*.npz"))
    for k in range(parts):
        group = files[k::parts]
        if not group:
            continue
        shards = [dict(np.load(p)) for p in group]
        offset, choices = 0, []
        for shard in shards:
            c = shard["choices"].copy()
            c[:, 0] += offset
            choices.append(c)
            offset += len(shard["mario"])
        joined = {key: np.concatenate([sh[key] for sh in shards]) for key in shards[0] if key != "choices"}
        joined["choices"] = np.concatenate(choices)
        np.savez_compressed(Path(folder) / f"part_{k:02d}.npz", **joined)
        for p in group:
            p.unlink()


def main(argv=None) -> int:
    import argparse
    import time

    parser = argparse.ArgumentParser(description="Sample every candidate choice's outcome on random levels")
    parser.add_argument("--levels", type=int, default=100)
    parser.add_argument("--first-seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--output", required=True, help="folder for the shard files")
    args = parser.parse_args(argv)
    start = time.time()
    rows = collect(range(args.first_seed, args.first_seed + args.levels), args.output, args.workers)
    pack(args.output)
    seconds = time.time() - start
    states, choices = sum(r[2] for r in rows), sum(r[3] for r in rows)
    print(f"{len(rows)} plays: {states} states, {choices} choices in {seconds:.0f} s"
          f" ({choices / seconds * 3600 / 1e6:.2f} million choices an hour)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
