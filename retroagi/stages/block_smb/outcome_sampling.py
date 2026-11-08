"""Outcome sampling (docs/choose-and-control-plan.md phase 3): the outcome
network's training data, measured, not taught.

At a decision point on a random level (game_levels.random_level) the
simulator's state is saved and every candidate choice (candidates) is played
from it by the same executor the agent uses (core/smb_controller.py), then
the state is put back (the choices: core/smb_choices.py). Its outcome (outcome): how it ended (the controller's event), whether Mario
died during it, or after it, buttons released, until he has stood WINDOW
frames (the chooser's first rule: not dying soon after either), how far right it took
him and in how many frames, the points it gained, whether he reached the
flag.

The states come from play on random levels by the scripted chooser
(stages/full_smb/scripted_chooser.py), which takes a random candidate
instead EXPLORE of the time so that the states include what a careful
player would not get into. Every decision point is kept: the state as
tokens relative to Mario (state_tokens: Mario; the surfaces and enemies on
screen), each candidate as (kind, the slot of its thing, modifier), each
outcome as numbers (OUTCOME_FIELDS).
"""

import random
from typing import Optional

import numpy as np

from retroagi.core.smb_choices import ENEMY_SLOTS, SURFACE_SLOTS, SURFACE_WIDTH, candidates, state_tokens
from retroagi.core.smb_controller import Choice, Controller

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
# The share of decisions the player makes at random.
EXPLORE = 0.3
ENDS = ("landed", "stomped", "arrived", "waited", "threat", "blocked", "infeasible", "timeout", "died", "finished")
OUTCOME_FIELDS = ("died", "died_after", "frames", "progress", "points", "finished", "end")


def outcome(env, source: BlockTruthState, choice: Choice) -> tuple:
    """``choice`` played by the controller from the simulator's present
    state, then WINDOW frames standing: the OUTCOME_FIELDS. The simulator is
    left where the play ended."""
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
        if controller.event is not None:
            end = controller.event
            break
        _, _, terminated, truncated, info = env.step(button)
        frames += 1
        if terminated or truncated:
            died, finished = bool(info["death"]), bool(env._goal_credited)
            end = "died" if died else "finished" if finished else "timeout"
            break
        state = source.read(env)
    progress = env.mario["x"] - x0
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
    return (float(died), float(died_after), float(frames), float(progress), float(env.score - points0),
            float(finished), float(ENDS.index(end)))


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
    marios, surfaces, enemies, rows, outcomes = [], [], [], [], []
    for _ in range(level["frame_budget"]):
        state = source.read(env)
        if (controller.choice is None or controller.event is not None) and state.grounded:
            mario, surface, enemy, slots = state_tokens(state)
            snapshot = snapshot_env_state(env)
            options = candidates(state)
            index = len(marios)
            marios.append(mario)
            surfaces.append(surface)
            enemies.append(enemy)
            for choice, kind, ref, modifier in options:
                outcomes.append(outcome(env, source, choice))
                rows.append((index, kind, slots.get(ref, -1) if ref is not None else -1, modifier))
                restore_env_state(env, snapshot)
            last_event = controller.event
            if rng.random() < explore:
                last_choice = rng.choice(options)[0]
            else:
                last_choice = chooser.decide(state, last_event, last_choice)
            controller = Controller()
            controller.start(last_choice, state)
        button = controller.press(state) if controller.event is None else 0
        _, _, terminated, truncated, _ = env.step(button)
        if terminated or truncated:
            break
    return {
        "mario": np.array(marios, np.float32).reshape(-1, 7),
        "surfaces": np.array(surfaces, np.float32).reshape(-1, SURFACE_SLOTS, SURFACE_WIDTH),
        "enemies": np.array(enemies, np.float32).reshape(-1, ENEMY_SLOTS, 7),
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

    shards = [dict(np.load(p)) for p in sorted(Path(folder).glob("level_*.npz"))]
    offset, choices = 0, []
    for shard in shards:
        c = shard["choices"].copy()
        c[:, 0] += offset
        choices.append(c)
        offset += len(shard["mario"])
    out = {k: np.concatenate([s[k] for s in shards]) for k in shards[0] if k != "choices"}
    out["choices"] = np.concatenate(choices)
    return out


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
    seconds = time.time() - start
    states, choices = sum(r[2] for r in rows), sum(r[3] for r in rows)
    print(f"{len(rows)} plays: {states} states, {choices} choices in {seconds:.0f} s"
          f" ({choices / seconds * 3600 / 1e6:.2f} million choices an hour)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
