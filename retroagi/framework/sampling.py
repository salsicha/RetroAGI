"""Outcome sampling: the outcome network's training data, measured, not
taught.

At a decision point in a practice world (a domain's Practice: a random
level) the world's state is saved and every candidate choice (the domain's
choice space) is played from it by the same controller the agent uses,
then the state is put back. Its outcome (outcome): how it ended (the
controller's event), whether the agent died during it, or after it, doing
nothing, until it has been settled WINDOW frames, with no probe choice from
where it ended saving it (the chooser's first rule: not dying soon after
either, when nothing can be done about it), how much progress it made and
in how many frames, the points it gained, whether it finished the task, and
whether it left the agent stuck: no probe choice from where it ended (the
domain's probes, the agent's speed kept) feasible, safe and getting it on
(one choice at a time, the chooser walked Mario into a valley between
staircases and waited there).

The states come from play by the domain's behaviour chooser, which takes
the domain's own exploration when it offers one (in Super Mario Bros: a
question block to bump, an item to collect) and a random candidate (a kind
of choice at random, then one of that kind) EXPLORE of the time, so that
the states include what a careful player would not get into. Every decision
point is kept: the state as the choice space's tokens, each candidate as
(kind, the slot of its thing, modifier), each outcome as numbers
(OUTCOME_FIELDS).

    python -m retroagi.framework.sampling --domain smb --levels 2000 --output FOLDER
"""

import random
from typing import Optional

import numpy as np

from . import registry
from .agent import Agent

# Frames after a choice ends, the agent settled, in which a death still
# counts against it (a landing an enemy walks into at once was no good
# landing), after up to FALLING frames for it to settle (a run ended over a
# pit arrives with Mario still falling into it).
WINDOW = 16
FALLING = 64
# A choice lasts at most this many frames.
LONGEST = 240
# The share of decisions the player makes at random.
EXPLORE = 0.3
# After a choice, the agent is stuck when no probe from where it left it is
# feasible, safe and makes at least this much progress.
STUCK_PROGRESS = 16
OUTCOME_FIELDS = ("died", "died_after", "frames", "progress", "points", "finished", "stuck", "end")


def outcome(world, source, domain, choice, probe: bool = True) -> tuple:
    """``choice`` played by the controller from the practice world's present
    state, then WINDOW frames settled: the OUTCOME_FIELDS (stuck only when
    ``probe``; a death counts as stuck). The world is left where the play
    ended."""
    ends = domain.ends
    state = source.read(world)
    start, points = world.progress, world.score
    controller = domain.controller()
    controller.start(choice, state)
    frames, died, finished, end = 0, False, False, controller.event
    while end is None:
        if frames >= LONGEST:
            end = "timeout"
            break
        action = controller.press(state)
        # The frame on which the choice ends is played too, with the action
        # it gives, as in play (agent.Agent): a run's "blocked" comes 2
        # pixels short of the wall, and that frame stops Mario against it.
        step = world.step(action)
        if controller.event is None or step.over:
            frames += 1
        if step.over:
            died, finished = step.died, step.finished
            end = "died" if died else "finished" if finished else "timeout"
            break
        if controller.event is not None:
            end = controller.event
            break
        state = source.read(world)
    progress = world.progress - start
    # A choice can end with the agent unsettled (a run off a ledge); nothing
    # is chosen until it settles, so it does nothing until then, as in play.
    # (Probed in mid-air, a "jump" seemed to save Mario from the Goomba below
    # 1-1's third pipe, and running off it looked safe.)
    falling = 0
    while not (died or finished or world.settled) and falling < FALLING:
        step = world.step(domain.idle)
        falling += 1
        if step.over:
            died, finished = step.died, step.finished
            end = "died" if died else "finished" if finished else end
    ended = world.snapshot() if probe else None  # where the probes start, the agent's speed kept
    died_after = False
    if not (died or finished):
        settled = 0
        for _ in range(FALLING + WINDOW):
            step = world.step(domain.idle)
            if step.over:
                died_after = step.died
                finished = finished or step.finished
                break
            settled = settled + 1 if world.settled else 0
            if settled >= WINDOW:
                break
    stuck = died or died_after
    if probe and not (died or finished):
        # Probed from where the choice ended, not after the window: settled
        # there first, Mario slid to rest, and a landing on a step at speed,
        # a jump across ready, was taken for a dead end. A death in the
        # window counts only when no probe (an escape among them) from
        # there survives: landing at speed on 1-1's two-column staircase
        # top, he slid off it standing, though jumping on across was safe.
        stuck, saved = True, False
        world.restore(ended)
        state = source.read(world)
        for option in domain.probes(state) + (domain.escapes(state) if died_after else []):
            r = outcome(world, source, domain, option, probe=False)
            world.restore(ended)
            safe = r[2] > 0 and ends[int(r[-1])] != "infeasible" and not (r[0] or r[1])
            saved = saved or safe
            if safe and r[3] >= STUCK_PROGRESS:
                stuck = False
            if saved and not stuck:
                break
        if died_after and saved:
            died_after = False
    return (float(died), float(died_after), float(frames), float(progress), float(world.score - points),
            float(finished), float(stuck), float(ends.index(end)))


class Sampler:
    """The chooser of a sampling play: at each decision it plays every
    candidate from the saved state and keeps the outcomes, then explores or
    lets the domain's behaviour chooser decide."""

    def __init__(self, domain, world, source, mode: str, rng: random.Random, explore: float = EXPLORE):
        self.domain, self.world, self.source, self.rng, self.explore = domain, world, source, rng, explore
        self.behaviour = domain.behaviour(mode)
        self.tokens = [[] for _ in domain.space.spec.groups]
        self.rows, self.outcomes = [], []

    def decide(self, state, last_event, last_choice):
        domain, space, world = self.domain, self.domain.space, self.world
        tokens, slots = space.tokens(state)
        saved = world.snapshot()
        options = space.candidates(state)
        index = len(self.tokens[0])
        for kept, token in zip(self.tokens, tokens):
            kept.append(token)
        for c in options:
            self.outcomes.append(outcome(world, self.source, domain, c.choice))
            self.rows.append((index, c.kind, slots.get(c.ref, -1) if c.ref is not None else -1, c.modifier))
            world.restore(saved)
        explored = domain.explore(state, options, self.rng)
        if explored is not None:
            return explored
        if self.rng.random() < self.explore:
            # a kind of choice at random, then one of that kind: bumps and
            # collects, a few among thirty landings, were almost never
            # tried, and no state had a mushroom out or Mario big
            kind = self.rng.choice(sorted({c.kind for c in options}))
            return self.rng.choice([c for c in options if c.kind == kind]).choice
        return self.behaviour.decide(state, last_event, last_choice)


def sample(domain, seed: int, mode: str, *, explore: float = EXPLORE, rng: Optional[random.Random] = None) -> dict:
    """Play the domain's practice world of ``seed`` in ``mode``, sampling
    every candidate's outcome at every decision point: a shard of arrays
    (see module notes)."""
    world = domain.practice(seed)
    world.reset()
    source = domain.truth_source()
    sampler = Sampler(domain, world, source, mode, rng or random.Random(seed), explore)
    agent = Agent(source, lambda _: sampler, domain.controller, domain.idle)
    agent.reset(seed, mode)
    for _ in range(world.budget):
        if world.step(agent.act(world)).over:
            break
    groups = domain.space.spec.groups
    states = len(sampler.tokens[0])
    shard = {
        g.name: np.array(kept, np.float32).reshape((-1, g.slots, g.width) if g.slots else (-1, g.width))
        for g, kept in zip(groups, sampler.tokens)
    }
    shard["choices"] = np.array(sampler.rows, np.int32).reshape(-1, 4)  # state, kind, slot, modifier
    shard["outcomes"] = np.array(sampler.outcomes, np.float32).reshape(-1, len(OUTCOME_FIELDS))
    shard["seed"] = np.full(states, seed, np.int32)
    shard["mode"] = np.full(states, domain.modes.index(mode), np.int8)
    return shard


def _sample_one(job: tuple) -> tuple:
    name, seed, mode, folder = job
    shard = sample(registry.load(name), seed, mode)
    np.savez_compressed(f"{folder}/level_{seed:06d}_{mode}.npz", **shard)
    return seed, mode, len(shard["seed"]), len(shard["choices"])


def collect(domain, seeds, folder: str, workers: int = 8) -> list:
    """sample for every seed and mode on ``workers`` processes, one shard
    file each in ``folder``: (seed, mode, states, choices) per shard."""
    import multiprocessing
    from pathlib import Path

    Path(folder).mkdir(parents=True, exist_ok=True)
    jobs = [(domain.name, seed, mode, folder) for seed in seeds for mode in domain.modes]
    with multiprocessing.get_context("spawn").Pool(workers) as pool:
        return list(pool.imap_unordered(_sample_one, jobs))


def _joined(shards: list) -> dict:
    """Shards joined, each choice's state index made global."""
    offset, choices = 0, []
    for shard in shards:
        c = shard["choices"].copy()
        c[:, 0] += offset
        choices.append(c)
        offset += len(shard["seed"])
    out = {k: np.concatenate([s[k] for s in shards]) for k in shards[0] if k != "choices"}
    out["choices"] = np.concatenate(choices)
    return out


def load(folder: str) -> dict:
    """Every shard in ``folder`` joined."""
    from pathlib import Path

    return _joined([dict(np.load(p)) for p in sorted(Path(folder).glob("*.npz"))])


def pack(folder: str, parts: int = 8) -> None:
    """The shard files in ``folder`` joined into ``parts`` larger ones
    (part_NN.npz; fewer files to keep), the shards removed."""
    from pathlib import Path

    files = sorted(Path(folder).glob("level_*.npz"))
    for k in range(parts):
        group = files[k::parts]
        if not group:
            continue
        np.savez_compressed(Path(folder) / f"part_{k:02d}.npz", **_joined([dict(np.load(p)) for p in group]))
        for p in group:
            p.unlink()


def main(argv=None) -> int:
    import argparse
    import time

    parser = argparse.ArgumentParser(description="Sample every candidate choice's outcome in practice worlds")
    parser.add_argument("--domain", default="smb")
    parser.add_argument("--levels", type=int, default=100)
    parser.add_argument("--first-seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--output", required=True, help="folder for the shard files")
    args = parser.parse_args(argv)
    start = time.time()
    rows = collect(registry.load(args.domain), range(args.first_seed, args.first_seed + args.levels), args.output,
                   args.workers)
    pack(args.output)
    seconds = time.time() - start
    states, choices = sum(r[2] for r in rows), sum(r[3] for r in rows)
    print(f"{len(rows)} plays: {states} states, {choices} choices in {seconds:.0f} s"
          f" ({choices / seconds * 3600 / 1e6:.2f} million choices an hour)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
