"""How well a dynamics model (framework/interfaces.Dynamics) predicts where
the game's other things go, measured in the real game. Nothing learns.

The agent plays from each distinct start state of an obstacle bench, its
state read from game memory. At every frame a model predicts each enemy's
box for the frames ahead from what the state source gave the controller
then; the prediction is compared with where the game had put that enemy
that many frames later (while it stayed the same enemy, in sight). The
errors are reported by the game's kind of enemy and by how far ahead: what
the controller's plans rest on, apart from the agent's play. Every one of
the domain's dynamics models is scored on the same plays.

    python -m retroagi.domains.smb.nes.dynamics_check --bench data/full_smb_obstacles/Level2-1 \\
        [--model OUTCOME.pt] [--plays PLAYS.pkl] [--output OUT.json]
"""

import argparse
import json
import pickle
from collections import defaultdict
from copy import copy
from pathlib import Path
from typing import Optional

import numpy as np

from retroagi.framework import registry
from retroagi.framework.agent import Agent
from retroagi.framework.bench import ATTEMPT_FRAMES
from retroagi.framework.chooser import network_choosers

from .truth import KIND_NAMES, true_frame

# Frames ahead the predictions are scored at.
HORIZONS = (8, 16, 32, 48)
# A prediction within this of the truth is near enough for a plan (pixels:
# the planner keeps enemies ENEMY_MARGIN 2 wider, and Mario is 10 wide).
NEAR = 4
# An enemy moving more than this in a frame is another enemy in its slot.
SAME = 8


def record(world, agent, frames: int = ATTEMPT_FRAMES) -> list:
    """One play of ``world`` by ``agent`` (both reset): per frame the
    enemies and the solids its state source gave, each enemy named by the
    game's kind of its slot."""
    seen = []
    for _ in range(frames):
        state = agent.source.read(world)
        action = agent.step(state)
        agent.source.acted(action)
        if state is not None:
            slots = true_frame(world.game.env.get_ram(), world.game.start)["slots"]
            names = [KIND_NAMES.get(slots[e.ref[1]]["kind"], f"object {slots[e.ref[1]]['kind']:02X}")
                     for e in state.enemies]
            seen.append(([copy(e) for e in state.enemies], state.solids, names))
        else:
            seen.append(([], [], []))
        if world.step(action).over:
            break
    return seen


def errors(seen: list, dynamics, horizons=HORIZONS) -> dict:
    """{(name, frames ahead): [(sideways error, vertical error), ...]} of
    ``dynamics``' predictions over a recorded play (shown each frame in
    turn, as in play): predicted minus true, of the box's left side and top
    (pixels)."""
    out = defaultdict(list)
    ahead = max(horizons)
    for t, (enemies, solids, names) in enumerate(seen):
        if not enemies:
            continue
        things = [copy(e) for e in enemies]
        dynamics.observe(things, solids)
        dynamics.paths(things, solids, ahead)
        for thing, name in zip(things, names):
            last = thing.box
            for k in range(1, ahead + 1):
                if t + k >= len(seen):
                    break
                now = next((e.box for e in seen[t + k][0] if e.ref == thing.ref), None)
                if now is None or abs(now[0] - last[0]) > SAME or abs(now[1] - last[1]) > SAME:
                    break  # out of sight, defeated, or another enemy in its slot
                last = now
                if k in horizons:
                    predicted = thing.at(k)
                    out[(name, k)].append((predicted[0] - now[0], predicted[1] - now[1]))
    return out


def table(found: dict) -> list:
    """Rows (name, frames ahead, predictions, median sideways and vertical
    error, the 90th percentile of the larger of the two, the share within
    NEAR) of ``errors``' answer, pooled over kinds as "all" too."""
    pooled = defaultdict(list)
    for (name, k), pairs in found.items():
        pooled[(name, k)] += pairs
        pooled[("all", k)] += pairs
    rows = []
    for (name, k), pairs in sorted(pooled.items()):
        e = np.abs(np.asarray(pairs, np.float64))
        worst = e.max(axis=1)
        rows.append({"kind": name, "ahead": k, "predictions": len(pairs),
                     "median_x": float(np.median(e[:, 0])), "median_y": float(np.median(e[:, 1])),
                     "p90": float(np.percentile(worst, 90)), "near": float((worst <= NEAR).mean())})
    return rows


def plays(bench: Path, model: Optional[str] = None) -> list:
    """A play (record) from each distinct start state of ``bench``, by the
    agent with its usual dynamics model (``model``: its outcome network;
    the hand-written chooser if None)."""
    domain = registry.load("smb")
    index = json.loads((bench / "index.json").read_text())
    make_chooser = network_choosers(domain, model) if model else domain.behaviour
    agent = Agent(domain.source("memory", index["level"]), make_chooser, domain.controller, domain.idle)
    out = []
    for state in sorted({(bench / f"{o['name']}.state").read_bytes() for o in index["obstacles"]}):
        world = domain.real_world(index["level"], state)
        world.reset()
        agent.reset(0, domain.modes[0])
        out.append(record(world, agent))
        world.close()
    return out


def scores(recorded: list, name: str, kept: bool = False) -> list:
    """``table`` of the domain's dynamics model called ``name`` over
    recorded plays: a new model each play, or (``kept``) one model through
    them all, keeping what it has seen."""
    domain = registry.load("smb", (f"dynamics={name}",))
    found = defaultdict(list)
    model = domain.dynamics()
    for seen in recorded:
        for key, pairs in errors(seen, model if kept else domain.dynamics()).items():
            found[key] += pairs
    return table(found)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--bench", type=Path, required=True)
    parser.add_argument("--model", help="an outcome network for the agent that plays (else the hand-written chooser)")
    parser.add_argument("--plays", type=Path, help="keep the recorded plays here (and use them if already there)")
    parser.add_argument("--kept", action="store_true", help="one model through all the plays, not a new one each")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    level = json.loads((args.bench / "index.json").read_text())["level"]
    if args.plays and args.plays.exists():
        recorded = pickle.loads(args.plays.read_bytes())
    else:
        recorded = plays(args.bench, args.model)
        if args.plays:
            args.plays.write_bytes(pickle.dumps(recorded))
    results = {"level": level}
    for name in registry.load("smb").parts["dynamics"]:
        rows = results[name] = scores(recorded, name, args.kept)
        print(f"{level}, {name}: predicted against true places, pixels")
        print(f"{'kind':28} {'ahead':>5} {'predictions':>11} {'median x':>8} {'median y':>8} {'90th pct':>8} "
              f"{'within ' + str(NEAR):>9}")
        for row in rows:
            print(f"{row['kind']:28} {row['ahead']:5d} {row['predictions']:11d} {row['median_x']:8.1f} "
                  f"{row['median_y']:8.1f} {row['p90']:8.1f} {row['near']:9.2f}")
    if args.output:
        args.output.write_text(json.dumps(results, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
