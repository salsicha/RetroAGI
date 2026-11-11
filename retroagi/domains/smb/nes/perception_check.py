"""What vision costs the controller, measured in the real game: the state
built from vision against the state built from game memory, on the same
frames. Nothing learns.

The agent plays from each distinct start state of an obstacle bench with
its state from game memory, and the vision state source is shown the same
frames. Each frame the two states are compared on the screen (vision keeps
its own count of how far the screen has scrolled: the two are laid over
each other by where the screen is), around Mario (BEHIND to AHEAD pixels
of him: what a plan reads):

- Mario: how far vision's body is from the game's, whether it agrees he
  stands, and how often it gives no state at all, or none to decide on
  where the game's is one;
- enemies: the share of the game's that vision gives (within MATCH of
  where they are), by the game's kind, how far off their places and
  speeds are, and how many vision gives that the game has not;
- ground: in each COLUMN-wide column, the top of what stands highest there,
  by each state: the share of columns where they agree (within LEVEL), and
  the two ways to be wrong that matter, a pit seen where there is ground and
  ground seen where there is a pit.

    python -m retroagi.domains.smb.nes.perception_check --bench data/full_smb_obstacles/Level2-1 \\
        [--model OUTCOME.pt] [--output OUT.json]
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from retroagi.framework import registry
from retroagi.framework.agent import Agent
from retroagi.framework.bench import ATTEMPT_FRAMES
from retroagi.framework.chooser import network_choosers

from .truth import KIND_NAMES, true_frame

# The stretch around Mario's middle compared (pixels behind and ahead of him).
BEHIND, AHEAD = 48, 128
# A vision enemy this near a game's (their boxes' middles, pixels) is that enemy.
MATCH = 12
COLUMN = 8
# Tops this close are the same ground (pixels).
LEVEL = 3
# No ground in a column: its top is put here.
PIT = 240


def _tops(solids, x0: float, x1: float) -> np.ndarray:
    """The top of the highest solid in each COLUMN-wide column from x0 to x1
    (PIT where there is none; the screen's edge is no ground)."""
    columns = np.arange(x0, x1, COLUMN) + COLUMN / 2
    tops = np.full(len(columns), float(PIT))
    for s in solids:
        if s.kind == "edge":
            continue
        over = (columns >= s.x0) & (columns <= s.x1)
        tops[over] = np.minimum(tops[over], s.top)
    return tops


def _moved(box, shift: float) -> tuple:
    return (box[0] + shift, box[1], box[2] + shift, box[3])


def compare(seen, true, names, found: dict, shift: float = 0.0) -> None:
    """Add one frame's comparison of vision's state ``seen`` (None: none;
    ``shift``: how far right of its own places the game's are) with game
    memory's ``true`` (its enemies named ``names``) to ``found``."""
    found["frames"] += 1
    found["to decide on"] += true.decidable
    if seen is None:
        found["no state"] += 1
        return
    middle = (true.body[0] + true.body[2]) / 2
    found["body off"].append(max(abs(a - b) for a, b in zip(_moved(seen.body, shift), true.body)))
    found["stands agreed"] += seen.grounded == true.grounded
    found["decided on"] += seen.decidable and true.decidable
    enemies = [(_moved(e.box, shift), e) for e in seen.enemies]
    near = [(box, e) for box, e in enemies if -BEHIND <= (box[0] + box[2]) / 2 - middle <= AHEAD]
    used = set()
    for enemy, name in zip(true.enemies, names):
        x, y = (enemy.box[0] + enemy.box[2]) / 2, (enemy.box[1] + enemy.box[3]) / 2
        if not -BEHIND <= x - middle <= AHEAD:
            continue
        best = min(
            ((abs((box[0] + box[2]) / 2 - x) + abs((box[1] + box[3]) / 2 - y), k)
             for k, (box, _) in enumerate(near) if k not in used
             and abs((box[0] + box[2]) / 2 - x) <= MATCH and abs((box[1] + box[3]) / 2 - y) <= MATCH),
            default=None,
        )
        kind = found["enemies"][name]
        kind["seen by the game"] += 1
        if best is None:
            continue
        used.add(best[1])
        box, e = near[best[1]]
        kind["given by vision"] += 1
        kind["box off"].append(max(abs(a - b) for a, b in zip(box, enemy.box)))
        kind["speed off"].append(abs(e.vx - enemy.vx) + abs(e.vy - enemy.vy))
        kind["kind agreed"] += e.kind == enemy.kind
    found["enemies vision made up"] += len(near) - len(used)
    ours = _tops(seen.solids, middle - BEHIND - shift, middle + AHEAD - shift)
    theirs = _tops(true.solids, middle - BEHIND, middle + AHEAD)
    found["columns"] += len(ours)
    found["ground agreed"] += int((np.abs(ours - theirs) <= LEVEL).sum())
    found["pit seen over ground"] += int(((ours >= PIT) & (theirs < PIT)).sum())
    found["ground seen over a pit"] += int(((ours < PIT) & (theirs >= PIT)).sum())
    found["frames with a ground error"] += bool((np.abs(ours - theirs) > LEVEL).any())


def check(bench: Path, model: str = None) -> dict:
    """``compare`` over a play from each distinct start state of ``bench``
    (by the agent with its state from game memory; ``model``: its outcome
    network, the hand-written chooser if None), summed up."""
    domain = registry.load("smb")
    index = json.loads((bench / "index.json").read_text())
    make_chooser = network_choosers(domain, model) if model else domain.behaviour
    agent = Agent(domain.source("memory", index["level"]), make_chooser, domain.controller, domain.idle)
    vision = domain.source("vision", index["level"])
    found = defaultdict(int)
    found["body off"] = []
    found["enemies"] = defaultdict(lambda: {"seen by the game": 0, "given by vision": 0, "kind agreed": 0,
                                            "box off": [], "speed off": []})
    for state in sorted({(bench / f"{o['name']}.state").read_bytes() for o in index["obstacles"]}):
        world = domain.real_world(index["level"], state)
        world.reset()
        agent.reset(0, domain.modes[0])
        vision.reset()
        for _ in range(ATTEMPT_FRAMES):
            true = agent.source.read(world)
            seen = vision.read(world)
            action = agent.step(true)
            agent.source.acted(action)
            vision.acted(action)
            if true is not None:
                truth = true_frame(world.game.env.get_ram(), world.game.start)
                slots = truth["slots"]
                names = [KIND_NAMES.get(slots[e.ref[1]]["kind"], f"object {slots[e.ref[1]]['kind']:02X}")
                         for e in true.enemies]
                compare(seen, true, names, found, truth["camera"] - vision.feedback.camera_position)
            if world.step(action).over:
                break
        world.close()
    frames, states = found["frames"], max(1, found["frames"] - found["no state"])
    enemies = {}
    for name, kind in sorted(found["enemies"].items(), key=lambda item: -item[1]["seen by the game"]):
        given = max(1, kind["given by vision"])
        enemies[name] = {
            "seen by the game": kind["seen by the game"],
            "given by vision": kind["given by vision"] / max(1, kind["seen by the game"]),
            "kind agreed": kind["kind agreed"] / given,
            "box off, median": float(np.median(kind["box off"])) if kind["box off"] else None,
            "box off, 90th percentile": float(np.percentile(kind["box off"], 90)) if kind["box off"] else None,
            "speed off, median": float(np.median(kind["speed off"])) if kind["speed off"] else None,
        }
    return {
        "level": index["level"], "frames": frames,
        "no state": found["no state"] / max(1, frames),
        "decided on, of the game's frames to decide on": found["decided on"] / max(1, found["to decide on"]),
        "stands agreed": found["stands agreed"] / states,
        "body off, median": float(np.median(found["body off"])),
        "body off, 90th percentile": float(np.percentile(found["body off"], 90)),
        "enemies": enemies,
        "enemies vision made up, a frame": found["enemies vision made up"] / states,
        "ground agreed": found["ground agreed"] / max(1, found["columns"]),
        "pit seen over ground": found["pit seen over ground"] / max(1, found["columns"]),
        "ground seen over a pit": found["ground seen over a pit"] / max(1, found["columns"]),
        "frames with a ground error": found["frames with a ground error"] / states,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--bench", type=Path, required=True)
    parser.add_argument("--model", help="an outcome network for the agent that plays (else the hand-written chooser)")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    results = check(args.bench, args.model)
    for key, value in results.items():
        if key != "enemies":
            print(f"{key:34} {value if isinstance(value, (int, str)) else round(value, 3)}")
    print(f"{'kind':28} {'seen':>6} {'given':>6} {'kind ok':>7} {'box off':>8} {'90th':>6} {'speed off':>9}")
    for name, row in results["enemies"].items():
        cells = [row["box off, median"], row["box off, 90th percentile"], row["speed off, median"]]
        print(f"{name:28} {row['seen by the game']:6d} {row['given by vision']:6.2f} {row['kind agreed']:7.2f} "
              + " ".join(f"{'-' if c is None else format(c, '.1f'):>8}" for c in cells))
    if args.output:
        args.output.write_text(json.dumps(results, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
