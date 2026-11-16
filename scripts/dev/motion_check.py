"""How far vision's estimate of Mario's motion is from the game's, field by
field, on the same frames of the real game. Nothing learns.

The agent plays from each distinct start state of an obstacle bench with its
state from game memory; the vision state source is shown the same frames and
told the same buttons. Each frame both states' motion (physics.NESPlayerMotion)
are compared: horizontal speed, vertical speed and its fraction, the jump's
gravity (held and released), where the jump started, the run timer, which way
he faces, whether he stands. Frames are counted apart on the ground and in the
air, and at the frames the agent decides on.

    python scripts/dev/motion_check.py --bench data/full_smb_obstacles/Level2-1 \\
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

FIELDS = ("x_speed", "y_speed", "y_force", "force", "force_down", "origin_y", "running_speed", "running_timer",
          "facing", "moving", "previous_jump")


def check(bench: Path, model: str | None) -> dict:
    domain = registry.load("smb")
    index = json.loads((bench / "index.json").read_text())
    make_chooser = network_choosers(domain, model) if model else domain.behaviour
    agent = Agent(domain.source("memory", index["level"]), make_chooser, domain.controller, domain.idle)
    vision = domain.source("vision", index["level"])
    off = defaultdict(list)
    counts = defaultdict(int)
    for state in sorted({(bench / f"{o['name']}.state").read_bytes() for o in index["obstacles"]}):
        world = domain.real_world(index["level"], state)
        world.reset()
        agent.reset(0, domain.modes[0])
        vision.reset()
        for _ in range(ATTEMPT_FRAMES):
            true = agent.source.read(world)
            seen = vision.read(world)
            c = agent.controller
            deciding = true is not None and (c.choice is None or c.event is not None) and true.decidable
            action = agent.step(true)
            agent.source.acted(action)
            vision.acted(action)
            if true is not None:
                counts["frames"] += 1
                if seen is None:
                    counts["no vision state"] += 1
                else:
                    where = "ground" if true.grounded else "air"
                    counts[where] += 1
                    counts[f"{where}: stands agreed"] += seen.grounded == true.grounded
                    counts[f"{where}: ready"] += bool(seen.ready)
                    for name in FIELDS:
                        a, b = getattr(seen.motion, name), getattr(true.motion, name)
                        off[f"{where}: {name}"].append(float(a) - float(b))
                        if deciding:
                            off[f"deciding: {name}"].append(float(a) - float(b))
                    off[f"{where}: body y"].append(seen.body[3] - true.body[3])
            if world.step(action).over:
                break
        world.close()
    out = {"level": index["level"], **counts}
    for key, values in sorted(off.items()):
        v = np.abs(np.array(values))
        out[key] = {"exact": float(np.mean(v == 0)), "median": float(np.median(v)),
                    "90th": float(np.percentile(v, 90)), "mean signed": float(np.mean(values))}
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--bench", type=Path, required=True)
    parser.add_argument("--model")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    result = check(args.bench, args.model)
    text = json.dumps(result, indent=1)
    if args.output:
        args.output.write_text(text)
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
