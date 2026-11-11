"""Which part of perception costs what: an obstacle bench played from
vision's state with some parts of it taken from game memory instead
(Mario's own body and motion, the solids, the enemies and items). With
none it is the vision bench; with all three, the game-memory bench. Nothing
learns.

    python scripts/dev/perception_parts.py --bench data/full_smb_obstacles/Level2-1 \\
        --true mario solids [--model OUTCOME.pt] [--runs 4] [--output OUT.json]
"""

import argparse
import json
from dataclasses import replace
from pathlib import Path

from retroagi.domains.smb.controller import State
from retroagi.domains.smb.nes.truth import true_frame
from retroagi.framework import registry
from retroagi.framework.agent import Agent
from retroagi.framework.bench import report, run_bench
from retroagi.framework.chooser import network_choosers

PARTS = ("mario", "solids", "enemies")


def _moved(box, shift: float) -> tuple:
    return (box[0] + shift, box[1], box[2] + shift, box[3])


class MixedSource:
    """Vision's state with the parts named in ``true`` from game memory,
    laid where vision has the screen (it keeps its own count of the
    scroll)."""

    def __init__(self, memory, vision, true):
        self.memory, self.vision, self.true = memory, vision, set(true)

    def reset(self) -> None:
        self.memory.reset()
        self.vision.reset()

    def acted(self, action: int) -> None:
        self.memory.acted(action)
        self.vision.acted(action)

    def read(self, world):
        true, seen = self.memory.read(world), self.vision.read(world)
        if seen is None or true is None or not self.true:
            return seen
        truth = true_frame(world.game.env.get_ram(), world.game.start)
        shift = self.vision.feedback.camera_position - truth["camera"]
        state = State(seen.body, seen.motion, seen.grounded, seen.solids, seen.enemies, seen.ready, seen.items)
        if "mario" in self.true:
            state.body, state.motion = _moved(true.body, shift), true.motion
            state.grounded, state.ready = true.grounded, true.ready
        if "solids" in self.true:
            state.solids = [replace(s, x0=s.x0 + shift, x1=s.x1 + shift) for s in true.solids]
        if "enemies" in self.true:
            def moved(e):
                track = None if e.track is None else tuple((x + shift, y) for x, y in e.track)
                return replace(e, box=_moved(e.box, shift), track=track)
            state.enemies, state.items = [moved(e) for e in true.enemies], [moved(e) for e in true.items]
        return state


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    registry.arguments(parser)
    parser.add_argument("--bench", type=Path, required=True)
    parser.add_argument("--true", nargs="*", default=(), choices=PARTS, help="the parts taken from game memory")
    parser.add_argument("--model")
    parser.add_argument("--runs", type=int, default=4)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    domain = registry.chosen(args)
    level = json.loads((args.bench / "index.json").read_text())["level"]
    make_chooser = network_choosers(domain, args.model) if args.model else domain.behaviour
    source = MixedSource(domain.source("memory", level), domain.source("vision", level), args.true)
    results = run_bench(args.bench, domain.real_world,
                        lambda: Agent(source, make_chooser, domain.controller, domain.idle), domain.modes, args.runs)
    report(results, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
