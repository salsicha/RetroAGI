"""Score an agent on each obstacle of a level (the obstacle test bench).

A bench is a folder with an index (index.json: the level and its obstacles,
each with where it ends and the saved state it starts from) made by the
domain (in Super Mario Bros: domains/smb/nes/obstacles.py). From each
obstacle's saved state the agent plays until it is settled PASSED beyond
the obstacle's end (passed), dies (died) or ATTEMPT_FRAMES pass (stalled).
Nothing learns.

    python -m retroagi.framework.bench --domain smb --bench DIR \\
        [--source memory | vision] [--model OUTCOME.pt] [--runs 4] [--output OUT.json]

``--source``: where the agent's state comes from (the domain's names: in
Super Mario Bros "memory", the game's own memory, testing the controller
and chooser alone, or "vision", what its vision shows). Without ``--model``
the domain's behaviour chooser plays; with it, the fixed priorities on that
outcome network's predictions.
"""

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Callable, Sequence

from . import registry

# Progress beyond an obstacle's end where the agent, settled, has passed it.
PASSED = 24
# Frames an agent has for one obstacle.
ATTEMPT_FRAMES = 900


def attempt(world, player, seed: int, mode: str, obstacle: dict) -> dict:
    """One try at ``obstacle`` from the world's start state: its end
    ("passed", "died", "stalled"), the frames it took and the most progress
    made. A player has ``reset(seed, mode)`` and ``act(world) -> action``
    (agent.Agent)."""
    world.reset()
    player.reset(seed, mode)
    goal = obstacle["x1"] + PASSED
    furthest = 0
    for frame in range(ATTEMPT_FRAMES):
        step = world.step(player.act(world))
        furthest = max(furthest, world.progress)
        if step.died:
            return {"end": "died", "frames": frame + 1, "furthest": furthest}
        if step.finished or (world.progress >= goal and world.settled):
            return {"end": "passed", "frames": frame + 1, "furthest": furthest}
    return {"end": "stalled", "frames": ATTEMPT_FRAMES, "furthest": furthest}


def run_bench(bench: Path, make_world: Callable, make_player: Callable, modes: Sequence[str], runs: int = 4,
              only: Sequence[str] = ()) -> dict:
    """Every obstacle of ``bench`` tried ``runs`` times in each mode (seeds 0
    to runs - 1); ``only``: obstacle names to try (all when empty).
    ``make_world(level, saved state)`` gives the world an obstacle is played
    in."""
    index = json.loads((bench / "index.json").read_text())
    player = make_player()
    results = {"level": index["level"], "runs": runs, "obstacles": []}
    for obstacle in index["obstacles"]:
        if only and obstacle["name"] not in only:
            continue
        world = make_world(index["level"], (bench / f"{obstacle['name']}.state").read_bytes())
        tries = {}
        try:
            for mode in modes:
                tries[mode] = [attempt(world, player, seed, mode, obstacle) for seed in range(runs)]
        finally:
            world.close()
        ends = Counter(t["end"] for mode_tries in tries.values() for t in mode_tries)
        total = sum(ends.values())
        results["obstacles"].append({**obstacle, "passed": ends["passed"] / max(1, total),
                                     "ends": dict(ends), "tries": tries})
        print(f"{obstacle['name']:24} passed {ends['passed']:2}/{total}  died {ends['died']:2}  "
              f"stalled {ends['stalled']:2}", flush=True)
    passed = [o["passed"] for o in results["obstacles"]]
    results["passed"] = sum(passed) / max(1, len(passed))
    return results


def report(results: dict, output) -> None:
    print(f"obstacles passed on average: {results['passed']:.2f}")
    if output:
        Path(output).write_text(json.dumps(results, indent=1) + "\n")


def main(argv=None) -> int:
    from .agent import Agent
    from .chooser import network_choosers

    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--domain", default="smb")
    parser.add_argument("--bench", type=Path, required=True)
    parser.add_argument("--source", default=None, help="where the state comes from (the domain's first by default)")
    parser.add_argument("--model", help="an outcome network: choose by the fixed priorities on its predictions")
    parser.add_argument("--runs", type=int, default=4)
    parser.add_argument("--only", nargs="*", default=())
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    domain = registry.load(args.domain)
    level = json.loads((args.bench / "index.json").read_text())["level"]
    make_chooser = network_choosers(domain, args.model) if args.model else domain.behaviour
    source = domain.source(args.source or domain.sources[0], level)
    results = run_bench(args.bench, domain.real_world,
                        lambda: Agent(source, make_chooser, domain.controller, domain.idle), domain.modes,
                        args.runs, only=args.only)
    report(results, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
