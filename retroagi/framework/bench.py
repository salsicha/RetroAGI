"""Score an agent on each obstacle of a level (the obstacle test bench).

A bench is a folder with an index (index.json: the level and its obstacles,
each with where it ends and the saved state it starts from) made by the
domain (in Super Mario Bros: domains/smb/nes/obstacles.py). From each
obstacle's saved state the agent plays until it is settled PASSED beyond
the obstacle's end (passed), dies (died) or ATTEMPT_FRAMES pass (stalled).
Nothing learns.

Play is deterministic, so the tries of one obstacle differ by how long the
world runs, the agent doing nothing, before it may act (agent.START_WAITS
by seed): the enemies are then elsewhere in their walks. Obstacles that start
from the same saved state are scored in one play (the nearer ones passed on
the way to the farthest; the ATTEMPT_FRAMES count again from each one
passed, so a patient wait at one does not stall the ones after it), and the
result also says how the plays from each distinct start went
(``passed_by_start``: obstacles sharing a start are not independent, and
one death there counts against all of them).

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
from .agent import START_WAITS, Agent

# Progress beyond an obstacle's end where the agent, settled, has passed it.
PASSED = 24
# Frames an agent has for one obstacle: with none passed for so long, a
# play has stalled.
ATTEMPT_FRAMES = 900


def attempts(world, player, seed: int, mode: str, obstacles: Sequence[dict]) -> list:
    """One play from the world's start state scored against each of
    ``obstacles`` (all starting from that state): for each its end
    ("passed", "died", or "stalled" when ATTEMPT_FRAMES went by since the
    play began or the last obstacle was passed), the frames it took and
    the most progress made by then. A player has ``reset(seed, mode)``,
    ``act(world) -> action`` and ``idle`` (agent.Agent)."""
    world.reset()
    player.reset(seed, mode)
    for _ in range(START_WAITS[seed % len(START_WAITS)]):
        world.step(player.idle)
    goals = [o["x1"] + PASSED for o in obstacles]
    results, furthest, frame, until = [None] * len(obstacles), 0, 0, ATTEMPT_FRAMES
    while frame < until and not all(r is not None for r in results):
        step = world.step(player.act(world))
        frame += 1
        furthest = max(furthest, world.progress)
        for k, goal in enumerate(goals):
            if results[k] is not None:
                continue
            if step.died:
                results[k] = {"end": "died", "frames": frame, "furthest": furthest}
            elif step.finished or (world.progress >= goal and world.settled):
                results[k] = {"end": "passed", "frames": frame, "furthest": furthest}
                until = frame + ATTEMPT_FRAMES
    return [r or {"end": "stalled", "frames": frame, "furthest": furthest} for r in results]


def run_bench(bench: Path, make_world: Callable, make_player: Callable, modes: Sequence[str], runs: int = 4,
              only: Sequence[str] = ()) -> dict:
    """Every obstacle of ``bench`` tried ``runs`` times in each mode (seeds 0
    to runs - 1); ``only``: obstacle names to try (all when empty).
    ``make_world(level, saved state)`` gives the world a start is played in."""
    index = json.loads((bench / "index.json").read_text())
    player = make_player()
    wanted = [o for o in index["obstacles"] if not only or o["name"] in only]
    starts = {}  # saved state -> the obstacles that start from it
    for obstacle in wanted:
        starts.setdefault((bench / f"{obstacle['name']}.state").read_bytes(), []).append(obstacle)
    scored, by_start = {}, []
    for state, group in starts.items():
        world = make_world(index["level"], state)
        tries = {o["name"]: {mode: [] for mode in modes} for o in group}
        try:
            for mode in modes:
                for seed in range(runs):
                    for obstacle, result in zip(group, attempts(world, player, seed, mode, group)):
                        tries[obstacle["name"]][mode].append(result)
        finally:
            world.close()
        for obstacle in group:
            ends = Counter(t["end"] for mode_tries in tries[obstacle["name"]].values() for t in mode_tries)
            total = sum(ends.values())
            scored[obstacle["name"]] = {**obstacle, "passed": ends["passed"] / max(1, total), "ends": dict(ends),
                                        "tries": tries[obstacle["name"]]}
            print(f"{obstacle['name']:24} passed {ends['passed']:2}/{total}  died {ends['died']:2}  "
                  f"stalled {ends['stalled']:2}", flush=True)
        by_start.append(scored[group[-1]["name"]]["passed"])  # the farthest obstacle: the whole play
    obstacles = [scored[o["name"]] for o in wanted]
    passed = [o["passed"] for o in obstacles]
    return {"level": index["level"], "runs": runs, "obstacles": obstacles,
            "passed": sum(passed) / max(1, len(passed)),
            "starts": len(by_start), "passed_by_start": sum(by_start) / max(1, len(by_start))}


def report(results: dict, output) -> None:
    print(f"obstacles passed on average: {results['passed']:.2f}"
          f" (the plays from {results['starts']} distinct starts: {results['passed_by_start']:.2f})")
    if output:
        Path(output).write_text(json.dumps(results, indent=1) + "\n")


def main(argv=None) -> int:
    from .chooser import network_choosers

    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    registry.arguments(parser)
    parser.add_argument("--bench", type=Path, required=True)
    parser.add_argument("--source", default=None, help="where the state comes from (the domain's first by default)")
    parser.add_argument("--model", help="an outcome network: choose by the fixed priorities on its predictions")
    parser.add_argument("--runs", type=int, default=4)
    parser.add_argument("--only", nargs="*", default=())
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    domain = registry.chosen(args)
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
