"""Play a level of a domain's real world from its start, and say how it
went. Nothing learns. Play is deterministic: with ``--runs`` each further
play waits a few frames more before the agent acts (agent.START_WAITS).

    python -m retroagi.framework.play --domain smb --level Level1-1 \\
        [--source memory | vision] [--model OUTCOME.pt] [--runs 4] [--frames 8000] [--output OUT.json]
"""

import argparse
import json
from pathlib import Path

from . import registry
from .agent import START_WAITS, Agent, play
from .chooser import network_choosers


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--domain", default="smb")
    parser.add_argument("--level", required=True)
    parser.add_argument("--source", default=None, help="where the state comes from (the domain's first by default)")
    parser.add_argument("--model", help="an outcome network: choose by the fixed priorities on its predictions")
    parser.add_argument("--modes", nargs="*", default=None)
    parser.add_argument("--frames", type=int, default=8000)
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    domain = registry.load(args.domain)
    source_name = args.source or domain.sources[0]
    make_chooser = network_choosers(domain, args.model) if args.model else domain.behaviour
    source = domain.source(source_name, args.level)
    rows = []
    for mode in args.modes or domain.modes:
        for seed in range(args.runs):
            world = domain.real_world(args.level)
            agent = Agent(source, make_chooser, domain.controller, domain.idle)
            agent.reset(seed, mode)
            result = play(world, agent, args.frames, START_WAITS[seed % len(START_WAITS)])
            rows.append({"level": args.level, "source": source_name, "mode": mode, "seed": seed, **result,
                         "progress": world.progress, "score": world.score})
            print(f"{args.level} {source_name} {mode} run {seed}: {result['end']} after {result['frames']} frames, "
                  f"progress {world.progress}, score {world.score}", flush=True)
            world.close()
    if args.output:
        args.output.write_text(json.dumps(rows, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
