"""Play a domain's practice worlds (its random levels) and count how they
end: the measure of a chooser where nothing is at stake.

    python -m retroagi.framework.practice --domain smb --levels 100 --first-seed 100000 [--model OUTCOME.pt]

Without ``--model`` the domain's own behaviour chooser plays (in Super
Mario Bros the hand-written one); with it, the fixed priorities on that
outcome network's predictions.
"""

from typing import Callable, Optional

from . import registry
from .agent import Agent, play
from .chooser import network_choosers


def run(domain, seeds, make_chooser: Optional[Callable] = None) -> list:
    """Each seed's practice world played in each of the domain's modes by
    ``make_chooser(mode)`` (the domain's behaviour chooser by default): one
    row per play (seed, mode, how it ended, frames, the most progress made,
    the score)."""
    rows = []
    for seed in seeds:
        for mode in domain.modes:
            world = domain.practice(seed)
            agent = Agent(domain.truth_source(), make_chooser or domain.behaviour, domain.controller, domain.idle)
            agent.reset(seed, mode)
            result = play(world, agent, world.budget)
            rows.append({"seed": seed, "mode": mode, **result, "x": world.furthest, "score": world.score})
            world.close()
    return rows


def main(argv=None) -> int:
    import argparse
    import json
    from collections import Counter

    parser = argparse.ArgumentParser(description="A chooser in a domain's practice worlds")
    registry.arguments(parser)
    parser.add_argument("--model", default=None,
                        help="an outcome network (training.py) to choose by the fixed priorities; "
                             "the domain's behaviour chooser if not given")
    parser.add_argument("--levels", type=int, default=100)
    parser.add_argument("--first-seed", type=int, default=0)
    parser.add_argument("--output", default=None)
    args = parser.parse_args(argv)
    domain = registry.chosen(args)
    make_chooser = network_choosers(domain, args.model) if args.model else None
    rows = run(domain, range(args.first_seed, args.first_seed + args.levels), make_chooser)
    for mode in sorted({r["mode"] for r in rows}):
        runs = [r for r in rows if r["mode"] == mode]
        ends = Counter(r["end"] for r in runs)
        print(f"{mode:12s} finished {ends['finished']}/{len(runs)}  died {ends['died']}  timeout {ends['timeout']}"
              f"  failed seeds {[r['seed'] for r in runs if r['end'] != 'finished']}")
    if args.output:
        with open(args.output, "w") as f:
            json.dump(rows, f, indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
