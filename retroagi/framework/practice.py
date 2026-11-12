"""Play a domain's practice worlds and count how they end: the measure of a
chooser where nothing is at stake. Whole random levels, and the domain's
skill scenes (a short stretch calling for one skill each), counted by kind:
which skill is weak shows by name.

    python -m retroagi.framework.practice --domain smb --levels 100 --scenes 50 --first-seed 100000 [--model OUTCOME.pt]

Without ``--model`` the domain's own behaviour chooser plays (in Super
Mario Bros the hand-written one); with it, the fixed priorities on that
outcome network's predictions.
"""

from typing import Callable, Optional, Sequence

from . import registry
from .agent import Agent, play
from .chooser import network_choosers

# How a play that got through ends: a level finished, a skill scene passed.
THROUGH = ("finished", "passed")


def run(domain, seeds, make_chooser: Optional[Callable] = None, kinds: Sequence[str] = ("level",),
        scenes: Sequence[int] = ()) -> list:
    """Practice worlds played by ``make_chooser(mode)`` (the domain's
    behaviour chooser by default) in each of the domain's modes: a whole
    level for each of ``seeds`` if "level" is among ``kinds``, and a skill
    scene of each other kind for each of ``scenes``. One row per play (kind,
    seed, mode, how it ended, frames, the most progress made, the score)."""
    rows = []
    worlds = [("level", seed) for seed in seeds if "level" in kinds]
    worlds += [(kind, seed) for kind in kinds if kind != "level" for seed in scenes]
    for kind, seed in worlds:
        for mode in domain.modes:
            world = domain.practice(seed, kind)
            agent = Agent(domain.truth_source(), make_chooser or domain.behaviour, domain.controller, domain.idle)
            agent.reset(seed, mode)
            result = play(world, agent, world.budget)
            rows.append({"kind": kind, "seed": seed, "mode": mode, **result, "x": world.furthest,
                         "score": world.score})
            world.close()
    return rows


def summary(rows: list) -> dict:
    """{kind: (plays got through, plays)} of ``run``'s rows."""
    out = {}
    for row in rows:
        through, plays = out.get(row["kind"], (0, 0))
        out[row["kind"]] = (through + (row["end"] in THROUGH), plays + 1)
    return out


def main(argv=None) -> int:
    import argparse
    import json
    from collections import Counter

    parser = argparse.ArgumentParser(description="A chooser in a domain's practice worlds")
    registry.arguments(parser)
    parser.add_argument("--model", default=None,
                        help="an outcome network (training.py) to choose by the fixed priorities; "
                             "the domain's behaviour chooser if not given")
    parser.add_argument("--levels", type=int, default=100, help="whole levels (seeds from --first-seed)")
    parser.add_argument("--scenes", type=int, default=0, help="skill scenes of each kind (seeds from --first-seed)")
    parser.add_argument("--first-seed", type=int, default=0)
    parser.add_argument("--output", default=None)
    args = parser.parse_args(argv)
    domain = registry.chosen(args)
    make_chooser = network_choosers(domain, args.model) if args.model else None
    rows = run(domain, range(args.first_seed, args.first_seed + args.levels), make_chooser, domain.practices,
               range(args.first_seed, args.first_seed + args.scenes))
    for mode in sorted({r["mode"] for r in rows}):
        runs = [r for r in rows if r["mode"] == mode and r["kind"] == "level"]
        if runs:
            ends = Counter(r["end"] for r in runs)
            print(f"{mode:12s} finished {ends['finished']}/{len(runs)}  died {ends['died']}  "
                  f"timeout {ends['timeout']}  failed seeds {[r['seed'] for r in runs if r['end'] != 'finished']}")
    scenes = {kind: counts for kind, counts in summary(rows).items() if kind != "level"}
    if scenes:
        through, plays = (sum(c[k] for c in scenes.values()) for k in (0, 1))
        print("skill scenes passed: " + "  ".join(f"{kind} {t}/{n}" for kind, (t, n) in scenes.items())
              + f"  all {through}/{plays} ({through / plays:.3f})")
    if args.output:
        with open(args.output, "w") as f:
            json.dump(rows, f, indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
