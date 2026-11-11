"""What ended each failed play of a Super Mario Bros obstacle bench: the
plays the bench makes (framework/bench.py: each distinct start state, every
mode and seed), and for each death the game's own account of it from game
memory: a fall, or the object whose box met Mario's (its kind, and whether
the state source reads that kind as an enemy at all), and the choice the
agent was carrying out.

    python scripts/dev/bench_deaths.py --bench data/full_smb_obstacles/Level2-1 \\
        [--source memory | vision] [--model OUTCOME.pt] [--runs 4] [--quiet]
"""

import argparse
import json
from collections import Counter
from pathlib import Path

from retroagi.domains.smb.nes.truth import KIND_NAMES, true_frame
from retroagi.framework import registry
from retroagi.framework.agent import START_WAITS, Agent
from retroagi.framework.bench import ATTEMPT_FRAMES, PASSED
from retroagi.framework.chooser import network_choosers

# Mario's box below this row has fallen out of the screen.
FALLEN = 224
# An object's box this near Mario's when he dies is what hit him (pixels).
NEAR = 4


def cause(before: dict, after: dict) -> str:
    """What killed Mario between two frames of the game's truth."""
    if after["mario"][3] >= FALLEN or before["mario"][3] >= FALLEN:
        return "fell"
    m = before["mario"]
    met = []
    for slot in before["slots"]:
        box = slot["box"]
        if not (slot["active"] and box) or slot["defeated"]:
            continue
        gap = max(box[0] - m[2], m[0] - box[2], box[1] - m[3], m[1] - box[3])
        if gap <= NEAR:
            met.append((gap, slot))
    if not met:
        return "something in no object slot (a thrown thing, a flame) or the clock"
    slot = min(met, key=lambda pair: pair[0])[1]
    name = KIND_NAMES.get(slot["kind"], f"object {slot['kind']:02X}")
    return name if slot["enemy"] else f"{name} (not read as an enemy)"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    registry.arguments(parser)
    parser.add_argument("--bench", type=Path, required=True)
    parser.add_argument("--source", default="memory")
    parser.add_argument("--model")
    parser.add_argument("--runs", type=int, default=4)
    parser.add_argument("--quiet", action="store_true", help="only the counts")
    args = parser.parse_args(argv)
    domain = registry.chosen(args)
    index = json.loads((args.bench / "index.json").read_text())
    make_chooser = network_choosers(domain, args.model) if args.model else domain.behaviour
    agent = Agent(domain.source(args.source, index["level"]), make_chooser, domain.controller, domain.idle)
    starts = {}
    for obstacle in index["obstacles"]:
        starts.setdefault((args.bench / f"{obstacle['name']}.state").read_bytes(), []).append(obstacle)
    causes, plays = Counter(), 0
    for state, group in starts.items():
        world = domain.real_world(index["level"], state)
        goals = [o["x1"] + PASSED for o in group]
        for mode in domain.modes:
            for seed in range(args.runs):
                world.reset()
                agent.reset(seed, mode)
                for _ in range(START_WAITS[seed % len(START_WAITS)]):
                    world.step(agent.idle)
                plays += 1
                left, frame, until = list(goals), 0, ATTEMPT_FRAMES
                while frame < until and left:
                    frame += 1
                    action = agent.act(world)
                    before = true_frame(world.game.env.get_ram(), world.game.start)
                    step = world.step(action)
                    if step.died:
                        why = cause(before, true_frame(world.game.env.get_ram(), world.game.start))
                        causes[why] += 1
                        if not args.quiet:
                            choice = agent.last_choice
                            doing = choice.verb + (f" ({choice.goal.kind})" if choice and choice.goal else "")
                            print(f"{group[0]['name']:22} {mode:10} seed {seed}: died at x {world.progress:4}: {why}; "
                                  f"during {doing}", flush=True)
                        break
                    if step.finished:
                        break
                    if world.settled and any(world.progress >= goal for goal in left):
                        # (as the bench: the frames allowed count again from each obstacle passed)
                        left, until = [goal for goal in left if world.progress < goal], frame + ATTEMPT_FRAMES
        world.close()
    print(f"{index['level']}: {sum(causes.values())} deaths in {plays} plays")
    for why, count in causes.most_common():
        print(f"  {count:3}  {why}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
