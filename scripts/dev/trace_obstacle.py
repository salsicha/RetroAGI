"""Play one obstacle of a Super Mario Bros bench and print what happened:
each choice the agent made and how it ended, and the game's own account of
the last frames (Mario's box and every object slot's kind and box, from
game memory), to see what a death or a stall was.

    python scripts/dev/trace_obstacle.py --bench data/full_smb_obstacles/Level2-1 --name 00_steps_320 \\
        [--source memory | vision] [--model OUTCOME.pt] [--mode speed_run] [--seed 0] [--last 30]
"""

import argparse
import json
from collections import deque
from pathlib import Path

from retroagi.domains.smb.nes.truth import true_frame
from retroagi.framework import registry
from retroagi.framework.agent import START_WAITS, Agent
from retroagi.framework.bench import ATTEMPT_FRAMES, PASSED
from retroagi.framework.chooser import network_choosers


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    registry.arguments(parser)
    parser.add_argument("--bench", type=Path, required=True)
    parser.add_argument("--name", required=True, help="the obstacle's name in the bench's index")
    parser.add_argument("--source", default="memory")
    parser.add_argument("--model")
    parser.add_argument("--mode", default="speed_run")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--last", type=int, default=30, help="frames of the game's account printed at the end")
    args = parser.parse_args(argv)
    domain = registry.chosen(args)
    index = json.loads((args.bench / "index.json").read_text())
    obstacle = next(o for o in index["obstacles"] if o["name"] == args.name)
    world = domain.real_world(index["level"], (args.bench / f"{args.name}.state").read_bytes())
    make_chooser = network_choosers(domain, args.model) if args.model else domain.behaviour
    agent = Agent(domain.source(args.source, index["level"]), make_chooser, domain.controller, domain.idle)
    world.reset()
    agent.reset(args.seed, args.mode)
    for _ in range(START_WAITS[args.seed % len(START_WAITS)]):
        world.step(agent.idle)
    print(f"{args.name}: {obstacle['kind']} from {obstacle['x0']} to {obstacle['x1']}, start at {world.progress}")
    recent, choice, end = deque(maxlen=args.last), None, "stalled"
    for frame in range(ATTEMPT_FRAMES):
        action = agent.act(world)
        if agent.last_choice is not choice:
            choice = agent.last_choice
            print(f"frame {frame:3} x {world.progress:4}  after {agent.last_event or 'nothing':10} chose {choice}")
        truth = true_frame(world.game.env.get_ram(), world.game.start)
        slots = "  ".join(f"{k}:{s['kind']:02X}{'' if s['enemy'] else '?'}{tuple(int(v) for v in s['box'])}"
                          for k, s in enumerate(truth["slots"]) if s["active"] and s["box"] and not s["defeated"])
        recent.append(f"  {frame:3} press {action} mario {tuple(int(v) for v in truth['mario'])}  {slots}")
        step = world.step(action)
        if step.died:
            end = "died"
            break
        if step.finished or (world.progress >= obstacle["x1"] + PASSED and world.settled):
            end = "passed"
            break
    print(f"{end} at frame {frame + 1}, x {world.progress}; the last frames (slot:kind(box), ? = not read as an enemy):")
    print("\n".join(recent))
    world.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
