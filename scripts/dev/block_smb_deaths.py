"""What ends Mario's plays of fresh Block SMB levels: for each death, what
killed him (a pit, or the kind of enemy he touched), what he was doing (the
choice in progress, how many frames into it, how its controller had last
reported), and the chance of death the outcome network had given that
choice when it was picked. Block SMB is where the policy may be debugged
(docs/requirements.md); nothing here touches the real game.

    python scripts/dev/block_smb_deaths.py --model data/outcome/NETWORK.pt [--levels 100] [--first-seed 100000]
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[2]))

from retroagi.domains.smb.domain import DOMAIN  # noqa: E402
from retroagi.framework.agent import Agent  # noqa: E402
from retroagi.framework.chooser import DEATH, network_choosers  # noqa: E402


def killer(env) -> str:
    """What killed Mario in the simulator, looked at just after the death."""
    import pygame

    m = env.mario
    if m["y"] > env.height:
        return "pit"
    rect = pygame.Rect(m["x"], m["y"], m["w"], m["h"])
    for e in env.enemies:
        if not e["dead"] and rect.colliderect(pygame.Rect(e["x"], e["y"], e["w"], e["h"])):
            return e.get("kind", "enemy")
    return "other"


def play(seed: int, mode: str, make_chooser) -> dict:
    world = DOMAIN.practice(seed, "level")
    agent = Agent(DOMAIN.truth_source(), make_chooser, DOMAIN.controller, DOMAIN.idle)
    agent.reset(seed, mode)
    world.reset()
    picked, risk, started = None, None, 0
    for frame in range(world.budget):
        before = agent.last_choice
        action = agent.act(world)
        if agent.last_choice is not before and agent.last_choice is not None:
            picked, started = agent.last_choice, frame
            last = getattr(agent.chooser, "last", None)
            risk = float(last[1][last[2], DEATH]) if last else None
        step = world.step(action)
        if step.over:
            end = "died" if step.died else "finished" if step.finished else "timeout"
            row = {"seed": seed, "mode": mode, "end": end, "frame": frame + 1, "x": round(world.progress)}
            if step.died:
                row.update(killer=killer(world.env), verb=picked.verb if picked else None,
                           care=getattr(picked, "care", None), into=frame - started,
                           event=agent.controller.event, risk=risk,
                           grounded=bool(world.env.mario.get("on_ground")))
            world.close()
            return row
    world.close()
    return {"seed": seed, "mode": mode, "end": "timeout", "frame": world.budget, "x": round(world.progress)}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--model", required=True)
    parser.add_argument("--levels", type=int, default=100)
    parser.add_argument("--first-seed", type=int, default=100000)
    parser.add_argument("--output", default=None)
    args = parser.parse_args(argv)
    make_chooser = network_choosers(DOMAIN, args.model)
    rows = [play(seed, mode, make_chooser) for seed in range(args.first_seed, args.first_seed + args.levels)
            for mode in DOMAIN.modes]
    deaths = [r for r in rows if r["end"] == "died"]
    print(f"plays {len(rows)}: " + ", ".join(f"{k} {v}" for k, v in Counter(r["end"] for r in rows).items()))
    for name in ("killer", "verb", "event"):
        print(f"deaths by {name}: " + ", ".join(f"{k} {v}" for k, v in Counter(r[name] for r in deaths).most_common()))
    print("deaths by killer and verb: " + ", ".join(
        f"{k[0]}/{k[1]} {v}" for k, v in Counter((r["killer"], r["verb"]) for r in deaths).most_common()))
    risks = [r["risk"] for r in deaths if r["risk"] is not None]
    if risks:
        print(f"chance of death given the fatal choice when picked: median {sorted(risks)[len(risks) // 2]:.3f}, "
              f"under 5% in {sum(r < 0.05 for r in risks)} of {len(risks)}")
    if args.output:
        Path(args.output).write_text(json.dumps(rows, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
