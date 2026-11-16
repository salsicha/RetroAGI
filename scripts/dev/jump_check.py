"""How well the jump planner (retroagi/domains/smb/planner.py) foresees what
Block SMB does: every jump the agent makes in fresh Block SMB levels, the
plan in force as Mario leaves the ground (its predicted end: landed on which
solid, where, after how many frames; a stomp on the way) against what then
happens (where he first stands again, on what, or how he dies). The planner
plays the game as the controller's model of the world; where it and the
practice world disagree, the controller carries out plans that fail
(docs/generalization.md). Block SMB only; nothing here touches the real game.

    python scripts/dev/jump_check.py --model OUTCOME.pt [--levels 100] [--first-seed 100000] [--output OUT.json]
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[2]))
sys.path.insert(0, str(Path(__file__).parent))

from block_smb_deaths import killer  # noqa: E402
from retroagi.domains.smb.domain import DOMAIN  # noqa: E402
from retroagi.framework.agent import Agent  # noqa: E402
from retroagi.framework.chooser import network_choosers  # noqa: E402

NEAR = 4  # pixels: a landing this close to the planned place counts as the plan's


def _ref(solid):
    return None if solid is None else list(solid.ref)


def play(seed: int, mode: str, make_chooser) -> list:
    world = DOMAIN.practice(seed, "level")
    env = world.env
    agent = Agent(DOMAIN.truth_source(), make_chooser, DOMAIN.controller, DOMAIN.idle)
    agent.reset(seed, mode)
    world.reset()
    jumps, flying = [], None
    for frame in range(world.budget):
        action = agent.act(world)
        c = agent.controller
        was_grounded = env.mario["on_ground"]
        step = world.step(action)
        m = env.mario
        if flying is None and was_grounded and not m["on_ground"] and c.choice is not None \
                and c.choice.verb == "jump" and c.plan is not None:
            o = c.plan.outcome
            flying = {"seed": seed, "mode": mode, "frame": frame, "goal": c.choice.goal.kind,
                      "care": c.choice.care, "planned": o.end, "planned_x": o.body[0], "planned_top": o.body[3],
                      "planned_solid": _ref(o.solid), "planned_frames": o.frame - c.plan.pre_run,
                      "planned_stomp": o.stomped is not None, "from_x": m["x"]}
        elif flying is not None and (m["on_ground"] or step.over):
            if step.over and step.died:
                flying.update(happened="died", by=killer(env))
            elif step.over:
                flying.update(happened="finished")
            else:
                platform = m.get("_platform")
                index = next((i for i, p in enumerate(env.platforms) if p is platform), None)
                flying.update(happened="landed", x=m["x"], top=m["y"] + m["h"],
                              solid=None if index is None else ["platforms", index])
            flying["frames"] = frame - flying["frame"]
            flying["stomped"] = bool(env.stomped) if hasattr(env, "stomped") else None
            jumps.append(flying)
            flying = None
        if step.over:
            break
    world.close()
    return jumps


def verdict(j: dict) -> str:
    if j["happened"] == "finished":
        return "finished"
    if j["planned"] == "landed":
        if j["happened"] == "died":
            return f"planned a landing, died ({j['by']})"
        same = j["solid"] == j["planned_solid"]
        if same and abs(j["x"] - j["planned_x"]) <= NEAR:
            return "as planned"
        return "landed as planned, elsewhere on it" if same else "landed on another solid"
    return f"planned {j['planned']}, {j['happened']}"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--model", required=True)
    parser.add_argument("--levels", type=int, default=100)
    parser.add_argument("--first-seed", type=int, default=100000)
    parser.add_argument("--output", default=None)
    args = parser.parse_args(argv)
    make_chooser = network_choosers(DOMAIN, args.model)
    jumps = [j for seed in range(args.first_seed, args.first_seed + args.levels) for mode in DOMAIN.modes
             for j in play(seed, mode, make_chooser)]
    for j in jumps:
        j["verdict"] = verdict(j)
    counts = Counter(j["verdict"] for j in jumps)
    print(f"jumps {len(jumps)}")
    for v, n in counts.most_common():
        print(f"  {n:5d}  {n / len(jumps):6.1%}  {v}")
    elsewhere = [abs(j["x"] - j["planned_x"]) for j in jumps if j["verdict"] == "landed as planned, elsewhere on it"]
    if elsewhere:
        print(f"landed elsewhere on the planned solid: median {sorted(elsewhere)[len(elsewhere) // 2]:.0f} px off")
    if args.output:
        Path(args.output).write_text(json.dumps(jumps, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
