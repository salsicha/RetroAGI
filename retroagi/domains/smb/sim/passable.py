"""Can a Block SMB practice world be got through at all? A search over
button presses, with the world's enemies taken out: from the start every
button combination is held for HOLD frames, from each new situation again
(the one furthest right first, of those the highest, of those the newest:
against a wall the search climbs before it tries anything else), until
Mario stands beyond the scene's end (or reaches the flag) or nothing new is
left to try. It knows nothing of the
agent: it tells the generator's mistakes (an arrangement no player could
pass) from the policy's.

    python -m retroagi.domains.smb.sim.passable [--kinds pair bridge ...] [--seeds 200] [--first-seed 100000] [--workers 14]
"""

import argparse
import heapq
from concurrent.futures import ProcessPoolExecutor
from itertools import count

from .levels import SCENE_KINDS, TILE, random_level, skill_scene
from .world import SimWorld

# Frames a button combination is held, and the situations tried before giving up.
HOLD = 4
EXPANSIONS = 40000
# The simulator's actions, tried in this order: right and jump, right, jump, nothing, left and jump, left.
ACTIONS = (2, 1, 5, 0, 4, 3)


def passable(scenario: dict, expansions: int = EXPANSIONS) -> dict:
    """Whether some sequence of presses gets Mario through ``scenario``
    without its enemies ("passable"), how far right the search got ("x")
    and the situations it tried ("tried")."""
    world = SimWorld({**scenario, "enemies": []})
    world.reset()
    env, order = world.env, count()
    queue, seen, furthest = [(0, 0, 0, world.snapshot())], set(), world.progress
    for tried in range(expansions):
        if not queue:
            return {"passable": False, "x": furthest, "tried": tried}
        saved = heapq.heappop(queue)[-1]
        for action in ACTIONS:
            world.restore(saved)
            step = None
            for _ in range(HOLD):
                step = world.step(action)
                if step.over:
                    break
            if step.finished or world.passed:
                return {"passable": True, "x": world.progress, "tried": tried}
            if step.over:
                continue
            mario, motion = env.mario, env.motion
            # (where the lifts are is part of the situation: the same place at
            # another point of a lift's swing is another situation; so is a
            # springboard's point of its squash, and whether it will throw
            # strongly)
            lifts = tuple((int(p["lift_x"]) // 4, int(p["lift_y"]) // 4, p.get("squash"), p.get("strong"))
                          for p in env.platforms if p.get("lift"))
            key = (int(mario["x"]) // 3, int(mario["y"]) // 3, motion.x_speed // 4, motion.y_speed, motion.force,
                   bool(mario["on_ground"]), bool(mario["big"]), bool(motion.previous_jump), lifts)
            if key in seen:
                continue
            seen.add(key)
            furthest = max(furthest, world.progress)
            heapq.heappush(queue, (-int(mario["x"]), int(mario["y"]), -next(order), world.snapshot()))
    return {"passable": False, "x": furthest, "tried": expansions}


def _one(job):
    kind, seed = job
    scenario = random_level(seed) if kind == "level" else skill_scene(seed, kind)
    return kind, seed, passable(scenario)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--kinds", nargs="*", default=SCENE_KINDS, help='skill scene kinds, or "level"')
    parser.add_argument("--seeds", type=int, default=200)
    parser.add_argument("--first-seed", type=int, default=100000)
    parser.add_argument("--workers", type=int, default=14)
    args = parser.parse_args(argv)
    jobs = [(kind, seed) for kind in args.kinds for seed in range(args.first_seed, args.first_seed + args.seeds)]
    with ProcessPoolExecutor(args.workers) as pool:
        found = list(pool.map(_one, jobs, chunksize=4))
    for kind in args.kinds:
        rows = [(seed, result) for k, seed, result in found if k == kind]
        stuck = [(seed, round(result["x"] / TILE, 1)) for seed, result in rows if not result["passable"]]
        print(f"{kind:10} passable {len(rows) - len(stuck)}/{len(rows)}"
              + (f"  not passed (seed, how far in tiles): {stuck}" if stuck else ""), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
