"""Collision lessons: every way Mario and an enemy can touch, for the collision
trackers alone (user, 2026-10-18: "Make a new set of families for all the
ways that mario can collide with an enemy ... we only want to use this new
dataset to train the gru and nothing else").

Each lesson is one action of Mario's, scripted: a run, a jump or a wait
(scenario "scripted_action", given at the first decision; the episode ends
when it ends: layered_train._Lane). It ends in a contact during the action
(Mario hit by the enemy, or hitting it: a stomp) or in a miss (the action
ends with none): the trackers learn both (user, 2026-10-18). Each family's
parameters reach both sides: waits that end before the enemy arrives, runs
that stop short, jumps that clear it. Every parameter is drawn
through ``rng`` from a short list (choice), so a full sweep
(monte_carlo.CombinationDraws) makes every combination, and a random draw
(monte_carlo.ParameterDraws) one of them.

The families, by Mario's action and what he meets:

- wait: a walker or a monster walking into him from either side; a plant
  rising under him as he stands over its pipe; a walker dropping onto him
  from a ledge above;
- run: into a walker or a monster coming at him or walking away; across a
  pipe whose plant is rising; off a ledge onto a walker below; under a
  ledge as a walker drops off it;
- jump: onto a walker (a stomp, or a hit when it is mistimed), coming at him
  or walking away; onto a monster (it cannot be stomped); up onto a ledge a
  walker walks on; over a pipe as its plant rises; down from a ledge onto a
  walker; a jump too short to clear a walker coming at him.

Difficulty sets the enemies' speeds (easy slow, hard fast) and how quickly a
plant rises.
"""

from retroagi.stages.block_smb.tactic_schedule import segment

FLOOR = 220
FEET = FLOOR - 16  # Mario's top when he stands on the floor (16 tall)
WALKER = FLOOR - 14  # a walker's top on the floor (env: 10 x 6 box, 4 below)
MONSTER = FLOOR - 20  # a monster's top on the floor (16 x 20)
MARIO_X = 120  # where Mario starts, mid screen: enemies on either side are seen
WATCH = 12  # pictures watched before the action, so the trackers see motion
SPEEDS = {
    "easy": (0.3, 0.4, 0.5),
    "medium": (0.6, 0.7, 0.8),
    "hard": (0.9, 1.05, 1.2),
}
# A plant's rise, in frames (hard: quickest).
RISES = {"easy": 20, "medium": 16, "hard": 12}
SIDES = (1, -1)  # the enemy is to Mario's right (1) or left (-1)


def _scenario(family, enemies, action, *, platforms=(), kinds=(), mario_x=MARIO_X, mario_top=FEET):
    """A collision lesson's scenario: a wide floor (and ``platforms``), Mario,
    ``enemies``, the scripted ``action`` and a goal out of reach (a stomp
    ends the lesson as won)."""
    world = 512
    return {
        "world_width": world,
        "mario": [mario_x, mario_top],
        "platforms": [[0, FLOOR, world, 20], *[list(p) for p in platforms]],
        "platform_kinds": ["ground", *kinds] if kinds else None,
        "enemies": enemies,
        "goal": [world - 24, FLOOR - 20, 16, 20],
        "goal_on_stomp": True,
        "watch_frames": WATCH,
        "frame_budget": WATCH + 300,
        "tactics": [segment("advance", 1)],
        "scripted_action": action,
        "metadata": {"block_smb_monte_carlo": {"family": family}},
    }


def _walker(x, speed, direction, top=WALKER, low=0, high=512):
    return [x, top, low, high, speed, direction]


def _monster(x, speed, direction):
    return {"kind": "monster", "x": x, "y": MONSTER, "patrol_min": 0, "patrol_max": 512,
            "speed": speed, "direction": direction}


def _plant(pipe_left, pipe_width, pipe_top, after, exposed, hidden, rise=16):
    """A plant in its pipe, hidden, that starts rising ``after`` frames into
    the action (its cycle: rise, exposed, fall, hidden; hidden must last past
    the watched pictures and ``after``)."""
    cycle = 2 * rise + exposed + hidden
    return {"kind": "piranha_plant", "x": pipe_left + (pipe_width - 12) // 2, "pipe_top": pipe_top,
            "plant_height": 20, "rise_frames": rise, "exposed_frames": exposed,
            "hidden_frames": hidden, "phase": (cycle - (WATCH + after)) % cycle}


def _hold(frames):
    return {"mode": "hold", "x": 0, "y": 0, "frames": frames}


def _move(mode, x, y=0):
    return {"mode": mode, "x": int(x), "y": int(y), "frames": 0}


# ── wait ──────────────────────────────────────────────────────────────────────


def wait_walker(rng, difficulty):
    side, gap, speed = rng.choice(SIDES), rng.choice((24, 48, 72, 96)), rng.choice(SPEEDS[difficulty])
    x = MARIO_X + 5 + side * (gap + 8) - 5
    frames = max(1, min(255, int(gap / speed) + rng.choice((-24, 32, 96))))
    params = dict(side=side, gap=gap, speed=speed, wait=frames)
    return _scenario("collide_wait_walker", [_walker(x, speed, -side)], _hold(frames)), params


def wait_monster(rng, difficulty):
    side, gap, speed = rng.choice(SIDES), rng.choice((24, 48, 72, 96)), rng.choice(SPEEDS[difficulty])
    x = MARIO_X + 5 + side * (gap + 10) - 8
    frames = max(1, min(255, int(gap / speed) + rng.choice((-24, 48, 128))))
    params = dict(side=side, gap=gap, speed=speed, wait=frames)
    return _scenario("collide_wait_monster", [_monster(x, speed, -side)], _hold(frames)), params


def wait_plant(rng, difficulty):
    """Mario stands on a pipe, over its mouth or beside it, as its plant rises."""
    height, width = rng.choice((24, 32, 40)), rng.choice((32, 40))
    offset = rng.choice((-6, 0, 6))  # Mario's middle from the mouth's middle
    exposed, after = rng.choice((32, 48)), rng.choice((4, 16, 32))  # it rises ``after`` frames in
    hidden = 64
    left = MARIO_X - width // 2 + 5 - offset
    top = FLOOR - height
    params = dict(pipe_height=height, pipe_width=width, offset=offset, exposed=exposed, after=after)
    phase = after
    frames = min(255, after + 48)
    return (
        _scenario(
            "collide_wait_plant",
            [_plant(left, width, top, phase, exposed, hidden, RISES[difficulty])],
            _hold(frames),
            platforms=[[left, top, width, height]],
            kinds=["pipe"],
            mario_top=top - 16,
        ),
        params,
    )


def wait_walker_drops(rng, difficulty):
    """A walker walks off a ledge above and falls onto the waiting Mario."""
    height = rng.choice((32, 48, 64))
    reach = rng.choice((8, 24, 40))  # how far the walker walks to the ledge's edge
    speed = rng.choice(SPEEDS[difficulty])
    side = rng.choice(SIDES)
    top = FLOOR - height
    edge = MARIO_X + 5 + side * 2  # the ledge ends right over Mario
    width = 96
    left = edge if side > 0 else edge - width
    x = (edge + reach) if side > 0 else (edge - reach - 10)
    walker = _walker(x, speed, -side, top=top - 14)
    frames = min(255, int((reach + 24) / speed) + 48)
    params = dict(ledge_height=height, reach=reach, speed=speed, side=side, wait=frames)
    return (
        _scenario("collide_wait_walker_drops", [walker], _hold(frames),
                  platforms=[[left, top, width, 8]], kinds=["brick"]),
        params,
    )


# ── run ───────────────────────────────────────────────────────────────────────


def run_walker(rng, difficulty):
    side, gap, speed = rng.choice(SIDES), rng.choice((32, 64, 96, 128)), rng.choice(SPEEDS[difficulty])
    toward = rng.choice((True, False))  # the walker comes at him, or walks away
    beyond = rng.choice((-24, 16, 48))  # where the run is aimed past the walker's start (short: < 0)
    x = MARIO_X + 5 + side * (gap + 8) - 5
    params = dict(side=side, gap=gap, speed=speed, toward=toward, beyond=beyond)
    return (
        _scenario("collide_run_walker", [_walker(x, speed, -side if toward else side)],
                  _move("run", side * (gap + beyond))),
        params,
    )


def run_monster(rng, difficulty):
    side, gap, speed = rng.choice(SIDES), rng.choice((32, 64, 96, 128)), rng.choice(SPEEDS[difficulty])
    toward = rng.choice((True, False))
    beyond = rng.choice((-24, 16, 48))
    x = MARIO_X + 5 + side * (gap + 10) - 8
    params = dict(side=side, gap=gap, speed=speed, toward=toward, beyond=beyond)
    return (
        _scenario("collide_run_monster", [_monster(x, speed, -side if toward else side)],
                  _move("run", side * (gap + beyond))),
        params,
    )


def run_plant(rng, difficulty):
    """Mario runs across a pipe's top as its plant rises."""
    height, width = rng.choice((24, 32)), rng.choice((48, 64))
    exposed, after = rng.choice((32, 48)), rng.choice((0, 4, 8, 16))
    hidden = 64
    phase = after
    start = rng.choice((4, 12))  # Mario's start from the pipe's left edge
    left = MARIO_X + 5 - start
    top = FLOOR - height
    params = dict(pipe_height=height, pipe_width=width, exposed=exposed, after=after, start=start)
    return (
        _scenario("collide_run_plant", [_plant(left, width, top, phase, exposed, hidden, RISES[difficulty])],
                  _move("run", width - start - 6), platforms=[[left, top, width, height]],
                  kinds=["pipe"], mario_top=top - 16),
        params,
    )


def run_off_ledge(rng, difficulty):
    """Mario runs off a ledge onto a walker below (a stomp, or a hit)."""
    height = rng.choice((24, 40, 56))
    beyond = rng.choice((8, 24, 40))  # where the walker is past the ledge's edge
    speed, toward = rng.choice(SPEEDS[difficulty]), rng.choice((True, False))
    run = rng.choice((24, 40, 56))  # how far past the edge the run is aimed
    top = FLOOR - height
    edge = MARIO_X + 5 + 16
    x = edge + beyond
    params = dict(ledge_height=height, beyond=beyond, speed=speed, toward=toward, run=run)
    return (
        _scenario("collide_run_off_ledge", [_walker(x, speed, -1 if toward else 1)],
                  _move("run", 16 + run, height), platforms=[[0, top, edge, height]],
                  kinds=["brick"], mario_top=top - 16),
        params,
    )


def run_under_drop(rng, difficulty):
    """Mario runs under a ledge as a walker walks off it onto him."""
    height = rng.choice((32, 48))
    speed = rng.choice(SPEEDS[difficulty])
    ahead = rng.choice((24, 48, 72))  # the ledge's edge ahead of Mario
    reach = rng.choice((8, 24))
    top = FLOOR - height
    edge = MARIO_X + 5 + ahead
    walker = _walker(edge + reach, speed, -1, top=top - 14)
    params = dict(ledge_height=height, speed=speed, ahead=ahead, reach=reach)
    return (
        _scenario("collide_run_under_drop", [walker], _move("run", ahead + rng.choice((8, 32))),
                  platforms=[[edge, top, 96, 8]], kinds=["brick"]),
        params,
    )


# ── jump ──────────────────────────────────────────────────────────────────────


def jump_walker(rng, difficulty):
    """A jump aimed at a walker's top, or a little short or long of it."""
    side, gap, speed = rng.choice(SIDES), rng.choice((24, 48, 72, 96)), rng.choice(SPEEDS[difficulty])
    toward = rng.choice((True, False))
    miss = rng.choice((-16, -8, 0, 8, 40))  # the aim from the walker's start (40: well past it)
    x = MARIO_X + 5 + side * (gap + 8) - 5
    params = dict(side=side, gap=gap, speed=speed, toward=toward, miss=miss)
    return (
        _scenario("collide_jump_walker", [_walker(x, speed, -side if toward else side)],
                  _move("jump", side * (gap + 8 + miss), -6)),
        params,
    )


def jump_monster(rng, difficulty):
    """A jump onto a monster, which cannot be stomped."""
    side, gap, speed = rng.choice(SIDES), rng.choice((24, 48, 72, 96)), rng.choice(SPEEDS[difficulty])
    toward = rng.choice((True, False))
    miss = rng.choice((-8, 0, 8))
    x = MARIO_X + 5 + side * (gap + 10) - 8
    params = dict(side=side, gap=gap, speed=speed, toward=toward, miss=miss)
    return (
        _scenario("collide_jump_monster", [_monster(x, speed, -side if toward else side)],
                  _move("jump", side * (gap + 10 + miss), -20)),
        params,
    )


def jump_up_onto(rng, difficulty):
    """Mario jumps from the floor up onto a ledge a walker walks on: onto it
    (a stomp), or into it at the ledge's height (a hit)."""
    height = rng.choice((24, 32, 40))
    ahead = rng.choice((16, 32, 48))  # the ledge's left edge ahead of Mario
    along = rng.choice((8, 24, 40))  # the walker's start from the ledge's left edge
    speed, toward = rng.choice(SPEEDS[difficulty]), rng.choice((True, False))
    miss = rng.choice((-8, 0, 8))
    top = FLOOR - height
    left = MARIO_X + 5 + ahead
    walker = _walker(left + along, speed, -1 if toward else 1, top=top - 14, low=left, high=left + 128)
    params = dict(ledge_height=height, ahead=ahead, along=along, speed=speed, toward=toward, miss=miss)
    return (
        _scenario("collide_jump_up_onto", [walker], _move("jump", ahead + along + 5 + miss, -(height + 6)),
                  platforms=[[left, top, 128, 8]], kinds=["brick"]),
        params,
    )


def jump_plant(rng, difficulty):
    """Mario jumps over a pipe, or onto it, as its plant rises."""
    height, width = rng.choice((24, 32, 40)), rng.choice((32, 40))
    gap = rng.choice((16, 32, 48))  # the pipe's left edge ahead of Mario
    exposed, after = rng.choice((32, 48)), rng.choice((0, 8, 16))
    hidden = 64
    phase = after
    onto = rng.choice((0, 1))
    left = MARIO_X + 10 + gap
    top = FLOOR - height
    aim = (gap + width // 2 + 5, -(height)) if onto else (gap + width + 24, 0)
    params = dict(pipe_height=height, pipe_width=width, gap=gap, exposed=exposed, after=after,
                  onto=onto)
    return (
        _scenario("collide_jump_plant", [_plant(left, width, top, phase, exposed, hidden, RISES[difficulty])],
                  _move("jump", *aim), platforms=[[left, top, width, height]], kinds=["pipe"]),
        params,
    )


def jump_down_onto(rng, difficulty):
    """Mario jumps from a ledge down onto a walker below."""
    height = rng.choice((24, 40, 56))
    beyond = rng.choice((16, 40, 64))
    speed, toward = rng.choice(SPEEDS[difficulty]), rng.choice((True, False))
    miss = rng.choice((-8, 0, 8))
    top = FLOOR - height
    edge = MARIO_X + 5 + 16
    x = edge + beyond
    params = dict(ledge_height=height, beyond=beyond, speed=speed, toward=toward, miss=miss)
    return (
        _scenario("collide_jump_down_onto", [_walker(x, speed, -1 if toward else 1)],
                  _move("jump", 16 + beyond + 5 + miss, height - 6),
                  platforms=[[0, top, edge, height]], kinds=["brick"], mario_top=top - 16),
        params,
    )


def jump_short(rng, difficulty):
    """A jump too short or too low to clear a walker coming at Mario."""
    gap, speed = rng.choice((24, 40, 56, 72)), rng.choice(SPEEDS[difficulty])
    short = rng.choice((-24, 8, 16, 24))  # how far short of clearing it the jump lands (< 0: clears)
    side = rng.choice(SIDES)
    x = MARIO_X + 5 + side * (gap + 8) - 5
    params = dict(gap=gap, speed=speed, short=short, side=side)
    return (
        _scenario("collide_jump_short", [_walker(x, speed, -side)],
                  _move("jump", side * max(4, gap - short), 0)),
        params,
    )


GENERATORS = {
    "collide_wait_walker": wait_walker,
    "collide_wait_monster": wait_monster,
    "collide_wait_plant": wait_plant,
    "collide_wait_walker_drops": wait_walker_drops,
    "collide_run_walker": run_walker,
    "collide_run_monster": run_monster,
    "collide_run_plant": run_plant,
    "collide_run_off_ledge": run_off_ledge,
    "collide_run_under_drop": run_under_drop,
    "collide_jump_walker": jump_walker,
    "collide_jump_monster": jump_monster,
    "collide_jump_up_onto": jump_up_onto,
    "collide_jump_plant": jump_plant,
    "collide_jump_down_onto": jump_down_onto,
    "collide_jump_short": jump_short,
}
COLLISION_FAMILIES = tuple(GENERATORS)
DIFFICULTIES = tuple(SPEEDS)


def collision_scenario(family: str, rng, difficulty: str):
    """(scenario, parameters) of one collision lesson; ``rng`` a
    random.Random-like source (ParameterDraws, CombinationDraws)."""
    scenario, params = GENERATORS[family](rng, difficulty)
    if scenario.get("platform_kinds") is None:
        scenario.pop("platform_kinds")
    return scenario, {**params, "difficulty": difficulty}


def every_layout(family: str, difficulty: str) -> list:
    """Every combination of a family's parameters at a difficulty (the full
    sweep): [(scenario, parameters)]."""
    from .monte_carlo import CombinationDraws

    out, path = [], []
    while path is not None:
        draws = CombinationDraws(path)
        out.append(collision_scenario(family, draws, difficulty))
        path = draws.next_path()
    return out


def random_layouts(family: str, difficulty: str, count: int, seed: int) -> list:
    """``count`` layouts drawn at random (the sweep's values), for measuring."""
    import random

    from .monte_carlo import ParameterDraws

    rng = ParameterDraws(random.Random(f"{family}:{difficulty}:{seed}"))
    return [collision_scenario(family, rng, difficulty) for _ in range(count)]
