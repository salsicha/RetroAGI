"""Lifts in Block SMB (sim/env.py; measured in Worlds 1-3 and 3-3's memory
with Mario riding them): one swinging across, one going up and down, one
falling while stood on; that a lift carries Mario both ways; how the state
shows a lift's speed; that the planner keeps a lift moving and carries
Mario on it; and that the ledges' gaps hold lifts."""

from retroagi.domains.smb.physics import NESPlayerMotion
from retroagi.domains.smb import planner
from retroagi.domains.smb.planner import Solid, rollout
from retroagi.domains.smb.sim import env as block
from retroagi.domains.smb.sim.levels import FLOOR, skill_scene
from retroagi.domains.smb.sim.world import SimSource, SimWorld

LIFT_TOP = 150


def scene(kind, mario_on=False, phase=0):
    lift = {"x": 300, "y": LIFT_TOP, "w": block.LIFT_W, "h": block.LIFT_H, "lift": kind, "phase": phase}
    mario = [lift["x"] + 16, LIFT_TOP - 12] if mario_on else [40.0, FLOOR - 12]
    return {
        "world_width": 2000, "scroll_x": 115, "mario": mario,
        "platforms": [[0, FLOOR, 200, 32], lift], "platform_kinds": ["ground", None], "enemies": [],
        "task": {"family": "game_level"},
    }


def lift_of(world):
    return next(p for p in world.env.platforms if p.get("lift"))


def test_lifts_swing_across_and_up_and_down_as_measured():
    for kind, axis, (reach, period) in (("across", 0, block.LIFT_ACROSS), ("updown", 1, block.LIFT_UPDOWN)):
        world = SimWorld(scene(kind))
        world.reset()
        lift, places = lift_of(world), []
        for _ in range(int(period)):
            world.step(0)
            places.append((lift["lift_x"] - 300, lift["lift_y"] - LIFT_TOP)[axis])
        sign = -1 if kind == "across" else 1
        assert abs(max(sign * p for p in places) - reach) < 0.5 and min(sign * p for p in places) >= -0.01


def test_a_falling_lift_stays_until_stood_on_then_drops_carrying_mario():
    world = SimWorld(scene("falls"))
    world.reset()
    for _ in range(30):
        world.step(0)
    assert lift_of(world)["lift_y"] == LIFT_TOP
    world = SimWorld(scene("falls", mario_on=True))
    world.reset()
    lift, mario = lift_of(world), world.env.mario
    for _ in range(20):
        world.step(0)
    assert lift["lift_y"] > LIFT_TOP + 30 and mario["on_ground"]
    assert mario["y"] + mario["h"] == lift["rect"].top


def test_a_lift_carries_mario_standing_on_it_across_and_up_and_down():
    for kind in ("across", "updown"):
        world = SimWorld(scene(kind, mario_on=True, phase=60))
        world.reset()
        lift, mario = lift_of(world), world.env.mario
        for _ in range(120):  # (until he has landed on it: at this point of its swing it may be below him)
            world.step(0)
            if mario["on_ground"]:
                break
        world.step(0)
        offset = mario["x"] - lift["rect"].x
        for _ in range(60):
            world.step(0)
            assert mario["on_ground"] and mario["y"] + mario["h"] == lift["rect"].top
        assert abs(mario["x"] - lift["rect"].x - offset) <= 1


def test_the_state_shows_a_lift_moving_and_the_planner_carries_mario_on_it():
    world = SimWorld(scene("across", mario_on=True, phase=30))
    world.reset()
    for _ in range(3):
        world.step(0)
    state = SimSource().read(world)
    lift = next(s for s in state.solids if s.kind == "moving")
    assert lift.vx < 0 and lift.vy == 0
    outcome = rollout(NESPlayerMotion(), state.body, 10, 0, 1, state.solids, [], horizon=10)
    assert outcome.end == "timeout" and outcome.body[0] < state.body[0] + 10 * lift.vx * 0.5


def test_ledges_gaps_hold_every_kind_of_lift():
    kinds = {p["lift"] for seed in range(80) for p in skill_scene(seed, "ledges")["platforms"] if isinstance(p, dict)}
    assert kinds == set(("across", "updown", "falls"))


def _put(kind, phase, dx, feet, y_speed, x_speed):
    """Mario put in the air ``dx`` from the lift's left and his feet
    ``feet`` below its top, with those speeds (the game's units)."""
    world = SimWorld(scene(kind, phase=phase))
    world.reset()
    lift, mario, motion = lift_of(world), world.env.mario, world.env.motion
    mario.update(x=float(lift["rect"].x + dx), y=float(lift["rect"].top + feet - mario["h"]), vy=y_speed,
                 on_ground=False)
    motion.y_speed, motion.y_force, motion.x_speed = y_speed, 0, x_speed
    motion.force = motion.force_down = 0x28
    return world, lift, mario


def test_lifts_meet_mario_as_the_games_do():
    # As measured in 1-3's memory (scripts/dev/lift_contact.py), the lift
    # coming down: rising into its underside, running right under its left
    # end, his rise stops with his feet some 21 to 24 below its top and he
    # falls from there; he is not pushed aside, nor is his run stopped.
    world, lift, mario = _put("updown", 40, -4, 36, -4, 16)
    rows = []
    for _ in range(12):
        world.step(1)  # (right)
        rows.append((mario["x"] - lift["rect"].x, mario["y"] + mario["h"] - lift["rect"].top, mario["vy"]))
    stop = next(i for i, r in enumerate(rows) if r[2] > 0)
    assert 20 <= rows[stop][1] <= 25 and rows[stop - 1][1] > 25
    assert all(b[0] > a[0] for a, b in zip(rows, rows[1:]))  # (on to the right all the while)
    assert world.env.motion.x_speed > 16
    # Falling into it: set on it with his feet up to 4 pixels into its top;
    # deeper, through it.
    for feet, on in ((-1, True), (1, True), (4, False)):
        world, lift, mario = _put("updown", 40, 16, feet, 3, 0)
        for _ in range(4):
            world.step(0)
        assert mario["on_ground"] == on
    # Run into from beside, it stops him as a wall.
    world, lift, mario = _put("updown", 40, -14, 6, 0, 24)
    for _ in range(8):
        world.step(1)
    assert mario["x"] + mario["w"] <= lift["rect"].x and world.env.motion.x_speed < 24
    # The planner foresees the same.
    assert (planner.LIFT_REACH, planner.LIFT_BUMP, planner.LIFT_DROP) == (block.LIFT_REACH, block.LIFT_BUMP,
                                                                          block.LIFT_DROP)
