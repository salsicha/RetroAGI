"""Lifts in Block SMB (sim/env.py; measured in Worlds 1-3 and 3-3's memory
with Mario riding them): one swinging across, one going up and down, one
falling while stood on; that a lift carries Mario both ways; how the state
shows a lift's speed; that the planner keeps a lift moving and carries
Mario on it; and that the ledges' gaps hold lifts."""

from retroagi.domains.smb.physics import NESPlayerMotion
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
