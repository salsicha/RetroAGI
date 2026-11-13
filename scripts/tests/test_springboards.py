"""Springboards in Block SMB (measured in World 2-1's memory with Mario set
down on one and walked into it; physics.SPRING_SQUASH): how one presses
Mario down and throws him, plainly or, the jump button held at the right
frames, strongly; that it stops a walking Mario; that the state
shows it and the planner plans a throw over a wall too tall to jump, which
the simulator then plays as planned; and that the generator's springboard
pieces can be passed."""

from retroagi.domains.smb import planner
from retroagi.domains.smb.physics import SPRING_LAUNCH, SPRING_REST, SPRING_SQUASH, SPRING_THROW
from retroagi.domains.smb.planner import Goal
from retroagi.domains.smb.sim.env import SPRING_W
from retroagi.domains.smb.sim.levels import FLOOR, TILE, random_level, skill_scene
from retroagi.domains.smb.sim.passable import passable
from retroagi.domains.smb.sim.world import SimSource, SimWorld

REST = FLOOR - TILE - SPRING_REST  # (the springboard's top at rest)
WALL_TOP = FLOOR - 10 * TILE


def scene(wall=True):
    platforms = [[0, FLOOR, 800, 32], {"x": 200, "y": REST, "w": SPRING_W, "h": SPRING_REST + TILE, "lift": "spring"}]
    kinds = ["ground", "spring"]
    if wall:
        platforms.append([232, WALL_TOP, 2 * TILE, 240 - WALL_TOP])
        kinds.append("ground")
    return {"world_width": 800, "scroll_x": 115, "mario": [40.0, FLOOR - 12], "frame_budget": 3000,
            "platforms": platforms, "platform_kinds": kinds, "enemies": [], "task": {"family": "game_level"}}


def dropped_on(world):
    """Mario dropped onto the springboard; frames until he lands on it."""
    mario = world.env.mario
    mario["x"], mario["y"] = 202, REST - mario["h"] - 20
    for frame in range(40):
        world.step(0)
        if mario["on_ground"]:
            return frame
    raise AssertionError("never landed")


def ride(jump_frames=()):
    """Mario's feet under the springboard's rest each frame from his
    landing, the jump button held on ``jump_frames`` of them."""
    world = SimWorld(scene(wall=False))
    world.reset()
    dropped_on(world)
    mario, feet = world.env.mario, [0]
    for frame in range(1, 30):
        world.step(5 if frame in jump_frames else 0)
        feet.append(mario["y"] + mario["h"] - REST)
    return feet


def test_a_springboard_presses_mario_down_and_throws_him_as_measured():
    feet = ride()
    # (the frame he lands, then the squash a frame each, then the throw)
    assert feet[:SPRING_THROW + 2] == [0, *SPRING_SQUASH[:SPRING_THROW + 1]]
    assert feet[SPRING_THROW + 2] - feet[SPRING_THROW + 1] == SPRING_LAUNCH[0]
    # The game's (2-1, the button not pressed): 3, -4, -11, -17, -24 after the throw.
    assert feet[SPRING_THROW + 2:SPRING_THROW + 7] in ([3, -4, -11, -17, -24], [3, -4, -11, -18, -24])


def test_the_jump_button_held_late_in_the_squash_throws_him_higher_and_early_does_nothing():
    strong = ride(jump_frames=range(16, 20))
    assert strong[SPRING_THROW + 2] - strong[SPRING_THROW + 1] == SPRING_LAUNCH[1]
    early = ride(jump_frames=range(2, 12))
    assert early == ride()  # (no ordinary jump off it either)


def test_it_stops_mario_walking_into_it():
    world = SimWorld(scene(wall=False))
    world.reset()
    mario = world.env.mario
    for _ in range(200):
        world.step(1)
    assert mario["x"] + mario["w"] <= 200 and mario["on_ground"] and mario["y"] + mario["h"] == FLOOR


def test_the_planner_throws_mario_over_a_wall_and_the_simulator_plays_it_so():
    world = SimWorld(scene())
    world.reset()
    dropped_on(world)
    for _ in range(4):
        world.step(0)
    state = SimSource().read(world)
    spring = planner.standing_spring(state.body, state.solids)
    assert spring is not None and spring.kind == "spring" and spring.spring == 3
    wall = next(s for s in state.solids if s.top == WALL_TOP)
    plan = planner.plan_jump(Goal("land", wall.ref, wall.x0, wall.x1, top=wall.top), state.motion, state.body,
                             state.solids, state.enemies, 1)
    assert plan is not None and plan.outcome.end == "landed" and plan.outcome.body[3] == WALL_TOP
    for button in plan.buttons:
        world.step(button)
    mario = world.env.mario
    assert mario["on_ground"] and mario["y"] + mario["h"] == WALL_TOP
    assert abs(mario["x"] - plan.outcome.body[0]) <= 1


def test_springboard_pieces_come_in_levels_and_scenes_and_can_be_passed():
    scenes = [skill_scene(seed, "spring") for seed in range(6)]
    for s in scenes:
        assert "spring" in s["platform_kinds"]
        assert passable(s)["passable"]
    walls = [s for s in scenes if any(p[1] == WALL_TOP for p in s["platforms"] if isinstance(p, list))]
    assert walls and len(walls) < len(scenes)  # (a wall too tall to jump, or a pit)
    assert any("spring" in random_level(seed)["platform_kinds"] for seed in range(40))


def test_a_drop_from_a_tall_wall_with_speed_is_planned_by_letting_go():
    # (on the wall's top, running; the floor beyond is seen to the screen's edge: every jump pressed
    # on toward it overshoots, so a jump pressed a while and then released is tried)
    from retroagi.domains.smb.physics import NESPlayerMotion
    from retroagi.domains.smb.planner import Solid

    wall = Solid(160, WALL_TOP, 192, 240, "ground", ("platforms", 1))
    floor = Solid(192, FLOOR, 309, 240, "ground", ("platforms", 2))
    motion = NESPlayerMotion(x_speed=25, moving=1, facing=1)
    body = (168, WALL_TOP - 12, 178, WALL_TOP)
    goal = Goal("land", floor.ref, floor.x0, floor.x1, prefer="near", top=FLOOR)
    plan = planner.plan_jump(goal, motion, body, [wall, floor], [], 1)
    assert plan is not None and plan.outcome.solid.ref == floor.ref
    assert plan.buttons[-1] in (0, 5)  # (released before landing: nothing pressed toward it)
