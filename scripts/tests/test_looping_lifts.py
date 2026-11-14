"""Looping lifts in Block SMB (measured in 1-2's and 2-4's memory): they go
up or down 15/16 of a pixel a frame without end, come round from the other
end of the game's 256 rows, carry Mario (but not round), and the planner
follows them; the generator puts them in bottomless pits as the game does,
and Mario can get across."""

from retroagi.domains.smb.planner import _solid_paths
from retroagi.domains.smb.sim import env as block
from retroagi.domains.smb.sim.levels import FLOOR, random_level, skill_scene
from retroagi.domains.smb.sim.passable import passable
from retroagi.domains.smb.sim.world import SimSource, SimWorld


def scene(way, top, width=block.LIFT_W, mario=(100, FLOOR - 12)):
    return {"world_width": 1000, "scroll_x": 115, "mario": list(mario), "frame_budget": 3000,
            "platforms": [[0, FLOOR, 200, 32], [312, FLOOR, 600, 32], {"x": 224, "y": top, "w": width, "h": block.LIFT_H, "lift": way}],
            "platform_kinds": ["ground", "ground", None], "enemies": [], "task": {"family": "game_level"}}


def lift_of(world):
    return next(p for p in world.env.platforms if p.get("lift") in ("up", "down"))


def test_a_looping_lift_goes_round_as_the_games_do():
    for way, step in (("up", -1), ("down", 1)):
        world = SimWorld(scene(way, 100))
        world.reset()
        lift = lift_of(world)
        tops = []
        for _ in range(600):
            world.step(0)
            tops.append(lift["rect"].y)
        moves = [b - a for a, b in zip(tops, tops[1:])]
        # (the game's: a pixel a frame its way, still one frame in sixteen;
        # round the 256 rows in 273 frames)
        assert set(moves) <= {0, step, -step * 255, -step * 256}
        assert 0.06 <= moves.count(0) / len(moves) <= 0.07
        rounds = [k for k, m in enumerate(moves) if abs(m) > 100]
        assert len(rounds) == 2 and rounds[1] - rounds[0] in (273, 274)
        assert lift["vy"] == step * block.LIFT_LOOP_SPEED and 0 <= lift["lift_y"] < block.LIFT_LOOP


def test_it_carries_mario_but_not_round():
    world = SimWorld(scene("up", FLOOR - 40, mario=(230, FLOOR - 60)))
    world.reset()
    lift = lift_of(world)
    for _ in range(30):
        world.step(0)
    assert world.env.mario["on_ground"] and world.env.mario["y"] + world.env.mario["h"] == lift["rect"].y
    ridden, top = [], lift["rect"].y
    while lift["rect"].y <= top:  # (until it comes round)
        top = lift["rect"].y
        ridden.append(world.env.mario["y"] + world.env.mario["h"] - top)
        world.step(0)
    assert len(ridden) > 100 and all(r == 0 for r in ridden)  # (carried up all the way)
    mario = world.env.mario
    assert mario["y"] + mario["h"] <= 1 and not mario["on_ground"]  # (left at the top as it comes round)


def test_the_planner_follows_it():
    world = SimWorld(scene("down", 60))
    world.reset()
    world.step(0)
    state = SimSource().read(world)
    k = next(i for i, s in enumerate(state.solids) if s.kind == "moving")
    paths = _solid_paths(state.solids, 40)
    lift = lift_of(world)
    for frame in range(1, 41):
        world.step(0)
        assert abs(paths[frame, k, 1] - lift["lift_y"]) < 1.0


def test_the_generator_puts_them_in_bottomless_pits_and_mario_gets_across():
    shapes = set()
    for seed in range(30):
        s = skill_scene(seed, "loop_lifts")
        lifts = [p for p in s["platforms"] if isinstance(p, dict) and p.get("lift") in ("up", "down")]
        assert lifts and all(0 <= p["y"] < block.LIFT_LOOP for p in lifts)
        shapes.add((len(lifts), lifts[0]["w"], tuple(sorted({p["lift"] for p in lifts}))))
        if seed < 8:
            assert passable(s)["passable"], seed
    assert {(1, block.LIFT_W, ("down",)), (2, block.LIFT_W, ("up",)),
            (2, block.LIFT_SHORT, ("down", "up"))} <= shapes
    assert any(isinstance(p, dict) and p.get("lift") in ("up", "down")
               for seed in range(80) for p in random_level(seed)["platforms"])
