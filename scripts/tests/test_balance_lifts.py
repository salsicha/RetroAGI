"""Balance lifts in Block SMB (measured in 3-3's memory, Mario riding one,
and read from their routine in the game's program): the one Mario stands on
sinks and the other rises as much, as fast as the game's; left alone they
slow to a stop; pulled over the top the rope breaks and both fall; the
generator hangs them over bottomless pits as the game does, and Mario can
get across."""

from retroagi.domains.smb.sim import env as block
from retroagi.domains.smb.sim.levels import FLOOR, random_level, skill_scene
from retroagi.domains.smb.sim.passable import passable
from retroagi.domains.smb.sim.world import SimWorld

HIGH, LOW = 94, 126  # (3-3's first pair)


def scene():
    return {"world_width": 1000, "scroll_x": 115, "mario": [230, HIGH - 30], "frame_budget": 3000,
            "platforms": [[0, FLOOR, 200, 32], [500, FLOOR, 500, 32],
                          {"x": 216, "y": HIGH, "w": block.LIFT_W, "h": block.LIFT_H, "lift": "balance", "partner": 1},
                          {"x": 328, "y": LOW, "w": block.LIFT_W, "h": block.LIFT_H, "lift": "balance", "partner": -1}],
            "platform_kinds": ["ground", "ground", None, None], "enemies": [], "task": {"family": "game_level"}}


def lifts(world):
    return [p for p in world.env.platforms if p.get("lift") == "balance"]


def test_the_one_mario_stands_on_sinks_and_the_other_rises_as_the_games_do():
    world = SimWorld(scene())
    world.reset()
    first, other = lifts(world)
    while not world.env.mario["on_ground"]:
        world.step(0)
    tops = []
    for _ in range(91):
        world.step(0)
        tops.append((first["lift_y"], other["lift_y"]))
        assert world.env.mario["y"] + world.env.mario["h"] == first["rect"].y  # (carried down)
    # (the game's: 82 pixels down in the 91 frames after it starts, the other as far up)
    assert 80 <= tops[-1][0] - HIGH <= 84 and abs((tops[-1][0] - HIGH) - (LOW - tops[-1][1])) < 1e-9
    assert first["balance"] <= block.BALANCE_FASTEST


def test_left_alone_they_slow_to_a_stop():
    world = SimWorld(scene())
    world.reset()
    first, other = lifts(world)
    while not world.env.mario["on_ground"]:
        world.step(0)
    for _ in range(40):
        world.step(0)
    world.env.mario.update(x=600.0, y=FLOOR - 40.0)  # (off them)
    speeds = []
    for _ in range(60):
        world.step(0)
        speeds.append(first["balance"])
    assert speeds[0] > 0 and speeds[-1] == 0.0
    slowing = [a - b for a, b in zip(speeds, speeds[1:]) if b > 0]
    assert all(abs(s - block.BALANCE_PULL) < 1e-9 for s in slowing)


def test_pulled_over_the_top_the_rope_breaks_and_both_fall():
    world = SimWorld(scene())
    world.reset()
    first, other = lifts(world)
    frames = 0
    while not first["falling"]:
        world.step(0)
        frames += 1
    # (the game's: the other came to 44 and both fell, Mario with his)
    assert other["lift_y"] <= block.BALANCE_TOP + 1
    tops = []
    for _ in range(40):
        world.step(0)
        tops.append((first["lift_y"], other["lift_y"]))
    falls = [b[0] - a[0] for a, b in zip(tops, tops[1:])]
    assert all(b > a for a, b in zip(falls[:20], falls[1:21])) and max(falls) == block.BALANCE_FASTEST
    assert world.env.mario["y"] + world.env.mario["h"] == first["rect"].y or not world.env.mario["on_ground"]


def test_the_generator_hangs_them_over_bottomless_pits_and_mario_gets_across():
    for seed in range(20):
        s = skill_scene(seed, "balance_lifts")
        pair = [(k, p) for k, p in enumerate(s["platforms"]) if isinstance(p, dict) and p.get("lift") == "balance"]
        assert len(pair) == 2 and pair[1][0] == pair[0][0] + 1
        (_, high), (_, low) = pair
        assert (high["partner"], low["partner"]) == (1, -1) and high["y"] < low["y"] and high["x"] < low["x"]
        under = [p for p in s["platforms"] if isinstance(p, list) and p[0] < low["x"] + 48 and high["x"] < p[0] + p[2]]
        assert not under  # (nothing under them: a bottomless pit)
        if seed < 8:
            assert passable(s)["passable"], seed
    assert any(isinstance(p, dict) and p.get("lift") == "balance"
               for seed in range(80) for p in random_level(seed)["platforms"])
