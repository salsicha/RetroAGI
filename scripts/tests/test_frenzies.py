"""Frenzies in Block SMB (sim/env.py; measured in Worlds 5-3 and 2-3's
memory): Bullet Bills from the right every 240 frames at three heights in
turn, leaping Cheep Cheeps from below near Mario, only while Mario is in
the frenzy's stretch; how the state and the planner show a Cheep Cheep, and
that a saved state carries a frenzy exactly."""

from retroagi.domains.smb.planner import enemy_paths
from retroagi.domains.smb.sim import env as block
from retroagi.domains.smb.sim.levels import FLOOR, random_level
from retroagi.domains.smb.sim.world import SimSource, SimWorld


def scene(kind, start=0.0, until=1e9):
    return {
        "world_width": 2000, "scroll_x": 115, "mario": [100.0, FLOOR - 12],
        "platforms": [[0, FLOOR, 2000, 32]], "platform_kinds": ["ground"], "enemies": [],
        "frenzies": [{"kind": kind, "from": start, "until": until, "turn": 7}],
        "task": {"family": "game_level"},
    }


def comers(world, kind, frames):
    """(frame, thing) of each ``kind`` that came in ``frames`` frames."""
    seen, out = set(), []
    for frame in range(frames):
        world.step(0)
        for e in world.env.enemies:
            if e.get("kind") == kind and id(e) not in seen:
                seen.add(id(e))
                out.append((frame, e))
    return out


def test_bullet_bills_come_from_the_right_every_240_frames_at_three_heights_in_turn():
    world = SimWorld(scene("bills"))
    world.reset()
    bills = comers(world, "bullet", 1000)
    assert [f for f, _ in bills] == [0, 240, 480, 720, 960]
    assert [b["y"] for _, b in bills] == [118.0, 166.0, 70.0, 118.0, 166.0]
    assert all(b["direction"] == -1 and b["speed"] == block.BULLET_SPEED for _, b in bills)


def test_nothing_comes_while_mario_is_outside_the_stretch():
    world = SimWorld(scene("bills", start=500.0))
    world.reset()
    assert not comers(world, "bullet", 300)


def test_cheep_cheeps_leap_up_near_mario_and_the_planner_follows_one():
    world = SimWorld(scene("cheeps"))
    world.reset()
    cheeps = comers(world, "cheep", 600)
    assert 4 <= len(cheeps) <= 40
    middle = world.env.mario["x"] + world.env.mario["w"] / 2
    _, first = cheeps[0]
    assert all(abs(c["x"] - middle) <= block.CHEEP_REACH + 1 for _, c in cheeps[:1])
    world = SimWorld(scene("cheeps"))
    world.reset()
    while True:  # (until one has come up onto the screen)
        world.step(0)
        state = SimSource().read(world)
        leapers = [e for e in state.enemies if e.kind == "leaper"]
        if leapers:
            break
    leaper = leapers[0]
    assert leaper.stompable
    real = world.env.enemies[leaper.ref[1]]
    enemy_paths([leaper], state.solids, 40)
    for frame in range(1, 41):
        world.step(0)
        assert abs(leaper.path[frame][1] - real["y"]) < 0.01 and abs(leaper.path[frame][0] - real["x"]) < 0.01


def test_a_saved_state_carries_a_frenzy_exactly():
    world = SimWorld(scene("cheeps"))
    world.reset()
    for _ in range(50):
        world.step(0)
    saved = world.snapshot()
    first = [(round(e["x"], 3), round(e["y"], 3)) for e in comers(world, "cheep", 300) for e in [e[1]]]
    world.restore(saved)
    again = [(round(e["x"], 3), round(e["y"], 3)) for e in comers(world, "cheep", 300) for e in [e[1]]]
    assert first == again


def test_random_levels_have_both_frenzies():
    kinds = [f["kind"] for seed in range(60) for f in random_level(seed)["frenzies"]]
    assert "bills" in kinds and "cheeps" in kinds
