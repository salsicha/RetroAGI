"""Bowser in Block SMB (measured in the castles' memory, 1-4's and 6-4's,
and read from his routine in the game's program): he paces about where he
starts at a quarter of a pixel a frame, jumps as high and as often as the
game's, breathes flames at the game's intervals that fly left and settle
at the game's heights, follows Mario who gets behind him, hurts and cannot
be stomped; in worlds 6 to 8 he throws hammers in the air, as many as the
game's; free flames come before him; the planner follows his fall and his
flames; the generator builds the castles' bridge, each of the game's eight
castles as often."""

import math
from collections import Counter

from retroagi.domains.smb.planner import enemy_paths
from retroagi.domains.smb.sim import env as block
from retroagi.domains.smb.sim.levels import BOWSER_BRIDGE, BOWSER_IN, FLOOR, TILE, random_level, skill_scene
from retroagi.domains.smb.sim.world import SimSource, SimWorld

BRIDGE = 640
ORIGIN = BRIDGE + 7 * TILE


def scene(mario_x=BRIDGE - 8, turn=1, frenzy=False, breathes=True, hammers=False):
    return {"world_width": 1400, "scroll_x": 115, "mario": [mario_x, FLOOR - 12], "frame_budget": 9000,
            "platforms": [[0, FLOOR, BRIDGE, 32], [BRIDGE, FLOOR, 13 * TILE, TILE],
                          [BRIDGE + 13 * TILE, FLOOR, 600, 32]],
            "platform_kinds": ["ground", "ground", "ground"],
            "enemies": [{"kind": "bowser", "x": ORIGIN, "y": FLOOR - block.BOWSER_FEET, "turn": turn,
                         "breathes": breathes, "hammers": hammers}],
            "frenzies": [{"kind": "flames", "from": 0, "until": 2000}] if frenzy else [],
            "task": {"family": "game_level"}}


def bowser_of(world):
    return next(e for e in world.env.enemies if e.get("kind") == "bowser")


def watch(world, frames):
    """Bowser's place each frame (from where he started and from his
    ground), and the frames flames came."""
    bowser, xs, ys, flames = bowser_of(world), [], [], []
    known = len(world.env.enemies)
    for frame in range(frames):
        world.step(0)
        xs.append(bowser["px"] - bowser["origin"])
        ys.append(bowser["py"] - bowser["ground"])
        if len(world.env.enemies) > known:
            flames.append(frame)
            known = len(world.env.enemies)
    return xs, ys, flames


def test_he_paces_and_jumps_as_the_game_does():
    gaps, air, tops, speeds = Counter(), set(), [], []
    for turn in range(1, 6):
        world = SimWorld(scene(turn=turn))
        world.reset()
        xs, ys, _ = watch(world, 1500)
        # (the game's: within 66 of where he started, 0.23 pixels a frame)
        assert max(abs(x) for x in xs) <= max(block.BOWSER_RANGES) + 1
        speeds.append(sum(abs(b - a) for a, b in zip(xs, xs[1:])) / len(xs))
        takeoffs = [k for k in range(1, len(ys)) if ys[k] < 0 <= ys[k - 1]]
        gaps.update(b - a for a, b in zip(takeoffs, takeoffs[1:]))
        for t in takeoffs:
            landing = next((k for k in range(t, len(ys)) if ys[k] >= 0), None)
            if landing is not None:
                air.add(landing - t)
        tops.append(min(ys))
    # the game's in 1-4, Mario before him: takeoff to takeoff 86, 102, 118
    # or 134 frames, 69 or 70 in the air, 37 pixels up
    assert set(gaps) == {86, 102, 118, 134}
    assert air <= {69, 70}
    assert all(-37.5 <= top <= -35.5 for top in tops)
    assert all(0.19 <= s <= 0.26 for s in speeds)


def test_his_flames_come_at_the_games_intervals_and_settle_at_its_heights():
    world = SimWorld(scene())
    world.reset()
    _, _, flames = watch(world, 1400)
    # (his mouth opens 223 frames after he comes on screen, shuts 32 later)
    assert flames[0] == block.BOWSER_FIRST_FLAME + block.BOWSER_MOUTH - 1
    assert [b - a for a, b in zip(flames, flames[1:])] == [w + block.BOWSER_MOUTH for w in block.FLAME_WAITS[:len(flames) - 1]]
    world = SimWorld(scene())
    world.reset()
    watch(world, flames[0] + 1)
    flame = next(e for e in world.env.enemies if e.get("kind") == "flame")
    bowser = bowser_of(world)
    assert flame["fx"] == bowser["px"] + block.FLAME_MOUTH[0] - flame["speed"]
    start = flame["fx"]
    for frame in range(60):
        world.step(0)
    assert abs(start - flame["fx"] - 60 * 1.25) < 1e-6
    assert flame["fy"] == flame["level"] and flame["level"] in block.FLAME_LEVELS
    # (a hard castle's: 16 frames sooner, faster)
    hard = scene()
    hard["enemies"][0]["hard"] = True
    world = SimWorld(hard)
    world.reset()
    _, _, flames = watch(world, 1400)
    assert [b - a for a, b in zip(flames, flames[1:])] == [
        w + block.BOWSER_MOUTH - block.FLAME_HARD for w in block.FLAME_WAITS[:len(flames) - 1]]
    assert next(e for e in world.env.enemies if e.get("kind") == "flame")["speed"] == 1.375


def test_mario_behind_him_he_follows_and_stops_short_of_the_axe():
    world = SimWorld(scene(mario_x=ORIGIN + 120))
    world.reset()
    xs, ys, flames = watch(world, 600)
    bowser = bowser_of(world)
    assert bowser["facing"] == 1 and bowser["x"] == bowser["px"] - block.BOWSER_BACK
    # (the game's: 89 right of where he started, then still; no jump, no flame)
    assert max(xs) in (block.BOWSER_CHASE_LIMIT, block.BOWSER_CHASE_LIMIT + 1) and xs[-1] == xs[-100]
    assert min(ys[60:]) == 0 and not flames


def test_he_hurts_and_cannot_be_stomped():
    world = SimWorld(scene(mario_x=ORIGIN + 8, breathes=False))
    world.reset()
    world.env.mario.update(y=FLOOR - 120.0, on_ground=False)
    for _ in range(200):
        step = world.step(0)
        if step.over:
            break
    assert step.over and step.died and not world.env.stomped


def test_in_worlds_6_to_8_he_throws_hammers_in_the_air():
    per_jump, flying = [], 0
    for turn in range(1, 7):
        world = SimWorld(scene(turn=turn, breathes=False, hammers=True))
        world.reset()
        bowser, came, jump = bowser_of(world), {}, None
        for frame in range(1500):
            known = len(world.env.enemies)
            world.step(0)
            for k in range(known, len(world.env.enemies)):
                hammer = world.env.enemies[k]
                # (only in the air, or as he lands)
                assert hammer["kind"] == "hammer" and (bowser["jump_timer"] == 0 or bowser["py"] == bowser["ground"])
                # (its place, the corner of its hurt box: 2 right of his and 10 up, as the game's)
                assert (hammer["x"], hammer["y"]) == (bowser["px"] + 2, math.floor(bowser["py"]) - 10)
                came[k] = [frame, hammer["x"]]
                if jump is not None:
                    assert (frame - jump[0]) % block.BOWSER_EVERY == 0  # (a try once in 4 frames)
                    jump[1] += 1
                else:
                    jump = [frame, 1]
            for k, (start, x) in came.items():
                hammer = world.env.enemies[k]
                if frame - start == 14:  # (held through the 15th frame, as the game's, then thrown his way: left)
                    assert hammer["x"] == bowser["px"] + 2 and hammer["held"] == 0
                if frame - start == 15:
                    assert hammer["x"] == bowser["px"] + 2 - 1 and hammer["direction"] == bowser["facing"] == -1
            flying = max(flying, sum(not world.env.enemies[k].get("gone") for k in came))
            if bowser["jump_timer"] and jump is not None:
                per_jump.append(jump[1])
                jump = None
    # (the game's draw of nine places: 6-4's, Mario mostly behind him so
    # that he jumped less often, 6 to 9 a jump; here, jumping as often as
    # in front of Mario, about 6)
    assert flying <= block.HAMMER_PLACES and max(per_jump) <= 9 and 5.5 <= sum(per_jump) / len(per_jump) <= 7
    world = SimWorld(scene(breathes=False))
    world.reset()
    for _ in range(600):
        world.step(0)
    assert not any(e.get("kind") == "hammer" for e in world.env.enemies)  # (worlds 1 to 5: none)


def test_free_flames_come_before_him_and_stop_when_he_comes():
    world = SimWorld(scene(mario_x=120, frenzy=True))
    world.reset()
    known, came = len(world.env.enemies), []
    for frame in range(700):
        world.step(0)
        if len(world.env.enemies) > known:
            came.append(frame)
            flame = world.env.enemies[-1]
            assert flame["kind"] == "flame" and flame["fy"] == flame["level"] in block.FLAME_LEVELS
            assert flame["fx"] == world.env.camera_x + block.FLAME_FROM
            known = len(world.env.enemies)
    assert not bowser_of(world).get("awake")
    assert [b - a for a, b in zip(came, came[1:])] == [w + block.BOWSER_MOUTH for w in block.FLAME_WAITS[:len(came) - 1]]
    # (Bowser on screen: only his own)
    world = SimWorld(scene(frenzy=True, breathes=False))
    world.reset()
    for _ in range(20):
        world.step(0)
    known = len(world.env.enemies)
    for _ in range(600):
        world.step(0)
    assert len(world.env.enemies) == known


def test_the_planner_follows_his_fall_and_his_flames():
    world = SimWorld(scene())
    world.reset()
    bowser = bowser_of(world)
    while bowser["jump_timer"] or bowser["vy"] < 0.5:  # (coming down)
        world.step(0)
    state = SimSource().read(world)
    seen = next(e for e in state.enemies if e.kind == "jumper")
    assert not seen.stompable and seen.vy > 0
    enemy_paths([seen], state.solids, 40)
    for frame in range(1, 41):
        world.step(0)
        assert abs(seen.path[frame][3] - (bowser["y"] + bowser["h"])) < 1e-6
    while not any(e.get("kind") == "flame" for e in world.env.enemies):
        world.step(0)
    world.step(0)
    state = SimSource().read(world)
    seen = next(e for e in state.enemies if e.kind == "flame")
    flame = next(e for e in world.env.enemies if e.get("kind") == "flame")
    assert not seen.stompable and seen.vx == -1.25
    enemy_paths([seen], state.solids, 50)
    for frame in range(1, 51):
        world.step(0)
        assert abs(seen.path[frame][0] - flame["x"]) < 1e-6 and abs(seen.path[frame][1] - flame["y"]) < 1e-6


def test_the_generator_builds_the_castles_bridge():
    for seed in range(20):
        s = skill_scene(seed, "bowser")
        bowser = next(e for e in s["enemies"] if e["kind"] == "bowser")
        bridge = next(p for p in s["platforms"] if isinstance(p, list) and p[1] == FLOOR and p[3] == TILE)
        assert bridge[2] == BOWSER_BRIDGE * TILE and bowser["x"] == bridge[0] + BOWSER_IN * TILE
        assert bowser["y"] + block.BOWSER_FEET == FLOOR
        floor = [p for p in s["platforms"] if isinstance(p, list) and p[1] == FLOOR and p[3] > TILE]
        assert not any(p[0] < bridge[0] + bridge[2] and bridge[0] < p[0] + p[2] for p in floor)  # (over lava)
        lift = next(p for p in s["platforms"] if isinstance(p, dict))
        assert lift["lift"] == "across" and bridge[0] < lift["x"] < bridge[0] + bridge[2]
        frenzy = next(f for f in s["frenzies"] if f["kind"] == "flames")
        assert frenzy["from"] < bridge[0] <= frenzy["until"]
    assert any(e.get("kind") == "bowser" for seed in range(60) for e in random_level(seed)["enemies"])
    # (the castle one of the game's eight: hard from world 5, hammers from 6, no flames in 6 and 7)
    castles = Counter()
    for seed in range(200):
        s = skill_scene(seed, "bowser")
        bowser = next(e for e in s["enemies"] if e["kind"] == "bowser")
        castles[(bowser["hard"], bowser["hammers"], bowser["breathes"])] += 1
        assert bowser["hard"] or not bowser["hammers"]
    assert set(castles) == {(False, False, True), (True, False, True), (True, True, False), (True, True, True)}
    assert 80 <= castles[(False, False, True)] <= 120  # (four in eight)
