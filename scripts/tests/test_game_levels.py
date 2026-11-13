"""Random levels of the real game's pieces in Block SMB
(domains/smb/sim/levels.py, docs/choose-and-control-plan.md phase 2)."""

from retroagi.domains.smb.domain import DOMAIN
from retroagi.domains.smb.sim.levels import FLOOR, SCROLL_X, random_level
from retroagi.domains.smb.sim.world import SimSource, SimWorld
from retroagi.framework import practice


def test_a_seed_always_makes_the_same_level_of_the_games_pieces():
    level = random_level(7)
    assert level == random_level(7) and level != random_level(8)
    kinds = set(level["platform_kinds"])
    assert kinds <= {"ground", "pipe", "brick", "question_block", "cannon", "spring", None}  # (None: a lift)
    # Every question block holds a coin or a mushroom.
    questions = {k for k, kind in enumerate(level["platform_kinds"]) if kind == "question_block"}
    assert set(level["block_contents"]) == questions
    assert set(level["block_contents"].values()) <= {"coin", "mushroom"}
    # Walkers under the game's rules; plants on pipes' tops; fire bars on blocks.
    blocks = [(p, k) for p, k in zip(level["platforms"], level["platform_kinds"]) if isinstance(p, list)]
    tops = {p[1] for p, k in blocks if k == "pipe"}
    for e in level["enemies"]:
        if e.get("kind") == "piranha_plant":
            assert e["pipe_top"] in tops
        elif e.get("kind") == "firebar":
            assert e["x"] % 16 == 0 and e["y"] % 16 == 0
        else:
            assert e["game_rules"]
    assert level["scroll_x"] == SCROLL_X
    # Pipes and bricks sit on the 16-pixel grid (lifts and springboards,
    # things that move, need not); the floor is cut where a pipe stands.
    for (x, y, w, h), kind in blocks:
        assert x % 16 == 0 and y % 16 == 0
        if kind == "pipe":
            assert not any(
                k == "ground" and fy == FLOOR and fx < x + w and x < fx + fw
                for (fx, fy, fw, _), k in blocks
            )
    assert level["goal"][0] < level["world_width"]


def test_the_state_shows_only_the_screen_and_its_left_edge():
    world = SimWorld(random_level(3))
    world.reset()
    env = world.env
    state = SimSource().read(world)
    assert state.grounded and state.body[3] == FLOOR
    assert all(s.x1 <= env.camera_x + env.width for s in state.solids)
    assert [s.kind for s in state.solids].count("edge") == 1
    # Goombas beyond the screen have not woken: none is shown.
    assert all(e.box[0] < env.camera_x + env.width for e in state.enemies)


def test_the_scripted_chooser_finishes_a_random_level():
    rows = practice.run(DOMAIN, [18])
    assert rows[0]["mode"] == "speed_run" and rows[0]["end"] == "finished"


def test_the_levels_hold_5_1s_plants_paratroopas_and_cannons():
    levels = [random_level(seed) for seed in range(40)]
    assert any("cannon" in level["platform_kinds"] for level in levels)
    kinds = {e.get("kind") for level in levels for e in level["enemies"]}
    assert {"piranha_plant", "paratroopa", "koopa", None} <= kinds


def test_a_skill_scene_is_one_piece_of_its_kind_with_no_flag_and_is_passed_beyond_it():
    from retroagi.domains.smb.sim.levels import SCENE_KINDS, TILE, skill_scene

    for kind in SCENE_KINDS:
        scene = skill_scene(7, kind)
        assert scene == skill_scene(7, kind) and "goal" not in scene
        assert scene["metadata"]["scene"]["kind"] == kind
        assert scene["world_width"] - scene["metadata"]["scene"]["passed"] >= 20 * TILE  # floor to spare beyond it
    # Its piece is of its kind: a pipe scene has a pipe, a pit scene a gap in the floor.
    assert "pipe" in skill_scene(7, "pipe")["platform_kinds"]
    floor = [p for p, k in zip(skill_scene(7, "pit")["platforms"], skill_scene(7, "pit")["platform_kinds"])
             if k == "ground" and p[1] == FLOOR]
    assert len(floor) == 2 and floor[1][0] - (floor[0][0] + floor[0][2]) in (2 * TILE, 3 * TILE, 4 * TILE)
    # The domain gives scenes as practice worlds; one is passed once Mario stands beyond its piece.
    # (and nothing else: no practice world has a real level's layout)
    assert DOMAIN.practices == ("level", *SCENE_KINDS)
    world = DOMAIN.practice(7, "pit")
    world.reset()
    assert not world.passed and world.passed_at == skill_scene(7, "pit")["metadata"]["scene"]["passed"]
    rows = practice.run(DOMAIN, [], kinds=("pit", "pipe"), scenes=[100003])
    assert [r["kind"] for r in rows] == ["pit", "pit", "pipe", "pipe"]
    assert practice.summary(rows) == {"pit": (2, 2), "pipe": (2, 2)}  # (the hand-written chooser passes these)


def test_the_levels_hold_the_pieces_measured_in_worlds_2_1_3_1_and_5_1():
    from retroagi.domains.smb.sim.levels import (
        BETWEEN_PITS,
        BOTTOM,
        BRIDGE_TILES,
        REACH,
        TILE,
        skill_scene,
    )

    def solids(scene):
        return [(p, kind) for p, kind in zip(scene["platforms"], scene["platform_kinds"])]

    tall = 0
    for seed in range(80):
        # A pipe too tall to jump onto from the floor has a lower thing to jump from just before it.
        scene = skill_scene(seed, "pipe")
        for p, kind in solids(scene):
            if kind == "pipe" and FLOOR - p[1] > REACH * TILE:
                tall += 1
                assert any(q[1] < FLOOR and 0 <= p[0] - (q[0] + q[2]) <= 4 * TILE and FLOOR - q[1] <= REACH * TILE
                           and q[1] - p[1] <= REACH * TILE for q, _ in solids(scene))
        # A bridge: one walkway in the air over a pit as long as it, steps up to it.
        scene = skill_scene(seed, "bridge")
        walkways = [p for p, kind in solids(scene) if kind == "ground" and p[1] + p[3] < FLOOR]
        assert len(walkways) == 1
        x0, top, width, _ = walkways[0]
        assert BRIDGE_TILES[0] <= width // TILE <= BRIDGE_TILES[1] and top == FLOOR - REACH * TILE
        assert not any(kind in ("ground", "pipe", "cannon") and p[1] + p[3] == BOTTOM and p[0] < x0 + width
                       and x0 < p[0] + p[2] for p, kind in solids(scene))
        heights = sorted(FLOOR - p[1] for p, kind in solids(scene) if kind == "ground" and p[0] + p[2] <= x0 and p[1] < FLOOR)
        assert heights == [TILE * k for k in range(1, REACH + 1)]
        # Enemies stand on what is under them: the floor, a step, a row or the walkway.
        for enemy in scene["enemies"]:
            assert enemy["y"] + 14 == top and x0 <= enemy["x"] < x0 + width
        # Two pits never run together, however close the pieces.
        pits = random_level(seed, 2, only="pit", flat=(0, 0), flag=False)
        floor = sorted(p for p, kind in solids(pits) if kind == "ground" and p[1] == FLOOR)
        assert len(floor) == 3 and floor[1][2] >= BETWEEN_PITS * TILE
    assert tall >= 5
    # Whole levels put pieces as close as the game does: some touch.
    assert any(level["world_width"] != random_level(seed, flat=(3, 8))["world_width"]
               for seed in range(5) for level in [random_level(seed)])


def test_every_kind_of_practice_world_can_be_got_through_by_some_presses():
    from retroagi.domains.smb.sim.levels import SCENE_KINDS, TILE, skill_scene
    from retroagi.domains.smb.sim.passable import passable

    for kind in SCENE_KINDS:
        for seed in range(100200, 100204):
            assert passable(skill_scene(seed, kind))["passable"], (kind, seed)
    assert passable(random_level(100200))["passable"]
    # It tells a world nobody could pass: the same pit made 14 tiles wide.
    scene = skill_scene(100200, "pit")
    far = max(range(len(scene["platforms"])), key=lambda k: scene["platforms"][k][0])
    edge = scene["platforms"][far][0]
    scene["platforms"][far][0] += 14 * TILE
    scene["platforms"][far][2] -= 14 * TILE
    scene["metadata"]["scene"]["passed"] += 14 * TILE
    found = passable(scene, expansions=1500)
    assert not found["passable"] and found["x"] < edge + 14 * TILE  # (no press carries him to its far side)
