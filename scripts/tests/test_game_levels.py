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
    assert kinds <= {"ground", "pipe", "brick", "question_block", "cannon"}
    # Every question block holds a coin or a mushroom.
    questions = {k for k, kind in enumerate(level["platform_kinds"]) if kind == "question_block"}
    assert set(level["block_contents"]) == questions
    assert set(level["block_contents"].values()) <= {"coin", "mushroom"}
    # Walkers under the game's rules; plants on pipes' tops.
    tops = {p[1] for p, k in zip(level["platforms"], level["platform_kinds"]) if k == "pipe"}
    for e in level["enemies"]:
        assert e["pipe_top"] in tops if e.get("kind") == "piranha_plant" else e["game_rules"]
    assert level["scroll_x"] == SCROLL_X
    # Pipes and bricks sit on the 16-pixel grid; the floor is cut where a
    # pipe stands.
    for (x, y, w, h), kind in zip(level["platforms"], level["platform_kinds"]):
        assert x % 16 == 0 and y % 16 == 0
        if kind == "pipe":
            assert not any(
                k == "ground" and fy == FLOOR and fx < x + w and x < fx + fw
                for (fx, fy, fw, _), k in zip(level["platforms"], level["platform_kinds"])
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
    rows = practice.run(DOMAIN, [1])
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
    assert DOMAIN.practices[0] == "level" and set(DOMAIN.practices[1:]) == set(SCENE_KINDS)
    world = DOMAIN.practice(7, "pit")
    world.reset()
    assert not world.passed and world.passed_at == skill_scene(7, "pit")["metadata"]["scene"]["passed"]
    rows = practice.run(DOMAIN, [], kinds=("pit", "pipe"), scenes=[100003])
    assert [r["kind"] for r in rows] == ["pit", "pit", "pipe", "pipe"]
    assert practice.summary(rows) == {"pit": (2, 2), "pipe": (2, 2)}  # (the hand-written chooser passes these)
