"""Random levels of the real game's pieces in Block SMB
(domains/smb/sim/levels.py, docs/choose-and-control-plan.md phase 2)."""

from retroagi.domains.smb.scripted_chooser import ScriptedChooser
from retroagi.domains.smb.sim.env import MarioScenarioEnv
from retroagi.domains.smb.sim.levels import FLOOR, SCROLL_X, BlockTruthState, play, random_level


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
    level = random_level(3)
    env = MarioScenarioEnv(world_width=level["world_width"])
    env.reset(scenario=level)
    state = BlockTruthState().read(env)
    assert state.grounded and state.body[3] == FLOOR
    assert all(s.x1 <= env.camera_x + env.width for s in state.solids)
    assert [s.kind for s in state.solids].count("edge") == 1
    # Goombas beyond the screen have not woken: none is shown.
    assert all(e.box[0] < env.camera_x + env.width for e in state.enemies)


def test_the_scripted_chooser_finishes_a_random_level():
    result = play(random_level(1), ScriptedChooser("speed_run"))
    assert result["end"] == "finished"


def test_the_levels_hold_5_1s_plants_paratroopas_and_cannons():
    levels = [random_level(seed) for seed in range(40)]
    assert any("cannon" in level["platform_kinds"] for level in levels)
    kinds = {e.get("kind") for level in levels for e in level["enemies"]}
    assert {"piranha_plant", "paratroopa", "koopa", None} <= kinds
