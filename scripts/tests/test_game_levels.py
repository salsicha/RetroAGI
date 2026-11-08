"""Random levels of the real game's pieces in Block SMB
(stages/block_smb/game_levels.py, docs/choose-and-control-plan.md phase 2)."""

from retroagi.stages.block_smb.env import MarioScenarioEnv
from retroagi.stages.block_smb.game_levels import (
    FLOOR,
    SCROLL_X,
    BlockTruthState,
    play,
    random_level,
)
from retroagi.stages.full_smb.scripted_chooser import ScriptedChooser


def test_a_seed_always_makes_the_same_level_of_the_games_pieces():
    level = random_level(7)
    assert level == random_level(7) and level != random_level(8)
    kinds = set(level["platform_kinds"])
    assert kinds <= {"ground", "pipe", "brick"}
    assert all(e["game_rules"] for e in level["enemies"])
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
