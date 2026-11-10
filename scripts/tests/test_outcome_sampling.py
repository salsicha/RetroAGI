"""Outcome sampling (stages/block_smb/outcome_sampling.py,
docs/choose-and-control-plan.md phase 3): every candidate choice played
from a saved state by the executor, its outcome measured."""

from retroagi.core.smb_controller import Choice
from retroagi.stages.block_smb.env import MarioScenarioEnv
from retroagi.stages.block_smb.env_state import restore_env_state, snapshot_env_state
from retroagi.stages.block_smb.game_levels import FLOOR, SCROLL_X, BlockTruthState
from retroagi.core.smb_choices import ENEMY_SLOTS, KINDS, SURFACE_SLOTS, candidates, state_tokens
from retroagi.stages.block_smb.outcome_sampling import ENDS, OUTCOME_FIELDS, outcome, sample_level

DIED, DIED_AFTER, FRAMES, PROGRESS = (OUTCOME_FIELDS.index(f) for f in ("died", "died_after", "frames", "progress"))


def scene(platforms, enemies=()):
    env = MarioScenarioEnv(world_width=1024)
    env.draws = False
    env.reset(scenario={
        "world_width": 1024,
        "scroll_x": SCROLL_X,
        "mario": [40, FLOOR - 12],
        "platforms": platforms,
        "platform_kinds": ["ground"] * len(platforms),
        "enemies": [{"x": x, "y": FLOOR - 14, "game_rules": True} for x in enemies],
        "task": {"family": "game_level"},
    })
    return env


def test_an_outcome_is_the_same_each_time_from_a_saved_state():
    env = scene([[0, FLOOR, 1024, 32]], enemies=[200])
    source, snapshot = BlockTruthState(), snapshot_env_state(env)
    first = outcome(env, source, Choice("run", x=150, keep_speed=True))
    restore_env_state(env, snapshot)
    assert outcome(env, source, Choice("run", x=150, keep_speed=True)) == first


def test_a_run_stops_at_a_pit_and_a_goomba_reaching_a_wait_is_jumped():
    # A pit from 120 to 200: a long run toward it stops at its edge.
    pit = scene([[0, FLOOR, 120, 32], [200, FLOOR, 824, 32]])
    result = outcome(pit, BlockTruthState(), Choice("run", x=140, keep_speed=True))
    assert not (result[DIED] or result[DIED_AFTER]) and ENDS[int(result[-1])] == "edge"
    # A Goomba walking at a waiting Mario reaches him in the window after
    # the wait, but a jump from there over it saves him: no death.
    walker = scene([[0, FLOOR, 1024, 32]], enemies=[54])
    result = outcome(walker, BlockTruthState(), Choice("wait", frames=200))
    assert not (result[DIED] or result[DIED_AFTER])
    # Nothing near: a short wait is safe and takes its frames.
    safe = scene([[0, FLOOR, 1024, 32]])
    result = outcome(safe, BlockTruthState(), Choice("wait", frames=8))
    assert not (result[DIED] or result[DIED_AFTER]) and result[FRAMES] == 8
    assert ENDS[int(result[-1])] == "waited"


def test_candidates_cover_every_kind_and_tokens_have_their_slots():
    env = scene([[0, FLOOR, 1024, 32], [160, FLOOR - 32, 32, 64]], enemies=[230])
    for _ in range(4):  # the Goomba wakes once on screen
        env.step(0)
    state = BlockTruthState().read(env)
    options = candidates(state)
    # (no block overhead nor item here: neither bump nor collect)
    assert {KINDS[kind] for _, kind, _, _ in options} == {"land", "stomp", "pass", "run", "wait"}
    mario, surfaces, enemies, items, slots = state_tokens(state)
    assert surfaces.shape[0] == SURFACE_SLOTS and enemies.shape[0] == ENEMY_SLOTS
    assert all(ref in slots for _, _, ref, _ in options if ref is not None)
    assert mario[6] == 1.0  # standing


def test_a_level_gives_states_choices_and_outcomes_that_line_up():
    shard = sample_level(2, "speed_run", explore=0.3)
    states = len(shard["mario"])
    assert states > 10 and len(shard["choices"]) == len(shard["outcomes"])
    assert shard["choices"][:, 0].max() == states - 1
