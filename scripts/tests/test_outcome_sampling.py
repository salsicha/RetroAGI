"""Outcome sampling (framework/sampling.py,
docs/choose-and-control-plan.md phase 3): every candidate choice played
from a saved state by the executor, its outcome measured."""

from retroagi.domains.smb.choices import ENEMY_SLOTS, KINDS, SPACE, SURFACE_SLOTS
from retroagi.domains.smb.controller import Choice
from retroagi.domains.smb.domain import DOMAIN
from retroagi.domains.smb.sim.levels import FLOOR, SCROLL_X
from retroagi.domains.smb.sim.world import SimSource, SimWorld
from retroagi.framework.sampling import OUTCOME_FIELDS, outcome, sample

ENDS = DOMAIN.ends

DIED, DIED_AFTER, FRAMES, PROGRESS = (OUTCOME_FIELDS.index(f) for f in ("died", "died_after", "frames", "progress"))


def scene(platforms, enemies=()):
    world = SimWorld({
        "world_width": 1024,
        "scroll_x": SCROLL_X,
        "mario": [40, FLOOR - 12],
        "platforms": platforms,
        "platform_kinds": ["ground"] * len(platforms),
        "enemies": [{"x": x, "y": FLOOR - 14, "game_rules": True} for x in enemies],
        "task": {"family": "game_level"},
    })
    world.reset()
    return world


def test_an_outcome_is_the_same_each_time_from_a_saved_state():
    world = scene([[0, FLOOR, 1024, 32]], enemies=[200])
    source, saved = SimSource(), world.snapshot()
    first = outcome(world, source, DOMAIN, Choice("run", x=150, keep_speed=True))
    world.restore(saved)
    assert outcome(world, source, DOMAIN, Choice("run", x=150, keep_speed=True)) == first


def test_a_run_stops_at_a_pit_and_a_goomba_reaching_a_wait_is_jumped():
    # A pit from 120 to 200: a long run toward it stops at its edge.
    pit = scene([[0, FLOOR, 120, 32], [200, FLOOR, 824, 32]])
    result = outcome(pit, SimSource(), DOMAIN, Choice("run", x=140, keep_speed=True))
    assert not (result[DIED] or result[DIED_AFTER]) and ENDS[int(result[-1])] == "edge"
    # A Goomba walking at a waiting Mario reaches him in the window after
    # the wait, but a jump from there over it saves him: no death.
    walker = scene([[0, FLOOR, 1024, 32]], enemies=[54])
    result = outcome(walker, SimSource(), DOMAIN, Choice("wait", frames=200))
    assert not (result[DIED] or result[DIED_AFTER])
    # Nothing near: a short wait is safe and takes its frames.
    safe = scene([[0, FLOOR, 1024, 32]])
    result = outcome(safe, SimSource(), DOMAIN, Choice("wait", frames=8))
    assert not (result[DIED] or result[DIED_AFTER]) and result[FRAMES] == 8
    assert ENDS[int(result[-1])] == "waited"


def test_candidates_cover_every_kind_and_tokens_have_their_slots():
    world = scene([[0, FLOOR, 1024, 32], [160, FLOOR - 32, 32, 64]], enemies=[230])
    for _ in range(4):  # the Goomba wakes once on screen
        world.step(0)
    state = SimSource().read(world)
    options = SPACE.candidates(state)
    # (no block overhead nor item here: neither bump nor collect)
    assert {KINDS[c.kind] for c in options} == {"land", "stomp", "pass", "run", "wait"}
    (mario, surfaces, enemies, items), slots = SPACE.tokens(state)
    assert surfaces.shape[0] == SURFACE_SLOTS and enemies.shape[0] == ENEMY_SLOTS
    assert all(c.ref in slots for c in options if c.ref is not None)
    assert mario[6] == 1.0  # standing


def test_a_level_gives_states_choices_and_outcomes_that_line_up():
    shard = sample(DOMAIN, 2, "speed_run", explore=0.3)
    states = len(shard["mario"])
    assert states > 10 and len(shard["choices"]) == len(shard["outcomes"])
    assert shard["choices"][:, 0].max() == states - 1


def test_a_skill_scene_is_sampled_until_it_is_passed_by_whatever_chooser_drives():
    from retroagi.domains.smb.scripted_chooser import ScriptedChooser

    shard = sample(DOMAIN, 5, "speed_run", kind="pit", explore=0.0)
    assert 1 <= len(shard["mario"]) < 30 and len(shard["choices"]) == len(shard["outcomes"])
    # The driver is whatever chooser is given (an outcome network's, in the making of training data).
    asked = []

    class Driver(ScriptedChooser):
        def decide(self, state, last_event, last):
            asked.append(last_event)
            return super().decide(state, last_event, last)

    again = sample(DOMAIN, 5, "speed_run", kind="pit", driver=Driver, explore=0.0)
    assert asked and len(again["mario"]) == len(shard["mario"])

    # A play whose driver gets no further is given up, not sampled to the end of its time.
    class Waiter(ScriptedChooser):
        def decide(self, state, last_event, last):
            return next(c.choice for c in SPACE.candidates(state) if KINDS[c.kind] == "wait")

    waited = sample(DOMAIN, 5, "speed_run", kind="pit", driver=Waiter, explore=0.0, stalled=60)
    assert 1 <= len(waited["mario"]) <= 10


def test_outcomes_sampled_before_a_kind_was_added_are_filled_out_with_zeros(tmp_path):
    import numpy as np

    from retroagi.framework.sampling import load

    shard = sample(DOMAIN, 5, "speed_run", kind="pit", explore=0.0)
    older = {k: (v[..., :-1] if k == "enemies" else v) for k, v in shard.items()}  # (one kind fewer)
    np.savez(tmp_path / "a.npz", **shard)
    np.savez(tmp_path / "b.npz", **older)
    joined = load(tmp_path)
    assert joined["enemies"].shape[-1] == shard["enemies"].shape[-1]
    assert len(joined["mario"]) == 2 * len(shard["mario"]) and not joined["enemies"][len(shard["mario"]):, :, -1].any()
