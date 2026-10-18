"""Collision lessons (user, 2026-10-18): one scripted action of Mario's, run,
jump or wait, against an enemy, swept over every parameter; for the
collision trackers alone."""

import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

from types import SimpleNamespace

from retroagi.stages.block_smb.collision_families import (
    COLLISION_FAMILIES,
    DIFFICULTIES,
    every_layout,
    random_layouts,
)
from retroagi.stages.block_smb.layered_train import learner_families


def test_every_family_sweeps_one_scripted_action_with_an_enemy():
    modes = {"run": 0, "jump": 0, "hold": 0}
    for family in COLLISION_FAMILIES:
        layouts = every_layout(family, "medium")
        assert len(layouts) >= 24, family
        assert len({repr(s) for s, _ in layouts}) == len(layouts), family  # no repeats
        for scenario, params in layouts:
            assert scenario["enemies"] and scenario["goal_on_stomp"]
            modes[scenario["scripted_action"]["mode"]] += 1
            assert params["difficulty"] == "medium"
    assert all(modes.values())  # runs, jumps and waits


def test_random_layouts_are_sweep_layouts():
    for family in COLLISION_FAMILIES[:4]:
        every = {repr(s) for d in DIFFICULTIES for s, _ in every_layout(family, d)}
        assert all(repr(s) in every for s, _ in random_layouts(family, "hard", 5, seed=3))


def test_collision_lessons_are_kept_out_of_the_layers_training():
    from retroagi.stages.block_smb.monte_carlo import BLOCK_SMB_MC_FAMILIES

    # They are not Block SMB families the layers train on (the trainers'
    # families: BLOCK_SMB_MC_FAMILIES, split by learner_families).
    assert not set(COLLISION_FAMILIES) & set(BLOCK_SMB_MC_FAMILIES)
    for learner in ("skill", "tactic"):
        assert not set(COLLISION_FAMILIES) & set(learner_families(learner, BLOCK_SMB_MC_FAMILIES))


def test_a_lesson_ending_with_its_action_is_a_miss_and_a_timeout_a_failure():
    from retroagi.stages.block_smb.collision_train import touched

    assert touched(SimpleNamespace(end="action_done")) == "miss"
    assert touched(SimpleNamespace(end="timeout")) is None
