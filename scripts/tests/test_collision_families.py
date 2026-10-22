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


def test_a_plant_seen_going_down_is_out_of_sight_when_the_action_starts():
    # The trackers must follow it from memory (user, 2026-10-21).
    from retroagi.core.smb_scene_labels import scene_from_labels
    from retroagi.stages.block_smb.env import MarioScenarioEnv

    def plants(labels):
        return [e for e in scene_from_labels(labels).enemies if e.kind == "plant"]

    for family in ("collide_run_plant", "collide_jump_plant"):
        seen = [(s, p) for s, p in every_layout(family, "medium") if p["gone"]]
        assert seen and len(seen) * 2 == len(every_layout(family, "medium"))
        scenario, params = seen[0]
        env = MarioScenarioEnv()
        try:
            env.reset(scenario=scenario, seed=0)
            watched = env.watched_labels
            assert plants(watched[0])  # seen at first
            assert not any(plants(labels) for labels in watched[-params["gone"] :])
            assert not plants(env.scene_labels())  # still down as the action starts
            for _ in range(params["after"] + 4):
                env.step(0)
            # Due up ``after`` frames in: up unless Mario, standing, is near
            # it (it waits for him, as the NES's plants do).
            plant, mario = env.enemies[0], env.mario
            near = abs(plant["x"] + plant["w"] / 2 - (mario["x"] + mario["w"] / 2)) <= 33
            assert bool(plants(env.scene_labels())) != near
        finally:
            env.close()


def test_held_out_layouts_are_none_of_the_sweeps():
    from retroagi.stages.block_smb.collision_families import held_out_layouts

    for family in COLLISION_FAMILIES:
        grid = {repr(s) for d in DIFFICULTIES for s, _ in every_layout(family, d)}
        held = held_out_layouts(family, "medium", 6, seed=0)
        assert not any(repr(s) in grid for s, _ in held), family
        assert held == held_out_layouts(family, "medium", 6, seed=0)  # the same each time


def test_a_plant_stays_down_while_mario_is_near_and_keeps_its_cycle_otherwise():
    # Every Block SMB plant, as the NES's (user, 2026-10-21).
    from retroagi.stages.block_smb.env import MarioScenarioEnv

    def time_up(mario_x):
        plant = {"kind": "piranha_plant", "x": 150, "pipe_top": 196, "plant_height": 20,
                 "rise_frames": 8, "exposed_frames": 16, "hidden_frames": 16, "phase": 0}
        env = MarioScenarioEnv()
        try:
            env.reset(scenario={"world_width": 400, "mario": [mario_x, 204],
                                "platforms": [[0, 220, 400, 20], [140, 196, 32, 24]],
                                "platform_kinds": ["ground", "pipe"], "enemies": [plant],
                                "goal": [380, 200, 16, 20]})
            up = 0
            for _ in range(200):
                env.step(0)
                up += env.enemies[0]["h"] > 0
            return up
        finally:
            env.close()

    assert time_up(130) < 40  # Mario's middle 21 pixels away: up once, then down
    assert time_up(40) > 80  # far: it keeps coming up


def test_pipe_wait_lessons_watch_the_plant_go_down_and_wait():
    layouts = every_layout("collide_pipe_wait", "easy")
    waits = {p["wait"] for _, p in layouts}
    assert waits == {60, 90, 120, 150, 210, 300}
    for scenario, params in layouts[:8]:
        (plant,) = scenario["enemies"]
        assert scenario["watch_frames"] == plant["rise_frames"] + params["wait"]
        assert plant["phase"] == plant["rise_frames"] + plant["exposed_frames"]  # starts going down
