"""Skill targets are executed without a learned action network."""

import dataclasses

import pytest
import torch

from retroagi.core.layered_policy import LayeredSMBPolicy
from retroagi.core.smb_agent import SMBAgents
from retroagi.core.smb_scene_labels import scene_from_labels
from retroagi.core.tokens import SkillToken
from retroagi.stages.block_smb.env import MarioScenarioEnv


@pytest.mark.parametrize("distance", [-60, -2, -1, 0, 1, 2, 60, 160])
def test_direct_run_target_is_held_until_arrival_and_brakes(distance):
    env = MarioScenarioEnv()
    try:
        start = 80 if distance < 0 else 40
        screen, _ = env.reset(
            scenario={"world_width": 340, "mario": [start, 208], "platforms": [[0, 220, 340, 20]]}
        )

        class Observer:
            def observe(self, screens):
                return [scene_from_labels(env.scene_labels())]

        agent = SMBAgents(Observer(), LayeredSMBPolicy().eval(), "cpu")

        def given(copies, scenes):
            return {"skill": [SkillToken("run", distance, 0)]}

        steps = []
        for _ in range(192):
            step = agent.act([screen], [0], given=given)[0]
            steps.append(step)
            screen, _, done, _, _ = env.step(step.button)
            assert not done
            if step.execution_status == "arrived":
                break
        assert steps[-1].execution_status == "arrived"
        assert abs(env.mario["x"] - (start + distance)) <= 1
        if abs(distance) in (1, 2):
            assert env.mario["x"] == start + distance
        assert abs(env.mario["vx"]) <= 0.3
        assert sum(s.decision is not None for s in steps) == 1
        assert not hasattr(agent.policy, "action")
        if distance == 0:
            assert all(s.button == 0 for s in steps)
    finally:
        env.close()


def test_direct_jump_plans_its_own_hold_and_completes_the_gap():
    env = MarioScenarioEnv()
    try:
        screen, _ = env.reset(
            scenario={
                "world_width": 256,
                "mario": [95, 208],
                "platforms": [[0, 220, 110, 20], [125, 220, 131, 20]],
                "goal": [138, 200, 30, 20],
                "goal_requires_support": True,
            }
        )

        class Observer:
            def observe(self, screens):
                return [scene_from_labels(env.scene_labels())]

        agents = SMBAgents(Observer(), LayeredSMBPolicy().eval(), "cpu")

        def given(copies, scenes):
            return {"skill": [SkillToken("jump", 45, 0)]}

        buttons = []
        for _ in range(100):
            step = agents.act([screen], [0], given=given)[0]
            buttons.append(step.button)
            screen, _, done, _, _ = env.step(step.button)
            if done:
                break
        assert env._goal_credited
        jump_frames = [i for i, button in enumerate(buttons) if button in (2, 4, 5)]
        assert jump_frames == list(range(jump_frames[0], jump_frames[-1] + 1))
    finally:
        env.close()


def test_legacy_checkpoint_discards_only_action_weights(tmp_path):
    from retroagi.core.smb_observer import observation_layout
    from retroagi.core.tokens import token_layout
    from retroagi.stages.block_smb.layered_train import (
        LayeredTrainConfig,
        load_layered_checkpoint,
        save_layered_checkpoint,
    )

    policy = LayeredSMBPolicy()
    state = dict(policy.state_dict())
    state["action.network.0.weight"] = torch.ones(2, 3)
    tokens = token_layout()
    tokens.pop("executor")
    tokens["action_input"] = "skill_only_v1"
    source = tmp_path / "old.pt"
    torch.save(
        {
            "settings": dataclasses.asdict(policy.settings),
            "state_dict": state,
            "observation_layout": observation_layout(),
            "token_layout": tokens,
            "trained_layers": ["action", "skill"],
        },
        source,
    )
    loaded, metadata = load_layered_checkpoint(source)
    assert metadata["trained_layers"] == []
    assert "maneuver_targets_require_requalification" in metadata["load_migrations"]
    assert "removed_action_network" in metadata["load_migrations"]
    for name, value in policy.state_dict().items():
        assert torch.equal(value, loaded.state_dict()[name])
    config = LayeredTrainConfig(vision_checkpoint=str(source))
    output = tmp_path / "new.pt"
    save_layered_checkpoint(output, loaded, config, ["skill"], [])
    reloaded, saved = load_layered_checkpoint(output)
    assert "action_input" not in saved["token_layout"]
    assert saved["token_layout"]["executor"] == "goal_following_v1"
    assert not any(name.startswith("action.") for name in reloaded.state_dict())


def test_cli_rejects_action_training():
    from retroagi.stages.block_smb.cli import build_parser

    with pytest.raises(SystemExit):
        build_parser().parse_args(["train-layer", "--learner", "action"])


def test_uncertain_visual_contact_always_returns_an_executable_control():
    from retroagi.core.smb_scene_labels import MarioView, SceneObservation
    from retroagi.core.smb_spatial_feedback import SpatialFeedback

    scene = SceneObservation(MarioView((40, 208, 50, 220), True, "ground", False))
    feedback = SpatialFeedback()
    plan = feedback.begin(SkillToken("jump", 40, 0), scene)
    assert plan is not None and plan.action == 6
    assert feedback.status == "observing_support"


@pytest.mark.parametrize("buttons", [[3] * 32, [1] * 24 + [3] * 24, [3] * 20 + [0] * 20])
def test_execution_motion_matches_actual_left_right_and_release_physics(buttons):
    from retroagi.core.smb_spatial_feedback import SpatialFeedback

    env = MarioScenarioEnv()
    try:
        env.reset(
            scenario={"world_width": 400, "mario": [140, 208], "platforms": [[0, 220, 400, 20]]}
        )
        feedback = SpatialFeedback()
        for button in buttons:
            scene = scene_from_labels(env.scene_labels())
            feedback.observe(scene)
            feedback.executed(button, scene)
            env.step(button)
            assert feedback.motion.x_speed == env.motion.x_speed
            assert feedback.motion.x_fraction == env.motion.x_fraction
    finally:
        env.close()


@pytest.mark.parametrize("family", ["platform_hop", "pit_leap"])
@pytest.mark.parametrize("difficulty,sample_index", [("easy", 100), ("medium", 103), ("hard", 106)])
def test_destination_only_jump_initializes_running_spawn_motion(family, difficulty, sample_index):
    from retroagi.stages.block_smb.monte_carlo import sample_block_smb_monte_carlo_scenario

    sample = sample_block_smb_monte_carlo_scenario(
        split="validation", seed=0, sample_index=sample_index, family=family, difficulty=difficulty
    )
    env = MarioScenarioEnv()
    try:
        screen, _ = env.reset(scenario=sample.scenario)
        assert env.mario["vx"] == 2.5

        class Observer:
            def observe(self, screens):
                return [scene_from_labels(env.scene_labels())]

        agents = SMBAgents(Observer(), LayeredSMBPolicy().eval(), "cpu")

        def given(copies, scenes):
            return {
                "skill": [
                    SkillToken(
                        "jump",
                        round(env.goal.centerx - env.mario["x"] - env.mario["w"] / 2),
                        round(env.goal.bottom - env.mario["y"] - env.mario["h"]),
                    )
                ]
            }

        launched = False
        for _ in range(120):
            step = agents.act([screen], [0], given=given)[0]
            if step.button in (2, 4, 5) and not launched:
                launched = True
                feedback = agents.copies[0].spatial
                assert feedback.motion_ready
                assert abs(feedback.flight.motion.x_speed / 16 - env.mario["vx"]) < 0.75
                assert env.mario["on_ground"]
            screen, _, done, truncated, _ = env.step(step.button)
            if done or truncated:
                break
        assert launched and env._goal_credited
    finally:
        env.close()


@pytest.mark.parametrize("family", ["enemy_gap", "low_choice_alternate_route", "tall_pipe_jump"])
def test_teacher_owns_approach_waypoints_before_the_jump(family):
    from retroagi.stages.block_smb.monte_carlo import sample_block_smb_monte_carlo_scenario
    from retroagi.stages.block_smb.teacher_tokens import (
        episode_teacher,
        teacher_plan,
        teacher_skill,
    )

    sample = sample_block_smb_monte_carlo_scenario(
        split="validation", seed=0, sample_index=100, family=family, difficulty="easy"
    )
    env = MarioScenarioEnv()
    try:
        env.reset(scenario=sample.scenario)
        teacher = episode_teacher(sample.scenario)
        plan, _ = teacher_plan(env, teacher, certify_holds=False)
        assert plan.action == 1
        target = teacher_skill(env, teacher, plan)
        assert target.mode == "run"
        goal_x = env.mario["x"] + env.mario["w"] / 2 + target.x
        goal_y = env.mario["y"] + env.mario["h"] + target.y
        assert any(
            p["rect"].left <= goal_x <= p["rect"].right and p["rect"].top == goal_y
            for p in env.platforms
        )
        for action in sample.oracle["actions"]:
            if action in (2, 4, 5):
                break
            env.step(action)
        plan, _ = teacher_plan(env, teacher, certify_holds=False)
        assert teacher_skill(env, teacher, plan).mode == "jump"
    finally:
        env.close()


def test_occluded_floor_edge_is_not_camera_scroll_and_bridge_support_uses_geometry():
    from retroagi.core.smb_scene_labels import MarioView, SceneObservation, Surface
    from retroagi.core.smb_spatial_feedback import camera_shift, moving_support_offset

    def scene(edge, mario):
        return SceneObservation(
            MarioView((mario, 208, mario + 10, 220), True, "ground", True),
            moving_platforms=((edge, 220, edge + 100, 232),),
            surfaces=(Surface(8, edge, 220, False), Surface(edge, edge + 100, 220, True)),
        )

    before, after = scene(80, 85), scene(78, 83)
    assert camera_shift(before, after) is None
    assert moving_support_offset(before) == moving_support_offset(after) == 10


@pytest.mark.parametrize("sample_index", [100, 101])
def test_teacher_destinations_board_ride_and_dismount_a_moving_bridge(sample_index):
    from retroagi.stages.block_smb.monte_carlo import sample_block_smb_monte_carlo_scenario
    from retroagi.stages.block_smb.teacher_tokens import (
        episode_teacher,
        teacher_plan,
        teacher_skill,
    )

    sample = sample_block_smb_monte_carlo_scenario(
        split="validation",
        seed=0,
        sample_index=sample_index,
        family="bridge_wait",
        difficulty="easy",
    )
    env = MarioScenarioEnv()
    try:
        screen, _ = env.reset(scenario=sample.scenario)
        teacher = episode_teacher(sample.scenario)

        class Observer:
            def observe(self, screens):
                return [scene_from_labels(env.scene_labels())]

        def given(copies, scenes):
            plan, _ = teacher_plan(env, teacher, certify_holds=False)
            return {"skill": [teacher_skill(env, teacher, plan)]}

        agents = SMBAgents(Observer(), LayeredSMBPolicy().eval(), "cpu")
        for _ in range(600):
            teacher.observe_frame(env)
            step = agents.act([screen], [0], given=given)[0]
            screen, _, done, _, info = env.step(step.button)
            assert not info.get("death")
            if done:
                break
        assert env._goal_credited
    finally:
        env.close()
