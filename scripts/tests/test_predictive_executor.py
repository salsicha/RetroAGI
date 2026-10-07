"""Skill targets are executed without a learned action network."""

import dataclasses

import pytest
import torch

from retroagi.core.layered_policy import LayeredSMBPolicy
from retroagi.core.smb_agent import SMBAgents
from retroagi.core.smb_scene_labels import scene_from_labels
from retroagi.core.tokens import SkillToken
from retroagi.stages.block_smb.env import MarioScenarioEnv


@pytest.mark.parametrize("distance", [-60, 0, 60, 160])
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
        assert abs(env.mario["x"] - (start + distance)) <= 2
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
    assert metadata["trained_layers"] == ["skill"]
    assert "removed_action_network" in metadata["load_migrations"]
    for name, value in policy.state_dict().items():
        assert torch.equal(value, loaded.state_dict()[name])
    config = LayeredTrainConfig(vision_checkpoint=str(source))
    output = tmp_path / "new.pt"
    save_layered_checkpoint(output, loaded, config, ["skill"], [])
    reloaded, saved = load_layered_checkpoint(output)
    assert "action_input" not in saved["token_layout"]
    assert saved["token_layout"]["executor"] == "predictive_spatial_v1"
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
