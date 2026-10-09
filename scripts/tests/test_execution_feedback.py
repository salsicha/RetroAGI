"""Requested destinations are attempted; measured failures reach skill training."""

from dataclasses import asdict

import numpy as np
import pytest
import torch

from retroagi.core.layered_policy import LayeredSMBPolicy
from retroagi.core.smb_agent import SMBAgents
from retroagi.core.smb_observer import observation_layout
from retroagi.core.smb_scene_labels import scene_from_labels
from retroagi.core.tokens import (
    EXECUTION_WIDTH,
    STRATEGY_WIDTH,
    TACTIC_WIDTH,
    SkillToken,
    token_layout,
)
from retroagi.stages.block_smb.env import MarioScenarioEnv
from retroagi.stages.block_smb.layered_train import load_layered_checkpoint


@pytest.mark.parametrize("lower_floor", [False, True])
def test_requested_run_walks_off_edge_and_does_not_veto_falling(lower_floor):
    env = MarioScenarioEnv()
    platforms = [[0, 172, 100, 10]] + ([[0, 220, 256, 20]] if lower_floor else [])
    screen, _ = env.reset(scenario={"mario": [75, 160], "platforms": platforms})

    class Observer:
        def observe(self, screens):
            return [scene_from_labels(env.scene_labels())]

    agent = SMBAgents(Observer(), LayeredSMBPolicy().eval(), "cpu")
    try:
        fell = False
        for _ in range(70):
            step = agent.act(
                [screen], [0], given=lambda c, s: {"skill": [SkillToken("run", 70, 48)]}
            )[0]
            screen, _, done, _, _ = env.step(step.button)
            fell |= env.mario["y"] > 160
            if done:
                break
        assert fell and env.mario["x"] > 100
    finally:
        env.close()


def test_wall_does_not_veto_buttons_and_stall_feedback_reaches_next_skill(monkeypatch):
    env = MarioScenarioEnv()
    screen, _ = env.reset(
        scenario={"mario": [40, 208], "platforms": [[0, 220, 256, 20], [70, 150, 20, 70]]}
    )

    class Observer:
        def observe(self, screens):
            return [scene_from_labels(env.scene_labels())]

    policy = LayeredSMBPolicy().eval()
    seen = []
    original = policy.run_skill

    def run(*args, **kwargs):
        seen.append(kwargs["feedback"].clone())
        return original(*args, **kwargs)

    monkeypatch.setattr(policy, "run_skill", run)
    agent = SMBAgents(Observer(), policy, "cpu")
    blocked_presses = 0
    try:
        for _ in range(100):
            step = agent.act(
                [screen],
                [0],
                given=lambda c, s: {"skill": [SkillToken("run", 100, 0)]},
                run_given=("skill",),
            )[0]
            if env.mario["x"] >= 60 and step.button == 1:
                blocked_presses += 1
            if step.ended == "no_progress":
                vector = step.decision.execution_feedback
                assert vector[0:3] == [1, 1, 0]
                assert vector[3] > 0 and vector[6] > 0
                assert np.allclose(seen[-1][0].numpy(), vector)
                assert blocked_presses >= 20
                break
            screen, *_ = env.step(step.button)
        else:
            pytest.fail("wall stall did not return control to skill")
    finally:
        env.close()


def test_feedback_migration_keeps_old_weights_and_clears_qualification(tmp_path):
    policy = LayeredSMBPolicy()
    state = policy.state_dict()
    before = TACTIC_WIDTH + STRATEGY_WIDTH
    state["skill.above.weight"] = state["skill.above.weight"][:, :before].clone()
    layout = token_layout()
    layout["executor"] = "predictive_spatial_v2"
    del layout["skill"]["execution_feedback"]
    path = tmp_path / "old.pt"
    torch.save(
        dict(
            settings=asdict(policy.settings),
            state_dict=state,
            token_layout=layout,
            observation_layout=observation_layout(),
            trained_layers=["skill", "tactic"],
        ),
        path,
    )
    loaded, metadata = load_layered_checkpoint(path)
    assert not metadata["trained_layers"]
    weight = loaded.state_dict()["skill.above.weight"]
    assert weight.shape[1] == before + EXECUTION_WIDTH
    assert torch.equal(weight[:, :before], state["skill.above.weight"])
    assert not weight[:, before:].any()
    for k, v in state.items():
        if k != "skill.above.weight":
            assert torch.equal(v, loaded.state_dict()[k])


def test_runup_executor_checkpoint_requires_requalification_without_changing_weights(tmp_path):
    policy = LayeredSMBPolicy()
    layout = token_layout()
    layout["executor"] = "goal_following_v1"
    path = tmp_path / "runup.pt"
    torch.save(
        dict(
            settings=asdict(policy.settings),
            state_dict=policy.state_dict(),
            token_layout=layout,
            observation_layout=observation_layout(),
            trained_layers=["skill", "tactic"],
        ),
        path,
    )
    loaded, metadata = load_layered_checkpoint(path)
    assert metadata["trained_layers"] == []
    assert all(torch.equal(v, loaded.state_dict()[k]) for k, v in policy.state_dict().items())
