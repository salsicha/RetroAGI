"""Spatial commands, isolated action inputs, and directional maneuver coverage."""

import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import pytest
import torch

from retroagi.core.layered_policy import LayeredSMBPolicy, choose
from retroagi.core.tokens import SKILL_WIDTH, SKILL_X, SKILL_Y, SkillToken, encode_skill
from retroagi.stages.block_smb.action_families import ACTION_FAMILIES
from retroagi.stages.block_smb.env import MarioScenarioEnv
from retroagi.stages.block_smb.env_state import snapshot_env_state
from retroagi.stages.block_smb.layered_train import (
    EpisodeTask,
    LayeredTrainConfig,
    _Lane,
    _teacher_given,
    learner_families,
    learner_losses,
)
from retroagi.stages.block_smb.monte_carlo import (
    BLOCK_SMB_MC_FAMILIES,
    sample_block_smb_monte_carlo_scenario,
)
from retroagi.stages.block_smb.teacher_tokens import episode_teacher, teacher_plan, teacher_skill


def test_skill_command_is_a_relative_pixel_destination():
    assert encode_skill(SkillToken("jump", -64, 32)).tolist() == [0, 1, 0, -0.25, 0.125]
    assert encode_skill(SkillToken("hold", 80, -40)).tolist() == [0, 0, 1, 0, 0]
    for mode, x, y in [("advance", 0, 0), ("run", 257, 0), ("jump", 1.5, 0)]:
        with pytest.raises(ValueError):
            SkillToken(mode, x, y)
    logits = {
        "mode": torch.tensor([-9.0, 9.0, -9.0]),
        "x": torch.full((len(SKILL_X),), -9.0),
        "y": torch.full((len(SKILL_Y),), -9.0),
    }
    logits["x"][SKILL_X.index(-40)] = 9
    logits["y"][SKILL_Y.index(-24)] = 9
    assert choose("skill", logits)[0] == SkillToken("jump", -40, -24)


@pytest.mark.parametrize("family", ACTION_FAMILIES[9:])
@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
def test_new_maneuvers_are_immediate_directional_jumps_with_spatial_landings(family, difficulty):
    sample = sample_block_smb_monte_carlo_scenario(
        split="validation",
        seed=3,
        sample_index=0,
        family=family,
        difficulty=difficulty,
    )
    env = MarioScenarioEnv()
    try:
        env.reset(scenario=sample.scenario)
        state = episode_teacher(sample.scenario)
        plan, _ = teacher_plan(env, state)
        before = snapshot_env_state(env)
        target = teacher_skill(env, state, plan)
        assert env.mario == {**before["mario"], "_platform": env.mario.get("_platform")}
        assert env.steps == before["steps"]
        assert target.mode == "jump"
        side = -1 if family.endswith("back") else 1
        assert plan.action == (4 if side < 0 else 2)
        assert target.x * side > 0
        if "up" in family:
            assert target.y < 0
        if "down" in family:
            assert target.y > 0
        for button in sample.oracle["actions"]:
            if env.step(button)[2]:
                break
        assert env._goal_credited
    finally:
        env.close()


@pytest.mark.parametrize("learner", ["action", "skill"])
@pytest.mark.parametrize("label", [True, False])
def test_teacher_collection_and_learning_use_the_new_layer_contract(learner, label, monkeypatch):
    from retroagi.core.smb_agent import SMBAgents
    from retroagi.core.smb_scene_labels import scene_from_labels

    # Exact scenes stand in for vision, while the agent still accepts only frames.
    task = EpisodeTask(0, "action_walk", "train", 7, 0, float(label), label, difficulty="easy")
    lane = _Lane(task, 0)
    if learner == "skill" and not label:
        from retroagi.stages.block_smb import teacher_tokens

        def unused(*args, **kwargs):
            raise AssertionError("skill evaluation searched for a teacher action route")

        monkeypatch.setattr(teacher_tokens, "teacher_plan", unused)
        monkeypatch.setattr(teacher_tokens, "teacher_skill", unused)

    class Observer:
        def observe(self, screens):
            return [scene_from_labels(lane.env.scene_labels())]

    policy = LayeredSMBPolicy().eval()
    agent = SMBAgents(Observer(), policy, "cpu")
    given = _teacher_given(learner, {0: lane})
    try:
        for _ in range(3):
            lane.teacher.observe_frame(lane.env)
            step = agent.act([lane.screen], [0], given=given, run_given=(learner,))[0]
            for key, row in zip("abc", step.rows):
                lane.frames[key].append(row)
            if step.decision:
                lane.note_decision(learner, step.decision)
                assert step.decision.skill is not None
                assert ("skill" in step.decision.chosen) == (learner == "skill")
                assert step.decision.tactic_step is None
            lane.frames["button"].append(step.button)
            lane.frames["reward"].append(0.0)
            lane.frames["potential"].append(0.0)
            lane.screen, _, _, _, _ = lane.env.step(step.button)
        record = lane.record()
        if learner == "skill" and not label:
            assert not record.labels["valid"].any()
            assert all(
                len(record.labels[head]) == len(record.decision_frames)
                for head in ("mode", "x", "y")
            )
            return
        if learner == "action":
            assert record.given.shape[1] == SKILL_WIDTH

            # Training is also prohibited from reading the scene/memory.
            def forbidden(*args, **kwargs):
                raise AssertionError("action training read vision")

            policy.scene.forward = forbidden
        losses, _ = learner_losses(policy, learner, [record], 1.0, "cpu")
        assert losses
        sum(losses.values()).backward()
        assert any(p.grad is not None for p in getattr(policy, learner).parameters())
    finally:
        lane.env.close()


def test_unlabelled_skill_teacher_execution_still_gets_a_destination():
    task = EpisodeTask(0, "action_walk", "train", 7, 0, 1.0, False, difficulty="easy")
    lane = _Lane(task, 0)
    try:
        asked = lane.ask_teacher("skill", None)
        assert asked["plays_teacher"]
        assert asked["skill"] is not None and asked["action"] is not None
    finally:
        lane.env.close()


def test_skill_stage_is_available_and_checkpoint_sequence_includes_it():
    from retroagi.stages.block_smb.cli import build_parser
    from retroagi.stages.block_smb.layered_train import GIVEN, LEARNERS

    assert LEARNERS == ("action", "skill", "tactic")
    assert GIVEN == {"action": "skill", "skill": "tactic", "tactic": None}
    assert LayeredTrainConfig(learner="skill").learner == "skill"
    assert set(ACTION_FAMILIES) <= set(learner_families("skill", BLOCK_SMB_MC_FAMILIES))
    assert (
        build_parser()
        .parse_args(["exam-layer", "--checkpoint", "some.pt", "--learner", "skill"])
        .learner
        == "skill"
    )


def test_skill_receives_strategy_context_in_runtime_and_training_records(monkeypatch):
    from retroagi.core.smb_agent import SMBAgents
    from retroagi.core.smb_scene_labels import scene_from_labels
    from retroagi.core.tokens import STRATEGY_WIDTH, TACTIC_WIDTH, StrategyToken, encode_strategy

    task = EpisodeTask(0, "action_walk", "train", 7, 0, 0.0, False, difficulty="easy")
    lane = _Lane(task, 0)
    switch = StrategyToken("max_points", 1)
    lane.switch = switch

    class Observer:
        def observe(self, screens):
            return [scene_from_labels(lane.env.scene_labels())]

    policy = LayeredSMBPolicy().eval()
    original = policy.run_skill
    seen = []

    def run(*args, **kwargs):
        seen.append(kwargs["strategy"].clone())
        return original(*args, **kwargs)

    monkeypatch.setattr(policy, "run_skill", run)
    agents = SMBAgents(Observer(), policy, "cpu", switch=switch)
    try:
        step = agents.act([lane.screen], [0], given=_teacher_given("skill", {0: lane}))[0]
        for key, row in zip("abc", step.rows):
            lane.frames[key].append(row)
        lane.note_decision("skill", step.decision)
        assert torch.equal(seen[0][0], encode_strategy(switch))
        recorded = lane.decisions["given"][0]
        assert len(recorded) == TACTIC_WIDTH + STRATEGY_WIDTH
        assert recorded[-STRATEGY_WIDTH:].tolist() == encode_strategy(switch).tolist()
    finally:
        lane.env.close()


def test_legacy_checkpoint_migration_preserves_action_and_existing_skill_weights(tmp_path):
    import copy
    from dataclasses import asdict

    from retroagi.core.smb_observer import observation_layout
    from retroagi.core.tokens import TACTIC_WIDTH, token_layout
    from retroagi.stages.block_smb.layered_train import load_layered_checkpoint

    policy = LayeredSMBPolicy()
    state = copy.deepcopy(policy.state_dict())
    state["skill.above.weight"] = state["skill.above.weight"][:, :TACTIC_WIDTH].clone()
    layout = token_layout()
    del layout["skill"]["strategy_context"]
    path = tmp_path / "legacy.pt"
    torch.save(
        {
            "settings": asdict(policy.settings),
            "state_dict": state,
            "token_layout": layout,
            "observation_layout": observation_layout(),
            "trained_layers": ["action"],
        },
        path,
    )
    loaded, metadata = load_layered_checkpoint(path)
    assert metadata["load_migrations"]
    for name, value in state.items():
        actual = loaded.state_dict()[name]
        if name == "skill.above.weight":
            assert torch.equal(actual[:, :TACTIC_WIDTH], value)
            assert not actual[:, TACTIC_WIDTH:].any()
        else:
            assert torch.equal(actual, value), name


def test_skill_curriculum_excludes_tactic_decisions_but_keeps_local_maneuvers():
    from retroagi.stages.block_smb.tactic_families import TACTIC_TRAINING_FAMILIES

    families = set(learner_families("skill", BLOCK_SMB_MC_FAMILIES))
    assert families == set(BLOCK_SMB_MC_FAMILIES) - set(TACTIC_TRAINING_FAMILIES)
    assert len(families) == 45
    assert not families & set(learner_families("tactic", BLOCK_SMB_MC_FAMILIES))
    assert "stomp_recovery" not in families
    assert {
        "enemy_hop",
        "enemy_patrol",
        "bridge_wait",
        "bridge_mount",
        "bridge_dismount",
        "platform_chain",
        "stair_gap",
        "landing_enemy",
        "enemy_on_platform",
        "choice_alternate_route",
        "skill_enemy_bypass",
        "skill_enemy_bypass_back",
    } <= families
    assert "skill_enemy_bypass" not in learner_families("action", BLOCK_SMB_MC_FAMILIES)


def test_enemy_bypass_sampler_keeps_drawing_after_rejected_duplicates():
    from collections import Counter

    rejected = Counter()
    sample = sample_block_smb_monte_carlo_scenario(
        split="train",
        seed=0,
        sample_index=0,
        family="skill_enemy_bypass",
        difficulty="easy",
        rejection_counter=rejected,
    )
    assert rejected["duplicate_regeneration"] > 0
    assert sample.reachability["reachable"]


@pytest.mark.parametrize("family", ["skill_enemy_bypass", "skill_enemy_bypass_back"])
@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
def test_enemy_bypass_lands_beyond_a_live_enemy_and_stomping_fails(family, difficulty):
    sample = sample_block_smb_monte_carlo_scenario(
        split="validation",
        seed=3,
        sample_index=0,
        family=family,
        difficulty=difficulty,
    )
    env = MarioScenarioEnv()
    try:
        env.reset(scenario=sample.scenario)
        teacher = episode_teacher(sample.scenario)
        plan, _ = teacher_plan(env, teacher, certify_holds=False)
        target = teacher_skill(env, teacher, plan)
        assert target.mode == "jump"
        assert target.x < 0 if family.endswith("back") else target.x > 0
        for action in sample.oracle["actions"]:
            if env.step(action)[2]:
                break
        assert env._goal_credited and env.mario["on_ground"]
        assert not any(e["dead"] for e in env.enemies)
        enemy = env.enemies[0]
        if family.endswith("back"):
            assert env.mario["x"] + env.mario["w"] < enemy["x"]
        else:
            assert env.mario["x"] > enemy["x"] + enemy["w"]
        env.reset(scenario=sample.scenario)
        env.enemies[0]["dead"] = True
        _, _, terminated, _, info = env.step(0)
        assert terminated and info["off_route"] and not env._goal_credited
    finally:
        env.close()
