"""Moving-bridge prerequisite tasks, progress credit, and curriculum ordering."""

from types import SimpleNamespace

import pytest
import torch

from retroagi.core.smb_learning import block_stage
from retroagi.core.smb_physics import NES_PHYSICS_PROFILE
from retroagi.stages.block_smb.demonstrations import (
    DemonstrationBatch,
    demonstration_sample_weights,
)
from retroagi.stages.block_smb.env import MarioScenarioEnv
from retroagi.stages.block_smb.geometry_expert import restore_env_state, snapshot_env_state
from retroagi.stages.block_smb.nes_curriculum import sample_nes_case
from scripts import smb_composable_training as training


@pytest.mark.parametrize("family", ["bridge_mount", "bridge_dismount"])
@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
def test_prerequisites_wait_jump_and_land_on_required_collision_surface(family, difficulty):
    sample = sample_nes_case(
        family=family,
        split="train",
        seed=20260910,
        index=100,
        difficulty=difficulty,
        max_rejections=0,
    )
    actions = sample.oracle["actions"]
    jump = actions.index(2)
    assert jump > 0 and set(actions[:jump]) == {0}
    stage = block_stage(sample)
    try:
        obs = stage.reset()
        geometry = stage.encode_observation(obs).metadata["smb_geometry"]
        assert geometry["objective"].kind == (
            "bridge_jump_mount" if family == "bridge_mount" else "bridge_jump_exit"
        )
        for action in actions:
            obs, _, done, truncated, info = stage.step(action)
            if done or truncated:
                break
        assert stage.env._goal_credited and not info["death"]
        assert stage.env.mario["on_ground"]
        assert bool(stage.env.mario["_platform"]["moving"]) == (family == "bridge_mount")
    finally:
        stage.env.close()


def touching_bridge(task):
    return dict(
        physics_profile=NES_PHYSICS_PROFILE,
        world_width=380,
        mario=[74 if task == "mount" else 175, 208],
        platforms=[
            [0, 220, 85, 20],
            dict(x=75, y=220, w=110, h=20, moving=[75, 145, 1], direction=1),
            [180, 220, 200, 20],
        ],
        goal=[240, 200, 16, 20],
        require_bridge_before_goal=True,
        bridge_jump_task=task,
        reward_wait_survival=0,
        reward_goal_distance_shaping=2,
    )


@pytest.mark.parametrize("task", ["mount", "dismount"])
def test_walking_does_not_satisfy_jump_task(task):
    env = MarioScenarioEnv()
    try:
        env.reset(scenario=touching_bridge(task))
        for _ in range(120):
            _, _, done, truncated, _ = env.step(1)
            if done or truncated:
                break
        assert env._bridge_boarded
        assert not env._goal_credited
    finally:
        env.close()


def test_passive_carry_receives_existing_progress_without_right_and_no_double_payment():
    env = MarioScenarioEnv()
    scenario = touching_bridge("dismount")
    scenario["mario"] = [110, 208]
    try:
        env.reset(scenario=scenario)
        before = env.mario["x"]
        _, _, _, _, info = env.step(0)
        assert env.mario["x"] > before and env.mario["vx"] == 0
        assert info["reward_terms"]["progress"] > 0 and info["reward_terms"]["goal_distance"] > 0
        assert info["bridge_carry_progress"] == pytest.approx(info["reward_terms"]["progress"])
        env._max_x_reached = 300
        _, _, _, _, info = env.step(0)
        assert info["bridge_carry_progress"] == 0
        state = snapshot_env_state(env)
        env.step(2)
        assert env._bridge_jump_launched
        restore_env_state(env, state)
        assert not env._bridge_jump_launched
    finally:
        env.close()


def test_passive_progress_changes_replay_allocation_without_dropping_other_families():
    data = DemonstrationBatch(
        torch.zeros(4, 8),
        torch.zeros(4, 16),
        torch.zeros(4, 64),
        torch.zeros(4, 8),
        torch.zeros(4, dtype=torch.long),
        torch.zeros(4, dtype=torch.long),
        torch.zeros(4, dtype=torch.long),
        torch.ones(4, dtype=torch.bool),
        torch.zeros(4, 64),
        torch.tensor([20, 20, 0, 0]),
        torch.ones(4, 16, dtype=torch.bool),
        torch.tensor([4, 4, 0, 0]),
        torch.tensor([0.0, 0.1, 0.0, 0.0]),
    )
    weights = demonstration_sample_weights(data)
    assert weights[1] == pytest.approx(3 * weights[0])
    assert weights[:2].sum() == pytest.approx(weights[2:].sum())


def test_bridge_dependencies_unlock_only_after_both_prerequisites_learned():
    cfg = dict(
        families=["flat_run", "bridge_mount", "bridge_dismount", "moving_bridge", "bridge_wait"],
        bridge_prerequisites=["bridge_mount", "bridge_dismount"],
        family_gate=0.99,
    )
    assert training.curriculum_families(cfg, False) == cfg["families"][:3]
    assert training.curriculum_families(cfg, True) == cfg["families"]
    result = dict(
        rates={
            "bridge_mount": dict(easy=1.0, medium=1.0, hard=1.0),
            "bridge_dismount": dict(easy=1.0, medium=1.0, hard=0.9),
        }
    )
    assert not training.bridge_prerequisites_learned(cfg, result)
    result["rates"]["bridge_dismount"]["hard"] = 1.0
    assert training.bridge_prerequisites_learned(cfg, result)


def test_duplicate_bridge_routes_removed_and_offsets_preserved():
    row = (torch.tensor([[1.0]]), 1, 0)
    episodes = [
        dict(id="a", family="bridge_wait", start=0, length=1, success=True),
        dict(id="a", family="bridge_wait", start=1, length=1, success=True, route_variant=True),
        dict(id="b", family="flat_run", start=2, length=1, success=True),
    ]
    rows, eps, count = training.deduplicate_bridge_routes([row, row, row], episodes)
    assert len(rows) == 2 and count == 1
    assert eps[1]["length"] == 0 and eps[1]["duplicate_of"] == "a"
    assert eps[2]["start"] == 1


def test_duplicate_physical_bridge_layouts_resampled_across_collection_calls(monkeypatch):
    def sample(**kwargs):
        index = kwargs["index"]
        return SimpleNamespace(
            scenario={
                "platforms": [index if index >= 1_000_000 else 0],
                "metadata": {"index": index},
            }
        )

    monkeypatch.setattr(training, "sample_nes_case", sample)
    seen = set()
    cfg = dict(families=["bridge_wait"], seed=42)
    a = training.samples(cfg, "train", 1, scenario_keys=seen)
    b = training.samples(cfg, "train", 1, offset=1, scenario_keys=seen)
    assert a[0].scenario["platforms"] != b[0].scenario["platforms"]
    assert len(seen) == 2


@pytest.mark.parametrize("objective", ["bridge_mount", "bridge_dismount"])
def test_full_pixel_adapter_accepts_matching_bridge_task_without_ram(objective):
    import numpy as np

    from retroagi.core.smb_learning import runtime
    from retroagi.core.smb_supervision import labels_to_vision
    from retroagi.stages.full_smb.adapter import FullSMBStage
    from scripts.tests.test_smb_transfer_contract import RAMEnv

    class Vision:
        def encode(self, observation):
            labels = np.zeros((240, 256), dtype=np.int64)
            labels[220:, :85] = 2
            labels[220:, 200:] = 2
            labels[220:, 110:180] = 6
            labels[208:220, 72:82] = 1
            return labels_to_vision(labels)

    stage = FullSMBStage(env=RAMEnv(), vision=Vision(), task_objective=objective)
    stage.configure_policy_runtime(runtime("perceived"))
    try:
        observation = stage.reset()

        def forbidden():
            raise AssertionError("Pixel bridge goals must not read RAM")

        stage.env.get_ram = forbidden
        batch = stage.encode_observation(observation)
        kind = batch.metadata["smb_geometry"]["objective"].kind
        assert kind == ("bridge_jump_mount" if objective == "bridge_mount" else "bridge_jump_exit")
    finally:
        stage.env.close()


@pytest.mark.parametrize("task", ["mount", "dismount"])
def test_jump_must_start_on_the_correct_surface(task):
    env = MarioScenarioEnv()
    try:
        env.reset(scenario=touching_bridge(task))
        support = env.platforms[1 if task == "mount" else 0]
        env.mario.update(
            x=110.0 if task == "mount" else 60.0, y=208.0, on_ground=True, _platform=support
        )
        env.step(2)
        assert not env.mario["on_ground"]
        assert not env._bridge_jump_launched
        assert not env._goal_credited
    finally:
        env.close()


def test_mount_credits_actual_edge_landing_without_full_body_containment():
    env = MarioScenarioEnv()
    try:
        env.reset(scenario=touching_bridge("mount"))
        bridge = env.platforms[1]
        bridge["rect"].x = 100
        bridge["move_x"] = 100.0
        env.mario.update(x=94.0, y=207.0, on_ground=False, _platform=None)
        env.motion.y_speed = 1
        env.motion.y_force = 0
        env._bridge_jump_launched = True
        env._airborne_started_with_jump = True
        _, _, _, _, info = env.step(0)
        assert env.mario["on_ground"] and env.mario["_platform"] is bridge
        assert env.mario["x"] < bridge["rect"].left
        assert env._goal_credited and not info["death"]
    finally:
        env.close()
