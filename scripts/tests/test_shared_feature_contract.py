"""The shared SMB observation: only what Full SMB can also show, in its screen frame."""

import numpy as np
import pytest
import torch

from retroagi.core.smb_enemy_history import HAZARD_NAMES
from retroagi.core.smb_geometry import FEATURE_NAMES
from retroagi.core.smb_physics import NES_JUMP_FRAMES
from retroagi.core.smb_runtime import SMBRuntimeContract
from retroagi.core.smb_scene import (
    AVAILABILITY_NAMES,
    C_SEMANTIC_LAYOUT_START,
    C_SPANS,
    HIDDEN_FIELDS,
    block_screen_scene,
    c_feature_index,
    observation_spec,
)
from retroagi.stages.block_smb.adapter import BLOCK_SMB_SPEC, BlockSMBStage
from retroagi.stages.block_smb.env import MarioScenarioEnv
from retroagi.stages.block_smb.local_traversal import local_objective
from retroagi.stages.block_smb.monte_carlo import (
    BLOCK_SMB_MC_ID,
    sample_block_smb_monte_carlo_scenario,
    validate_block_smb_monte_carlo_oracle,
)
from retroagi.stages.block_smb.train import BlockSMBTrainingConfig, block_smb_evaluation_target
from scripts.tests.test_block_smb_training import StaticBlockVision

WIDE = {
    "world_width": 512,
    "mario": [40, 204],
    "platforms": [[0, 220, 330, 20], [370, 220, 142, 20]],
    "goal": [480, 200, 16, 20],
}
PATROL = {**WIDE, "enemies": [[200, 204, 150, 260]]}
LIFT = {"x": 120, "y": 160, "w": 40, "h": 10, "moving": [100, 180, 1.0]}


def stage(scenario):
    return BlockSMBStage(env=MarioScenarioEnv(), scenario=scenario, vision=StaticBlockVision())


def run(block, actions):
    block.reset(seed=0)
    rows = [block.state_features()]
    for action in actions:
        block.step(action)
        rows.append(block.state_features())
    return np.array(rows)


def test_policy_features_are_only_those_full_smb_can_supply():
    absent = {
        "coyote",
        "jump_buffer",
        "elapsed",
        "enemy_patrol_min",
        "enemy_patrol_max",
        "bridge_min",
        "bridge_max",
    }
    assert not absent & set(FEATURE_NAMES)
    assert FEATURE_NAMES[-3:] == ("death", "terminated", "truncated")


def test_c_layout_is_contiguous_and_fills_the_stage():
    spans = list(C_SPANS.values())
    assert spans[0][0] == 0
    assert all(a[1] == b[0] for a, b in zip(spans, spans[1:]))
    assert spans[-1][1] == C_SEMANTIC_LAYOUT_START < BLOCK_SMB_SPEC.seq_len_c
    assert C_SPANS["c_state"][1] - C_SPANS["c_state"][0] == len(FEATURE_NAMES)
    assert C_SPANS["c_enemy_history"][1] - C_SPANS["c_enemy_history"][0] == len(HAZARD_NAMES)
    availability = C_SPANS["c_availability"]
    assert availability[1] - availability[0] == len(AVAILABILITY_NAMES)
    assert observation_spec()["features"] == list(FEATURE_NAMES)


def test_block_scene_hides_simulator_ground_truth():
    env = MarioScenarioEnv()
    try:
        env.reset(scenario={**PATROL, "platforms": [*WIDE["platforms"], LIFT]})
        scene = block_screen_scene(env)
        assert scene.enemies and any(p.get("moving") for p in scene.platforms)
        assert all(not set(HIDDEN_FIELDS) & set(e) for e in scene.enemies)
        assert all(not set(HIDDEN_FIELDS) & set(p) for p in scene.platforms)
        assert scene.goal.left == 240  # the NES observer's placeholder, not the finish flag
    finally:
        env.close()


def test_block_features_use_the_nes_screen_frame():
    block = stage(WIDE)
    try:
        rows = run(block, [1] * 120)
        env = block.env
        assert env.camera_x > 0 and env.mario["on_ground"]
        m, last = env.mario, rows[-1]
        assert last[FEATURE_NAMES.index("x")] == pytest.approx((m["x"] - int(env.camera_x)) / 256)
        assert last[FEATURE_NAMES.index("vx")] == pytest.approx(m["vx"] / 3.0)
        # The goal slots hold the visible local objective, never the finish marker.
        scene = block_screen_scene(env)
        target = local_objective(scene)
        assert target.kind == "gap"
        dx = ((target.left + target.right) / 2 - scene.mario["x"] - m["w"] / 2) / 256
        assert last[FEATURE_NAMES.index("goal_dx")] == pytest.approx(dx)
    finally:
        block.env.close()


def test_goal_is_kept_from_takeoff_while_airborne():
    block = stage(WIDE)
    try:
        run(block, [1] * 30)
        takeoff = block.objective_memory.target
        for _ in range(6):
            block.step(2)
            block.state_features()
            assert not block.env.mario["on_ground"]
            assert block.objective_memory.target == takeoff
    finally:
        block.env.close()


def test_block_env_runs_nes_physics_only():
    env = MarioScenarioEnv()
    try:
        env.reset(scenario=WIDE)
        assert (env.mario["w"], env.mario["h"]) == (10, 12) and env.mario["on_ground"]
        assert "coyote_frames" not in env.mario and "jump_buffer" not in env.mario
        # Holding jump through a landing does not jump again.
        airborne = []
        for _ in range(90):
            env.step(5)
            airborne.append(not env.mario["on_ground"])
        first_landing = airborne.index(False, airborne.index(True))
        assert not any(airborne[first_landing:])
    finally:
        env.close()


def test_evaluation_target_is_screen_relative():
    env = MarioScenarioEnv()
    try:
        env.reset(scenario=WIDE)
        env.camera_x = 100.0
        target = block_smb_evaluation_target(env)
        goal_center = (env.goal.left + env.goal.right) / 2 - env.mario["w"] / 2
        assert target[0, 0].item() == pytest.approx((goal_center - 100) / 256)
    finally:
        env.close()


def test_runtime_contract_holds_only_control_settings():
    contract = SMBRuntimeContract.from_block_config(
        {"adaptive_duration_control": False, "learned_skill_goals": True}
    )
    assert contract.learned_skill_goals and not contract.adaptive_duration
    assert SMBRuntimeContract.from_manifest(contract.manifest()) == contract
    with pytest.raises(ValueError, match="retrain"):
        SMBRuntimeContract.from_manifest({**contract.manifest(), "schema": "old"})
    for removed in ("motion_observations", "hazard_observations", "monte_carlo_distribution_id"):
        assert not hasattr(BlockSMBTrainingConfig(), removed)


@pytest.mark.parametrize("family", ["single_gap", "enemy_stomp", "pit_leap", "bridge_mount"])
def test_every_layout_has_a_route_verified_under_nes_physics(family):
    def draw():
        return sample_block_smb_monte_carlo_scenario(
            split="validation", seed=3, sample_index=0, family=family, difficulty="hard"
        )

    sample = draw()
    assert sample.scenario_id.startswith(BLOCK_SMB_MC_ID)
    assert validate_block_smb_monte_carlo_oracle(sample.scenario, sample.oracle["actions"])[
        "reachable"
    ]
    assert sample.scenario == draw().scenario
    if family == "enemy_stomp":
        width = sample.scenario["world_width"]
        assert all(enemy[2:4] == [0, width] for enemy in sample.scenario["enemies"])
        assert sample.scenario["task_objective"] == "stomp"
    if family == "pit_leap":
        assert sample.scenario["mario_velocity"] == [2.5, 0.0]


def test_platform_hop_jumps_from_the_first_frame_with_a_menu_hold():
    sample = sample_block_smb_monte_carlo_scenario(
        split="validation", seed=3, sample_index=0, family="platform_hop", difficulty="medium"
    )
    actions = sample.oracle["actions"]
    hold = next(i for i, a in enumerate(actions) if a != 2)
    assert actions[0] == 2 and hold in NES_JUMP_FRAMES


def test_stage_encoding_places_observed_features_in_their_spans():
    block = stage(PATROL)
    try:
        observation = block.reset(seed=0)
        batch = block.encode_observation(observation)
        start, end = C_SPANS["c_state"][0], C_SEMANTIC_LAYOUT_START
        expected = torch.as_tensor(block.state_features(), dtype=batch.src_c.dtype)
        assert batch.src_c.shape == (1, BLOCK_SMB_SPEC.seq_len_c)
        assert torch.allclose(batch.src_c[0, start:end], expected)
        assert batch.src_c[0, c_feature_index("death")] == 0
        assert batch.metadata["vision_fusion"]["c_state"] == C_SPANS["c_state"]
    finally:
        block.env.close()


def test_enemy_relative_motion_includes_passive_carry():
    from types import SimpleNamespace

    from retroagi.core.smb_scene import enemy_relative_motion

    scene = SimpleNamespace(
        mario=dict(x=60, w=10, vx=2, _platform=dict(moving=True, move_speed=0.5, move_dir=1)),
        enemies=[dict(x=120, w=12, speed=1, direction=-1)],
    )
    # Enemy -1 px/frame, minus Mario's own 2 px/frame and the lift's 0.5 carry.
    assert enemy_relative_motion(dict(scene=scene)) == (-3.5, True)
