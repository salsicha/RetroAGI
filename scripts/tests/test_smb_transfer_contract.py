"""Semantic and physical contracts that tensor-shape compatibility cannot test."""

from types import SimpleNamespace

import numpy as np
import pygame
import pytest
import torch

from retroagi.core.interfaces import VisionOutput
from retroagi.core.smb_geometry import geometry_features
from retroagi.core.smb_runtime import SMBRuntimeContract, attach_runtime, make_smb_executor
from retroagi.stages.block_smb.env import MarioScenarioEnv
from retroagi.stages.block_smb.local_traversal import local_objective
from retroagi.stages.full_smb.geometry import (
    NESGeometry,
    _platforms,
    remap_full_vision,
    visible_tiles,
)
from retroagi.stages.full_smb.train import _smb_forward_kwargs


def ram_scene():
    ram = np.zeros(0x800, dtype=np.uint8)
    ram[0x86] = 40
    ram[0xCE] = 176
    ram[0xB5] = 1
    ram[0x499] = 1
    ram[0x33] = 1
    ram[0x500 + 11 * 16 : 0x500 + 13 * 16] = 0x54
    ram[0x5D0 + 11 * 16 : 0x5D0 + 13 * 16] = 0x54
    return ram


def test_visual_categories_are_mapped_by_meaning():
    logits = torch.full((1, 13, 1, 3), -20.0)
    logits[0, 8, 0, 0] = 20  # mario
    logits[0, 1, 0, 1] = 20  # ground
    logits[0, 6, 0, 2] = 20  # goomba
    original = VisionOutput(torch.zeros(1, 2), logits, logits.argmax(1), torch.ones(1, 2, 4))
    mapped = remap_full_vision(original)
    assert mapped.semantic_ids.tolist() == [[[1, 2, 5]]]
    assert torch.allclose(mapped.semantic_logits.softmax(1).sum(1), torch.ones(1, 1, 3))
    assert mapped.tokens.equal(original.tokens)
    assert remap_full_vision(original, visual_tokens="zero_ablation").tokens.count_nonzero() == 0


def test_pipe_tiles_remain_a_solid_body_not_an_overhead_platform():
    tiles = [(pygame.Rect(140, y, 16, 16), 0x10) for y in (176, 192)]
    tiles += [(pygame.Rect(x, 208, 16, 16), 0x54) for x in range(0, 256, 16)]
    platforms = _platforms(tiles)
    scene = SimpleNamespace(
        mario=dict(x=110, y=196, w=10, h=12),
        platforms=platforms,
        enemies=[],
        goal=pygame.Rect(240, 188, 16, 20),
    )
    objective = local_objective(scene)
    assert objective.kind == "mount"
    assert any(p["rect"] == pygame.Rect(140, 176, 16, 32) for p in platforms)


def test_tile_page_parity_and_camera_coordinates():
    ram = ram_scene()
    ram[0x5D0] = 0x51
    ram[0x500] = 0x52
    assert (pygame.Rect(0, 32, 16, 16), 0x51) in visible_tiles(ram, 256)
    assert (pygame.Rect(0, 32, 16, 16), 0x52) in visible_tiles(ram, 512)


def test_motion_does_not_mistake_camera_scroll_for_velocity():
    ram = ram_scene()
    ram[0x0F] = 1
    ram[0x16] = 6
    ram[0x87] = 150
    ram[0xCF] = 184
    ram[0xB6] = 1
    ram[0x49A] = 9
    observer = NESGeometry()
    first = observer.observe(ram, frame=0)
    ram[0x71C] = 10
    ram[0x87] = 151
    second = observer.observe(ram, frame=1)
    assert second["features"]["motion_vec"][0] == pytest.approx(1 / 3)
    assert "enemy_patrol_min" in second["unavailable_features"]
    assert first["world_x"] == second["world_x"]
    assert first["player_box"][0] - second["player_box"][0] == 10
    assert observer.observe(ram, frame=1) is second  # re-encoding must not advance tracking


def test_objective_stays_in_world_coordinates_during_jump():
    ram = ram_scene()
    ram[0x500 + 9 * 16 + 9] = 0x10
    ram[0x500 + 10 * 16 + 9] = 0x10
    observer = NESGeometry()
    first = observer.observe(ram, frame=0)
    assert first["objective"].kind == "mount"
    ram[0x1D] = 1
    ram[0x71C] = 7
    second = observer.observe(ram, frame=1)
    assert second["objective"].left + 7 == first["objective"].left


def test_shared_feature_encoder_preserves_block_geometry():
    env = MarioScenarioEnv()
    try:
        _, info = env.reset(
            scenario={
                "width": 256,
                "mario": [40, 200],
                "platforms": [[0, 220, 256, 20]],
                "goal": [230, 200, 16, 20],
            }
        )
        features = geometry_features(env)
        np.testing.assert_array_equal(features["state_vec"], info["state_vec"])
        assert features["state_vec"][0] == pytest.approx(40 / 256)
        assert features["state_vec"][18] == pytest.approx((256 - 40 - env.mario["w"]) / 256)
        assert len(features["state_vec"]) == 27 and len(features["motion_vec"]) == 8
    finally:
        env.close()


def test_runtime_validates_duration_units_and_schema():
    with pytest.raises(ValueError, match="schema"):
        SMBRuntimeContract(schema="wrong")
    with pytest.raises(ValueError, match="frame"):
        SMBRuntimeContract(frame_skip=4)
    with pytest.raises(ValueError, match="16 duration"):
        SMBRuntimeContract(jump_hold_frames=(1, 2))
    with pytest.raises(ValueError, match="fixed commitments"):
        SMBRuntimeContract(adaptive_duration=True, jump_hold_frames=tuple(range(2, 34, 2)))


def test_executor_uses_checkpoint_settings_and_calibrated_jump_units():
    model = SimpleNamespace(
        smb_runtime_contract=SMBRuntimeContract(jump_hold_frames=tuple(range(2, 34, 2)))
    )
    executor = make_smb_executor(model)
    assert not executor.adaptive_duration and not executor.walk_primitives
    logits = torch.full((1, 1, 16), -20.0)
    logits[0, 0, 10] = 20
    motor = SimpleNamespace(hold_duration_logits=logits, duration_bins=torch.arange(1, 17).float())
    started = executor.execute(2, motor_primitives=motor, support_override="ground")
    assert started.duration_bin_index == 10
    assert started.hold_frames == 22
    executor.reset()
    assert (
        executor.execute(1, motor_primitives=motor, support_override="ground").hold_frames is None
    )


def test_missing_semantic_contract_is_rejected_before_inference():
    model = SimpleNamespace(smb_runtime_contract=SMBRuntimeContract())
    with pytest.raises(ValueError, match="semantics"):
        _smb_forward_kwargs(model, SimpleNamespace(metadata={}), True)


def test_model_runtime_restores_deterministic_critic_and_memory_policy():
    model = SimpleNamespace(ranked_candidate_search=True, deterministic_critic_slots=None)
    attach_runtime(model, SMBRuntimeContract().manifest())
    assert not model.ranked_candidate_search
    assert model.deterministic_critic_slots["death"] == 36
    assert not model.smb_runtime_contract.recurrent_state


class RAMEnv:
    buttons = ("B", "A", "SELECT", "START", "UP", "DOWN", "LEFT", "RIGHT")

    def __init__(self):
        self.ram = ram_scene()

    def get_ram(self):
        return self.ram.copy()

    def reset(self, seed=None):
        self.ram = ram_scene()
        return np.zeros((240, 256, 3), dtype=np.uint8), {}

    def step(self, action):
        self.ram[0x86] += 1
        return np.zeros((240, 256, 3), dtype=np.uint8), 0.0, False, False, {}

    def get_state(self):
        return self.ram.copy()

    def set_state(self, state):
        self.ram = state.copy()

    def close(self):
        pass


class ContractVision:
    def encode(self, observation):
        logits = torch.zeros(1, 13, 2, 2)
        support = torch.tensor([[-10.0, 10.0, -10.0]])
        return VisionOutput(
            torch.zeros(1, 2),
            logits,
            logits.argmax(1),
            torch.zeros(1, 4, 8),
            support_logits=support,
        )


def shared_stage():
    from retroagi.stages.full_smb.adapter import FullSMBStage

    stage = FullSMBStage(env=RAMEnv(), vision=ContractVision())
    stage.configure_policy_runtime(SMBRuntimeContract())
    stage.reset()
    return stage


def test_snapshot_restores_observation_history_and_objective():
    stage = shared_stage()
    try:
        first = stage.encode_observation(stage._last_observation)
        snapshot = stage.save_emulator_state()
        stage.step(1)
        stage.encode_observation(stage._last_observation)
        restored = stage.load_emulator_state(snapshot)
        second = stage.encode_observation(restored)
        assert first.src_c.equal(second.src_c)
        assert first.metadata["smb_geometry"]["frame"] == second.metadata["smb_geometry"]["frame"]
        assert stage.smb_geometry.frames == 1
    finally:
        stage.close()


def test_shared_episodes_end_on_death_and_level_clear_without_backend_game_over():
    stage = shared_stage()
    try:
        stage.env.ram[0x0E] = 11
        _, _, terminal, _, info = stage.step(1)
        assert terminal and info["full_smb_signals"]["death"]
        stage.reset()
        stage.env.ram[0x0E] = 5
        stage.env.ram[0x746] = 1
        _, _, terminal, _, info = stage.step(1)
        assert terminal and info["full_smb_signals"]["completion"]
    finally:
        stage.close()


def test_full_forward_matches_the_block_action_path_on_same_observation():
    from retroagi.stages.block_smb.train import (
        BlockSMBTrainingConfig,
        _action_from_model,
        block_smb_evaluation_target,
        make_block_smb_model,
    )
    from retroagi.stages.full_smb.train import _policy_action_logits_and_state

    stage = shared_stage()
    try:
        model = make_block_smb_model(
            BlockSMBTrainingConfig(hidden_dim=8, architecture_config={"hidden_dim": 8})
        )
        model.eval()
        attach_runtime(model, SMBRuntimeContract().manifest())
        batch = stage.encode_observation(stage._last_observation)
        g = batch.metadata["smb_geometry"]
        full_executor = make_smb_executor(model)
        forward = _policy_action_logits_and_state(model, batch, device=torch.device("cpu"))
        full_action = full_executor.execute(
            int(forward.logits.argmax(-1)), batch=batch, motor_primitives=forward.motor_primitives
        ).action
        block_executor = make_smb_executor(model)
        block = _action_from_model(
            model,
            batch,
            deterministic=True,
            tau=1.0,
            primitive_executor=block_executor,
            skill_goal=g["skill_goal"],
            support_override=g["support"],
            enemy_contact_override=False,
            evaluation_target=block_smb_evaluation_target(g["scene"], g["objective"]),
        )
        assert int(block[0]) == full_action
    finally:
        stage.close()


def test_stomp_recovery_releases_jump_without_inventing_a_new_primitive():
    model = SimpleNamespace(smb_runtime_contract=SMBRuntimeContract())
    executor = make_smb_executor(model)
    batch = SimpleNamespace(metadata={"smb_geometry": {"support": "air", "bouncing": True}})
    execution = executor.execute(2, batch=batch)
    assert execution.action == 1 and not execution.started


def test_dynamic_platform_motion_and_unknown_bounds_are_explicit():
    ram = ram_scene()
    ram[0x0F] = 1
    ram[0x16] = 0x29
    ram[0x49A] = 6
    ram[0x87] = 160
    ram[0xCF] = 176
    ram[0xB6] = 1
    observer = NESGeometry()
    first = observer.observe(ram, frame=0)
    assert "bridge_vx" in first["unavailable_features"]
    ram[0x87] = 162
    ram[0x71C] = 7
    second = observer.observe(ram, frame=1)
    assert second["features"]["motion_vec"][5] == pytest.approx(2 / 3)
    assert "bridge_min" in second["unavailable_features"]
    assert second["unsupported_objects"] == []


def test_capture_shape_change_discards_incompatible_frame_history():
    from dataclasses import replace

    stage = shared_stage()
    try:
        stage.observation_config = replace(stage.observation_config, resize_shape=None)
        stage._reset_frame_stack(np.zeros((224, 256, 3), dtype=np.uint8))
        stage._append_frame(np.zeros((224, 240, 3), dtype=np.uint8), valid=True)
        assert len({tuple(f.shape) for f in stage._frame_stack}) == 1
        assert list(stage._frame_mask) == [False, False, False, True]
        stage.encode_observation(np.zeros((224, 240, 3), dtype=np.uint8))
    finally:
        stage.close()


def test_shared_dynamics_slots_cover_geometry_and_real_terminal_features():
    from retroagi.stages.full_smb.train import _full_smb_c_stream_slot_spans

    stage = shared_stage()
    try:
        batch = stage.encode_observation(stage._last_observation)
        spans = _full_smb_c_stream_slot_spans(batch)
        assert spans["emulator_state"] == (12, 47)
        assert spans["terminal_outcome"] == (36, 39)
        assert spans["camera_state"] == (47, 47)
    finally:
        stage.close()


def test_comparison_rejects_mixed_contracts_and_clears_old_commitments():
    from retroagi.stages.full_smb.compare import _configure_comparison_contract

    stage = shared_stage()
    try:
        shared = SimpleNamespace(smb_runtime_contract=SMBRuntimeContract(), smb_executor=object())
        with pytest.raises(ValueError, match="identical"):
            _configure_comparison_contract(stage, [shared, SimpleNamespace()])
        _configure_comparison_contract(stage, [shared, shared])
        assert not hasattr(shared, "smb_executor")
    finally:
        stage.close()


def test_shared_training_rejects_legacy_warm_start_labels_before_collection():
    from retroagi.stages.full_smb.train import _run_full_smb_imitation_warm_start_phase

    with pytest.raises(ValueError, match="Legacy scripted"):
        _run_full_smb_imitation_warm_start_phase(
            SimpleNamespace(imitation_warm_start=True),
            SimpleNamespace(smb_runtime_contract=SMBRuntimeContract()),
            None,
            device=torch.device("cpu"),
            make_stage=None,
        )


def test_standalone_legacy_imitation_also_rejects_shared_checkpoint():
    from retroagi.stages.full_smb.imitation import train_full_smb_imitation_warm_start

    with pytest.raises(ValueError, match="Legacy imitation"):
        train_full_smb_imitation_warm_start(
            SimpleNamespace(smb_runtime_contract=SMBRuntimeContract()),
            {},
            device=torch.device("cpu"),
        )
