"""Plant timing must be observable, shared with NES, and checkpoint-versioned."""

from types import SimpleNamespace

import numpy as np
import pygame
import pytest
import torch

from retroagi.core.smb_enemy_history import HAZARD_NAMES, EnemyObservationHistory
from retroagi.core.smb_runtime import SMBRuntimeContract
from retroagi.stages.block_smb.adapter import BlockSMBObservationConfig, BlockSMBStage
from retroagi.stages.block_smb.train import (
    make_block_smb_model,
    restore_block_smb_checkpoint,
    save_block_smb_checkpoint,
)
from retroagi.stages.full_smb.geometry import NESGeometry
from scripts.tests.test_block_smb_training import StaticBlockVision, tiny_config
from scripts.tests.test_piranha_avoidance import contact_scenario
from scripts.tests.test_smb_transfer_contract import ram_scene


def test_identical_plant_pixels_have_distinct_observed_motion_inputs():
    states, frames = [], []
    for phase in (1, 77):
        stage = BlockSMBStage(
            scenario=contact_scenario(phase=phase, mario=(40, 204)),
            vision=StaticBlockVision(),
            observation_config=BlockSMBObservationConfig(
                motion_observations=True,
                hazard_observations=True,
            ),
        )
        try:
            stage.reset()
            frame, _, _, _, info = stage.step(0)
            frames.append(frame)
            states.append(stage.state_features(info))
            assert states[-1].shape == (35 + len(HAZARD_NAMES),)
            assert states[-1][-4] == 1  # Measured velocity, not a guessed phase.
        finally:
            stage.env.close()
    np.testing.assert_array_equal(frames[0], frames[1])
    np.testing.assert_array_equal(states[0][:-6], states[1][:-6])
    assert states[0][-5] < 0 < states[1][-5]


def test_history_tracks_visibility_and_missing_velocity_without_hidden_timers():
    history = EnemyObservationHistory()
    enemy = dict(x=120, y=180, w=12, h=24, kind="piranha_plant")
    scene = SimpleNamespace(mario={"x": 40}, enemies=[enemy], platforms=[])
    first = history.observe(scene, 10)
    assert first[0] == 1 and first[2] == 0
    enemy["plant_tick"] = 100000  # Must have no effect on an observed feature.
    np.testing.assert_array_equal(history.observe(scene, 10), first)
    stationary = history.observe(scene, 11)
    assert stationary[1] == 0 and stationary[2] == 1
    scene.platforms = [{"rect": pygame.Rect(112, 180, 32, 40)}]
    hidden = history.observe(scene, 12)
    assert hidden[0] == 0 and hidden[2] == 0 and hidden[3] == -1
    assert hidden[5] == pytest.approx(1 / 64)
    scene.platforms = []
    assert history.observe(scene, 13)[2] == 0
    assert history.observe(scene, 15)[2] == 0  # A missing frame is not unit-time velocity.
    history.reset()
    assert history.observe(scene, 0)[2] == 0
    enemy["x"] = 300
    assert history.observe(scene, 1)[0] == 0  # Offscreen boxes are unavailable.


def test_nes_and_block_use_the_same_vertical_motion_history():
    ram = ram_scene()
    ram[0x0F], ram[0x16] = 1, 0x0D  # Piranha slot.
    ram[0x87], ram[0xCF], ram[0xB6], ram[0x49A] = 150, 164, 1, 9
    observer = NESGeometry()
    assert observer.observe(ram, frame=0)["features"]["hazard_vec"][2] == 0
    ram[0xCF] -= 2
    rising = observer.observe(ram, frame=1)["features"]["hazard_vec"]
    assert rising[1] == -0.25 and rising[2] == 1
    ram[0xCF] += 2
    falling = observer.observe(ram, frame=2)["features"]["hazard_vec"]
    assert falling[1] == 0.25 and falling[2] == 1
    observer.reset()
    assert observer.observe(ram, frame=0)["features"]["hazard_vec"][2] == 0


@pytest.mark.parametrize("enabled", [False, True])
def test_history_checkpoint_declares_layout_and_rejects_semantic_mismatch(tmp_path, enabled):
    config = tiny_config(motion_observations=True, hazard_observations=enabled)
    model = make_block_smb_model(config)
    path = tmp_path / "policy.pth"
    save_block_smb_checkpoint(
        path,
        model,
        torch.optim.Adam(model.parameters()),
        config=config,
        epoch=1,
        global_step=1,
        metrics={},
    )
    checkpoint = restore_block_smb_checkpoint(path, model, hazard_observations=enabled)
    names = checkpoint["specs"]["smb_observation"]["features"]
    assert (names[-6:] == list(HAZARD_NAMES)) == enabled
    with pytest.raises(ValueError, match="enemy-history"):
        restore_block_smb_checkpoint(path, model, hazard_observations=not enabled)
    assert SMBRuntimeContract.from_block_config(checkpoint["config"]).hazard_observations == enabled


def test_history_contract_requires_motion_and_supported_projection():
    with pytest.raises(ValueError):
        BlockSMBObservationConfig(hazard_observations=True)
    with pytest.raises(ValueError):
        SMBRuntimeContract(hazard_observations=True, motion_observations=False)
    with pytest.raises(ValueError):
        BlockSMBObservationConfig(
            hazard_observations=True, motion_observations=True, scene_schema="smb_scene_v2"
        )


@pytest.mark.parametrize(
    "contract,history,message", [(8, True, "plant labels"), (9, False, "enemy-history")]
)
def test_cached_demonstrations_require_new_labels_and_matching_history(
    tmp_path, monkeypatch, contract, history, message
):
    import json
    import sys

    from scripts import block_smb_joint_learning as learning

    source = tmp_path / "source"
    source.mkdir()
    (source / "config.json").write_text(
        json.dumps(dict(motion_observations=True, hazard_observations=True))
    )
    (source / "demonstration_manifest.json").write_text(
        json.dumps(dict(contract_version=contract, hazard_observations=history))
    )
    dataset = source / "demonstrations.pth"
    torch.save(
        SimpleNamespace(family=torch.tensor([27]), forced_release=torch.tensor([False])), dataset
    )
    monkeypatch.setattr(learning, "_make_vision_factory", lambda *_: (StaticBlockVision, None))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "joint",
            "--output-dir",
            str(tmp_path / "output"),
            "--dataset",
            str(dataset),
            "--motion-observations",
            "--hazard-observations",
        ],
    )
    threads = torch.get_num_threads()
    deterministic = torch.are_deterministic_algorithms_enabled()
    try:
        with pytest.raises(ValueError, match=message):
            learning.main()
    finally:
        torch.set_num_threads(threads)
        torch.use_deterministic_algorithms(deterministic)


def test_full_smb_history_projection_forward_and_snapshot_restore():
    from retroagi.core.smb_runtime import attach_runtime
    from retroagi.stages.full_smb.adapter import FullSMBStage
    from retroagi.stages.full_smb.train import _policy_action_logits_and_state
    from scripts.tests.test_smb_transfer_contract import ContractVision, RAMEnv

    contract = SMBRuntimeContract(hazard_observations=True)
    stage = FullSMBStage(env=RAMEnv(), vision=ContractVision())
    stage.configure_policy_runtime(contract)
    try:
        stage.reset()
        ram = stage.env.ram
        ram[0x0F], ram[0x16] = 1, 0x0D
        ram[0x87], ram[0xCF], ram[0xB6], ram[0x49A] = 150, 164, 1, 9
        first = stage.encode_observation(stage._last_observation)
        assert tuple(first.metadata["vision_fusion"]["c_state"]) == (12, 53)
        saved = stage.save_emulator_state()
        ram[0xCF] -= 2
        stage.step(1)
        second = stage.encode_observation(stage._last_observation)
        assert second.src_c[0, 48] == -0.25
        model = make_block_smb_model(
            tiny_config(motion_observations=True, hazard_observations=True)
        )
        attach_runtime(model, contract.manifest())
        output = _policy_action_logits_and_state(model, second, device=torch.device("cpu"))
        assert torch.isfinite(output.logits).all()
        restored = stage.load_emulator_state(saved)
        torch.testing.assert_close(first.src_c, stage.encode_observation(restored).src_c)
    finally:
        stage.close()
