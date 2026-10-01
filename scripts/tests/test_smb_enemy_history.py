"""Plant timing must be observable, shared with NES, and recorded in checkpoints."""

from types import SimpleNamespace

import numpy as np
import pygame
import pytest
import torch

from retroagi.core.smb_enemy_history import HAZARD_NAMES, EnemyObservationHistory
from retroagi.core.smb_geometry import FEATURE_NAMES
from retroagi.core.smb_runtime import SMBRuntimeContract
from retroagi.core.smb_scene import C_SPANS
from retroagi.stages.block_smb.adapter import BlockSMBStage
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
        )
        try:
            stage.reset()
            frame, _, _, _, info = stage.step(0)
            frames.append(frame)
            states.append(stage.state_features(info))
            assert history(states[-1])[2] == 1  # Measured velocity, not a guessed phase.
        finally:
            stage.env.close()
    np.testing.assert_array_equal(frames[0], frames[1])
    geometry = len(FEATURE_NAMES)
    np.testing.assert_array_equal(states[0][:geometry], states[1][:geometry])
    assert history(states[0])[1] < 0 < history(states[1])[1]


def history(observed):
    start = len(FEATURE_NAMES)
    return observed[start : start + len(HAZARD_NAMES)]


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


def test_peak_exposure_memory_remembers_what_disappeared():
    history = EnemyObservationHistory()
    plant = dict(x=120, y=180, w=12, h=0, kind="piranha_plant")
    scene = SimpleNamespace(mario={"x": 40}, enemies=[plant], platforms=[])

    def seen(frame, height):
        plant.update(h=height, y=180 - height)
        features = history.observe(scene, frame)
        return features, history.memory_features()[0]

    features, memory = seen(0, 0)
    assert features[5] == 1 and memory == 0  # Never seen: no height is known.
    assert seen(1, 20)[1] == pytest.approx(20 / 64)
    assert seen(2, 80)[1] == 1  # Saturates beyond 64 exposed pixels.
    for frame in range(3, 40):
        features, memory = seen(frame, 0)
        assert features[0] == 0 and memory == 1  # Held while the plant hides.
    assert seen(40, 7)[1] == 1  # A re-emerging plant keeps its identity.
    np.testing.assert_array_equal(history.memory_features(), history.memory_features())
    history.reset()
    assert history.memory_features()[0] == 0
    # Only the part above a pipe counts, as for the other history features.
    scene.platforms = [{"rect": pygame.Rect(112, 170, 32, 50)}]
    assert seen(0, 30)[1] == pytest.approx(20 / 64)


def test_peak_exposure_is_a_memory_target_not_a_policy_input():
    stage = BlockSMBStage(
        scenario=contact_scenario(phase=20, mario=(40, 204)), vision=StaticBlockVision()
    )
    try:
        stage.reset()
        stage.step(0)
        assert stage._hazard_memory[0] > 0
        observed = stage.state_features()
        assert observed.shape == (C_SPANS["c_enemy_relative_motion"][1] - C_SPANS["c_state"][0],)
    finally:
        stage.env.close()


def test_nes_and_block_use_the_same_vertical_motion_history():
    ram = ram_scene()
    ram[0x0F], ram[0x16] = 1, 0x0D  # Piranha slot.
    ram[0x87], ram[0xCF], ram[0xB6], ram[0x49A] = 150, 164, 1, 9
    observer = NESGeometry()
    assert observer.observe(ram, frame=0)["enemy_history"][2] == 0
    ram[0xCF] -= 2
    rising = observer.observe(ram, frame=1)["enemy_history"]
    assert rising[1] == -0.25 and rising[2] == 1
    ram[0xCF] += 2
    falling = observer.observe(ram, frame=2)["enemy_history"]
    assert falling[1] == 0.25 and falling[2] == 1
    observer.reset()
    assert observer.observe(ram, frame=0)["enemy_history"][2] == 0


def test_checkpoint_records_the_observation_and_rejects_another(tmp_path):
    config = tiny_config()
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
    checkpoint = restore_block_smb_checkpoint(path, model)
    spec = checkpoint["specs"]["smb_observation"]
    assert spec["enemy_history"] == list(HAZARD_NAMES)
    assert spec["features"] == list(FEATURE_NAMES)
    checkpoint["specs"]["smb_observation"] = {**spec, "features": spec["features"][:-1]}
    torch.save(checkpoint, path)
    with pytest.raises(ValueError, match="retrain"):
        restore_block_smb_checkpoint(path, model)


def test_cached_demonstrations_from_other_teacher_code_are_rejected(tmp_path, monkeypatch):
    import json
    import sys

    from scripts import block_smb_joint_learning as learning

    source = tmp_path / "source"
    source.mkdir()
    (source / "config.json").write_text(json.dumps({}))
    (source / "demonstration_manifest.json").write_text(json.dumps(dict(teacher_digest="old")))
    dataset = source / "demonstrations.pth"
    torch.save(
        SimpleNamespace(family=torch.tensor([27]), forced_release=torch.tensor([False])), dataset
    )
    monkeypatch.setattr(learning, "_make_vision_factory", lambda *_: (StaticBlockVision, None))
    monkeypatch.setattr(
        sys,
        "argv",
        ["joint", "--output-dir", str(tmp_path / "output"), "--dataset", str(dataset)],
    )
    threads = torch.get_num_threads()
    deterministic = torch.are_deterministic_algorithms_enabled()
    try:
        with pytest.raises(ValueError, match="different teacher code"):
            learning.main()
    finally:
        torch.set_num_threads(threads)
        torch.use_deterministic_algorithms(deterministic)


def test_full_smb_history_projection_forward_and_snapshot_restore():
    from retroagi.core.smb_runtime import attach_runtime
    from retroagi.stages.full_smb.adapter import FullSMBStage
    from retroagi.stages.full_smb.train import _policy_action_logits_and_state
    from scripts.tests.test_smb_transfer_contract import ContractVision, RAMEnv

    contract = SMBRuntimeContract()
    stage = FullSMBStage(env=RAMEnv(), vision=ContractVision())
    stage.configure_policy_runtime(contract)
    try:
        stage.reset()
        ram = stage.env.ram
        ram[0x0F], ram[0x16] = 1, 0x0D
        ram[0x87], ram[0xCF], ram[0xB6], ram[0x49A] = 150, 164, 1, 9
        first = stage.encode_observation(stage._last_observation)
        assert first.metadata["vision_fusion"]["c_enemy_history"] == C_SPANS["c_enemy_history"]
        saved = stage.save_emulator_state()
        ram[0xCF] -= 2
        stage.step(1)
        second = stage.encode_observation(stage._last_observation)
        assert second.src_c[0, C_SPANS["c_enemy_history"][0] + 1] == -0.25
        model = make_block_smb_model(tiny_config())
        attach_runtime(model, contract.manifest())
        output = _policy_action_logits_and_state(model, second, device=torch.device("cpu"))
        assert torch.isfinite(output.logits).all()
        restored = stage.load_emulator_state(saved)
        torch.testing.assert_close(first.src_c, stage.encode_observation(restored).src_c)
    finally:
        stage.close()


def test_full_smb_reports_peak_exposure_as_a_memory_target():
    from retroagi.stages.full_smb.adapter import FullSMBStage
    from scripts.tests.test_smb_transfer_contract import ContractVision, RAMEnv

    stage = FullSMBStage(env=RAMEnv(), vision=ContractVision())
    stage.configure_policy_runtime(SMBRuntimeContract())
    try:
        stage.reset()
        ram = stage.env.ram
        ram[0x0F], ram[0x16] = 1, 0x0D
        ram[0x87], ram[0xCF], ram[0xB6], ram[0x49A] = 150, 164, 1, 9
        batch = stage.encode_observation(stage._last_observation)
        assert batch.metadata["smb_geometry"]["enemy_memory"][0] > 0
    finally:
        stage.close()


@pytest.mark.parametrize("supervision_start", [0, 12])
def test_demonstrations_and_recovery_match_live_history_on_every_frame(supervision_start):
    from dataclasses import replace

    from retroagi.stages.block_smb.demonstrations import collect_demonstrations
    from retroagi.stages.block_smb.monte_carlo import sample_block_smb_monte_carlo_scenario

    sample = sample_block_smb_monte_carlo_scenario(
        family="piranha_avoidance",
        split="train",
        seed=13,
        difficulty="medium",
        sample_index=0,
    )
    sample = replace(
        sample,
        oracle={
            **sample.oracle,
            "supervision_start_frame": supervision_start,
            "recovery": bool(supervision_start),
        },
    )
    config = tiny_config(
        walk_duration_primitives=False,
    )
    data = collect_demonstrations([(27, sample)], config, StaticBlockVision)
    stage = BlockSMBStage(scenario=sample.scenario, vision=StaticBlockVision())
    expected = []
    try:
        frame = stage.reset(seed=sample.sample_seed % (2**31))
        for action in sample.oracle["actions"]:
            expected.append(stage.encode_observation(frame).src_c)
            frame, _, done, truncated, _ = stage.step(action)
            if done or truncated:
                break
        lo, _ = C_SPANS["c_enemy_history"]
        expected = torch.cat(expected)[supervision_start:]
        assert expected[:, lo + 2].any()  # Velocity is available after observation.
        assert expected[:, lo + 1].abs().max() > 0  # Rising/retracting motion.
        # Replayed recovery prefixes remain as unsupervised context rows.
        assert int(data.context.sum()) == supervision_start
        torch.testing.assert_close(data.c[data.context.logical_not()], expected)
        torch.testing.assert_close(data.next_c[-1:], stage.encode_observation(frame).src_c)
    finally:
        stage.env.close()
