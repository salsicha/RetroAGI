"""Component swaps preserve meaning, gradients, and the physical interface."""

from dataclasses import replace

import pytest
import torch

from retroagi.core.interfaces import VisionOutput
from retroagi.core.smb_components import (
    SMBComponentContract,
    component_states,
    export_bundle,
    load_component,
    parameter_digest,
    trainable_components,
    verify_frozen_components,
)
from retroagi.core.smb_scene import CanonicalSMBProjector, canonical_vision
from retroagi.stages.block_smb.adapter import BLOCK_SMB_SPEC
from retroagi.stages.block_smb.train import BlockSMBTrainingConfig, make_block_smb_model


def model():
    return make_block_smb_model(
        BlockSMBTrainingConfig(hidden_dim=8, architecture_config={"hidden_dim": 8})
    )


def test_component_roundtrip_and_world_model_swap(tmp_path):
    first, second = model(), model()
    contract = SMBComponentContract()
    architecture = {"hidden_dim": 8}
    export_bundle(first, tmp_path / "bundle", contract=contract, architecture=architecture)
    untouched = parameter_digest(component_states(second)["actor"])
    load_component(
        second, tmp_path / "bundle", "world_model", contract=contract, architecture=architecture
    )
    assert parameter_digest(component_states(second)["actor"]) == untouched
    assert parameter_digest(component_states(first)["world_model"]) == parameter_digest(
        component_states(second)["world_model"]
    )
    assert not second.full_level_qualified
    for name in component_states(first):
        load_component(
            second, tmp_path / "bundle", name, contract=contract, architecture=architecture
        )
    assert parameter_digest(first.state_dict()) == parameter_digest(second.state_dict())
    with pytest.raises(ValueError, match="contract mismatch"):
        load_component(
            second,
            tmp_path / "bundle",
            "actor",
            contract=replace(contract, recurrent_state=True),
            architecture=architecture,
        )


def test_world_model_update_preserves_frozen_actor_and_critic():
    m = model()
    frozen = trainable_components(m, ["world_model"])
    before = parameter_digest(component_states(m)["world_model"])
    optimizer = torch.optim.SGD([p for p in m.parameters() if p.requires_grad], lr=0.01)
    # Exercise the complete recurrent component, including its decoder.
    state = torch.randn(4, 64)
    prediction = m.world_model(
        state, torch.ones_like(state), torch.zeros_like(state), torch.zeros_like(state)
    )
    loss = torch.nn.functional.mse_loss(prediction, state + 0.1)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    verify_frozen_components(m, frozen)
    assert parameter_digest(component_states(m)["world_model"]) != before


def test_latent_basis_cannot_change_canonical_policy_inputs():
    logits = torch.randn(1, 7, 15, 16)
    a = VisionOutput(
        torch.zeros(1, 2),
        logits,
        logits.argmax(1),
        torch.randn(1, 240, 64),
        support_logits=torch.ones(1, 3),
    )
    b = replace(a, tokens=torch.randn(1, 240, 192) * 100)
    projector = CanonicalSMBProjector(BLOCK_SMB_SPEC)
    left = projector.project(canonical_vision(a, "block"), torch.zeros(1, 35), availability=[1] * 8)
    right = projector.project(
        canonical_vision(b, "block"), torch.zeros(1, 35), availability=[1] * 8
    )
    assert torch.equal(left.src_c, right.src_c)
    assert left.src_c.shape == (1, 64)
    assert left.metadata["vision_fusion"]["c_availability"] == (47, 55)
    with pytest.raises(ValueError, match="availability"):
        projector.project(canonical_vision(a, "block"), torch.zeros(1, 35))


def test_dense_perception_bundle_roundtrip_and_tamper(tmp_path):
    from retroagi.core.smb_components import load_bundle
    from retroagi.core.smb_learning import make_model
    from retroagi.core.smb_perception import DenseSMBPerception

    vision = DenseSMBPerception(dim=32, depth=1)
    path = tmp_path / "vision.pth"
    vision.save(path)
    m = make_model(hidden_dim=8, provider="perceived")
    export_bundle(
        m,
        tmp_path / "pack",
        contract=SMBComponentContract(),
        architecture={"hidden_dim": 8, "controller_schedule": "constant"},
        perception=path,
    )
    loaded, v, manifest = load_bundle(tmp_path / "pack")
    assert parameter_digest(m.state_dict()) == parameter_digest(loaded.state_dict())
    assert parameter_digest(vision.state_dict()) == parameter_digest(v.state_dict())
    assert loaded.smb_runtime_contract.observation_provider == "perceived"
    with (tmp_path / "pack" / "world_model.pth").open("ab") as stream:
        stream.write(b"bad")
    with pytest.raises(ValueError, match="checksum"):
        load_bundle(tmp_path / "pack")


def test_perceived_motion_separates_camera_and_objects():
    import numpy as np

    from retroagi.core.smb_supervision import labels_to_vision
    from retroagi.core.smb_tracking import PerceivedSMBScene

    a = np.zeros((240, 256), dtype=np.uint8)
    a[208:, :] = 2
    a[160:208, 120:152] = 2
    a[196:208, 60:70] = 1
    a[196:202, 170:180] = 5
    a[175:182, 200:230] = 6
    b = np.zeros_like(a)
    b[208:, :] = 2
    b[160:208, 117:149] = 2
    b[196:208, 59:69] = 1
    b[196:202, 166:176] = 5
    b[175:182, 199:229] = 6
    tracker = PerceivedSMBScene()
    first = tracker.observe(labels_to_vision(a))
    second = tracker.observe(labels_to_vision(b))
    assert first["availability"][2] == 0
    assert second["scroll"] == 3
    assert second["scene"].mario["vx"] == 2
    assert second["features"]["motion_vec"][0] == pytest.approx(-1 / 3)
    assert second["features"]["motion_vec"][5] == pytest.approx(2 / 3)
    assert second["availability"][3] == second["availability"][5] == 1
    missing = tracker.observe(labels_to_vision(np.zeros_like(a)))
    assert missing["availability"][0:3] == [0, 0, 0]


def test_nes_snapshot_restores_fractional_physics():
    from retroagi.stages.block_smb.env import MarioScenarioEnv
    from retroagi.stages.block_smb.geometry_expert import restore_env_state, snapshot_env_state

    env = MarioScenarioEnv(physics_profile="nes_land_v1")
    try:
        env.reset(
            scenario={"mario": [43, 196], "platforms": [[0, 208, 512, 32]], "world_width": 512}
        )
        for _ in range(11):
            env.step(1)
        saved = snapshot_env_state(env)
        for _ in range(8):
            env.step(2)
        expected = (dict(env.motion.__dict__), env.mario["x"], env.mario["y"])
        restore_env_state(env, saved)
        for _ in range(8):
            env.step(2)
        assert expected == (dict(env.motion.__dict__), env.mario["x"], env.mario["y"])
    finally:
        env.close()


def test_perceived_emulator_snapshot_is_idempotent_and_does_not_read_ram():
    import numpy as np

    from retroagi.core.smb_learning import runtime
    from retroagi.core.smb_supervision import labels_to_vision
    from retroagi.stages.full_smb.adapter import FullSMBStage
    from scripts.tests.test_smb_transfer_contract import RAMEnv

    class Vision:
        def encode(self, _):
            labels = np.zeros((240, 256), dtype=np.uint8)
            labels[208:] = 2
            labels[196:208, 43:53] = 1
            return labels_to_vision(labels)

    stage = FullSMBStage(env=RAMEnv(), vision=Vision())
    stage.configure_policy_runtime(runtime("perceived"))
    obs = stage.reset()
    try:

        def forbidden():
            raise AssertionError("Pixel policy read RAM")

        stage.env.get_ram = forbidden
        first = stage.encode_observation(obs)
        snapshot = stage.save_emulator_state()
        stage.scene_tracker.frames += 10
        obs = stage.load_emulator_state(snapshot)
        second = stage.encode_observation(obs)
        assert torch.equal(first.src_c, second.src_c)
        assert stage.scene_tracker.frames == 1
    finally:
        stage.close()


def test_nes_primitive_units_include_one_frame_wait():
    from retroagi.core.smb_learning import make_model, primitive
    from retroagi.core.smb_runtime import make_smb_executor

    m = make_model(hidden_dim=8)
    executor = make_smb_executor(m)
    wait = executor.execute(0, motor_primitives=primitive(0), support_override="ground")
    assert wait.hold_frames == 1 and executor.committed_action is None
    jump = executor.execute(2, motor_primitives=primitive(15), support_override="ground")
    assert jump.hold_frames == 32
    assert m.motor_controller.duration_bin_values.tolist() == list(
        SMBComponentContract().jump_frames
    )


def test_centered_crop_preserves_pixel_coordinates():
    import numpy as np

    from retroagi.core.smb_scene import canonical_rgb

    crop = np.zeros((224, 240, 3), dtype=np.uint8)
    crop[20, 30] = 255
    full = canonical_rgb(crop)
    assert full.shape == (240, 256, 3)
    assert (full[28, 38] == 255).all()


def test_ordered_dynamics_adaptation_has_causal_actor_path_and_frozen_weights():
    from retroagi.core.smb_learning import (
        fit_ordered_sequences,
        make_model,
        recurrent_causal_probe,
        rows_to_data,
        runtime,
    )
    from retroagi.core.smb_runtime import attach_runtime

    m = make_model(hidden_dim=8)
    attach_runtime(m, replace(runtime(), recurrent_state=True).manifest())
    rows = []
    for i in range(4):
        c = torch.randn(1, 64)
        rows.append(
            (
                torch.ones(1, 8, dtype=torch.long),
                torch.ones(1, 16, dtype=torch.long),
                c,
                torch.zeros(1, 8),
                1,
                1,
                0,
                True,
                c + 0.1,
                0,
                [True] + [False] * 15,
            )
        )
    # Use the core's actual goal width, not an assumed feature allocation.
    from retroagi.core.skills import SKILL_GOAL_ENCODING_DIM

    rows = [(*r[:3], torch.zeros(1, SKILL_GOAL_ENCODING_DIM), *r[4:]) for r in rows]
    data = rows_to_data(rows)
    result = fit_ordered_sequences(m, data, [dict(start=0, length=4)], world_model_only=True)
    verify_frozen_components(m, result["frozen_hashes"])
    assert result["updates"] == 4
    assert recurrent_causal_probe(m, data)["causal_path_present"]


def test_goomba_flattened_state_triggers_one_bounce_and_removes_hazard():
    from retroagi.stages.full_smb.geometry import NESGeometry
    from scripts.tests.test_smb_transfer_contract import ram_scene

    ram = ram_scene()
    ram[0x0F] = 1
    ram[0x16] = 6
    ram[0x87] = 80
    ram[0xCF] = 184
    ram[0xB6] = 1
    ram[0x49A] = 9
    observer = NESGeometry()
    assert len(observer.observe(ram, frame=0)["scene"].enemies) == 1
    ram[0x1E] = 4
    ram[0x9F] = 252
    ram[0x1D] = 1
    contact = observer.observe(ram, frame=1)
    assert contact["enemy_contact"] and contact["bouncing"] and not contact["scene"].enemies
    assert not observer.observe(ram, frame=2)["enemy_contact"]


def test_landing_clears_fractional_vertical_force():
    from retroagi.core.smb_physics import NESPlayerMotion

    motion = NESPlayerMotion(y_speed=3, y_force=224, y_fraction=64)
    motion.vertical_contact()
    assert motion.y_speed == motion.y_force == 0
    assert motion.y_fraction == 64
    motion.bounce()
    assert motion.y_speed == -4


def test_nes_enemy_damage_body_is_distinct_from_floor_probe():
    from retroagi.stages.block_smb.env import MarioScenarioEnv

    env = MarioScenarioEnv(physics_profile="nes_land_v1")
    try:
        env.reset(
            scenario={
                "mario": [43, 196],
                "platforms": [[0, 208, 256, 32]],
                "enemies": [[120, 194, 90, 200, 0.5, -1]],
            }
        )
        for _ in range(4):
            env.step(0)
        enemy = env.enemies[0]
        assert (enemy["w"], enemy["h"]) == (10, 6)
        assert enemy["y"] + enemy["h"] == 204
        assert enemy["on_ground"]
    finally:
        env.close()


def test_retreat_is_an_explicit_task_input_without_rendered_goal_markers():
    import numpy as np

    from retroagi.core.smb_scene import apply_local_target
    from retroagi.core.smb_supervision import labels_to_vision
    from retroagi.core.smb_tracking import PerceivedSMBScene

    labels = np.zeros((240, 256), dtype=np.uint8)
    labels[220:] = 2
    labels[208:220, 195:205] = 1
    left = apply_local_target(
        PerceivedSMBScene().observe(labels_to_vision(labels), goal_direction=-1)
    )
    right = apply_local_target(
        PerceivedSMBScene().observe(labels_to_vision(labels), goal_direction=1)
    )
    assert left["objective"].direction == -1 and right["objective"].direction == 1
    assert left["features"]["state_vec"][15] < 0 < right["features"]["state_vec"][15]
