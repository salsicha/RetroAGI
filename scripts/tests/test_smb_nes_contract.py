"""Canonical policy inputs and NES physics that both games share."""

from dataclasses import replace

import pytest
import torch

from retroagi.core.interfaces import VisionOutput
from retroagi.core.smb_scene import C_SPANS, SMBProjector, canonical_vision
from retroagi.stages.block_smb.adapter import BLOCK_SMB_SPEC


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
    projector = SMBProjector(BLOCK_SMB_SPEC)
    observed = torch.zeros(1, C_SPANS["c_enemy_relative_motion"][1] - C_SPANS["c_state"][0])
    left = projector.project(canonical_vision(a, "block"), observed)
    right = projector.project(canonical_vision(b, "block"), observed)
    assert torch.equal(left.src_c, right.src_c)
    assert left.src_c.shape == (1, 64)
    assert left.metadata["vision_fusion"]["c_availability"] == C_SPANS["c_availability"]
    with pytest.raises(ValueError, match="canonical"):
        projector.project(a, observed)


def test_nes_snapshot_restores_fractional_physics():
    from retroagi.stages.block_smb.env import MarioScenarioEnv
    from retroagi.stages.block_smb.geometry_expert import restore_env_state, snapshot_env_state

    env = MarioScenarioEnv()
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


def test_centered_crop_preserves_pixel_coordinates():
    import numpy as np

    from retroagi.core.smb_scene import canonical_rgb

    crop = np.zeros((224, 240, 3), dtype=np.uint8)
    crop[20, 30] = 255
    full = canonical_rgb(crop)
    assert full.shape == (240, 256, 3)
    assert (full[28, 38] == 255).all()


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

    env = MarioScenarioEnv()
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
