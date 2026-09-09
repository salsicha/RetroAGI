"""Regressions for coaching under the exact composable observation/executor contract."""

from types import SimpleNamespace

import pytest
import torch

from retroagi.core.smb_coaching import (
    bridge_walk_reached,
    safe_bridge_wait_indices,
    safe_jump_indices,
)
from retroagi.core.smb_learning import block_stage, collect_case, make_model, primitive
from retroagi.core.smb_objectives import objective_goal, observable_objective
from retroagi.core.smb_runtime import make_smb_executor
from retroagi.core.smb_scene import block_oracle_scene
from retroagi.core.smb_supervision import collision_labels, labels_to_vision
from retroagi.core.smb_tracking import PerceivedSMBScene
from retroagi.stages.block_smb.geometry_expert import restore_env_state, snapshot_env_state
from retroagi.stages.block_smb.nes_curriculum import sample_nes_case


def sample(family, index=10000, difficulty="easy"):
    return sample_nes_case(
        family=family, split="train", seed=20260908, index=index, difficulty=difficulty
    )


def test_bridge_target_exists_with_far_shore_outside_view_and_no_hidden_bounds():
    stage = block_stage(sample("bridge_wait"))
    try:
        stage.reset()
        scene = block_oracle_scene(stage.env)["scene"]
        for p in scene.platforms:
            p.pop("move_min", None)
            p.pop("move_max", None)
        objective = observable_objective(scene)
        assert objective.kind.startswith("bridge_")
        assert objective_goal(objective).any()
        tracker = PerceivedSMBScene()
        perceived = tracker.observe(labels_to_vision(collision_labels(scene)))
        assert perceived["objective"].kind == objective.kind
        assert perceived["skill_goal"].equal(objective_goal(objective))
    finally:
        stage.env.close()


def test_stomp_request_reaches_pixel_goal_and_differs_from_optional_clear():
    stage = block_stage(sample("enemy_stomp"))
    try:
        stage.reset()
        scene = block_oracle_scene(stage.env)["scene"]
        vision = labels_to_vision(collision_labels(scene))
        optional = PerceivedSMBScene().observe(vision)
        required = PerceivedSMBScene().observe(vision, objective_kind="stomp")
        assert optional["objective"].kind == "enemy"
        assert required["objective"].kind == "stomp"
        assert not torch.equal(optional["skill_goal"], required["skill_goal"])
        batch = stage.encode_observation(stage.env.render())
        assert batch.metadata["smb_geometry"]["objective"].kind == "stomp"
    finally:
        stage.env.close()


def test_wait_and_landing_boundaries_are_resolved_before_actor_label():
    executor = make_smb_executor(make_model(hidden_dim=8))

    def batch(phase, support="ground", bouncing=False):
        return SimpleNamespace(
            metadata={
                "smb_geometry": dict(
                    objective=SimpleNamespace(kind=phase),
                    support=support,
                    bouncing=bouncing,
                )
            }
        )

    executor.execute(0, batch=batch("bridge_wait"), motor_primitives=primitive(15))
    assert executor.committed_action is None  # bridge waits reobserve after one frame
    assert executor.prepare(batch("bridge_board")) is None
    assert executor.prepare(batch("bridge_board")) is None
    executor.execute(2, batch=batch("mount"), motor_primitives=primitive(0))
    executor.execute(2, batch=batch("mount", "air"), motor_primitives=primitive(0))
    assert executor.committed_action == 2
    assert executor.prepare(batch("finish")) is None
    executor.execute(2, batch=batch("stomp"), motor_primitives=primitive(15))
    assert executor.prepare(batch("stomp", "air", True)) is None
    assert executor.execute(2, batch=batch("stomp", "air", True)).action == 1


def test_bridge_wait_sets_are_actual_collision_windows_and_restore_state():
    stage = block_stage(sample("bridge_wait", index=10002, difficulty="hard"))
    try:
        stage.reset()
        env = stage.env
        saved = snapshot_env_state(env)
        _, departures, _ = safe_bridge_wait_indices(env)
        assert snapshot_env_state(env) == saved
        # Advance to a state with a departure inside the physical menu.
        for _ in range(160):
            if departures:
                break
            env.step(0)
            _, departures, _ = safe_bridge_wait_indices(env)
        assert departures
        saved = snapshot_env_state(env)
        from retroagi.core.smb_physics import NES_JUMP_FRAMES

        for index in departures:
            restore_env_state(env, saved)
            for _ in range(NES_JUMP_FRAMES[index]):
                env.step(0)
                assert env.mario["on_ground"]
            reached = False
            for _ in range(48):
                _, _, done, _, info = env.step(1)
                assert not info["death"] and env.mario["on_ground"]
                if bridge_walk_reached(env, "board"):
                    reached = True
                    break
                assert not done
            assert reached
    finally:
        stage.env.close()


@pytest.mark.parametrize(
    "family", ["tall_pipe_jump", "pit_leap", "enemy_stomp", "stomp_mount", "platform_hop"]
)
def test_coached_routes_have_safe_initiations_and_exclude_bounce_decisions(family):
    item = sample(family)
    stage = block_stage(item)
    try:
        rows, result = collect_case(make_model(hidden_dim=8), stage, item.oracle["actions"])
        if not rows:
            from retroagi.core.smb_learning import collect_reactive_case

            rows, result = collect_reactive_case(make_model(hidden_dim=8), stage)
        assert result["success"], result
        assert rows and any(r[7] and r[4] in (2, 4, 5) for r in rows)
        assert all(r[10][r[6]] for r in rows)
        if family in ("enemy_stomp", "stomp_mount"):
            assert stage.env._stomp_credited
    finally:
        stage.env.close()


def test_jump_probes_preserve_fractional_motion_and_credit():
    stage = block_stage(sample("stomp_mount"))
    try:
        stage.reset()
        saved = snapshot_env_state(stage.env)
        safe_jump_indices(make_model(hidden_dim=8), stage.env, 2)
        assert snapshot_env_state(stage.env) == saved
    finally:
        stage.env.close()


def test_playback_never_calls_coaching(monkeypatch):
    from retroagi.core import smb_coaching
    from retroagi.core.smb_learning import playback

    def forbidden(*args, **kwargs):
        raise AssertionError("Teacher called during policy playback")

    monkeypatch.setattr(smb_coaching, "coach_choice", forbidden)
    monkeypatch.setattr(smb_coaching, "safe_jump_indices", forbidden)
    monkeypatch.setattr(smb_coaching, "safe_bridge_wait_indices", forbidden)
    stage = block_stage(sample("bridge_wait"))
    try:
        assert playback(make_model(hidden_dim=8), stage, max_steps=3)["frames"] == 3
    finally:
        stage.env.close()


def test_component_loader_rejects_old_goal_meanings(tmp_path):
    import json

    from retroagi.core.smb_components import SMBComponentContract, export_bundle, load_component

    model = make_model(hidden_dim=8)
    path = tmp_path / "bundle"
    contract = SMBComponentContract()
    export_bundle(model, path, contract=contract, architecture={"hidden_dim": 8})
    manifest_path = path / "bundle.json"
    manifest = json.loads(manifest_path.read_text())
    del manifest["contract"]["objective_contract"]
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="contract mismatch"):
        load_component(model, path, "actor", contract=contract, architecture={"hidden_dim": 8})


def test_bridge_exit_target_survives_overlap_with_far_shore():
    import pygame

    stage = block_stage(sample("bridge_wait"))
    try:
        stage.reset()
        scene = block_oracle_scene(stage.env)["scene"]
        bridge = dict(rect=pygame.Rect(100, 220, 100, 20), moving=True)
        shore = dict(rect=pygame.Rect(188, 220, 68, 20), moving=False)
        scene.platforms = [bridge, shore]
        scene.mario.update(x=189, y=208, w=10, h=12, on_ground=True, _platform=bridge)
        objective = observable_objective(scene)
        assert objective.kind == "bridge_exit"
        assert objective.left == 188
        scene.mario["x"] = 101
        assert observable_objective(scene).kind == "bridge_board"
    finally:
        stage.env.close()


def test_inference_releases_landed_commitment_before_forcing_actor_context():
    from retroagi.stages.full_smb.train import _smb_forward_kwargs

    model = make_model(hidden_dim=8)
    stage = block_stage(sample("flat_run"))
    try:
        obs = stage.reset()
        batch = stage.encode_observation(obs)
        executor = make_smb_executor(model)
        executor.execute(2, batch=batch, motor_primitives=primitive(0))
        executor._released = executor._left_support = True
        assert "forced_action" not in _smb_forward_kwargs(model, batch, True)
        assert executor.committed_action is None
    finally:
        stage.env.close()


@pytest.mark.parametrize("provider", ["oracle", "perceived"])
@pytest.mark.parametrize("direction", [-1, 1])
def test_full_adapter_accepts_same_explicit_task_direction(provider, direction):
    import numpy as np

    from retroagi.core.smb_learning import runtime
    from retroagi.stages.full_smb.adapter import FullSMBStage
    from scripts.tests.test_smb_transfer_contract import RAMEnv

    class Vision:
        def encode(self, observation):
            labels = np.zeros((240, 256), dtype=np.uint8)
            labels[208:] = 2
            labels[196:208, 43:53] = 1
            return labels_to_vision(labels)

    stage = FullSMBStage(env=RAMEnv(), vision=Vision(), task_direction=direction)
    stage.configure_policy_runtime(runtime(provider))
    try:
        batch = stage.encode_observation(stage.reset())
        geometry = batch.metadata["smb_geometry"]
        assert geometry["objective"].direction == direction
        assert geometry["features"]["state_vec"][15] * direction > 0
    finally:
        stage.close()
