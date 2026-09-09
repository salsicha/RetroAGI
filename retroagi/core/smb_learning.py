"""Shared canonical curriculum collection, playback and component adaptation.

No evaluator calls a teacher. All policy paths use the checkpoint-owned
executor. Oracle and perceived lanes use separate, explicitly named providers.
"""

from dataclasses import asdict
from types import SimpleNamespace

import torch
from torch.nn import functional as F

from retroagi.core.smb_components import (
    trainable_components,
    verify_frozen_components,
)
from retroagi.core.smb_physics import NES_JUMP_FRAMES, NES_PHYSICS_PROFILE
from retroagi.core.smb_runtime import SMBRuntimeContract, attach_runtime, make_smb_executor
from retroagi.core.smb_scene import block_oracle_scene
from retroagi.core.smb_supervision import OracleSceneVision
from retroagi.stages.block_smb.adapter import BlockSMBObservationConfig, BlockSMBStage
from retroagi.stages.block_smb.demonstrations import DemonstrationBatch
from retroagi.stages.block_smb.env import MarioScenarioEnv
from retroagi.stages.block_smb.train import BlockSMBTrainingConfig, make_block_smb_model
from retroagi.stages.full_smb.train import _policy_action_logits_and_state


def runtime(provider="oracle"):
    return SMBRuntimeContract(
        objective_contract="observable_traversal_v2",
        schema="smb_scene_v2",
        visual_tokens="canonical",
        physics_profile=NES_PHYSICS_PROFILE,
        observation_provider=provider,
        geometry_source="canonical_pixels" if provider == "perceived" else "canonical_oracle",
        jump_hold_frames=NES_JUMP_FRAMES,
        recurrent_state=False,
        adaptive_duration=False,
        walk_primitives=False,
        steady_primitives=True,
        critic_feedback=False,
        wait_duration_scale=1.0,
        min_wait_frames=1,
        max_wait_frames=32,
    )


def make_model(*, hidden_dim=128, device="cpu", provider="oracle"):
    config = BlockSMBTrainingConfig(
        hidden_dim=hidden_dim,
        architecture_config={"hidden_dim": hidden_dim, "controller_schedule": "constant"},
        ranked_candidate_search=False,
    )
    model = make_block_smb_model(config).to(device)
    attach_runtime(model, runtime(provider).manifest())
    return model


def block_stage(sample, *, vision=None, device="cpu"):
    env = MarioScenarioEnv(physics_profile=NES_PHYSICS_PROFILE)
    provider = "perceived" if vision is not None else "oracle"
    encoder = vision or OracleSceneVision(lambda: block_oracle_scene(env)["scene"], device)
    return BlockSMBStage(
        env=env,
        scenario=sample.scenario,
        vision=encoder,
        observation_config=BlockSMBObservationConfig(
            motion_observations=True, scene_schema="smb_scene_v2", observation_provider=provider
        ),
    )


def primitive(index):
    logits = torch.full((1, 1, 16), -20.0)
    logits[..., index] = 20.0
    return SimpleNamespace(
        hold_duration_logits=logits, duration_bin_values=torch.tensor(NES_JUMP_FRAMES)
    )


def rows_to_data(rows):
    if not rows:
        raise ValueError("No executable successful demonstrations")
    return DemonstrationBatch(
        *(
            (
                torch.cat([r[i] for r in rows])
                if i in (0, 1, 2, 3, 8)
                else torch.tensor([r[i] for r in rows])
            )
            for i in range(11)
        )
    )


def _collect_coached(
    model, stage, actions=None, *, family=0, seed=0, max_steps=320, takeoff_distance=50, variant=0
):
    from retroagi.core.smb_coaching import (
        COACHING_CONTRACT,
        coach_choice,
        safe_jump_indices,
    )

    obs = stage.reset(seed=seed)
    executor = make_smb_executor(model)
    if stage.env._require_bridge_before_goal:
        actions = None
    rows = []
    selected_index = 0
    selected_valid = [0]
    decisions = safe_sets = recoveries = 0
    info = {}
    for frame in range(max_steps if actions is None else min(max_steps, len(actions))):
        batch = stage.encode_observation(obs)
        committed = executor.prepare(batch)
        bouncing = bool(batch.metadata["smb_geometry"].get("bouncing"))
        decision = (
            committed is None
            and not bouncing
            and batch.metadata["smb_geometry"].get("support") != "air"
        )
        if decision:
            if actions is None or stage.env._require_bridge_before_goal:
                intended, selected_index, selected_valid = coach_choice(
                    model, stage.env, takeoff_distance=takeoff_distance, variant=variant
                )
            else:
                intended = int(actions[frame])
                count = 1
                while frame + count < len(actions) and actions[frame + count] == intended:
                    count += 1
                # A rounded-up wait can skip the only safe departure window.
                selected_index = max(i for i, n in enumerate(NES_JUMP_FRAMES) if n <= count)
                selected_valid = [selected_index]
                if intended in (2, 4, 5):
                    selected_valid = safe_jump_indices(model, stage.env, intended)
                    if not selected_valid:
                        return [], dict(
                            success=False,
                            frames=len(rows),
                            executor_verified=True,
                            coaching=COACHING_CONTRACT,
                            reason="no_safe_takeoff",
                        )
                    selected_index = min(
                        selected_valid, key=lambda i: abs(NES_JUMP_FRAMES[i] - count)
                    )
                elif intended in (1, 3):
                    selected_valid = list(range(16))  # Walk durations are not trained.
            decisions += 1
            safe_sets += intended not in (1, 3) and len(selected_valid) > 1
        elif committed is not None:
            intended = int(committed)
        else:
            # The physical bounce owns this continuation. A supplied script
            # must not teach another jump while the executor releases A.
            from retroagi.core.actions import smb_jump_release_action

            intended = int(smb_jump_release_action(actions[frame] if actions is not None else 1))
            recoveries += int(bouncing)
        execution = executor.execute(
            intended, batch=batch, motor_primitives=primitive(selected_index)
        )
        obs, _, done, truncated, info = stage.step(execution.action)
        following = stage.encode_observation(obs)
        rows.append(
            (
                batch.src_a.cpu(),
                batch.src_b.cpu(),
                batch.src_c.cpu(),
                batch.metadata["smb_geometry"]["skill_goal"].cpu(),
                intended,
                int(execution.action) if bouncing else intended,
                selected_index,
                decision,
                following.src_c.cpu(),
                family,
                [i in selected_valid for i in range(16)],
            )
        )
        if done or truncated:
            break
    success = bool(stage.env._goal_credited)
    return rows if success else [], dict(
        success=success,
        frames=len(rows),
        executor_verified=True,
        coaching=COACHING_CONTRACT,
        decisions=decisions,
        safe_duration_sets=safe_sets,
        bounce_continuations=recoveries,
        teacher=(
            "reactive_collision"
            if actions is None or stage.env._require_bridge_before_goal
            else "verified_route"
        ),
    )


@torch.no_grad()
def collect_case(model, stage, actions, *, family=0, seed=0):
    """Retain complete executable routes with collision-coached decision labels."""
    return _collect_coached(model, stage, actions, family=family, seed=seed, max_steps=320)


@torch.no_grad()
def playback(model, stage, *, max_steps=320, seed=0, start=None, target=None):
    obs = stage.reset(seed=seed) if start is None else stage.load_emulator_state(start)
    if start is not None:
        stage._scene_start_frame = stage._geometry_frame
    executor = make_smb_executor(model)
    model.eval()
    memory = None
    actions = []
    reached = False
    death = False
    max_x = -float("inf")
    stable_support = 0
    for frame in range(max_steps):
        batch = stage.encode_observation(obs)
        forward = _policy_action_logits_and_state(
            model, batch, device=next(model.parameters()).device, world_model_state=memory
        )
        memory = (
            forward.next_world_model_state if model.smb_runtime_contract.recurrent_state else None
        )
        action = executor.execute(
            int(forward.logits.argmax(-1)), batch=batch, motor_primitives=forward.motor_primitives
        ).action
        obs, _, done, truncated, info = stage.step(action)
        actions.append(int(action))
        if hasattr(stage.env, "_goal_credited"):
            reached = bool(stage.env._goal_credited)
            death = bool(info.get("death"))
            max_x = max(max_x, stage.env.mario["x"])
        else:
            signals = info["full_smb_signals"]
            death = bool(signals["death"])
            # Instrumentation is an evaluator, never an undeclared policy input.
            from scripts.full_smb_adapt_geometry import geometry

            actual = geometry(stage)
            max_x = max(max_x, actual["world_x"])
            safe = bool(
                target and actual["scene"].mario["on_ground"] and actual["world_x"] >= target
            )
            stable_support = stable_support + 1 if safe else 0
            reached = bool(signals["completion"]) or stable_support >= 3
        if done or truncated or reached or death:
            break
    return dict(
        success=reached and not death, death=death, frames=frame + 1, max_x=max_x, actions=actions
    )


def save_dataset(data, path, *, episodes, provider):
    torch.save(
        dict(
            kind="canonical_demonstrations_v3",
            coaching="canonical_collision_coaching_v2",
            data=asdict(data),
            episodes=episodes,
            runtime=runtime(provider).manifest(),
        ),
        path,
    )


def adapt_world_model(model, data, *, steps=100, learning_rate=3e-4, seed=0):
    """Train the whole dynamics component, verifying actor/critic identity.

    The baseline contract resets history each physical frame. Ordered data is
    retained for future sequence qualification, but memory is not enabled by
    this adaptation. Greedy decisions therefore need not improve.
    """
    frozen = trainable_components(model, ["world_model"])
    model.eval()
    model.world_model.train()
    optimizer = torch.optim.AdamW(model.world_model.parameters(), lr=learning_rate)
    rng = torch.Generator().manual_seed(seed)
    device = next(model.parameters()).device
    losses = []
    for _ in range(steps):
        ids = torch.randint(len(data.action), (64,), generator=rng)
        a, b, c, g = (getattr(data, k)[ids].to(device) for k in ("a", "b", "c", "goal"))
        out = model(
            a,
            b,
            c,
            skill_goal=g,
            forced_action=data.motor_action[ids].to(device),
            critic_feedback_enabled=False,
            world_model_state=None,
        )
        loss = F.mse_loss(out[1], data.next_c[ids].to(device)) + 0.1 * physical_outcome_loss(
            model, c, data.next_c[ids].to(device)
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.world_model.parameters(), 1.0)
        optimizer.step()
        losses.append(float(loss.detach()))
    verify_frozen_components(model, frozen)
    return dict(
        loss=sum(losses) / len(losses), frozen_hashes=frozen, decision_improvement_claimed=False
    )


@torch.no_grad()
def collect_reactive_case(
    model, stage, *, family=0, seed=0, max_steps=320, takeoff_distance=50, variant=0
):
    return _collect_coached(
        model,
        stage,
        family=family,
        seed=seed,
        max_steps=max_steps,
        takeoff_distance=takeoff_distance,
        variant=variant,
    )


def fit_ordered_sequences(
    model, data, episodes, *, rounds=1, world_model_only=False, learning_rate=3e-4
):
    """One-frame truncated BPTT, carried history, and explicit episode resets.

    Update cadence matches playback: one transition per physical frame. Memory
    is detached between updates; it is never shared between demonstration clips.
    Actor training is allowed here only on Block replay. Full adaptation selects
    the dynamics component and verifies every frozen parameter digest.
    """
    if not model.smb_runtime_contract.recurrent_state:
        raise ValueError("Sequence training requires the same recurrent playback contract")
    frozen = trainable_components(
        model,
        ["world_model"] if world_model_only else ["actor", "world_model", "critic", "auxiliary"],
    )
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=learning_rate
    )
    model.eval()
    model.world_model.train()
    device = next(model.parameters()).device
    losses = []
    for _ in range(rounds):
        for episode in episodes:
            memory = None
            for j in range(episode["length"]):
                i = episode["start"] + j
                a, b, c, g = (
                    getattr(data, k)[i : i + 1].to(device) for k in ("a", "b", "c", "goal")
                )
                output = model(
                    a,
                    b,
                    c,
                    skill_goal=g,
                    forced_action=data.motor_action[i : i + 1].to(device),
                    critic_feedback_enabled=False,
                    world_model_state=memory,
                    episode_mask=torch.tensor([int(j > 0)], device=device),
                    return_world_model_state=True,
                )
                memory = output[-1].detach()
                loss = F.mse_loss(
                    output[1], data.next_c[i : i + 1].to(device)
                ) + 0.1 * physical_outcome_loss(model, c, data.next_c[i : i + 1].to(device))
                if not world_model_only and bool(data.actor_mask[i]):
                    loss = loss + F.cross_entropy(
                        output[4][:, -1, :6], data.action[i : i + 1].to(device)
                    )
                    allowed = data.valid_durations[i : i + 1].to(device)
                    duration = model.last_motor_primitives.hold_duration_logits[:, -1, :]
                    loss = (
                        loss
                        - torch.logsumexp(
                            F.log_softmax(duration, -1).masked_fill(~allowed, -1e9), -1
                        ).mean()
                    )
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                losses.append(float(loss.detach()))
    verify_frozen_components(model, frozen)
    return dict(
        mean_loss=sum(losses) / max(1, len(losses)),
        updates=len(losses),
        tbptt_frames=1,
        frozen_hashes=frozen,
        history_reset="each_episode",
    )


@torch.no_grad()
def recurrent_causal_probe(model, data):
    """Compare actor logits using memory produced by the current vs changed LSTM."""
    model.eval()
    device = next(model.parameters()).device
    other = make_model(hidden_dim=model.agent.d_model, device=device)
    other.load_state_dict(model.state_dict())
    other.eval()
    other.world_model.lstm.weight_ih_l0.add_(0.1)
    device = next(model.parameters()).device
    memories = [None, None]
    differences = []
    for i in range(min(32, len(data.action))):
        a, b, c, g = (getattr(data, k)[i : i + 1].to(device) for k in ("a", "b", "c", "goal"))
        logits = []
        for index, network in enumerate((model, other)):
            result = network(
                a,
                b,
                c,
                skill_goal=g,
                forced_action=data.motor_action[i : i + 1].to(device),
                critic_feedback_enabled=False,
                world_model_state=memories[index],
                return_world_model_state=True,
            )
            memories[index] = result[-1]
            logits.append(result[4][:, -1, :6])
        differences.append(float((logits[0] - logits[1]).abs().max()))
    return dict(
        max_actor_logit_change=max(differences, default=0.0),
        causal_path_present=max(differences, default=0.0) > 1e-7,
        behavior_improvement_proven=False,
    )


def physical_outcome_loss(model, current, following):
    """Supervise observable one-frame outcomes; do not invent cancel labels."""
    outcome = model.world_model.last_primitive_outcome
    progress = following[:, 12] - current[:, 12]
    loss = F.mse_loss(outcome.progress_delta, progress)
    support_loss = ((current[:, 16] > 0.5) & (following[:, 16] < 0.5)).float()
    death = following[:, 36].clamp(0, 1)
    terminal = following[:, 37:39].amax(-1).clamp(0, 1)
    return (
        loss
        + F.binary_cross_entropy_with_logits(outcome.support_loss_logit, support_loss)
        + F.binary_cross_entropy_with_logits(outcome.collision_death_logit, death)
        + F.binary_cross_entropy_with_logits(outcome.terminal_logit, terminal)
    )


def collect_stomp_recovery_case(model, stage, *, family, seed=0):
    """Teach the observable turn-back state after overshooting a required stomp.

    This is a training-only reset variation; accept it only after the shared
    pixel executor actually stomps the enemy and completes the task.
    """
    import copy

    original = stage.scenario
    stage.reset(seed=seed)
    enemy = next((e for e in stage.env.enemies if not e["dead"]), None)
    if enemy is None:
        return [], dict(success=False, reason="no_recovery_target")
    try:
        stage.scenario = copy.deepcopy(original)
        stage.scenario["mario"] = [enemy["x"] + enemy["w"] + 36, stage.env.mario["y"]]
        stage.scenario["mario_velocity"] = [0.0, 0.0]
        return collect_reactive_case(model, stage, family=family, seed=seed)
    finally:
        stage.scenario = original
