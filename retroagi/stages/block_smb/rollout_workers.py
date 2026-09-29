"""Training rollouts on CPU workers, with the policy terms recomputed by the learner.

Policy updates backpropagate through each step's policy call, which cannot
leave the worker that ran it. Workers therefore play an update batch's
episodes without gradients and record every step's policy inputs and
decisions; the learner re-runs all of the batch's policy calls as one batched
forward pass and rebuilds the graph-attached terms the losses consume. Carried
world-model states are detached in rollouts, so each step's call depends only
on its recorded inputs. Dropout and Gumbel noise are drawn afresh in the
batched pass; everything else matches the in-process rollout.
"""

import copy
from dataclasses import fields, is_dataclass, replace

import torch

from retroagi.core.interfaces import StageBatch
from retroagi.core.models import TACTIC_STANCES, WorldModelState


class _Array:
    """A tensor sent between processes as a copied array.

    Tensors travel as shared file descriptors, and a batch of pending
    episodes exceeds a systemd service's descriptor limit.
    """

    __slots__ = ("value",)

    def __init__(self, value):
        self.value = value


def _pack(value):
    if torch.is_tensor(value):
        return _Array(value.detach().cpu().numpy())
    if isinstance(value, dict):
        return {key: _pack(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_pack(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_pack(item) for item in value)
    return value


def _unpack(value):
    if isinstance(value, _Array):
        return torch.from_numpy(value.value)
    if isinstance(value, dict):
        return {key: _unpack(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_unpack(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_unpack(item) for item in value)
    return value


def _slim_batch(batch):
    """The parts of an observation the losses and the policy call read."""
    metadata = batch.metadata if isinstance(batch.metadata, dict) else {}
    episode = metadata.get("episode", {})
    return {
        "src_a": batch.src_a,
        "src_b": batch.src_b,
        "src_c": batch.src_c,
        "vision_fusion": metadata.get("vision_fusion", {}),
        "mask": episode.get("mask") if isinstance(episode, dict) else None,
    }


def _stage_batch(packed):
    return StageBatch(
        src_a=packed["src_a"],
        target_a=None,
        src_b=packed["src_b"],
        target_b=None,
        src_c=packed["src_c"],
        target_c=None,
        metadata={"vision_fusion": packed["vision_fusion"], "episode": {"mask": packed["mask"]}},
    )


def pack_trajectory(trajectory, policy_scenario):
    """A worker trajectory without graph tensors, ready to send to the learner."""
    next_batch_index = {
        id(step.next_batch): index for index, step in enumerate(trajectory.transitions)
    }
    transitions = []
    for step in trajectory.transitions:
        info = dict(step.info)
        outcome = info.get("primitive_outcome_batch")
        if outcome is not None:
            info["primitive_outcome_batch"] = ("next_batch", next_batch_index[id(outcome)])
        record = dict(step.policy_record)
        state = record.get("world_model_state")
        if state is not None:
            record["world_model_state"] = (state.hidden, state.cell, state.stance)
        transitions.append(
            {
                "batch": _slim_batch(step.batch),
                "next_batch": _slim_batch(step.next_batch),
                "action": step.action,
                "reward": step.reward,
                "done": step.done,
                "episode_mask": step.episode_mask,
                "info": info,
                "oracle_action": step.oracle_action,
                "step_index": step.step_index,
                "noop_allowed": step.noop_allowed,
                "tactic_target": step.tactic_target,
                "tactic_actions": step.tactic_actions,
                "memory_target": step.memory_target,
                "objective_target": step.objective_target,
                "record": record,
            }
        )
    return _pack(
        {
            "scenario_name": trajectory.scenario_name,
            "spans": trajectory.spans,
            "policy_scenario": policy_scenario,
            "transitions": transitions,
        }
    )


def unpack_trajectory(payload):
    """(trajectory, policy scenario); graph fields stay None until recomputed."""
    from .train import BlockSMBTrajectory, BlockSMBTransition

    payload = _unpack(payload)
    next_batches = [_stage_batch(step["next_batch"]) for step in payload["transitions"]]
    trajectory = BlockSMBTrajectory(scenario_name=payload["scenario_name"], spans=payload["spans"])
    for step, next_batch in zip(payload["transitions"], next_batches):
        info = step["info"]
        outcome = info.get("primitive_outcome_batch")
        if outcome is not None:
            info["primitive_outcome_batch"] = next_batches[outcome[1]]
        record = step["record"]
        state = record.get("world_model_state")
        if state is not None:
            record["world_model_state"] = WorldModelState(*state)
        trajectory.transitions.append(
            BlockSMBTransition(
                batch=_stage_batch(step["batch"]),
                next_batch=next_batch,
                action=step["action"],
                reward=step["reward"],
                done=step["done"],
                episode_mask=step["episode_mask"],
                scenario_name=payload["scenario_name"],
                info=info,
                log_prob=None,
                entropy=None,
                actions1=None,
                actions2=None,
                next_state_pred=None,
                criticism=None,
                logits_a=None,
                oracle_action=step["oracle_action"],
                step_index=step["step_index"],
                noop_allowed=step["noop_allowed"],
                tactic_target=step["tactic_target"],
                tactic_actions=step["tactic_actions"],
                memory_target=step["memory_target"],
                objective_target=step["objective_target"],
                policy_record=record,
            )
        )
    return trajectory, payload["policy_scenario"]


def rollout_task(task):
    """Play one training episode on a worker, recording its policy calls."""
    from .adapter import BlockSMBObservationConfig, BlockSMBStage
    from .env import MarioScenarioEnv
    from .parallel import worker_policy, worker_vision
    from .train import block_smb_policy_scenario, collect_trajectory

    config, policy_path, version, job = task
    model = worker_policy(config, policy_path, version, training=True)
    torch.manual_seed(job["seed"])
    stage = BlockSMBStage(
        env=MarioScenarioEnv(reward_config=config.reward_config),
        scenario=block_smb_policy_scenario(
            copy.deepcopy(job["scenario"]), config.autonomous_policy
        ),
        vision=worker_vision(),
        observation_config=BlockSMBObservationConfig(
            motion_observations=config.motion_observations,
            hazard_observations=config.hazard_observations,
            hazard_memory_observations=config.hazard_memory_observations,
        ),
    )
    try:
        with torch.no_grad():
            trajectory = collect_trajectory(
                model,
                stage,
                job["scenario_name"],
                rollout_steps=job["rollout_steps"],
                seed=job["seed"],
                deterministic=job["deterministic"],
                device=torch.device("cpu"),
                ablation=config.ablation,
                use_oracle_actions=job["use_oracle_actions"],
                adaptive_duration_control=config.adaptive_duration_control,
                skill_goal_conditioning=config.skill_goal_conditioning,
                steady_duration_primitives=config.steady_duration_primitives,
                walk_duration_primitives=config.walk_duration_primitives,
                engine_support=config.engine_support_override,
                demonstration_actions=job["demonstration_actions"],
                record_policy_inputs=True,
            )
    finally:
        stage.env.close()
    return pack_trajectory(trajectory, stage.scenario)


def _row(value, index, batch_size):
    if torch.is_tensor(value) and value.ndim > 0 and value.size(0) == batch_size:
        return value[index : index + 1]
    return value


def _motor_row(motor, index, batch_size, duration_bin_values, device):
    if motor is None:
        return None
    rows = {
        field.name: _row(getattr(motor, field.name), index, batch_size)
        for field in fields(motor)
        if field.name != "duration_bin_values"
    }
    if duration_bin_values is not None:
        rows["duration_bin_values"] = duration_bin_values.to(device)
    return replace(motor, **rows) if is_dataclass(motor) else motor


def recompute_policy_terms(model, trajectories, config, device):
    """Rebuild the graph-attached policy terms of recorded steps in one forward pass."""
    from .train import (
        BLOCK_SMB_ACTION_COUNT,
        _action_step_terms,
        _smb_primitive_auxiliary_loss,
        finite_or_raise,
    )

    steps = [step for trajectory in trajectories for step in trajectory.transitions]
    if not steps:
        return
    records = [step.policy_record for step in steps]
    if any(record is None or record.get("forced_action") is None for record in records):
        raise ValueError("Recomputed steps need a recorded policy call with its chosen action")
    count = len(steps)
    # The losses read observations directly, as in-process rollouts leave them on `device`.
    for step in steps:
        for batch in (step.batch, step.next_batch):
            batch.src_a, batch.src_b, batch.src_c = (
                batch.src_a.to(device),
                batch.src_b.to(device),
                batch.src_c.to(device),
            )
    src_a, src_b, src_c = (
        torch.cat([getattr(step.batch, name) for step in steps])
        for name in ("src_a", "src_b", "src_c")
    )
    world_model_state = None
    if records[0]["world_model_state"] is not None:
        states = [record["world_model_state"] for record in records]
        empty_stance = torch.zeros(1, model.strategy_network.history, len(TACTIC_STANCES))
        world_model_state = WorldModelState(
            torch.cat([state.hidden for state in states], dim=1).to(device),
            torch.cat([state.cell for state in states], dim=1).to(device),
            torch.cat(
                [empty_stance if state.stance is None else state.stance for state in states]
            ).to(device),
        )
    episode_mask = None
    if records[0]["episode_mask"] is not None:
        episode_mask = torch.cat([record["episode_mask"].reshape(-1) for record in records]).to(
            device
        )
    skill_goal = None
    if records[0]["skill_goal"] is not None:
        skill_goal = torch.cat([record["skill_goal"] for record in records]).to(device)
    ablation = config.ablation
    actions1, next_state_pred, criticism, actions2, logits_a, _w, _b, _next_state = model(
        src_a,
        src_b,
        src_c,
        tau=1.0,
        world_model_state=world_model_state,
        episode_mask=episode_mask,
        return_world_model_state=True,
        critic_feedback_enabled=ablation.critic_feedback_enabled,
        world_model_enabled=ablation.world_model_enabled,
        skill_goal=skill_goal,
        forced_action=torch.tensor([record["forced_action"] for record in records], device=device),
    )
    policy_logits = getattr(model, "last_policy_logits_a", None)
    finite_or_raise("action_logits", logits_a[:, -1, :BLOCK_SMB_ACTION_COUNT])
    motor = getattr(model, "last_motor_primitives", None)
    tactic = getattr(model, "last_tactic_logits", None)
    memory = getattr(model, "last_memory_prediction", None)
    objective = getattr(model, "last_objective_logits", None)
    strategy = getattr(model, "last_strategy_logits", None)
    for index, (step, record) in enumerate(zip(steps, records)):
        step_logits = (
            _row(policy_logits, index, count)
            if record["supplied_action"] is not None and policy_logits is not None
            else logits_a[index : index + 1]
        )
        motor_row = _motor_row(motor, index, count, record["duration_bin_values"], device)
        log_prob, entropy, aux_loss, expected_hold, release_logit = _action_step_terms(
            step_logits[:, -1, :BLOCK_SMB_ACTION_COUNT],
            motor_row,
            intent=record["intent"],
            action=record["action"],
            execution=record["execution"],
            supplied_action=record["supplied_action"],
            committed_action=record["committed_action"],
            deterministic=record["deterministic"],
            oracle_action=record["oracle_action"],
            recovering_from_stomp=record["recovering_from_stomp"],
        )
        if record["aux_override"]:
            aux_loss = _smb_primitive_auxiliary_loss(
                motor_row,
                torch.tensor([record["oracle_action"]], device=device),
                replace(record["execution"], duration_bin_index=None),
                action_count=BLOCK_SMB_ACTION_COUNT,
                device=device,
                dtype=log_prob.dtype,
            )
        step.log_prob = log_prob
        step.entropy = entropy
        step.primitive_aux_loss = aux_loss
        step.expected_hold = expected_hold
        step.release_logit = release_logit
        step.actions1 = actions1[index : index + 1]
        step.actions2 = actions2[index : index + 1]
        step.next_state_pred = next_state_pred[index : index + 1]
        step.criticism = _row(criticism, index, count)
        step.logits_a = step_logits
        step.hold_duration_logits = getattr(motor_row, "hold_duration_logits", None)
        step.duration_bin_values = getattr(motor_row, "duration_bin_values", None)
        step.tactic_logits = _row(tactic, index, count)
        step.memory_prediction = _row(memory, index, count)
        step.objective_logits = _row(objective, index, count)
        step.strategy_logits = _row(strategy, index, count)
