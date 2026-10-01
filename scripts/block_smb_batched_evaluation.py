"""Batched autonomous evaluation with the normal primitive executor.

Every physics frame still runs; batching removes repeated GPU launch overhead
across independent levels. With recurrent state each level carries its own
world-model state (LSTM memory and stance history) from frame to frame.
"""

from collections import Counter
from types import SimpleNamespace

import numpy as np
import torch

from retroagi.core.actions import (
    SMBPrimitiveExecution,
    smb_jump_release_action,
)
from retroagi.core.models import WorldModelState, skill_goal_objective
from retroagi.core.skills import SKILL_GOAL_ENCODING_DIM, skill_goal_encoding
from retroagi.core.smb_scene import canonical_vision
from retroagi.stages.block_smb.adapter import BlockSMBStage
from retroagi.stages.block_smb.bridge_traversal import bridge_phase, bridge_safe_wait_frames
from retroagi.stages.block_smb.env import MarioScenarioEnv
from retroagi.stages.block_smb.hierarchy import bridge_training_active
from retroagi.stages.block_smb.local_traversal import LOCAL_TRAVERSAL_FAMILIES, local_objective
from retroagi.stages.block_smb.pipe_traversal import TallPipeTraversal
from retroagi.stages.block_smb.primitive_execution import BlockSMBPrimitiveExecutor
from retroagi.stages.block_smb.skills import requested_block_smb_skill_goal
from retroagi.stages.block_smb.tasks import scenario_family
from retroagi.stages.block_smb.train import (
    block_smb_policy_scenario,
    block_smb_single_jump_scenario,
)
from retroagi.stages.block_smb.vision import BlockVisionTransformer


def _stack_states(model, memories, device):
    """One batched world-model state; levels without one start empty."""
    history = model.strategy_network.history
    stances = model.strategy_network.input_projection.in_features
    states = []
    for memory in memories:
        if memory is None:
            empty = model.initial_world_model_state(1, device)
            memory = WorldModelState(
                empty.hidden, empty.cell, torch.zeros((1, history, stances), device=device)
            )
        states.append(memory)
    return WorldModelState(
        torch.cat([m.hidden for m in states], dim=1),
        torch.cat([m.cell for m in states], dim=1),
        torch.cat([m.stance for m in states], dim=0),
    )


def _record_memory_and_strategy(model, active, goal):
    prediction = getattr(model, "last_memory_prediction", None)
    logits = getattr(model, "last_objective_logits", None)
    objectives = skill_goal_objective(goal).cpu()
    for i, s in enumerate(active):
        if prediction is not None:
            target = torch.as_tensor(s.stage._hazard_memory, dtype=torch.float32)
            s.memory_error[0] += 1
            s.memory_error[1] += float((prediction[i].cpu() - target).abs().mean())
        if logits is not None:
            s.objectives[0] += 1
            s.objectives[1] += int(int(logits[i].argmax()) == int(objectives[i]))


def evaluate_batched(model, cases, config, vision_factory, *, return_actions=False):
    if (
        not config.engine_support_override
        or not config.skill_goal_conditioning
        or config.ranked_candidate_search
    ):
        raise ValueError(
            "Batched qualification requires autonomous control with engine support and local goals"
        )
    recurrent = bool(config.ablation.recurrent_state_enabled)
    if any((config.ablation.vision_enabled is False, config.ablation.hierarchy_enabled is False)):
        raise ValueError("Batched qualification requires the complete observation")
    model.eval()
    vision = vision_factory()
    states = []
    device = torch.device(config.device)
    try:
        for sample in cases:
            stage = BlockSMBStage(
                env=MarioScenarioEnv(reward_config=config.reward_config),
                scenario=block_smb_policy_scenario(sample.scenario, True),
                vision=vision,
            )
            observation = stage.reset(seed=sample.sample_seed % (2**31))
            pipe = TallPipeTraversal.from_stage(stage.scenario, stage.env)
            if pipe:
                pipe.observe(stage.env)
            request = requested_block_smb_skill_goal(stage.scenario)
            if request is None:
                request = torch.zeros(1, SKILL_GOAL_ENCODING_DIM)
            states.append(
                SimpleNamespace(
                    sample=sample,
                    stage=stage,
                    observation=observation,
                    pipe=pipe,
                    goal=request,
                    local=scenario_family(stage.scenario) in LOCAL_TRAVERSAL_FAMILIES,
                    target=None,
                    enemy=stage.env._require_stomp_before_goal,
                    bridge=stage.env._require_bridge_before_goal
                    and not stage.env._bridge_jump_task,
                    bridge_jump=stage.env._bridge_jump_task,
                    phase="stomp" if stage.env._require_stomp_before_goal else None,
                    recovery=False,
                    opening=stage.env._require_bridge_before_goal,
                    exit_committed=False,
                    contact=False,
                    executor=BlockSMBPrimitiveExecutor(
                        stage.env,
                        duration_sampling=False,
                        adaptive_duration=config.adaptive_duration_control,
                        steady_primitives=config.steady_duration_primitives,
                        walk_primitives=config.walk_duration_primitives,
                    ),
                    actions=[],
                    memory=None,
                    memory_error=[0, 0.0],
                    objectives=[0, 0],
                    done=False,
                    death=False,
                    last_phase=None,
                )
            )
        with torch.no_grad():
            for _ in range(config.evaluation_max_steps):
                active = [s for s in states if not s.done]
                if not active:
                    break
                goals = []
                for s in active:
                    env = s.stage.env
                    s.bridge = bridge_training_active(env) and not s.bridge_jump
                    phase = s.pipe.phase if s.pipe else s.phase
                    target = local_objective(env) if s.local else None
                    if s.local:
                        if (
                            not env.mario["on_ground"]
                            and s.target is not None
                            and s.executor.active
                        ):
                            target = s.target
                        phase = (
                            s.pipe.phase
                            if s.pipe
                            else "bounce_recovery"
                            if s.recovery
                            else target.kind
                        )
                    safe = bridge_safe_wait_frames(env) if s.bridge else []
                    if s.bridge:
                        phase = bridge_phase(env, s.opening)
                        if s.exit_committed and not env._bridge_crossed:
                            phase = "exit"
                    if s.bridge_jump:
                        phase = "board" if s.bridge_jump == "mount" else "exit"
                    goal = s.goal.clone()
                    if phase in ("finish", "bounce_recovery") or (
                        s.bridge and phase in ("approach", "board", "exit")
                    ):
                        goal.zero_()
                    if s.local and not s.bridge and phase not in ("finish", "bounce_recovery"):
                        goal = skill_goal_encoding(
                            {
                                "gap": "clear_gap",
                                "mount": "mount_platform",
                                "enemy": "enemy_clear",
                                "retreat": "retreat_recover",
                            }[phase]
                        )
                    s.step_target = target
                    s.last_phase = phase
                    s.wait_event = 1 in safe if s.bridge else False
                    goals.append(goal)
                if isinstance(vision, BlockVisionTransformer):
                    features = canonical_vision(
                        vision.encode(np.stack([s.observation for s in active])), "block"
                    )
                    batch = active[0].stage.vision_projector.project(
                        features, np.stack([s.stage.state_features() for s in active])
                    )
                    a, b, c = (t.to(device) for t in (batch.src_a, batch.src_b, batch.src_c))
                else:
                    batches = [s.stage.encode_observation(s.observation) for s in active]
                    a, b, c = (
                        torch.cat([getattr(b, k) for b in batches]).to(device)
                        for k in ("src_a", "src_b", "src_c")
                    )
                goal = torch.cat(goals).to(device)
                carried = (
                    _stack_states(model, [s.memory for s in active], device) if recurrent else None
                )
                # First pass reads A. Second pass conditions B on each actual
                # action, including commitments already owned by the executor.
                model(
                    a,
                    b,
                    c,
                    skill_goal=goal,
                    forced_action=torch.zeros(len(active), device=device, dtype=torch.long),
                    critic_feedback_enabled=False,
                    world_model_state=carried,
                    world_model_enabled=config.ablation.world_model_enabled,
                )
                chosen = model.last_policy_logits_a[:, -1, :6].argmax(-1)
                for i, s in enumerate(active):
                    if not (
                        s.recovery
                        or s.phase == "bounce_recovery"
                        or block_smb_single_jump_scenario(s.stage.scenario)
                    ):
                        s.executor.resolve_landing(
                            support_override="ground" if s.stage.env.mario["on_ground"] else "air",
                            enemy_contact_override=s.contact,
                        )
                    commitment = s.executor.committed_action
                    if commitment is not None and not (s.recovery or s.phase == "bounce_recovery"):
                        chosen[i] = commitment
                outputs = model(
                    a,
                    b,
                    c,
                    skill_goal=goal,
                    forced_action=chosen,
                    critic_feedback_enabled=False,
                    world_model_state=carried,
                    world_model_enabled=config.ablation.world_model_enabled,
                    return_world_model_state=recurrent,
                )
                motor = model.last_motor_primitives
                _record_memory_and_strategy(model, active, goal)
                if recurrent and outputs[-1] is not None:
                    for i, s in enumerate(active):
                        s.memory = WorldModelState(
                            outputs[-1].hidden[:, i : i + 1],
                            outputs[-1].cell[:, i : i + 1],
                            outputs[-1].stance[i : i + 1],
                        )
                for i, s in enumerate(active):
                    env = s.stage.env
                    if s.recovery or s.phase == "bounce_recovery":
                        execution = SMBPrimitiveExecution(
                            action=int(smb_jump_release_action(int(chosen[i])))
                        )
                    else:
                        primitive = SimpleNamespace(
                            hold_duration_logits=motor.hold_duration_logits[i : i + 1],
                            duration_bin_values=motor.duration_bin_values,
                            hold_duration=motor.hold_duration[i : i + 1],
                        )
                        primitive = s.executor.motor_parameters(int(chosen[i]), primitive)
                        execution = s.executor.execute(
                            int(chosen[i]),
                            motor_primitives=primitive,
                            wait_event=s.wait_event,
                            support_override="ground" if env.mario["on_ground"] else "air",
                            enemy_contact_override=s.contact,
                        )
                    action = execution.action
                    if s.local and execution.started:
                        s.target = s.step_target
                    s.observation, _, done, truncated, info = s.stage.step(action)
                    s.actions.append(action)
                    stomp = (
                        bool(info.get("stomp_geometry", {}).get("stomp"))
                        or info["reward_terms"]["enemy_stomp"] > 0
                    )
                    s.contact = bool(info.get("death")) or stomp
                    s.death = bool(info.get("death"))
                    if s.bridge:
                        if s.last_phase == "exit" and (
                            (execution.released and s.wait_event) or action in (1, 2)
                        ):
                            s.exit_committed = True
                        if action in (3, 4) or (action == 0 and execution.started):
                            s.exit_committed = False
                        if action != 0 or (execution.released and s.wait_event):
                            s.opening = False
                    if s.local or s.enemy:
                        if stomp:
                            s.executor.reset()
                            s.contact = False
                            if s.local:
                                s.recovery = True
                            if s.enemy:
                                s.phase = "bounce_recovery"
                        elif env.mario["on_ground"]:
                            s.recovery = False
                            if s.phase == "bounce_recovery":
                                s.phase = "finish"
                    if s.pipe:
                        s.pipe.observe(env)
                    s.done = bool(
                        done
                        or truncated
                        or (
                            block_smb_single_jump_scenario(s.stage.scenario)
                            and (execution.landed or execution.cancelled)
                        )
                    )
        counts = {d: [0, 0] for d in ("easy", "medium", "hard")}
        failures = []
        for s in states:
            success = bool(s.stage.env._goal_credited)
            counts[s.sample.difficulty_bin][0] += int(success)
            counts[s.sample.difficulty_bin][1] += 1
            if not success:
                failures.append(
                    dict(
                        scenario=s.sample.scenario_id,
                        difficulty=s.sample.difficulty_bin,
                        actions=dict(Counter(s.actions)),
                        x=s.stage.env.mario["x"],
                        y=s.stage.env.mario["y"],
                        phase=s.last_phase,
                        death=s.death,
                        mounted=s.pipe.mounted if s.pipe else None,
                        frames=len(s.actions),
                    )
                )
        rates = {d: yes / total for d, (yes, total) in counts.items() if total}
        frames = sum(s.memory_error[0] for s in states)
        decisions = sum(s.objectives[0] for s in states)
        result = dict(
            rates=rates,
            counts=counts,
            passed=all(r >= 0.9 for r in rates.values()),
            failures=failures,
            memory_mean_absolute_error=(
                sum(s.memory_error[1] for s in states) / frames if frames else None
            ),
            strategy_accuracy=(
                sum(s.objectives[1] for s in states) / decisions if decisions else None
            ),
        )
        if return_actions:
            result["actions"] = [s.actions for s in states]
        return result
    finally:
        for s in states:
            s.stage.env.close()
