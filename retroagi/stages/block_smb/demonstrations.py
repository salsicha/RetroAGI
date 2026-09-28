"""Batched learning from physically completed trajectories.

Frame observations come from the same frozen vision and engine encoder as
rollouts. Teachers supervise actions AND commitment lengths, including walks.
Evaluation uses only the learned policy and normal primitive controller.
"""

import copy
import time
from dataclasses import dataclass, fields, replace

import numpy as np
import torch
import torch.nn.functional as F

from retroagi.core.models import (
    WorldModelState,
    clip_policy_and_objective_gradients,
    skill_goal_objective,
)
from retroagi.core.skills import SKILL_GOAL_ENCODING_DIM, skill_goal_encoding
from retroagi.core.smb_enemy_history import HAZARD_MEMORY_NAMES, EnemyObservationHistory

from .adapter import BlockSMBObservationConfig, BlockSMBStage
from .bridge_traversal import bridge_phase, bridge_safe_wait_frames
from .env import MarioScenarioEnv
from .hierarchy import bridge_training_active
from .local_traversal import (
    LOCAL_TRAVERSAL_FAMILIES,
    local_objective,
    local_target_distance,
    plant_clearance_target,
    safe_jump_holds,
    support_edge_distance,
)
from .pipe_traversal import TallPipeTraversal
from .skills import requested_block_smb_skill_goal
from .tasks import scenario_family
from .vision import BlockVisionTransformer


@dataclass
class DemonstrationBatch:
    a: torch.Tensor
    b: torch.Tensor
    c: torch.Tensor
    goal: torch.Tensor
    action: torch.Tensor
    motor_action: torch.Tensor
    duration: torch.Tensor
    actor_mask: torch.Tensor
    next_c: torch.Tensor
    family: torch.Tensor
    valid_durations: torch.Tensor
    phase: torch.Tensor | None = None
    carry_progress: torch.Tensor | None = None
    recovery: torch.Tensor | None = None
    forced_release: torch.Tensor | None = None
    tactic: torch.Tensor | None = None
    tactic_actions: torch.Tensor | None = None
    duration_consumed: torch.Tensor | None = None
    # Episodic memory: position within each stored episode (0 starts one),
    # the observable hazard memory the world model must hold at each row, and
    # the carried world-model state entering each row (refreshed in training).
    frame_index: torch.Tensor | None = None
    memory_target: torch.Tensor | None = None
    memory_state: torch.Tensor | None = None
    # Replayed prefixes of recovery and correction routes: never supervised or
    # sampled, but they rebuild the memory a policy carried into the suffix.
    context: torch.Tensor | None = None
    # Parameter-free world-model LSTM inputs of each row (refreshed with the
    # carried states), so fitting can unroll the LSTM with gradients.
    world_model_inputs: torch.Tensor | None = None

    def __post_init__(self):
        if self.frame_index is None:
            # Legacy rows: every row starts its own episode with empty memory.
            self.frame_index = torch.zeros_like(self.family)
        if self.memory_target is None:
            self.memory_target = torch.zeros((len(self.family), len(HAZARD_MEMORY_NAMES)))
        if self.memory_state is None:
            self.memory_state = torch.zeros((len(self.family), 0))
        if self.context is None:
            self.context = torch.zeros_like(self.family, dtype=torch.bool)
        if self.world_model_inputs is None:
            self.world_model_inputs = torch.zeros((len(self.family), 0))
        if self.tactic_actions is None:
            self.tactic_actions = torch.zeros((len(self.family), 6), dtype=torch.bool)
        if self.duration_consumed is None:
            self.duration_consumed = torch.ones_like(self.family, dtype=torch.bool)
        if self.tactic is None:
            self.tactic = torch.full_like(self.family, -1)
        if self.forced_release is None:
            self.forced_release = torch.zeros_like(self.family, dtype=torch.bool)
        if self.recovery is None:
            self.recovery = torch.zeros_like(self.family, dtype=torch.bool)
        if self.carry_progress is None:
            self.carry_progress = torch.zeros_like(self.family, dtype=torch.float32)
        if self.phase is None:
            self.phase = torch.zeros_like(self.family)


def demonstration_rows(data, mask):
    """The rows of `data` selected by `mask`; whole episodes keep their clock."""
    return DemonstrationBatch(
        **{field.name: getattr(data, field.name)[mask] for field in fields(DemonstrationBatch)}
    )


def log_demonstration_progress(config, event, **payload):
    """Expose long CPU setup and fitting phases in both run logs."""
    if config.log_path is None:
        return
    from .train import _log_block_smb_event

    _log_block_smb_event(config, event, **payload)
    print(f"{event}: {payload}", flush=True)


class _CollectionProgress:
    def __init__(self, config, phase, total):
        self.config, self.phase, self.total = config, phase, total
        self.started = self.last_report = time.monotonic()
        self.update(0, force=True)

    def update(self, completed, *, force=False, **payload):
        now = time.monotonic()
        if force or completed == self.total or now - self.last_report >= 30:
            log_demonstration_progress(
                self.config,
                "demonstration_collection_progress",
                phase=self.phase,
                completed=completed,
                total=self.total,
                elapsed_seconds=round(now - self.started, 2),
                **payload,
            )
            self.last_report = now


def collect_demonstrations(cases, config, vision_factory, *, vision_batch_size=32):
    cases = list(cases)
    progress = _CollectionProgress(config, "encode_trajectories", len(cases))
    rows = []
    recovery_rows = []
    release_rows = []
    tactic_rows = []
    tactic_action_rows = []
    duration_consumed_rows = []
    memory_rows = []
    context_rows = []
    episode_starts = []
    frame_wait_episodes = set()
    for case_index, (family_index, sample) in enumerate(cases):
        stage = BlockSMBStage(
            env=MarioScenarioEnv(reward_config=config.reward_config),
            scenario=sample.scenario,
            vision=vision_factory(),
            observation_config=BlockSMBObservationConfig(
                motion_observations=config.motion_observations,
                hazard_observations=config.hazard_observations,
                hazard_memory_observations=config.hazard_memory_observations,
            ),
        )
        episode = []
        episode_release = []
        episode_tactics = []
        episode_tactic_actions = []
        episode_duration_consumed = []
        episode_memory = []
        from .primitive_execution import JumpReleaseState

        # The same observer a live policy has, from the replayed prefix onward.
        memory_history = EnemyObservationHistory()

        release = JumpReleaseState()
        try:
            observation = stage.reset(seed=sample.sample_seed % (2**31))
            actions = list(sample.oracle["actions"])
            supervision_start = int(sample.oracle.get("supervision_start_frame", 0))
            pipe = TallPipeTraversal.from_stage(stage.scenario, stage.env)
            request = requested_block_smb_skill_goal(stage.scenario)
            request = request if request is not None else torch.zeros(1, SKILL_GOAL_ENCODING_DIM)
            from retroagi.core.smb_physics import NES_JUMP_FRAMES, NES_PHYSICS_PROFILE

            # The hold-duration menu depends on the physics profile: legacy
            # bins are the values 1..16, NES bins are NES_JUMP_FRAMES (up
            # to 32 frames). Both have 16 entries; the demonstration tensors
            # store menu INDICES, so value-1 is only correct for legacy.
            duration_menu = (
                NES_JUMP_FRAMES
                if stage.env.physics_profile == NES_PHYSICS_PROFILE
                else tuple(range(1, 17))
            )

            def menu_index(frames: int) -> int:
                index = 0
                for slot, entry in enumerate(duration_menu):
                    if entry <= max(1, frames):
                        index = slot
                    else:
                        break
                return index

            jump_intent = None
            jump_hold = 1
            jump_rows = []
            bridge_jump = stage.env._bridge_jump_task
            from .piranha import has_plants

            # Must match BlockSMBPrimitiveExecutor: these layouts re-decide
            # waits every frame, so no wait duration is ever consumed.
            frame_waits = bool(bridge_jump) or has_plants(stage.env)
            bridge = stage.env._require_bridge_before_goal and bridge_jump is None
            enemy = stage.env._require_stomp_before_goal
            opening = True
            bridge_exit_committed = False
            recovering_stomp = False
            intent = -1
            observations = [observation]
            states = [stage.state_features(stage.last_info)]
            for frame, action in enumerate(actions):
                env = stage.env
                bridge = bridge_training_active(env) and bridge_jump is None
                memory_history.observe(env, env.steps)
                episode_memory.append(memory_history.memory_features())
                if jump_intent is not None and env.mario["on_ground"]:
                    jump_intent = None
                    jump_rows = []
                if recovering_stomp and env.mario["on_ground"]:
                    recovering_stomp = False
                episode_release.append(bool(release.remaining))
                actor_mask = jump_intent is None and not recovering_stomp and not release.remaining
                from .tactics import compatible_actions, hierarchy_intent

                free_decision = actor_mask
                episode_duration_consumed.append(not (action == 0 and frame_waits))
                end = frame + 1
                while end < len(actions) and actions[end] == action:
                    end += 1
                hold = min(duration_menu[-1], end - frame)
                if jump_intent is None and action in (2, 4, 5):
                    jump_intent = action
                    jump_hold = hold
                    from retroagi.core.smb_coaching import training_target

                    objective = training_target(env)
                    jump_valid = (
                        safe_jump_holds(
                            env,
                            objective,
                            1 if action == 2 else -1,
                            plant_history=stage._hazard_features,
                        )
                        if action in (2, 4) and env.mario["on_ground"]
                        else []
                    )
                motor_action = jump_intent if jump_intent is not None else action
                duration = jump_hold if jump_intent is not None else hold
                if motor_action == 0:
                    duration = 1 if frame_waits else max(1, min(16, round((end - frame) / 4)))
                    duration_index = duration - 1
                elif motor_action in (2, 4, 5):
                    duration_index = menu_index(duration)
                else:
                    duration_index = min(15, duration - 1)
                goal = request.clone()
                phase = None
                if pipe is not None:
                    pipe.observe(env)
                    phase = pipe.phase
                elif scenario_family(stage.scenario) in LOCAL_TRAVERSAL_FAMILIES and not bridge:
                    target = local_objective(env)
                    phase = target.kind
                    goal = skill_goal_encoding(
                        {
                            "gap": "clear_gap",
                            "mount": "mount_platform",
                            "enemy": "enemy_clear",
                            "retreat": "retreat_recover",
                        }.get(phase, "mount_platform")
                    )
                elif enemy and env._stomp_credited:
                    phase = "finish"
                if recovering_stomp:
                    phase = "bounce_recovery"
                if bridge:
                    phase = bridge_phase(env, opening)
                    if bridge_exit_committed and not env._bridge_crossed:
                        phase = "exit"
                if bridge_jump:
                    phase = "board" if bridge_jump == "mount" else "exit"
                if phase in ("finish", "bounce_recovery") or (
                    bridge and phase in ("approach", "board", "exit")
                ):
                    goal = torch.zeros_like(request)
                # Match the live collector: a committed arc keeps the local
                # objective from takeoff, even while no surface is underneath.
                if jump_intent is not None:
                    if not jump_rows:
                        jump_goal = goal.clone()
                    else:
                        goal = jump_goal
                if bridge:
                    next_action = actions[frame + 1] if frame + 1 < len(actions) else None
                    if phase == "exit" and (action in (1, 2) or (action == 0 and next_action == 1)):
                        bridge_exit_committed = True
                    if action in (3, 4) or (
                        action == 0 and (frame == 0 or actions[frame - 1] != 0)
                    ):
                        bridge_exit_committed = False
                opening_ready = (
                    bridge and opening and action == 0 and 1 in bridge_safe_wait_frames(env)
                )
                intent = hierarchy_intent(
                    env,
                    stage._hazard_features,
                    action,
                    family=scenario_family(stage.scenario),
                    phase=phase,
                    decision=free_decision,
                    previous=intent,
                )
                tactic = intent
                episode_tactics.append(tactic)
                allowed_tactic_actions = compatible_actions(env, tactic if free_decision else -1)
                if tactic >= 0 and free_decision:
                    allowed_tactic_actions[action] = True
                episode_tactic_actions.append(allowed_tactic_actions)
                # Advance adapter-owned observation history just as live rollouts do.
                # Direct env.step leaves all temporal features frozen at reset.
                observation, reward, done, truncated, info = stage.step(action)
                release.observe(env, action, info)
                observations.append(observation)
                states.append(stage.state_features(info))
                valid = [False] * 16
                if jump_intent is not None and jump_valid:
                    # safe_jump_holds returns exact menu values; record their
                    # menu positions (NES values exceed 16 by design).
                    for value in jump_valid:
                        valid[duration_menu.index(value)] = True
                else:
                    valid[duration_index] = True
                episode.append(
                    [
                        frame,
                        frame,
                        frame,
                        goal.cpu(),
                        action,
                        motor_action,
                        duration_index,
                        actor_mask,
                        frame + 1,
                        family_index,
                        valid,
                        {
                            "enemy": 1,
                            "stomp": 1,
                            "gap": 2,
                            "wait": 2,
                            "mount": 3,
                            "board": 3,
                            "ride": 4,
                            "exit": 5,
                        }.get(phase, 0),
                    ]
                )
                if jump_intent is not None:
                    jump_rows.append(len(episode) - 1)
                    if env.mario["on_ground"] or info["reward_terms"]["enemy_stomp"] > 0 or done:
                        for i in jump_rows:
                            episode[i][8] = frame + 1
                        jump_intent = None
                        jump_rows = []
                if info["reward_terms"]["enemy_stomp"] > 0:
                    recovering_stomp = True
                if bridge and (action != 0 or opening_ready):
                    opening = False
                if done or truncated:
                    break
            if not stage.env._goal_credited:
                raise ValueError(f"Failed demonstration: {sample.scenario_id}")
            # The frozen encoder is feedforward. Batch the exact rendered
            # frames through the normal projector, avoiding duplicate vision
            # calls for each next/current observation pair.
            streams = [[], [], []]
            size = vision_batch_size if isinstance(stage.vision, BlockVisionTransformer) else 1
            with torch.no_grad():
                for start in range(0, len(observations), size):
                    images = np.stack(observations[start : start + size])
                    vision = stage.vision.encode(images if size > 1 else images[0])
                    batch = stage.vision_projector.project(
                        vision,
                        state=torch.as_tensor(
                            np.stack(states[start : start + size]), device=vision.position.device
                        ),
                    )
                    for stream, value in zip(streams, (batch.src_a, batch.src_b, batch.src_c)):
                        stream.append(value.detach().cpu())
            a, b, c = (torch.cat(values) for values in streams)
            for row in episode:
                frame = row[0]
                row[0], row[1], row[2] = (
                    a[frame : frame + 1],
                    b[frame : frame + 1],
                    c[frame : frame + 1],
                )
                row[8] = c[row[8] : row[8] + 1]
            if len(episode) <= supervision_start:
                raise ValueError("A recovery demonstration must have a supervised suffix")
            # The replayed prefix stays as unsupervised context rows.
            for row in episode[:supervision_start]:
                row[7] = False
            if frame_waits:
                frame_wait_episodes.add(len(rows))
            episode_starts.append(len(rows))
            rows.extend(episode)
            context_rows.extend(frame < supervision_start for frame in range(len(episode)))
            release_rows.extend(episode_release[: len(episode)])
            recovery_rows.extend([bool(sample.oracle.get("recovery"))] * len(episode))
            tactic_action_rows.extend(episode_tactic_actions[: len(episode)])
            duration_consumed_rows.extend(episode_duration_consumed[: len(episode)])
            tactic_rows.extend(
                -1 if frame < supervision_start else tactic
                for frame, tactic in enumerate(episode_tactics[: len(episode)])
            )
            memory_rows.extend(episode_memory[: len(episode)])
        finally:
            stage.env.close()
        progress.update(case_index + 1, family_index=family_index, frames=len(rows))
    log_demonstration_progress(config, "demonstration_dataset_assembly", frames=len(rows))
    columns = list(zip(*rows))
    data = DemonstrationBatch(
        *(torch.cat(v) if i in (0, 1, 2, 3, 8) else torch.tensor(v) for i, v in enumerate(columns))
    )

    data.recovery = torch.tensor(recovery_rows, dtype=torch.bool)
    data.forced_release = torch.tensor(release_rows, dtype=torch.bool)
    data.tactic = torch.tensor(tactic_rows, dtype=torch.long)
    data.tactic_actions = torch.tensor(tactic_action_rows, dtype=torch.bool)
    data.duration_consumed = torch.tensor(duration_consumed_rows, dtype=torch.bool)
    data.memory_target = torch.as_tensor(np.stack(memory_rows), dtype=torch.float32)
    data.context = torch.tensor(context_rows, dtype=torch.bool)
    frame_index = torch.arange(len(rows))
    for start in episode_starts:
        frame_index[start:] -= frame_index[start].clone()
    data.frame_index = frame_index
    data = align_steady_demonstrations(
        data, episode_starts, frame_wait_episodes=frame_wait_episodes
    )
    return (
        data if config.walk_duration_primitives else without_walk_commitments(data, episode_starts)
    )


def align_steady_demonstrations(data, episode_starts=None, *, frame_wait_episodes=()):
    """Migrate frame labels to the executor's actual walk/wait commitments.

    Call once per dataset. Old cached real-ViT data retain the episode clock
    at C slot 26; freshly collected data pass their exact episode boundaries.
    """
    data = replace(
        data,
        actor_mask=data.actor_mask.clone(),
        duration=data.duration.clone(),
        valid_durations=data.valid_durations.clone(),
        next_c=data.next_c.clone(),
    )
    if episode_starts is None:
        episode_starts = (data.c[:, 26] == 0).nonzero().flatten().tolist()
        if not episode_starts or episode_starts[0] != 0:
            raise ValueError("Cannot locate episode boundaries in legacy demonstration cache")
    ends = episode_starts[1:] + [len(data.action)]
    for begin, end in zip(episode_starts, ends):
        frame = begin
        while frame < end:
            action = int(data.action[frame])
            if (
                (action == 0 and begin in frame_wait_episodes)
                or action not in (0, 1, 3)
                or not bool(data.actor_mask[frame])
            ):
                frame += 1
                continue
            stop = frame + 1
            while stop < end and int(data.action[stop]) == action and bool(data.actor_mask[stop]):
                stop += 1
            maximum = 64 if action == 0 else 16
            for start in range(frame, stop, maximum):
                finish = min(stop, start + maximum)
                frames = finish - start
                label = ((frames + 3) // 4 if action == 0 else frames) - 1
                data.actor_mask[start:finish] = False
                data.actor_mask[start] = True
                data.duration[start:finish] = label
                data.valid_durations[start:finish] = False
                data.valid_durations[start:finish, label] = True
                data.next_c[start:finish] = data.next_c[finish - 1].clone()
            frame = stop
    return data


DEMONSTRATION_CONTRACT_VERSION = 17


def without_walk_commitments(data, episode_starts=None):
    """Walking reconsiders A every frame; jump and wait commitments remain."""
    data = replace(data, actor_mask=data.actor_mask.clone(), next_c=data.next_c.clone())
    walking = ((data.motor_action == 1) | (data.motor_action == 3)) & (data.c[:, 16] > 0.5)
    data.actor_mask[walking] = True
    indices = walking.nonzero().flatten()
    indices = indices[indices < len(data.action) - 1]
    indices = indices[data.c[indices + 1, 26] != 0]
    if episode_starts is not None:
        boundaries = torch.tensor(episode_starts, device=indices.device)
        indices = indices[~torch.isin(indices + 1, boundaries)]
    data.next_c[indices] = data.c[indices + 1]
    # Only jumps still held at landing own release frames. Motor intent stays
    # a jump throughout flight, so its transition to walking cannot tell us
    # whether the button was already released. Use the physical replay mask.
    forced_release = getattr(data, "forced_release", None)
    if forced_release is not None:
        data.actor_mask[forced_release] = False
    context = getattr(data, "context", None)
    if context is not None:
        data.actor_mask[context] = False
    return data


def demonstration_groups(data):
    # 0=routine, 1=approach, 2=wait, 3=board, 4=ride, 5=exit.
    return (data.family.long() * 6 + data.phase.long()) * 7 + torch.where(
        data.actor_mask, data.action, 6
    )


def demonstration_sample_weights(data, family_weights=None):
    weights = torch.zeros(len(data.action))
    recovery = getattr(data, "recovery", None)
    if recovery is None:
        recovery = torch.zeros_like(data.family, dtype=torch.bool)
    context = getattr(data, "context", None)
    if context is None:
        context = torch.zeros_like(data.family, dtype=torch.bool)
    for family in data.family.unique():
        family_mask = data.family == family
        for phase in data.phase[family_mask].unique():
            phase_mask = family_mask & (data.phase == phase)
            for action in data.action[phase_mask & data.actor_mask].unique():
                mask = phase_mask & data.actor_mask & (data.action == action)
                weights[mask] = 1.0 / mask.sum()
                repaired = mask & recovery
                retained = mask & ~recovery
                if repaired.any() and retained.any():
                    # Reserve practice for both actual-policy corrections and
                    # original routes, regardless of their dataset sizes.
                    weights[repaired] = 0.5 / repaired.sum()
                    weights[retained] = 0.5 / retained.sum()
            continuation = phase_mask & ~data.actor_mask & ~context
            if continuation.any():
                weights[continuation] = 0.25 / continuation.sum()
            # Passive carry toward the goal is useful behavior, including NOOP
            # and LEFT braking. The environment pays this progress independent
            # of RIGHT; retain it in imitation replay without inventing labels.
            productive = phase_mask & data.actor_mask & ((data.action == 0) | (data.action == 3))
            weights[productive] *= 1 + (40 * data.carry_progress[productive]).clamp(0, 2)
            weights[phase_mask] /= weights[phase_mask].sum().clamp_min(1e-12)
        weights[family_mask] /= weights[family_mask].sum().clamp_min(1e-12)
        if family_weights:
            weight = float(family_weights.get(int(family), 1.0))
            if weight <= 0:
                raise ValueError("Every family must retain positive practice weight")
            weights[family_mask] *= weight
    return weights


def observed_decision_groups(data):
    cached = getattr(data, "_observed_decision_groups", None)
    if cached is not None:
        return cached
    keys = torch.cat((data.a.float(), data.b.float(), data.c, data.goal), dim=1).numpy()
    keys = np.ascontiguousarray(np.round(keys, 6))
    packed = keys.view(np.dtype((np.void, keys.dtype.itemsize * keys.shape[1]))).ravel()
    _, inverse = np.unique(packed, return_inverse=True)
    data._observed_decision_groups = torch.from_numpy(inverse)
    return data._observed_decision_groups


def demonstrated_tactic_sets(data):
    """Do not contradict successful early/late departures from one observation."""
    groups = observed_decision_groups(data)
    labels = torch.zeros((int(groups.max()) + 1, 4), dtype=torch.bool)
    actions = torch.zeros((int(groups.max()) + 1, 6), dtype=torch.bool)
    mask = (data.tactic >= 0) & ~data.context
    labels[groups[mask], data.tactic[mask]] = True
    for action in range(6):
        actions[groups[mask & data.tactic_actions[:, action]], action] = True
    return labels[groups], actions[groups]


def demonstrated_action_sets(data):
    """Union valid demonstrated choices at the same observed decision state.

    Successful route variants can walk OR jump from an identical prefix.
    Rejecting either demonstrated choice creates contradictory actor targets.
    """
    cached = getattr(data, "_valid_action_sets", None)
    if cached is not None:
        return cached
    groups = observed_decision_groups(data)
    choices = torch.zeros((int(groups.max()) + 1, 6), dtype=torch.bool)
    choices[groups[data.actor_mask], data.action[data.actor_mask]] = True
    allowed = choices[groups]
    allowed[~data.actor_mask, data.action[~data.actor_mask]] = True
    data._valid_action_sets = allowed
    return allowed


def priority_sample_weights(base, groups, priorities):
    """Focus difficult states while preserving every family/action's mass."""
    mass = torch.bincount(groups, weights=base)
    weighted = base * priorities
    current = torch.bincount(groups, weights=weighted, minlength=len(mass))
    return weighted * (mass / current.clamp_min(1e-12))[groups]


def adaptive_group_weights(base, groups, counts, errors):
    """Increase difficult decision-group practice without dropping any family.

    Group support limits the influence of tiny correction sets. Keep 35% of
    the original group allocation for retention; redistribute the rest within
    each family. Only training minibatch errors drive this allocation.
    """
    group_ids = torch.arange(len(counts))
    phase = (group_ids // 7) % 6
    action = group_ids % 7
    critical = ((phase == 4) & (action != 6)) | (
        (phase >= 1) & (phase <= 3) & ((action == 0) | (action == 3))
    )
    confidence = counts / (counts + 16)
    factors = 1 + torch.where(critical, 8.0, 3.0) * errors.clamp(0, 1) * confidence
    weighted = base * factors[groups]
    families = groups // 42
    mass = torch.bincount(families, weights=base)
    current = torch.bincount(families, weights=weighted, minlength=len(mass))
    weighted *= (mass / current.clamp_min(1e-12))[families]
    return 0.35 * base + 0.65 * weighted


# Weight of the interior-duration preference added to safe-set likelihood, in
# demonstration fitting and in online coaching alike.
INTERIOR_DURATION_WEIGHT = 0.2


def interior_duration_targets(allowed):
    """Prefer the interior of each collision-safe run, without bridging holes."""
    left = torch.zeros_like(allowed, dtype=torch.float32)
    right = torch.zeros_like(left)
    for i in range(allowed.shape[1]):
        left[:, i] = allowed[:, i] * (1 + (left[:, i - 1] if i else 0))
        j = allowed.shape[1] - 1 - i
        right[:, j] = allowed[:, j] * (1 + (right[:, j + 1] if i else 0))
    weights = torch.minimum(left, right).square()
    return weights / weights.sum(-1, keepdim=True).clamp_min(1)


@torch.no_grad()
def refresh_demonstration_memory(model, data, *, chunk_size=1024, progress=None):
    """Store the carried state entering every demonstration row.

    Each stored episode, including unsupervised context rows, is replayed in
    order through the current model with its demonstrated actions from an
    empty state. Batched updates then see the memory (LSTM state and distinct
    stance history) a policy would carry rather than a reset. The parameter-
    free LSTM inputs of each row are stored too, so fitting can unroll the
    LSTM with gradients. States go stale as weights change; training
    refreshes them periodically.
    """
    device = next(model.parameters()).device
    world_model = model.world_model
    layers, hidden = world_model.num_layers, world_model.hidden_size
    history = model.strategy_network.history
    stances = model.strategy_network.input_projection.in_features
    starts = (data.frame_index == 0).nonzero().flatten().tolist()
    ends = starts[1:] + [len(data.action)]
    episodes = sorted(zip(starts, ends), key=lambda span: span[0] - span[1])
    stored = torch.zeros((len(data.action), 2 * layers * hidden + history * stances))
    inputs = None
    completed_rows = 0
    last_report = time.monotonic()
    for first in range(0, len(episodes), chunk_size):
        chunk = episodes[first : first + chunk_size]
        state = model.initial_world_model_state(len(chunk), device)
        state = WorldModelState(
            state.hidden, state.cell, torch.zeros((len(chunk), history, stances), device=device)
        )
        for t in range(chunk[0][1] - chunk[0][0]):
            # Longest episodes come first, so the active ones form a prefix.
            active = sum(end - start > t for start, end in chunk)
            rows = torch.tensor([start + t for start, _ in chunk[:active]])
            state = WorldModelState(
                state.hidden[:, :active], state.cell[:, :active], state.stance[:active]
            )
            stored[rows] = _flatten_memory_state(state).cpu()
            a, b, c, g, motor = (
                getattr(data, key)[rows].to(device)
                for key in ("a", "b", "c", "goal", "motor_action")
            )
            state = model(
                a,
                b,
                c,
                skill_goal=g,
                forced_action=motor,
                critic_feedback_enabled=False,
                world_model_state=state,
                return_world_model_state=True,
            )[-1]
            step_inputs = model.last_world_model_inputs
            if inputs is None:
                inputs = torch.zeros((len(data.action), step_inputs.size(1)))
            inputs[rows] = step_inputs.cpu()
            completed_rows += active
            now = time.monotonic()
            if progress is not None and now - last_report >= 30:
                progress(
                    "memory_refresh_rows", frames=completed_rows, total_frames=len(data.action)
                )
                last_report = now
    data.memory_state = stored
    data.world_model_inputs = inputs if inputs is not None else torch.zeros((0, 0))
    return data


def _flatten_memory_state(state):
    return torch.cat(
        (
            state.hidden.transpose(0, 1).flatten(1),
            state.cell.transpose(0, 1).flatten(1),
            state.stance.flatten(1),
        ),
        dim=1,
    )


def carried_memory_state(model, data, ids, device):
    """The stored state entering rows `ids` (LSTM state and stance history)."""
    layers = model.world_model.num_layers
    hidden = model.world_model.hidden_size
    history = model.strategy_network.history
    size = layers * hidden
    carried = data.memory_state[ids].to(device)
    return WorldModelState(
        carried[:, :size].view(len(ids), layers, hidden).transpose(0, 1).contiguous(),
        carried[:, size : 2 * size].view(len(ids), layers, hidden).transpose(0, 1).contiguous(),
        carried[:, 2 * size :].view(len(ids), history, -1),
    )


def unrolled_memory_state(model, data, ids, device, steps):
    """Carried state entering `ids`, rebuilt over the last `steps` rows with gradients.

    Starts from the stored state up to `steps` rows earlier in the same
    episode and replays the stored LSTM inputs, so losses at `ids` train
    what the LSTM keeps. Also returns the memory head's predictions and
    targets at the replayed rows.
    """
    stored = carried_memory_state(model, data, ids, device)
    if steps <= 0:
        return stored, [], []
    back = data.frame_index[ids].clamp(max=steps)
    state = carried_memory_state(model, data, ids - back, device)
    hidden, cell = state.hidden, state.cell
    predictions, targets = [], []
    world_model = model.world_model
    for offset in range(steps, 0, -1):
        active = back >= offset
        if not active.any():
            continue
        rows = (ids - offset).clamp_min(0)
        out, advanced = world_model.advance(
            data.world_model_inputs[rows].to(device),
            data.c[rows].to(device),
            WorldModelState(hidden, cell),
        )
        keep = active.to(device).view(1, -1, 1)
        hidden = torch.where(keep, advanced.hidden, hidden)
        cell = torch.where(keep, advanced.cell, cell)
        if world_model.memory_head is not None:
            predictions.append(world_model.memory_head(out)[active.to(device)])
            targets.append(data.memory_target[rows[active]].to(device))
    return WorldModelState(hidden, cell, stored.stance), predictions, targets


def fit_demonstrations(
    model,
    optimizer,
    data,
    *,
    steps,
    batch_size=128,
    seed=0,
    decision_durations_only=False,
    walk_durations=True,
    prioritized=False,
    family_weights=None,
    adaptive_groups=False,
    tactic_loss_weight=0.5,
    memory_weight=0.0,
    memory_refresh_interval=0,
    memory_unroll=0,
    strategy_loss_weight=0.0,
    strategy_intent_loss_weight=0.5,
    progress=None,
):
    """Balance families and decision actions; never train on validation data.

    With a memory world model and a positive refresh interval, every row is
    fitted from the carried state entering it, rebuilt with gradients over up
    to `memory_unroll` earlier rows, and the LSTM is trained to report the
    observable hazard memory. The strategy network learns each row's
    objective: the skill goal the teacher supplied, or none.
    """
    # Supervise the same soft A context and deterministic transformer used
    # during greedy execution. Gumbel draws in the other A slots and dropout
    # otherwise change B's conditioning despite forcing the final action.
    # cuDNN recurrent backward requires training mode; recurrent dropout is
    # absent in this feedforward demonstration path.
    started = last_report = time.monotonic()
    if progress is not None:
        progress("started", updates=0, total_updates=steps, frames=len(data.action))
    model.eval()
    for module in model.modules():
        if isinstance(module, torch.nn.RNNBase):
            module.train()
    rng = torch.Generator().manual_seed(seed)
    weights = demonstration_sample_weights(data, family_weights)
    base_weights = weights.clone()
    priorities = torch.ones_like(weights)
    groups = demonstration_groups(data)
    group_counts = torch.bincount(groups).float()
    group_errors = torch.ones_like(group_counts)
    group_observations = torch.zeros_like(group_counts)
    allowed_actions = demonstrated_action_sets(data)
    allowed_tactics, allowed_tactic_actions = demonstrated_tactic_sets(data)
    device = next(model.parameters()).device
    memory = memory_refresh_interval > 0 and (
        getattr(getattr(model, "world_model", None), "memory_head", None) is not None
    )
    objectives = skill_goal_objective(data.goal)
    losses = []
    components = []
    for update in range(steps):
        if memory and update % memory_refresh_interval == 0:
            if progress is not None:
                progress("memory_refresh_started", updates=update, total_updates=steps)
            refresh_demonstration_memory(model, data, progress=progress)
            if progress is not None:
                progress(
                    "memory_refresh_completed",
                    updates=update,
                    total_updates=steps,
                    elapsed_seconds=round(time.monotonic() - started, 2),
                )
        if update % 32 == 0:
            weights = (
                adaptive_group_weights(base_weights, groups, group_counts, group_errors)
                if adaptive_groups
                else base_weights
            )
            if prioritized:
                weights = priority_sample_weights(weights, groups, priorities)
        ids = torch.multinomial(weights, batch_size, replacement=True, generator=rng)
        a, b, c, g = (getattr(data, k)[ids].to(device) for k in ("a", "b", "c", "goal"))
        motor = data.motor_action[ids].to(device)
        mask = data.actor_mask[ids].to(device)
        carried, window_predictions, window_targets = (
            unrolled_memory_state(model, data, ids, device, memory_unroll)
            if memory
            else (None, [], [])
        )
        outputs = model(
            a,
            b,
            c,
            skill_goal=g,
            forced_action=motor,
            critic_feedback_enabled=False,
            world_model_state=carried,
        )
        memory_loss = outputs[4].new_zeros(())
        if memory and memory_weight:
            memory_loss = F.mse_loss(
                torch.cat((model.last_memory_prediction, *window_predictions)),
                torch.cat((data.memory_target[ids].to(device), *window_targets)),
            )
        strategy_loss = outputs[4].new_zeros(())
        objective_logits = getattr(model, "last_objective_logits", None)
        if strategy_loss_weight and objective_logits is not None:
            strategy_loss = F.cross_entropy(objective_logits, objectives[ids].to(device))
        tactic_targets = data.tactic[ids].to(device)
        tactic_mask = tactic_targets >= 0
        intent_logits = getattr(model, "last_strategy_logits", None)
        intent_loss = outputs[4].new_zeros(())
        if tactic_mask.any() and intent_logits is not None:
            intent_loss = -torch.logsumexp(
                F.log_softmax(intent_logits, dim=-1).masked_fill(
                    ~allowed_tactics[ids].to(device), -1e9
                ),
                dim=-1,
            )[tactic_mask].mean()
        tactic_logits = getattr(model, "last_tactic_logits", None)
        tactic_loss = outputs[4].new_zeros(())
        tactic_action_loss = outputs[4].new_zeros(())
        if tactic_mask.any() and tactic_logits is not None:
            tactic_loss = -torch.logsumexp(
                F.log_softmax(tactic_logits, dim=-1).masked_fill(
                    ~allowed_tactics[ids].to(device), -1e9
                ),
                dim=-1,
            )[tactic_mask].mean()
        tactic_actions = allowed_tactic_actions[ids].to(device)
        tactic_action_mask = tactic_mask & mask & tactic_actions.any(dim=-1)
        if tactic_action_mask.any():
            tactic_action_loss = -torch.logsumexp(
                F.log_softmax(outputs[4][:, -1, :6], dim=-1).masked_fill(~tactic_actions, -1e9),
                dim=-1,
            )[tactic_action_mask].mean()
        logits = outputs[4][:, -1, :6]
        action_losses = -torch.logsumexp(
            F.log_softmax(logits, dim=-1).masked_fill(~allowed_actions[ids].to(device), -1e9),
            dim=-1,
        )
        action_loss = (action_losses * mask).sum() / mask.sum().clamp_min(1)
        duration_logits = model.last_motor_primitives.hold_duration_logits[:, -1, :]
        allowed = data.valid_durations[ids].to(device)
        duration_log_probs = F.log_softmax(duration_logits, dim=-1)
        duration_losses = -torch.logsumexp(duration_log_probs.masked_fill(~allowed, -1e9), dim=-1)
        # Safe-set likelihood alone can settle on a boundary hold: eight
        # frames worked in training but a nearby hard gap required nine.
        # Favor an interior duration so small timing errors retain a margin.
        duration_losses -= INTERIOR_DURATION_WEIGHT * (
            interior_duration_targets(allowed) * duration_log_probs
        ).sum(-1)
        # A feedforward policy without the executor's commitment state cannot
        # infer a past chosen duration from an arbitrary continuation frame.
        # Fixed commitments only consume this head at initiation.
        duration_mask = mask.clone() if decision_durations_only else torch.ones_like(mask)
        if not walk_durations:
            duration_mask = duration_mask & (motor != 1) & (motor != 3)
        # Timed plant waits reobserve every frame; no duration head is
        # consumed for those decisions, only the tactic and motor action.
        duration_mask = duration_mask & data.duration_consumed[ids].to(device)
        duration_loss = (duration_losses * duration_mask).sum() / duration_mask.sum().clamp_min(1)
        dynamics = F.mse_loss(outputs[1], data.next_c[ids].to(device))
        loss = (
            action_loss
            + duration_loss
            + 0.1 * dynamics
            + tactic_loss_weight * (tactic_loss + tactic_action_loss)
            + memory_weight * memory_loss
            + strategy_loss_weight * strategy_loss
            + strategy_intent_loss_weight * intent_loss
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        norm = clip_policy_and_objective_gradients(model, 1.0)
        if not torch.isfinite(norm):
            raise FloatingPointError("Nonfinite demonstration gradient")
        optimizer.step()
        if adaptive_groups:
            with torch.no_grad():
                predicted = logits.argmax(-1).cpu()
                actor_wrong = ~allowed_actions[ids, predicted]
                duration_wrong = ~data.valid_durations[ids, duration_logits.argmax(-1).cpu()]
                wrong = (actor_wrong & mask.cpu()) | (duration_wrong & duration_mask.cpu())
                sampled_groups = groups[ids]
                counts = torch.bincount(sampled_groups, minlength=len(group_counts)).float()
                failures = torch.bincount(
                    sampled_groups, weights=wrong.float(), minlength=len(group_counts)
                )
                seen = counts > 0
                group_errors[seen] = 0.9 * group_errors[seen] + 0.1 * (
                    failures[seen] / counts[seen]
                )
                group_observations += counts
        if prioritized:
            errors = action_losses.detach() * mask + duration_losses.detach() * duration_mask
            priorities[ids] = 0.05 + errors.cpu().clamp(0, 5)
        losses.append(float(loss.detach()))
        components.append(
            torch.stack(
                (
                    action_loss.detach(),
                    duration_loss.detach(),
                    dynamics.detach(),
                    tactic_loss.detach(),
                    tactic_action_loss.detach(),
                    memory_loss.detach(),
                    strategy_loss.detach(),
                    intent_loss.detach(),
                )
            )
        )
        now = time.monotonic()
        if progress is not None and (
            (update + 1) % 250 == 0 or update + 1 == steps or now - last_report >= 30
        ):
            progress(
                "updates",
                updates=update + 1,
                total_updates=steps,
                loss=float(np.mean(losses[-250:])),
                memory_loss=float(memory_loss.detach()),
                strategy_loss=float(strategy_loss.detach()),
                strategy_intent_loss=float(intent_loss.detach()),
                elapsed_seconds=round(now - started, 2),
            )
            last_report = now
    means = torch.stack(components).mean(0).cpu().tolist()
    model.last_demonstration_metrics = dict(
        zip(
            (
                "action_loss",
                "duration_loss",
                "dynamics_loss",
                "tactic_loss",
                "tactic_action_loss",
                "memory_loss",
                "strategy_loss",
                "strategy_intent_loss",
            ),
            means,
        )
    )
    if adaptive_groups:
        final = adaptive_group_weights(base_weights, groups, group_counts, group_errors)
        masses = torch.bincount(groups, weights=final) / final.sum()
        model.last_demonstration_groups = [
            dict(
                family=int(group) // 42,
                phase=(int(group) // 7) % 6,
                action=int(group) % 7,
                rows=int(group_counts[group]),
                sampled=int(group_observations[group]),
                error_ema=float(group_errors[group]),
                sampling_mass=float(masses[group]),
            )
            for group in (group_counts > 0).nonzero().flatten()
            if int(group) % 7 != 6
        ]
    return sum(losses) / len(losses)


def varied_demonstration(sample, seed, *, robust=False):
    """A second successful route, using current geometry and varied safe holds.

    Canonical scripts alone miss states reached when a learned jump lands a few
    pixels earlier or later. Only complete successful alternative routes enter
    the demonstration pool.
    """
    import random

    if any(
        isinstance(e, dict) and e.get("kind") == "piranha_plant"
        for e in sample.scenario.get("enemies", [])
    ):
        from .piranha import plant_oracle

        actions = plant_oracle(sample.scenario, variant=1 + seed % 3)
        return replace(sample, oracle={**sample.oracle, "actions": actions}) if actions else None
    if sample.scenario.get("bridge_jump_task"):
        from .bridge_curriculum import bridge_jump_oracle
        from .monte_carlo import validate_block_smb_monte_carlo_oracle

        actions = bridge_jump_oracle(sample.scenario, variant=1 + seed % 3)
        if validate_block_smb_monte_carlo_oracle(sample.scenario, actions)["reachable"]:
            return replace(sample, oracle={**sample.oracle, "actions": actions})
        return None

    from .bridge_traversal import bridge_walk_state

    rng = random.Random(seed)
    env = MarioScenarioEnv()
    actions = []
    remaining = 0
    airborne = False
    from .primitive_execution import JumpReleaseState

    release = JumpReleaseState()
    direction = 1
    takeoff_distance = rng.randint(12, 75)
    gap_lead = rng.randint(10, 24) if robust else 0
    stomp_lead = rng.randint(48, 68) if robust else 0
    try:
        env.reset(scenario=sample.scenario)
        env.render = lambda: None
        for _ in range(320):
            if airborne and env.mario["on_ground"]:
                airborne = False
                if robust:
                    remaining = 0
            if bridge_training_active(env):
                state = bridge_walk_state(env)
                action = (
                    1
                    if env._bridge_crossed
                    or (state is not None and 0 in bridge_safe_wait_frames(env))
                    else 0
                )
            elif release.remaining:
                remaining = 0
                action = release.action
            elif remaining:
                action = 2 if direction > 0 else 4
                remaining -= 1
            elif airborne or not env.mario["on_ground"]:
                action = 1 if direction > 0 else 3
            else:
                target = plant_clearance_target(env, local_objective(env))
                if (
                    (env._goal_on_stomp or env._require_stomp_before_goal)
                    and not env._stomp_credited
                    and target.kind == "enemy"
                ):
                    target = replace(target, kind="stomp")
                direction = target.direction
                distance = local_target_distance(env, target)
                ready = distance < takeoff_distance
                if robust:
                    support = env.mario.get("_platform")
                    if target.kind == "gap" and support is not None:
                        ready = support_edge_distance(env, direction) <= gap_lead
                    else:
                        ready = distance < (stomp_lead if target.kind == "stomp" else 40)
                    if (
                        target.kind in ("enemy", "stomp")
                        and support is not None
                        and target.top > env.mario["y"] + env.mario["h"] + 1
                    ):
                        # A lower enemy can still be far away when the safe
                        # takeoff surface ends. Jump before walking off it.
                        ready |= support_edge_distance(env, direction) <= 24
                ready |= (
                    env._single_jump_attempt
                    or env._goal_on_stomp
                    or (robust and bool(sample.parameters.get("single_jump")))
                )
                valid = (
                    safe_jump_holds(env, target, direction)
                    if target.kind not in ("finish", "retreat") and ready
                    else []
                )
                if (
                    robust
                    and target.kind == "stomp"
                    and not env._goal_on_stomp
                    and not env._single_jump_attempt
                    and len(valid) < 3
                ):
                    valid = []
                if valid:
                    if robust:
                        runs = []
                        for value in valid:
                            if not runs or value != runs[-1][-1] + 1:
                                runs.append([])
                            runs[-1].append(value)
                        longest = max(runs, key=len)
                        remaining = longest[len(longest) // 2] - 1
                    else:
                        remaining = rng.choice(valid) - 1
                    airborne = True
                    action = 2 if direction > 0 else 4
                    takeoff_distance = rng.randint(12, 75)
                    if robust:
                        gap_lead = rng.randint(10, 24)
                else:
                    action = 1 if direction > 0 else 3
            _, _, done, truncated, info = env.step(action)
            release.observe(env, action, info)
            actions.append(action)
            if info["reward_terms"]["enemy_stomp"] > 0:
                remaining = 0
                airborne = True
            if done or truncated:
                break
        if env._goal_credited:
            return replace(sample, oracle={**sample.oracle, "actions": actions})
        return None
    finally:
        env.close()


ENEMY_WAIT_FRAMES = (8, 12, 16, 20, 24)


def _enemy_ahead(env, near=32.0, far=112.0):
    """A live moving walker ahead toward the goal, within approach range.

    Waiting for a stationary enemy changes nothing but the clock.
    """
    from .tactics import goal_direction

    direction = goal_direction(env)
    center = env.mario["x"] + env.mario["w"] / 2
    for enemy in env.enemies:
        if (
            enemy.get("dead")
            or enemy.get("kind") == "piranha_plant"
            or not float(enemy.get("speed", 0.0))
        ):
            continue
        ahead = (enemy["x"] + enemy["w"] / 2 - center) * direction
        if near <= ahead <= far:
            return True
    return False


def enemy_wait_demonstration(sample, seed):
    """A successful route that holds while an enemy approaches, then stomps it.

    Canonical enemy routes never wait, so the tactical layer never sees a hold
    near an enemy, and stomp takeoffs cover only the enemy positions of one
    timing. Only layouts that require a stomp qualify: waiting lets the enemy
    walk into stomp range, whereas in avoidance layouts it only shifts timing
    and lowered success in the probe. The canonical prefix runs to the first
    grounded walking decision with an enemy in approach range; Mario then
    waits a multiple of four frames (the executor's wait granularity) while
    grounded and alive, and the recovery teacher completes the level from
    there. Only complete routes are kept.
    """
    import random

    from .monte_carlo import validate_block_smb_monte_carlo_oracle
    from .policy_recovery import coached_suffix
    from .primitive_execution import JumpReleaseState

    scenario = sample.scenario
    if scenario.get("single_jump_attempt") or scenario.get("bridge_jump_task"):
        return None
    actions = list(sample.oracle["actions"])
    env = MarioScenarioEnv()
    release = JumpReleaseState()
    try:
        env.reset(scenario=copy.deepcopy(dict(scenario)), seed=0)
        env.render = lambda: None
        if not (env._require_stomp_before_goal or env._goal_on_stomp):
            return None
        for frame, action in enumerate(actions):
            if (
                action in (1, 3)
                and env.mario["on_ground"]
                and not release.remaining
                and _enemy_ahead(env)
            ):
                break
            _, _, done, truncated, info = env.step(action)
            release.observe(env, action, info)
            if done or truncated:
                return None
        else:
            return None
        waits = random.Random(seed).choice(ENEMY_WAIT_FRAMES)
        for _ in range(waits):
            _, _, done, truncated, info = env.step(0)
            release.observe(env, 0, info)
            if info["death"] or done or truncated or not env.mario["on_ground"]:
                return None
        suffix = coached_suffix(env, release_state=release)
        if not suffix:
            return None
        route = actions[:frame] + [0] * waits + list(suffix)
    finally:
        env.close()
    if not validate_block_smb_monte_carlo_oracle(scenario, route)["reachable"]:
        return None
    return replace(sample, oracle={**sample.oracle, "actions": route})


def with_robust_demonstrations(cases, seed):
    return [
        (index, varied_demonstration(sample, seed + i, robust=True) or sample)
        for i, (index, sample) in enumerate(cases)
    ]


def with_varied_demonstrations(cases, seed, *, robust=False):
    """Keep canonical cases and add only physically successful alternate routes."""
    return list(cases) + [
        (family_index, alternative)
        for i, (family_index, sample) in enumerate(cases)
        if (alternative := varied_demonstration(sample, seed + i, robust=robust)) is not None
    ]


def build_balanced_demonstrations(config, vision_factory, *, families=None):
    """Independent training layouts and successful route variants for every family."""
    from .monte_carlo import BLOCK_SMB_MC_FAMILIES, sample_block_smb_monte_carlo_scenario

    families = set(BLOCK_SMB_MC_FAMILIES if families is None else families)
    if not families or families - set(BLOCK_SMB_MC_FAMILIES):
        raise ValueError("Demonstration families must be a nonempty subset of the family catalog")
    cases = []
    completed_layouts = 0
    progress = _CollectionProgress(
        config,
        "generate_routes",
        len(families) * config.demonstration_layouts_per_family,
    )
    for family_index, family in enumerate(BLOCK_SMB_MC_FAMILIES):
        if family not in families:
            continue
        for i in range(config.demonstration_layouts_per_family):
            progress.update(
                completed_layouts,
                force=i == 0,
                family=family,
                trajectories=len(cases),
            )
            completed_layouts += 1
            sample = sample_block_smb_monte_carlo_scenario(
                family=family,
                seed=config.seed,
                split="train",
                sample_index=i,
                difficulty=("easy", "medium", "hard")[i % 3],
            )
            if (sample.scenario.get("bridge_jump_task") or family == "piranha_avoidance") and (
                config.demonstration_robust_routes or config.demonstration_varied_routes
            ):
                # Difficulty already cycles with i % 3. Reusing that index to
                # select a route permanently omits a boundary in each tier.
                # Plant variants use the same modulo-three selection, so keep
                # the canonical route and every alternate in every tier for
                # both families, including during later rehearsal.
                cases.append((family_index, sample))
                for variant in (1, 2, 3):
                    alternative = varied_demonstration(sample, variant - 1, robust=True)
                    if alternative is not None:
                        cases.append((family_index, alternative))
                if family == "piranha_avoidance":
                    from .piranha_tactics import arrival_demonstrations

                    for corrected in arrival_demonstrations(sample, i):
                        cases.append((family_index, corrected))
                continue
            if config.demonstration_robust_routes:
                sample = varied_demonstration(sample, config.seed + i, robust=True) or sample
            cases.append((family_index, sample))
            if config.demonstration_enemy_wait_routes:
                waiting = enemy_wait_demonstration(sample, config.seed + i + 200000)
                if waiting is not None:
                    cases.append((family_index, waiting))
            if config.demonstration_varied_routes:
                alternative = varied_demonstration(
                    sample, config.seed + i + 100000, robust=config.demonstration_robust_routes
                )
                if alternative is not None:
                    cases.append((family_index, alternative))
    progress.update(progress.total, trajectories=len(cases))
    return collect_demonstrations(cases, config, vision_factory)
