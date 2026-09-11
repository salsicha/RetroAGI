"""Batched learning from physically completed trajectories.

Frame observations come from the same frozen vision and engine encoder as
rollouts. Teachers supervise actions AND commitment lengths, including walks.
Evaluation uses only the learned policy and normal primitive controller.
"""

from dataclasses import dataclass, replace

import numpy as np
import torch
import torch.nn.functional as F

from retroagi.core.skills import SKILL_GOAL_ENCODING_DIM, skill_goal_encoding

from .adapter import BlockSMBObservationConfig, BlockSMBStage
from .bridge_traversal import bridge_phase, bridge_safe_wait_frames
from .env import MarioScenarioEnv
from .local_traversal import (
    LOCAL_TRAVERSAL_FAMILIES,
    local_objective,
    local_target_distance,
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

    def __post_init__(self):
        if self.carry_progress is None:
            self.carry_progress = torch.zeros_like(self.family, dtype=torch.float32)
        if self.phase is None:
            self.phase = torch.zeros_like(self.family)


def collect_demonstrations(cases, config, vision_factory, *, vision_batch_size=32):
    rows = []
    episode_starts = []
    frame_wait_episodes = set()
    for family_index, sample in cases:
        stage = BlockSMBStage(
            env=MarioScenarioEnv(reward_config=config.reward_config),
            scenario=sample.scenario,
            vision=vision_factory(),
            observation_config=BlockSMBObservationConfig(
                motion_observations=config.motion_observations
            ),
        )
        episode = []
        try:
            observation = stage.reset(seed=sample.sample_seed % (2**31))
            actions = list(sample.oracle["actions"])
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
            bridge = stage.env._require_bridge_before_goal and bridge_jump is None
            enemy = stage.env._require_stomp_before_goal
            opening = True
            bridge_exit_committed = False
            recovering_stomp = False
            observations = [observation]
            states = [stage.state_features(stage.last_info)]
            for frame, action in enumerate(actions):
                env = stage.env
                if jump_intent is not None and env.mario["on_ground"]:
                    jump_intent = None
                    jump_rows = []
                if recovering_stomp and env.mario["on_ground"]:
                    recovering_stomp = False
                actor_mask = jump_intent is None and not recovering_stomp
                end = frame + 1
                while end < len(actions) and actions[end] == action:
                    end += 1
                hold = min(duration_menu[-1], end - frame)
                if jump_intent is None and action in (2, 4, 5):
                    jump_intent = action
                    jump_hold = hold
                    objective = local_objective(env)
                    if (
                        (env._goal_on_stomp or env._require_stomp_before_goal)
                        and not env._stomp_credited
                        and objective.kind == "enemy"
                    ):
                        objective = replace(objective, kind="stomp")
                    jump_valid = (
                        safe_jump_holds(env, objective, 1 if action == 2 else -1)
                        if action in (2, 4) and env.mario["on_ground"]
                        else []
                    )
                motor_action = jump_intent if jump_intent is not None else action
                duration = jump_hold if jump_intent is not None else hold
                if motor_action == 0:
                    duration = 1 if bridge_jump else max(1, min(16, round((end - frame) / 4)))
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
                elif scenario_family(stage.scenario) in LOCAL_TRAVERSAL_FAMILIES:
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
                observation, reward, done, truncated, info = env.step(action)
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
            if bridge_jump:
                frame_wait_episodes.add(len(rows))
            episode_starts.append(len(rows))
            rows.extend(episode)
        finally:
            stage.env.close()
    columns = list(zip(*rows))
    data = DemonstrationBatch(
        *(torch.cat(v) if i in (0, 1, 2, 3, 8) else torch.tensor(v) for i, v in enumerate(columns))
    )

    data = align_steady_demonstrations(
        data, episode_starts, frame_wait_episodes=frame_wait_episodes
    )
    return data if config.walk_duration_primitives else without_walk_commitments(data)


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


DEMONSTRATION_CONTRACT_VERSION = 4


def without_walk_commitments(data):
    """Walking reconsiders A every frame; jump and wait commitments remain."""
    data = replace(data, actor_mask=data.actor_mask.clone(), next_c=data.next_c.clone())
    walking = ((data.motor_action == 1) | (data.motor_action == 3)) & (data.c[:, 16] > 0.5)
    data.actor_mask[walking] = True
    indices = walking.nonzero().flatten()
    indices = indices[indices < len(data.action) - 1]
    indices = indices[data.c[indices + 1, 26] != 0]
    data.next_c[indices] = data.c[indices + 1]
    return data


def demonstration_groups(data):
    # 0=routine, 1=approach, 2=wait, 3=board, 4=ride, 5=exit.
    return (data.family.long() * 6 + data.phase.long()) * 7 + torch.where(
        data.actor_mask, data.action, 6
    )


def demonstration_sample_weights(data, family_weights=None):
    weights = torch.zeros(len(data.action))
    for family in data.family.unique():
        family_mask = data.family == family
        for phase in data.phase[family_mask].unique():
            phase_mask = family_mask & (data.phase == phase)
            for action in data.action[phase_mask & data.actor_mask].unique():
                mask = phase_mask & data.actor_mask & (data.action == action)
                weights[mask] = 1.0 / mask.sum()
            continuation = phase_mask & ~data.actor_mask
            if continuation.any():
                weights[continuation] = 0.25 / continuation.sum()
            # Passive carry toward the goal is useful behavior, including NOOP
            # and LEFT braking. The environment pays this progress independent
            # of RIGHT; retain it in imitation replay without inventing labels.
            productive = phase_mask & data.actor_mask & ((data.action == 0) | (data.action == 3))
            weights[productive] *= 1 + (40 * data.carry_progress[productive]).clamp(0, 2)
            weights[phase_mask] /= weights[phase_mask].sum()
        weights[family_mask] /= weights[family_mask].sum()
        if family_weights:
            weight = float(family_weights.get(int(family), 1.0))
            if weight <= 0:
                raise ValueError("Every family must retain positive practice weight")
            weights[family_mask] *= weight
    return weights


def demonstrated_action_sets(data):
    """Union valid demonstrated choices at the same observed decision state.

    Successful route variants can walk OR jump from an identical prefix.
    Rejecting either demonstrated choice creates contradictory actor targets.
    """
    cached = getattr(data, "_valid_action_sets", None)
    if cached is not None:
        return cached
    keys = torch.cat((data.a.float(), data.b.float(), data.c, data.goal), dim=1).numpy()
    keys = np.ascontiguousarray(np.round(keys, 6))
    packed = keys.view(np.dtype((np.void, keys.dtype.itemsize * keys.shape[1]))).ravel()
    _, inverse = np.unique(packed, return_inverse=True)
    groups = torch.from_numpy(inverse)
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
):
    """Balance families and decision actions; never train on validation data."""
    # Supervise the same soft A context and deterministic transformer used
    # during greedy execution. Gumbel draws in the other A slots and dropout
    # otherwise change B's conditioning despite forcing the final action.
    # cuDNN recurrent backward requires training mode; recurrent dropout is
    # absent in this feedforward demonstration path.
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
    device = next(model.parameters()).device
    losses = []
    components = []
    for update in range(steps):
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
        outputs = model(
            a,
            b,
            c,
            skill_goal=g,
            forced_action=motor,
            critic_feedback_enabled=False,
            world_model_state=None,
        )
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
        duration_losses -= 0.2 * (interior_duration_targets(allowed) * duration_log_probs).sum(-1)
        # A feedforward policy without the executor's commitment state cannot
        # infer a past chosen duration from an arbitrary continuation frame.
        # Fixed commitments only consume this head at initiation.
        duration_mask = mask.clone() if decision_durations_only else torch.ones_like(mask)
        if not walk_durations:
            duration_mask = duration_mask & (motor != 1) & (motor != 3)
        duration_loss = (duration_losses * duration_mask).sum() / duration_mask.sum().clamp_min(1)
        dynamics = F.mse_loss(outputs[1], data.next_c[ids].to(device))
        loss = action_loss + duration_loss + 0.1 * dynamics
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
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
            torch.stack((action_loss.detach(), duration_loss.detach(), dynamics.detach()))
        )
    means = torch.stack(components).mean(0).cpu().tolist()
    model.last_demonstration_metrics = dict(
        zip(("action_loss", "duration_loss", "dynamics_loss"), means)
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
    settling = 0
    bouncing = False
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
                    settling = 0 if bouncing else 2
                bouncing = False
            if env._require_bridge_before_goal:
                state = bridge_walk_state(env)
                action = (
                    1
                    if env._bridge_crossed
                    or (state is not None and 0 in bridge_safe_wait_frames(env))
                    else 0
                )
            elif settling:
                action = 1 if direction > 0 else 3
                settling -= 1
            elif remaining:
                action = 2 if direction > 0 else 4
                remaining -= 1
            elif airborne or not env.mario["on_ground"]:
                action = 1 if direction > 0 else 3
            else:
                target = local_objective(env)
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
            actions.append(action)
            if info["reward_terms"]["enemy_stomp"] > 0:
                remaining = 0
                airborne = True
                bouncing = True
            if done or truncated:
                break
        if env._goal_credited:
            return replace(sample, oracle={**sample.oracle, "actions": actions})
        return None
    finally:
        env.close()


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


def build_balanced_demonstrations(config, vision_factory):
    """Independent training layouts and successful route variants for every family."""
    from .monte_carlo import BLOCK_SMB_MC_FAMILIES, sample_block_smb_monte_carlo_scenario

    cases = []
    for family_index, family in enumerate(BLOCK_SMB_MC_FAMILIES):
        for i in range(config.demonstration_layouts_per_family):
            sample = sample_block_smb_monte_carlo_scenario(
                family=family,
                seed=config.seed,
                split="train",
                sample_index=i,
                difficulty=("easy", "medium", "hard")[i % 3],
            )
            if config.demonstration_robust_routes:
                sample = varied_demonstration(sample, config.seed + i, robust=True) or sample
            cases.append((family_index, sample))
            if config.demonstration_varied_routes:
                alternative = varied_demonstration(
                    sample, config.seed + i + 100000, robust=config.demonstration_robust_routes
                )
                if alternative is not None:
                    cases.append((family_index, alternative))
    return collect_demonstrations(cases, config, vision_factory)
