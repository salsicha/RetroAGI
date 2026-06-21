"""Training utilities for the Block SMB stage."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Optional

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from retroagi.core import (
    AgentWorldModelCritic,
    StageBatch,
    VisionEncoder,
    build_checkpoint,
    load_checkpoint,
    save_checkpoint,
    select_device,
    to_plain_data,
)

from .adapter import BLOCK_SMB_SPEC, SCENARIOS_DIR, BlockSMBStage
from .env import MarioScenarioEnv
from .vision import BlockVisionTransformer

BLOCK_SMB_MODEL_NAME = "block_smb_actor_world_model_critic"
BLOCK_SMB_CHECKPOINT_KIND = "block_smb_trainer"
BLOCK_SMB_ACTION_COUNT = 6


@dataclass(frozen=True)
class BlockSMBTrainingConfig:
    seed: int = 0
    epochs: int = 1
    episodes_per_epoch: int = 2
    rollout_steps: int = 32
    learning_rate: float = 3e-4
    gamma: float = 0.95
    entropy_weight: float = 0.01
    world_model_weight: float = 0.1
    action_aux_weight: float = 0.01
    critic_loss_weight: float = 0.001
    gradient_clip_norm: float = 1.0
    hidden_dim: int = 32
    device: str = "auto"
    deterministic: bool = True
    fixed_scenarios: tuple[str, ...] = (
        "level_1_flat.json",
        "level_2_gap.json",
        "level_3_stairs.json",
        "level_4_platforms.json",
    )
    generated_scenarios: int = 0
    generated_seed: int = 50_000
    evaluation_episodes: int = 1
    evaluation_max_steps: int = 200
    checkpoint_path: Optional[Path] = None
    resume_path: Optional[Path] = None
    save_checkpoints: bool = False
    video_dir: Optional[Path] = None
    record_videos: bool = False
    num_envs: int = 1

    def __post_init__(self) -> None:
        positive_ints = (
            "epochs",
            "episodes_per_epoch",
            "rollout_steps",
            "hidden_dim",
            "evaluation_episodes",
            "evaluation_max_steps",
            "num_envs",
        )
        for name in positive_ints:
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        for name in ("learning_rate", "gamma", "gradient_clip_norm"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.generated_scenarios < 0:
            raise ValueError("generated_scenarios must be non-negative")
        loss_weights = (
            self.entropy_weight,
            self.world_model_weight,
            self.action_aux_weight,
            self.critic_loss_weight,
        )
        if any(weight < 0 for weight in loss_weights):
            raise ValueError("loss weights must be non-negative")
        if self.save_checkpoints and self.checkpoint_path is None:
            raise ValueError("checkpoint_path is required when save_checkpoints is true")


@dataclass
class BlockSMBTransition:
    batch: StageBatch
    next_batch: StageBatch
    action: int
    reward: float
    done: bool
    episode_mask: float
    scenario_name: str
    info: Mapping[str, Any]
    log_prob: torch.Tensor
    entropy: torch.Tensor
    actions1: torch.Tensor
    actions2: torch.Tensor
    next_state_pred: torch.Tensor
    criticism: torch.Tensor
    logits_a: torch.Tensor


@dataclass
class BlockSMBTrajectory:
    scenario_name: str
    transitions: list[BlockSMBTransition] = field(default_factory=list)
    frames: list[np.ndarray] = field(default_factory=list)

    @property
    def total_return(self) -> float:
        return float(sum(step.reward for step in self.transitions))

    @property
    def success(self) -> bool:
        return bool(
            self.transitions
            and self.transitions[-1].done
            and self.transitions[-1].info.get("goal_reached", False)
        )


class BlockSMBReplayBuffer:
    def __init__(self) -> None:
        self.trajectories: list[BlockSMBTrajectory] = []

    def add(self, trajectory: BlockSMBTrajectory) -> None:
        if trajectory.transitions:
            self.trajectories.append(trajectory)

    def clear(self) -> None:
        self.trajectories.clear()

    def transitions(self) -> list[BlockSMBTransition]:
        return [
            step
            for trajectory in self.trajectories
            for step in trajectory.transitions
        ]

    def episode_masks(self) -> torch.Tensor:
        values = [step.episode_mask for step in self.transitions()]
        return torch.tensor(values, dtype=torch.float32)


class SequentialBlockSMBVectorEnv:
    """Deterministic vector-env scaffold that steps independent envs sequentially."""

    def __init__(self, scenarios: list[tuple[str, dict]], num_envs: int = 1):
        if num_envs <= 0:
            raise ValueError("num_envs must be positive")
        if not scenarios:
            raise ValueError("scenarios must be non-empty")
        self.scenarios = scenarios
        self.envs = [MarioScenarioEnv() for _ in range(num_envs)]

    def reset(self, seed: int = 0) -> list[tuple[np.ndarray, Mapping[str, Any]]]:
        outputs = []
        for index, env in enumerate(self.envs):
            _name, scenario = self.scenarios[index % len(self.scenarios)]
            outputs.append(env.reset(scenario=scenario, seed=seed + index))
        return outputs

    def step(
        self, actions: list[int]
    ) -> list[tuple[np.ndarray, float, bool, bool, Mapping[str, Any]]]:
        if len(actions) != len(self.envs):
            raise ValueError("actions length must match num_envs")
        return [env.step(action) for env, action in zip(self.envs, actions)]

    def close(self) -> None:
        for env in self.envs:
            env.close()


def seed_everything(seed: int, deterministic: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(deterministic)
    torch.backends.cudnn.deterministic = deterministic
    if deterministic:
        torch.backends.cudnn.benchmark = False


def load_fixed_scenarios(names: tuple[str, ...]) -> list[tuple[str, dict]]:
    scenarios = []
    for name in names:
        path = SCENARIOS_DIR / name
        with path.open("r", encoding="utf-8") as handle:
            scenarios.append((name, json.load(handle)))
    return scenarios


def build_curriculum(config: BlockSMBTrainingConfig) -> list[tuple[str, dict]]:
    scenarios = load_fixed_scenarios(config.fixed_scenarios)
    for index in range(config.generated_scenarios):
        seed = config.generated_seed + index
        scenarios.append(
            (f"generated_{index:03d}", MarioScenarioEnv.generate_scenario(seed=seed))
        )
    return scenarios


def make_block_smb_model(config: BlockSMBTrainingConfig) -> AgentWorldModelCritic:
    return AgentWorldModelCritic(
        BLOCK_SMB_SPEC.vocab_size,
        BLOCK_SMB_SPEC.seq_len_a,
        BLOCK_SMB_SPEC.seq_len_c,
        BLOCK_SMB_SPEC.ratio_bc,
        d_model=config.hidden_dim,
    )


def finite_or_raise(name: str, tensor: torch.Tensor) -> None:
    if not torch.isfinite(tensor).all().item():
        raise FloatingPointError(f"{name} contains NaN or infinite values")


def check_model_gradients(model: torch.nn.Module) -> None:
    saw_gradient = False
    for name, parameter in model.named_parameters():
        if parameter.grad is None:
            continue
        saw_gradient = True
        finite_or_raise(f"gradient {name}", parameter.grad)
    if not saw_gradient:
        raise FloatingPointError("no gradients were produced")


def discounted_returns(
    rewards: list[float], masks: list[float], gamma: float, device: torch.device
) -> torch.Tensor:
    returns = []
    running = 0.0
    for reward, mask in zip(reversed(rewards), reversed(masks)):
        running = reward + gamma * running * mask
        returns.append(running)
    returns.reverse()
    values = torch.tensor(returns, dtype=torch.float32, device=device)
    if values.numel() > 1:
        values = (values - values.mean()) / values.std().clamp_min(1e-6)
    return values


def _goal_reached(env: MarioScenarioEnv) -> bool:
    if env.goal is None:
        return False
    import pygame

    mario_rect = pygame.Rect(
        env.mario["x"], env.mario["y"], env.mario["w"], env.mario["h"]
    )
    return bool(mario_rect.colliderect(env.goal))


def _action_from_model(
    model: AgentWorldModelCritic,
    batch: StageBatch,
    *,
    deterministic: bool,
    tau: float,
) -> tuple[int, torch.Tensor, torch.Tensor, tuple[torch.Tensor, ...]]:
    actions1, next_state_pred, criticism, actions2, logits_a, _w, _b = model(
        batch.src_a, batch.src_b, batch.src_c, tau=tau
    )
    action_logits = logits_a[:, -1, :BLOCK_SMB_ACTION_COUNT]
    finite_or_raise("action_logits", action_logits)
    distribution = torch.distributions.Categorical(logits=action_logits)
    action_tensor = action_logits.argmax(dim=-1) if deterministic else distribution.sample()
    log_prob = distribution.log_prob(action_tensor).squeeze(0)
    entropy = distribution.entropy().squeeze(0)
    return (
        int(action_tensor.item()),
        log_prob,
        entropy,
        (actions1, actions2, next_state_pred, criticism, logits_a),
    )


def collect_trajectory(
    model: AgentWorldModelCritic,
    stage: BlockSMBStage,
    scenario_name: str,
    *,
    rollout_steps: int,
    seed: int,
    deterministic: bool,
    device: torch.device,
    record_frames: bool = False,
) -> BlockSMBTrajectory:
    observation = stage.reset(seed=seed)
    trajectory = BlockSMBTrajectory(scenario_name=scenario_name)
    if record_frames:
        trajectory.frames.append(np.asarray(observation).copy())

    for _ in range(rollout_steps):
        batch = stage.encode_observation(observation)
        batch.src_a = batch.src_a.to(device)
        batch.src_b = batch.src_b.to(device)
        batch.src_c = batch.src_c.to(device)
        action, log_prob, entropy, outputs = _action_from_model(
            model, batch, deterministic=deterministic, tau=1.0
        )
        next_observation, reward, terminated, truncated, info = stage.step(action)
        info = dict(info)
        info["goal_reached"] = _goal_reached(stage.env)
        next_batch = stage.encode_observation(next_observation, info)
        next_batch.src_a = next_batch.src_a.to(device)
        next_batch.src_b = next_batch.src_b.to(device)
        next_batch.src_c = next_batch.src_c.to(device)
        done = bool(terminated or truncated)
        episode_mask = 0.0 if done else 1.0
        actions1, actions2, next_state_pred, criticism, logits_a = outputs
        trajectory.transitions.append(
            BlockSMBTransition(
                batch=batch,
                next_batch=next_batch,
                action=action,
                reward=float(reward),
                done=done,
                episode_mask=episode_mask,
                scenario_name=scenario_name,
                info=info,
                log_prob=log_prob,
                entropy=entropy,
                actions1=actions1,
                actions2=actions2,
                next_state_pred=next_state_pred,
                criticism=criticism,
                logits_a=logits_a,
            )
        )
        observation = next_observation
        if record_frames:
            trajectory.frames.append(np.asarray(observation).copy())
        if done:
            break
    return trajectory


def compute_block_smb_losses(
    transitions: list[BlockSMBTransition],
    config: BlockSMBTrainingConfig,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    if not transitions:
        raise ValueError("transitions must be non-empty")
    returns = discounted_returns(
        [step.reward for step in transitions],
        [step.episode_mask for step in transitions],
        config.gamma,
        device,
    )
    policy_terms = []
    entropy_terms = []
    world_terms = []
    action1_terms = []
    action2_terms = []
    critic_terms = []
    for index, step in enumerate(transitions):
        reward_target = torch.full_like(step.actions2, returns[index])
        policy_terms.append(-step.log_prob * returns[index].detach())
        entropy_terms.append(step.entropy)
        world_terms.append(
            F.mse_loss(step.next_state_pred, step.next_batch.src_c.detach())
        )
        action1_terms.append(F.mse_loss(step.actions1, reward_target.detach()))
        action2_terms.append(F.mse_loss(step.actions2, reward_target.detach()))
        critic_terms.append(step.criticism.pow(2).mean())
    loss_actor_pass1 = torch.stack(action1_terms).mean()
    loss_actor_pass2 = (
        torch.stack(policy_terms).mean()
        + config.action_aux_weight * torch.stack(action2_terms).mean()
    )
    loss_world_model = torch.stack(world_terms).mean()
    loss_critic = torch.stack(critic_terms).mean()
    entropy_bonus = torch.stack(entropy_terms).mean()
    loss_total = (
        loss_actor_pass1 * config.action_aux_weight
        + loss_actor_pass2
        + loss_world_model * config.world_model_weight
        + loss_critic * config.critic_loss_weight
        - config.entropy_weight * entropy_bonus
    )
    losses = {
        "loss_actor_pass1": loss_actor_pass1,
        "loss_actor_pass2": loss_actor_pass2,
        "loss_world_model": loss_world_model,
        "loss_critic": loss_critic,
        "loss_entropy": entropy_bonus,
        "loss_total": loss_total,
    }
    for name, value in losses.items():
        finite_or_raise(name, value)
    return losses


def train_block_smb_epoch(
    model: AgentWorldModelCritic,
    optimizer: optim.Optimizer,
    curriculum: list[tuple[str, dict]],
    config: BlockSMBTrainingConfig,
    epoch: int,
    *,
    device: torch.device,
    vision_factory: Callable[[], VisionEncoder] = BlockVisionTransformer,
) -> tuple[dict[str, float], BlockSMBReplayBuffer]:
    model.train()
    replay = BlockSMBReplayBuffer()
    for episode in range(config.episodes_per_epoch):
        scenario_name, scenario = curriculum[
            (epoch * config.episodes_per_epoch + episode) % len(curriculum)
        ]
        stage = BlockSMBStage(scenario=scenario, vision=vision_factory())
        try:
            trajectory = collect_trajectory(
                model,
                stage,
                scenario_name,
                rollout_steps=config.rollout_steps,
                seed=config.seed + epoch * 10_000 + episode,
                deterministic=False,
                device=device,
            )
        finally:
            stage.env.close()
        replay.add(trajectory)

    losses = compute_block_smb_losses(replay.transitions(), config, device)
    optimizer.zero_grad(set_to_none=True)
    losses["loss_total"].backward()
    check_model_gradients(model)
    grad_norm = torch.nn.utils.clip_grad_norm_(
        model.parameters(), config.gradient_clip_norm
    )
    if not torch.isfinite(grad_norm).item():
        raise FloatingPointError("gradient norm is NaN or infinite")
    optimizer.step()
    epoch_losses = {key: float(value.detach().cpu()) for key, value in losses.items()}
    epoch_losses["gradient_norm"] = float(grad_norm.detach().cpu())
    epoch_losses["mean_return"] = float(
        np.mean([t.total_return for t in replay.trajectories])
    )
    epoch_losses["episodes"] = float(len(replay.trajectories))
    return epoch_losses, replay


def evaluate_block_smb(
    model: AgentWorldModelCritic,
    config: BlockSMBTrainingConfig,
    *,
    device: torch.device,
    vision_factory: Callable[[], VisionEncoder] = BlockVisionTransformer,
    record_dir: Optional[Path] = None,
) -> dict[str, Any]:
    model.eval()
    fixed = load_fixed_scenarios(config.fixed_scenarios)
    results = {}
    returns = []
    successes = []
    with torch.no_grad():
        for scenario_index, (scenario_name, scenario) in enumerate(fixed):
            scenario_returns = []
            scenario_successes = []
            for episode in range(config.evaluation_episodes):
                stage = BlockSMBStage(scenario=scenario, vision=vision_factory())
                try:
                    trajectory = collect_trajectory(
                        model,
                        stage,
                        scenario_name,
                        rollout_steps=config.evaluation_max_steps,
                        seed=config.seed + 1_000_000 + scenario_index * 100 + episode,
                        deterministic=True,
                        device=device,
                        record_frames=record_dir is not None,
                    )
                finally:
                    stage.env.close()
                scenario_returns.append(trajectory.total_return)
                scenario_successes.append(float(trajectory.success))
                if record_dir is not None:
                    record_dir.mkdir(parents=True, exist_ok=True)
                    frames = (
                        np.stack(trajectory.frames)
                        if trajectory.frames
                        else np.empty((0,))
                    )
                    np.savez_compressed(
                        record_dir / f"{scenario_name}_episode{episode}.npz",
                        frames=frames,
                        actions=np.array(
                            [step.action for step in trajectory.transitions], dtype=np.int64
                        ),
                        rewards=np.array(
                            [step.reward for step in trajectory.transitions],
                            dtype=np.float32,
                        ),
                    )
            mean_return = float(np.mean(scenario_returns))
            success_rate = float(np.mean(scenario_successes))
            results[scenario_name] = {
                "return": mean_return,
                "success_rate": success_rate,
            }
            returns.extend(scenario_returns)
            successes.extend(scenario_successes)
    return {
        "fixed_scenarios": results,
        "mean_return": float(np.mean(returns)) if returns else 0.0,
        "success_rate": float(np.mean(successes)) if successes else 0.0,
    }


def save_block_smb_checkpoint(
    path: Path,
    model: AgentWorldModelCritic,
    optimizer: optim.Optimizer,
    *,
    epoch: int,
    global_step: int,
    config: BlockSMBTrainingConfig,
    metrics: Mapping[str, float],
) -> dict[str, Any]:
    checkpoint = build_checkpoint(
        stage=BLOCK_SMB_SPEC.name,
        model_name=BLOCK_SMB_MODEL_NAME,
        checkpoint_kind=BLOCK_SMB_CHECKPOINT_KIND,
        epoch=epoch,
        global_step=global_step,
        metrics=metrics,
        config=to_plain_data(config),
        specs={
            "stage": {
                "name": BLOCK_SMB_SPEC.name,
                "seq_len_a": BLOCK_SMB_SPEC.seq_len_a,
                "seq_len_b": BLOCK_SMB_SPEC.seq_len_b,
                "seq_len_c": BLOCK_SMB_SPEC.seq_len_c,
                "ratio_bc": BLOCK_SMB_SPEC.ratio_bc,
                "vocab_size": BLOCK_SMB_SPEC.vocab_size,
            }
        },
        states={
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "torch_rng": torch.get_rng_state(),
            "python_rng": random.getstate(),
            "numpy_rng": np.random.get_state(),
        },
    )
    save_checkpoint(path, checkpoint)
    return checkpoint


def restore_block_smb_checkpoint(
    path: Path,
    model: AgentWorldModelCritic,
    optimizer: Optional[optim.Optimizer] = None,
    *,
    map_location: Any = "cpu",
) -> dict[str, Any]:
    checkpoint = load_checkpoint(path, map_location=map_location)
    if checkpoint["stage"] != BLOCK_SMB_SPEC.name:
        raise ValueError("checkpoint stage does not match block_smb")
    if checkpoint["model_name"] != BLOCK_SMB_MODEL_NAME:
        raise ValueError("checkpoint model does not match Block SMB trainer")
    if checkpoint["checkpoint_kind"] != BLOCK_SMB_CHECKPOINT_KIND:
        raise ValueError("checkpoint kind does not match Block SMB trainer")
    states = checkpoint["states"]
    model.load_state_dict(states["model"])
    if optimizer is not None:
        optimizer.load_state_dict(states["optimizer"])
    if "torch_rng" in states:
        torch.set_rng_state(states["torch_rng"].cpu())
    if "python_rng" in states:
        random.setstate(states["python_rng"])
    if "numpy_rng" in states:
        np.random.set_state(states["numpy_rng"])
    return checkpoint


def train_and_evaluate_block_smb(
    config: Optional[BlockSMBTrainingConfig] = None,
    *,
    vision_factory: Callable[[], VisionEncoder] = BlockVisionTransformer,
) -> dict[str, Any]:
    config = config or BlockSMBTrainingConfig()
    seed_everything(config.seed, config.deterministic)
    device = select_device(config.device)
    model = make_block_smb_model(config).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate)
    start_epoch = 0
    global_step = 0
    if config.resume_path is not None:
        checkpoint = restore_block_smb_checkpoint(
            config.resume_path, model, optimizer, map_location=device
        )
        start_epoch = int(checkpoint["epoch"])
        global_step = int(checkpoint["global_step"])
    curriculum = build_curriculum(config)
    vector_env = SequentialBlockSMBVectorEnv(curriculum, num_envs=config.num_envs)
    vector_env.close()
    history: list[dict[str, float]] = []
    last_metrics: dict[str, float] = {}
    for epoch in range(start_epoch, config.epochs):
        losses, _replay = train_block_smb_epoch(
            model,
            optimizer,
            curriculum,
            config,
            epoch,
            device=device,
            vision_factory=vision_factory,
        )
        global_step += int(losses["episodes"])
        evaluation = evaluate_block_smb(
            model,
            config,
            device=device,
            vision_factory=vision_factory,
            record_dir=config.video_dir if config.record_videos else None,
        )
        last_metrics = {
            **losses,
            "eval_mean_return": float(evaluation["mean_return"]),
            "eval_success_rate": float(evaluation["success_rate"]),
        }
        history.append(last_metrics)
        if config.save_checkpoints and config.checkpoint_path is not None:
            save_block_smb_checkpoint(
                config.checkpoint_path,
                model,
                optimizer,
                epoch=epoch + 1,
                global_step=global_step,
                config=config,
                metrics=last_metrics,
            )
    evaluation = evaluate_block_smb(
        model,
        config,
        device=device,
        vision_factory=vision_factory,
        record_dir=config.video_dir if config.record_videos else None,
    )
    return {
        "history": history,
        "metrics": last_metrics,
        "evaluation": evaluation,
        "curriculum": [name for name, _scenario in curriculum],
        "model": model,
    }
