"""Scripted known-good policy for the fixed Block SMB scenarios."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Optional

import numpy as np
import pygame

from retroagi.core import build_checkpoint, load_checkpoint, save_checkpoint

from .adapter import BLOCK_SMB_SPEC
from .env import MarioScenarioEnv
from .success import evaluate_fixed_success_thresholds
from .train import load_fixed_scenarios

SCRIPTED_BLOCK_SMB_POLICY_NAME = "block_smb_scripted_known_good"
SCRIPTED_BLOCK_SMB_CHECKPOINT_KIND = "scripted_policy"
SCRIPTED_BLOCK_SMB_SEED = 20260622
SCRIPTED_BLOCK_SMB_EVALUATION_EPISODES = 3
SCRIPTED_BLOCK_SMB_EVALUATION_MAX_STEPS = 200


def fixed_scenario_action_scripts(
    max_steps: int = SCRIPTED_BLOCK_SMB_EVALUATION_MAX_STEPS,
) -> dict[str, list[int]]:
    """Return deterministic action scripts for the fixed scenarios.

    Actions use the shared Block SMB IDs:
    1 = RIGHT, 2 = RIGHT_JUMP.
    """
    right = [1] * max_steps
    scripts = {
        "level_1_flat.json": list(right),
        "level_2_gap.json": [1] * 10 + [2] * 17 + [1] * max_steps,
        "level_3_stairs.json": [2] * 8 + [1] * 2 + [2] * 6 + [1] * max_steps,
        "level_4_platforms.json": [1] * 8 + [2] * 16 + [1] * max_steps,
        "level_5_enemy_hop.json": [1] * 20 + [2] * 18 + [1] * max_steps,
        "level_6_enemy_patrol.json": (
            [1] * 12 + [2] * 18 + [1] * 18 + [2] * 18 + [1] * max_steps
        ),
        "level_7_moving_bridge.json": (
            [1] * 10 + [2] * 14 + [1] * 8 + [2] * 14 + [1] * max_steps
        ),
        "level_8_enemy_gap.json": (
            [1] * 10 + [2] * 17 + [1] * 8 + [2] * 18 + [1] * max_steps
        ),
        "level_9_enemy_stomp.json": [1] * 8 + [2] * 14 + [1] * max_steps,
    }
    return {
        scenario_name: actions[:max_steps]
        for scenario_name, actions in scripts.items()
    }


class BlockSMBScriptedPolicy:
    """Step-indexed action script policy for deterministic fixed scenarios."""

    def __init__(self, action_scripts: Optional[Mapping[str, list[int]]] = None):
        self.action_scripts = {
            scenario_name: list(actions)
            for scenario_name, actions in (
                action_scripts or fixed_scenario_action_scripts()
            ).items()
        }

    def action(self, scenario_name: str, step_index: int) -> int:
        script = self.action_scripts[scenario_name]
        if step_index < len(script):
            return int(script[step_index])
        return int(script[-1])


def _goal_reached(env: MarioScenarioEnv) -> bool:
    if env.goal is None:
        return False
    mario_rect = pygame.Rect(
        env.mario["x"], env.mario["y"], env.mario["w"], env.mario["h"]
    )
    return bool(mario_rect.colliderect(env.goal))


def evaluate_scripted_block_smb_policy(
    policy: BlockSMBScriptedPolicy,
    *,
    seed: int = SCRIPTED_BLOCK_SMB_SEED,
    evaluation_episodes: int = SCRIPTED_BLOCK_SMB_EVALUATION_EPISODES,
    evaluation_max_steps: int = SCRIPTED_BLOCK_SMB_EVALUATION_MAX_STEPS,
    record_dir: Optional[Path] = None,
) -> dict[str, Any]:
    """Evaluate a scripted policy and optionally write trajectory recordings."""
    fixed = load_fixed_scenarios(tuple(policy.action_scripts.keys()))
    results: dict[str, dict[str, Any]] = {}
    returns = []
    successes = []
    for scenario_index, (scenario_name, scenario) in enumerate(fixed):
        scenario_returns = []
        scenario_successes = []
        for episode in range(evaluation_episodes):
            env = MarioScenarioEnv()
            frames = []
            actions = []
            rewards = []
            try:
                observation, _info = env.reset(
                    scenario=scenario,
                    seed=seed + scenario_index * 100 + episode,
                )
                frames.append(np.asarray(observation).copy())
                total_return = 0.0
                success = False
                for step_index in range(evaluation_max_steps):
                    action = policy.action(scenario_name, step_index)
                    observation, reward, terminated, truncated, _info = env.step(action)
                    total_return += float(reward)
                    frames.append(np.asarray(observation).copy())
                    actions.append(action)
                    rewards.append(float(reward))
                    if terminated or truncated:
                        success = _goal_reached(env)
                        break
                scenario_returns.append(total_return)
                scenario_successes.append(float(success))
                if record_dir is not None:
                    record_dir.mkdir(parents=True, exist_ok=True)
                    np.savez_compressed(
                        record_dir / f"{scenario_name}_episode{episode}.npz",
                        frames=np.stack(frames),
                        actions=np.asarray(actions, dtype=np.int64),
                        rewards=np.asarray(rewards, dtype=np.float32),
                    )
            finally:
                env.close()
        results[scenario_name] = {
            "return": float(np.mean(scenario_returns)),
            "success_rate": float(np.mean(scenario_successes)),
        }
        returns.extend(scenario_returns)
        successes.extend(scenario_successes)

    threshold_results = evaluate_fixed_success_thresholds(
        results,
        evaluation_episodes=evaluation_episodes,
        evaluation_max_steps=evaluation_max_steps,
    )
    for scenario_name, threshold_result in threshold_results.items():
        results[scenario_name]["threshold"] = threshold_result["threshold"]
        results[scenario_name]["threshold_met"] = threshold_result["threshold_met"]
        results[scenario_name]["threshold_diagnostics"] = {
            key: value
            for key, value in threshold_result.items()
            if key not in {"threshold", "threshold_met"}
        }
    return {
        "fixed_scenarios": results,
        "mean_return": float(np.mean(returns)) if returns else 0.0,
        "success_rate": float(np.mean(successes)) if successes else 0.0,
        "success_thresholds_met": all(
            threshold_result["threshold_met"]
            for threshold_result in threshold_results.values()
        )
        if threshold_results
        else False,
    }


def save_scripted_block_smb_checkpoint(
    path: Path,
    *,
    policy: BlockSMBScriptedPolicy,
    seed: int,
    evaluation: Mapping[str, Any],
    evaluation_episodes: int,
    evaluation_max_steps: int,
    record_dir: Optional[Path],
) -> dict[str, Any]:
    """Save the scripted known-good policy in the shared checkpoint envelope."""
    checkpoint = build_checkpoint(
        stage=BLOCK_SMB_SPEC.name,
        model_name=SCRIPTED_BLOCK_SMB_POLICY_NAME,
        checkpoint_kind=SCRIPTED_BLOCK_SMB_CHECKPOINT_KIND,
        epoch=0,
        global_step=0,
        metrics={
            "mean_return": float(evaluation["mean_return"]),
            "success_rate": float(evaluation["success_rate"]),
            "success_thresholds_met": float(evaluation["success_thresholds_met"]),
        },
        config={
            "seed": seed,
            "evaluation_episodes": evaluation_episodes,
            "evaluation_max_steps": evaluation_max_steps,
            "record_dir": str(record_dir) if record_dir is not None else None,
        },
        specs={
            "stage": {
                "name": BLOCK_SMB_SPEC.name,
                "seq_len_a": BLOCK_SMB_SPEC.seq_len_a,
                "seq_len_b": BLOCK_SMB_SPEC.seq_len_b,
                "seq_len_c": BLOCK_SMB_SPEC.seq_len_c,
                "vocab_size": BLOCK_SMB_SPEC.vocab_size,
            }
        },
        states={"action_scripts": policy.action_scripts},
        metadata={"policy_type": "scripted_known_good"},
    )
    save_checkpoint(path, checkpoint)
    return checkpoint


def load_scripted_block_smb_checkpoint(path: Path) -> BlockSMBScriptedPolicy:
    """Load a scripted known-good policy checkpoint."""
    checkpoint = load_checkpoint(path)
    if checkpoint["stage"] != BLOCK_SMB_SPEC.name:
        raise ValueError("checkpoint stage does not match block_smb")
    if checkpoint["model_name"] != SCRIPTED_BLOCK_SMB_POLICY_NAME:
        raise ValueError("checkpoint model does not match scripted Block SMB policy")
    if checkpoint["checkpoint_kind"] != SCRIPTED_BLOCK_SMB_CHECKPOINT_KIND:
        raise ValueError("checkpoint kind does not match scripted Block SMB policy")
    return BlockSMBScriptedPolicy(checkpoint["states"]["action_scripts"])
