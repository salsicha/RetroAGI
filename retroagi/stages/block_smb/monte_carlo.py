"""Versioned Monte Carlo scenario families for Block SMB."""

from __future__ import annotations

import copy
import hashlib
import random
from collections import Counter
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable, Mapping, Optional

import numpy as np
import pygame

from .env import MarioScenarioEnv

BLOCK_SMB_MC_SCHEMA_VERSION = "block_smb_monte_carlo.v1"
DEFAULT_BLOCK_SMB_MC_DISTRIBUTION_ID = "block_smb_mc_v1"
BLOCK_SMB_MC_SPLITS = ("train", "validation", "test", "stress")
BLOCK_SMB_MC_FAMILIES = (
    "flat_run",
    "single_gap",
    "stair_climb",
    "platform_chain",
    "moving_bridge",
    "enemy_hop",
    "enemy_patrol",
    "enemy_gap",
    "enemy_stomp",
    "retreat_recovery",
    "wait_timing",
    "mixed_section",
)
DEFAULT_BLOCK_SMB_MC_MAX_STEPS = 200


@dataclass(frozen=True)
class BlockSMBScenarioFamilySpec:
    """Schema entry describing one parameterized Block SMB family."""

    schema_version: str
    distribution_id: str
    family: str
    parameter_schema: Mapping[str, Any]
    constraints: Mapping[str, Any]
    oracle: Mapping[str, Any]

    def __post_init__(self) -> None:
        if self.schema_version != BLOCK_SMB_MC_SCHEMA_VERSION:
            raise ValueError("unsupported Block SMB Monte Carlo schema_version")
        if not self.distribution_id:
            raise ValueError("distribution_id must be non-empty")
        if self.family not in BLOCK_SMB_MC_FAMILIES:
            raise ValueError(f"unknown Block SMB Monte Carlo family {self.family!r}")
        if not isinstance(self.parameter_schema, Mapping):
            raise TypeError("parameter_schema must be a mapping")
        if not isinstance(self.constraints, Mapping):
            raise TypeError("constraints must be a mapping")
        if not isinstance(self.oracle, Mapping):
            raise TypeError("oracle must be a mapping")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class BlockSMBScenarioSample:
    """One deterministic sampled scenario plus replay metadata."""

    schema_version: str
    distribution_id: str
    family: str
    split: str
    seed: int
    sample_seed: int
    sample_index: int
    scenario_id: str
    parameters: Mapping[str, Any]
    constraints: Mapping[str, Any]
    oracle: Mapping[str, Any]
    reachability: Mapping[str, Any]
    scenario: Mapping[str, Any]

    def __post_init__(self) -> None:
        if self.schema_version != BLOCK_SMB_MC_SCHEMA_VERSION:
            raise ValueError("unsupported Block SMB Monte Carlo schema_version")
        if self.family not in BLOCK_SMB_MC_FAMILIES:
            raise ValueError(f"unknown Block SMB Monte Carlo family {self.family!r}")
        if self.split not in BLOCK_SMB_MC_SPLITS:
            raise ValueError(f"split must be one of {BLOCK_SMB_MC_SPLITS}")
        if self.sample_index < 0:
            raise ValueError("sample_index must be non-negative")
        if not self.scenario_id:
            raise ValueError("scenario_id must be non-empty")

    @property
    def difficulty_bin(self) -> str:
        return str(self.parameters.get("difficulty_bin", "default"))

    def metadata(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "distribution_id": self.distribution_id,
            "family": self.family,
            "split": self.split,
            "seed": self.seed,
            "sample_seed": self.sample_seed,
            "sample_index": self.sample_index,
            "scenario_id": self.scenario_id,
            "parameters": dict(self.parameters),
            "constraints": dict(self.constraints),
            "oracle": dict(self.oracle),
            "reachability": dict(self.reachability),
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self.metadata(), "scenario": copy.deepcopy(dict(self.scenario))}


@dataclass(frozen=True)
class BlockSMBMonteCarloSampleSet:
    """A deterministic split manifest and its sampled scenarios."""

    schema_version: str
    distribution_id: str
    split: str
    seed: int
    samples: tuple[BlockSMBScenarioSample, ...] = field(default_factory=tuple)
    rejected_counts: Mapping[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.schema_version != BLOCK_SMB_MC_SCHEMA_VERSION:
            raise ValueError("unsupported Block SMB Monte Carlo schema_version")
        if self.split not in BLOCK_SMB_MC_SPLITS:
            raise ValueError(f"split must be one of {BLOCK_SMB_MC_SPLITS}")
        if any(sample.split != self.split for sample in self.samples):
            raise ValueError("all samples must belong to the sample-set split")

    @property
    def sample_count(self) -> int:
        return len(self.samples)

    def scenarios(self) -> list[tuple[str, dict]]:
        return [
            (sample.scenario_id, copy.deepcopy(dict(sample.scenario)))
            for sample in self.samples
        ]

    def manifest(self, *, include_scenarios: bool = False) -> dict[str, Any]:
        sample_records = [
            sample.to_dict() if include_scenarios else sample.metadata()
            for sample in self.samples
        ]
        return {
            "schema_version": self.schema_version,
            "distribution_id": self.distribution_id,
            "split": self.split,
            "seed": self.seed,
            "sample_count": self.sample_count,
            "samples": sample_records,
            "coverage": summarize_block_smb_monte_carlo_samples(self.samples),
            "rejected_counts": dict(self.rejected_counts),
            "rejected_sample_count": int(sum(self.rejected_counts.values())),
            "scenario_ids": [sample.scenario_id for sample in self.samples],
        }


def block_smb_monte_carlo_family_specs(
    distribution_id: str = DEFAULT_BLOCK_SMB_MC_DISTRIBUTION_ID,
) -> dict[str, BlockSMBScenarioFamilySpec]:
    """Return the supported family schema for a distribution version."""

    if distribution_id != DEFAULT_BLOCK_SMB_MC_DISTRIBUTION_ID:
        raise ValueError(f"unsupported Block SMB Monte Carlo distribution {distribution_id!r}")
    base_constraints = {
        "spawn_safe": True,
        "minimum_landing_width": 40,
        "max_gap_width": 56,
        "max_enemy_density": 2,
        "requires_oracle_reachability": True,
    }
    oracle = {
        "kind": "scripted_action_sequence",
        "max_steps": DEFAULT_BLOCK_SMB_MC_MAX_STEPS,
    }
    schemas = {
        "flat_run": {
            "world_width": [256, 320],
            "goal_distance": [200, 240],
            "coin_spacing": [80, 150],
        },
        "single_gap": {
            "gap_x": [92, 104],
            "gap_width": [42, 52],
            "landing_width": [96, 120],
        },
        "stair_climb": {
            "step_count": [3, 4],
            "step_width": [36, 44],
            "step_height": [28, 32],
        },
        "platform_chain": {
            "platform_count": [4, 5],
            "gap_spacing": [55, 75],
            "vertical_variance": [40, 100],
        },
        "moving_bridge": {
            "platform_speed": [0.4, 1.0],
            "travel_range": [36, 56],
            "gap_width": [28, 48],
        },
        "enemy_hop": {
            "enemy_x": [96, 112],
            "approach_distance": [70, 90],
        },
        "enemy_patrol": {
            "enemy_count": [2, 3],
            "patrol_width": [36, 52],
            "speed": [0.4, 0.8],
        },
        "enemy_gap": {
            "gap_width": [42, 52],
            "enemy_gap_offset": [18, 36],
        },
        "enemy_stomp": {
            "enemy_x": [104, 116],
            "stomp_window": [12, 18],
        },
        "retreat_recovery": {
            "start_x": [188, 208],
            "goal_x": [30, 90],
            "safe_fallback": [60, 120],
        },
        "wait_timing": {
            "wait_window": [12, 24],
            "moving_platform_phase": [0, 0],
            "jump_window": [14, 18],
        },
        "mixed_section": {
            "section_count": [2, 3],
            "families": ["single_gap", "enemy_hop", "moving_bridge"],
        },
    }
    return {
        family: BlockSMBScenarioFamilySpec(
            schema_version=BLOCK_SMB_MC_SCHEMA_VERSION,
            distribution_id=distribution_id,
            family=family,
            parameter_schema=schema,
            constraints={**base_constraints, "family": family},
            oracle=oracle,
        )
        for family, schema in schemas.items()
    }


def stable_block_smb_monte_carlo_seed(
    distribution_id: str,
    split: str,
    seed: int,
    sample_index: int,
    *,
    attempt: int = 0,
) -> int:
    """Return a stable 63-bit seed for the replay tuple."""

    if split not in BLOCK_SMB_MC_SPLITS:
        raise ValueError(f"split must be one of {BLOCK_SMB_MC_SPLITS}")
    key = f"{distribution_id}|{split}|{int(seed)}|{int(sample_index)}|{int(attempt)}"
    digest = hashlib.sha256(key.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") & ((1 << 63) - 1)


def sample_block_smb_monte_carlo_scenario(
    *,
    distribution_id: str = DEFAULT_BLOCK_SMB_MC_DISTRIBUTION_ID,
    split: str,
    seed: int,
    sample_index: int,
    family: Optional[str] = None,
    family_weights: Optional[Mapping[str, float]] = None,
    validate_reachability: bool = True,
    max_rejections: int = 32,
) -> BlockSMBScenarioSample:
    """Sample one replayable scenario from the versioned distribution."""

    if split not in BLOCK_SMB_MC_SPLITS:
        raise ValueError(f"split must be one of {BLOCK_SMB_MC_SPLITS}")
    if sample_index < 0:
        raise ValueError("sample_index must be non-negative")
    if max_rejections < 0:
        raise ValueError("max_rejections must be non-negative")
    specs = block_smb_monte_carlo_family_specs(distribution_id)
    rejected: Counter[str] = Counter()
    for attempt in range(max_rejections + 1):
        sample_seed = stable_block_smb_monte_carlo_seed(
            distribution_id,
            split,
            seed,
            sample_index,
            attempt=attempt,
        )
        rng = random.Random(sample_seed)
        selected_family = family or _select_family(sample_index, rng, family_weights)
        if selected_family not in specs:
            raise ValueError(f"unknown Block SMB Monte Carlo family {selected_family!r}")
        scenario, parameters, actions = _generate_family_scenario(
            selected_family,
            rng,
            split=split,
        )
        constraints = specs[selected_family].constraints
        scenario_id = _scenario_id(
            distribution_id,
            split,
            seed,
            sample_index,
            selected_family,
        )
        reachability = (
            validate_block_smb_monte_carlo_oracle(
                scenario,
                actions,
                max_steps=DEFAULT_BLOCK_SMB_MC_MAX_STEPS,
            )
            if validate_reachability
            else {"reachable": True, "validation_skipped": True}
        )
        oracle = {
            "kind": "scripted_action_sequence",
            "actions": list(actions[:DEFAULT_BLOCK_SMB_MC_MAX_STEPS]),
            "action_source": f"{distribution_id}:{selected_family}:oracle_v1",
            "expected_completion_steps": reachability.get("completion_steps"),
            "expected_min_progress": reachability.get("max_progress"),
        }
        sample = BlockSMBScenarioSample(
            schema_version=BLOCK_SMB_MC_SCHEMA_VERSION,
            distribution_id=distribution_id,
            family=selected_family,
            split=split,
            seed=int(seed),
            sample_seed=sample_seed,
            sample_index=int(sample_index),
            scenario_id=scenario_id,
            parameters=parameters,
            constraints=constraints,
            oracle=oracle,
            reachability=reachability,
            scenario=_with_sample_metadata(
                scenario,
                schema_version=BLOCK_SMB_MC_SCHEMA_VERSION,
                distribution_id=distribution_id,
                family=selected_family,
                split=split,
                seed=seed,
                sample_seed=sample_seed,
                sample_index=sample_index,
                scenario_id=scenario_id,
                parameters=parameters,
                constraints=constraints,
                oracle=oracle,
                reachability=reachability,
            ),
        )
        if bool(reachability.get("reachable", False)):
            return sample
        rejected[str(reachability.get("rejection_reason", "unreachable"))] += 1
    reasons = ", ".join(f"{key}={value}" for key, value in sorted(rejected.items()))
    raise ValueError(
        "failed to sample a reachable Block SMB Monte Carlo scenario "
        f"for {distribution_id}/{split}/{seed}/{sample_index}; rejected {reasons}"
    )


def sample_block_smb_monte_carlo_split(
    *,
    distribution_id: str = DEFAULT_BLOCK_SMB_MC_DISTRIBUTION_ID,
    split: str,
    seed: int,
    sample_count: int,
    family_weights: Optional[Mapping[str, float]] = None,
    validate_reachability: bool = True,
    max_rejections: int = 32,
) -> BlockSMBMonteCarloSampleSet:
    """Sample a deterministic split manifest."""

    if sample_count < 0:
        raise ValueError("sample_count must be non-negative")
    samples: list[BlockSMBScenarioSample] = []
    rejected_counts: Counter[str] = Counter()
    for sample_index in range(sample_count):
        before = len(samples)
        try:
            sample = sample_block_smb_monte_carlo_scenario(
                distribution_id=distribution_id,
                split=split,
                seed=seed,
                sample_index=sample_index,
                family_weights=family_weights,
                validate_reachability=validate_reachability,
                max_rejections=max_rejections,
            )
        except ValueError as exc:
            rejected_counts[str(exc)] += 1
            raise
        samples.append(sample)
        if len(samples) == before:
            rejected_counts["unknown"] += 1
    return BlockSMBMonteCarloSampleSet(
        schema_version=BLOCK_SMB_MC_SCHEMA_VERSION,
        distribution_id=distribution_id,
        split=split,
        seed=int(seed),
        samples=tuple(samples),
        rejected_counts=dict(rejected_counts),
    )


def validate_block_smb_monte_carlo_oracle(
    scenario: Mapping[str, Any],
    actions: Iterable[int],
    *,
    max_steps: int = DEFAULT_BLOCK_SMB_MC_MAX_STEPS,
) -> dict[str, Any]:
    """Run the scripted oracle and return reachability diagnostics."""

    env = MarioScenarioEnv()
    total_return = 0.0
    completion_steps: int | None = None
    last_info: Mapping[str, Any] = {}
    try:
        env.reset(scenario=copy.deepcopy(dict(scenario)), seed=0)
        for step_index, action in enumerate(list(actions)[:max_steps]):
            _observation, reward, terminated, truncated, info = env.step(int(action))
            last_info = info
            total_return += float(reward)
            if terminated or truncated:
                completion_steps = step_index + 1
                break
        reachable = _goal_reached(env)
        if not reachable and completion_steps is None:
            completion_steps = max_steps
        max_progress = float(last_info.get("max_x_reached", env._max_x_reached))
        reason = None if reachable else _oracle_rejection_reason(env, last_info)
        return {
            "reachable": bool(reachable),
            "completion_steps": completion_steps,
            "max_steps": int(max_steps),
            "total_return": float(total_return),
            "max_progress": max_progress,
            "rejection_reason": reason,
        }
    finally:
        env.close()


def block_smb_monte_carlo_metadata(scenario: Mapping[str, Any]) -> dict[str, Any]:
    """Return Monte Carlo metadata from a scenario dict, if present."""

    metadata = scenario.get("metadata") if isinstance(scenario, Mapping) else None
    if not isinstance(metadata, Mapping):
        return {}
    value = metadata.get("block_smb_monte_carlo")
    return dict(value) if isinstance(value, Mapping) else {}


def block_smb_monte_carlo_oracle_actions(
    scenario: Mapping[str, Any],
    *,
    max_steps: int = DEFAULT_BLOCK_SMB_MC_MAX_STEPS,
) -> list[int]:
    """Return padded oracle actions stored in a sampled scenario."""

    metadata = block_smb_monte_carlo_metadata(scenario)
    oracle = metadata.get("oracle") if isinstance(metadata, Mapping) else None
    actions = oracle.get("actions") if isinstance(oracle, Mapping) else None
    if not isinstance(actions, list) or not actions:
        raise ValueError("scenario does not contain Monte Carlo oracle actions")
    parsed = [int(action) for action in actions]
    if len(parsed) < max_steps:
        parsed.extend([parsed[-1]] * (max_steps - len(parsed)))
    return parsed[:max_steps]


def summarize_block_smb_monte_carlo_samples(
    samples: Iterable[BlockSMBScenarioSample | Mapping[str, Any]],
) -> dict[str, Any]:
    """Return coverage histograms for sampled scenario metadata."""

    family_counts: Counter[str] = Counter()
    split_counts: Counter[str] = Counter()
    bin_counts: Counter[str] = Counter()
    scenario_ids: list[str] = []
    for sample in samples:
        metadata: Mapping[str, Any]
        if isinstance(sample, BlockSMBScenarioSample):
            metadata = sample.metadata()
        else:
            metadata = block_smb_monte_carlo_metadata(sample)
            if not metadata:
                metadata = sample
        family = str(metadata.get("family", "unknown"))
        split = str(metadata.get("split", "unknown"))
        params = metadata.get("parameters", {})
        difficulty_bin = (
            str(params.get("difficulty_bin", "default"))
            if isinstance(params, Mapping)
            else "default"
        )
        scenario_id = str(metadata.get("scenario_id", ""))
        family_counts[family] += 1
        split_counts[split] += 1
        bin_counts[f"{family}:{difficulty_bin}"] += 1
        if scenario_id:
            scenario_ids.append(scenario_id)
    expected = set(BLOCK_SMB_MC_FAMILIES)
    present = set(family_counts)
    return {
        "family_counts": dict(sorted(family_counts.items())),
        "split_counts": dict(sorted(split_counts.items())),
        "difficulty_bin_counts": dict(sorted(bin_counts.items())),
        "missing_families": sorted(expected - present),
        "scenario_ids": scenario_ids,
    }


def summarize_block_smb_monte_carlo_action_counts(
    actions: Iterable[int],
) -> dict[str, int]:
    counts: Counter[str] = Counter(str(int(action)) for action in actions)
    return {str(index): int(counts.get(str(index), 0)) for index in range(6)}


def evaluate_block_smb_monte_carlo_gates(
    evaluation: Mapping[str, Any],
    *,
    pass_rate_gate: float,
    family_pass_rate_gate: float,
) -> dict[str, Any]:
    """Evaluate held-out Monte Carlo promotion gates."""

    pass_rate = float(evaluation.get("success_rate", 0.0))
    families = evaluation.get("families", {})
    family_results = families if isinstance(families, Mapping) else {}
    per_family = {
        family: float(result.get("success_rate", 0.0))
        for family, result in family_results.items()
        if isinstance(result, Mapping)
    }
    missing_families = list(
        evaluation.get("coverage", {}).get("missing_families", [])
        if isinstance(evaluation.get("coverage"), Mapping)
        else []
    )
    pass_rate_met = pass_rate >= float(pass_rate_gate)
    per_family_met = all(
        value >= float(family_pass_rate_gate) for value in per_family.values()
    ) and bool(per_family)
    coverage_met = not missing_families
    return {
        "pass_rate": pass_rate,
        "pass_rate_gate": float(pass_rate_gate),
        "pass_rate_gate_met": bool(pass_rate_met),
        "family_pass_rates": per_family,
        "family_pass_rate_gate": float(family_pass_rate_gate),
        "family_pass_rate_gate_met": bool(per_family_met),
        "coverage_gate_met": bool(coverage_met),
        "missing_families": missing_families,
        "gate_met": bool(pass_rate_met and per_family_met and coverage_met),
    }


def block_smb_transfer_gate_metrics_from_evaluation(
    evaluation: Mapping[str, Any],
) -> dict[str, Any]:
    """Return the combined Block SMB transfer-source gate summary."""

    tuning = evaluation.get("tuning_metrics", {})
    fixed_pass_rate = (
        float(tuning.get("threshold_pass_rate", 0.0))
        if isinstance(tuning, Mapping)
        else 0.0
    )
    fixed_gate_met = bool(evaluation.get("success_thresholds_met", False)) and (
        fixed_pass_rate >= 1.0
    )
    monte_carlo = evaluation.get("monte_carlo_validation", {})
    mc_gate_met = True
    mc_pass_rate = None
    if isinstance(monte_carlo, Mapping) and monte_carlo:
        gates = monte_carlo.get("gates", {})
        mc_gate_met = bool(gates.get("gate_met", False)) if isinstance(gates, Mapping) else False
        mc_pass_rate = float(monte_carlo.get("success_rate", 0.0))
    return {
        "fixed_threshold_pass_rate": fixed_pass_rate,
        "fixed_gate_met": bool(fixed_gate_met),
        "monte_carlo_validation_success_rate": mc_pass_rate,
        "monte_carlo_validation_gate_met": bool(mc_gate_met),
        "transfer_source_gate_met": bool(fixed_gate_met and mc_gate_met),
    }


def _scenario_id(
    distribution_id: str,
    split: str,
    seed: int,
    sample_index: int,
    family: str,
) -> str:
    return f"{distribution_id}.{split}.{int(seed)}.{int(sample_index):06d}.{family}"


def _select_family(
    sample_index: int,
    rng: random.Random,
    family_weights: Optional[Mapping[str, float]],
) -> str:
    if not family_weights:
        return BLOCK_SMB_MC_FAMILIES[sample_index % len(BLOCK_SMB_MC_FAMILIES)]
    families = []
    weights = []
    for family in BLOCK_SMB_MC_FAMILIES:
        weight = float(family_weights.get(family, 0.0))
        if weight > 0:
            families.append(family)
            weights.append(weight)
    if not families:
        raise ValueError("family_weights must contain at least one positive weight")
    return str(rng.choices(families, weights=weights, k=1)[0])


def _with_sample_metadata(
    scenario: Mapping[str, Any],
    **metadata: Any,
) -> dict[str, Any]:
    enriched = copy.deepcopy(dict(scenario))
    existing = enriched.get("metadata", {})
    if not isinstance(existing, Mapping):
        existing = {}
    enriched["metadata"] = {
        **dict(existing),
        "block_smb_monte_carlo": {
            key: copy.deepcopy(value) for key, value in metadata.items()
        },
    }
    return enriched


def _goal_reached(env: MarioScenarioEnv) -> bool:
    if env.goal is None:
        return False
    mario_rect = pygame.Rect(
        env.mario["x"],
        env.mario["y"],
        env.mario["w"],
        env.mario["h"],
    )
    return bool(mario_rect.colliderect(env.goal))


def _oracle_rejection_reason(env: MarioScenarioEnv, info: Mapping[str, Any]) -> str:
    if env.mario["y"] > env.height:
        return "fall_death"
    terms = info.get("reward_terms", {}) if isinstance(info, Mapping) else {}
    if isinstance(terms, Mapping) and float(terms.get("enemy_hit", 0.0)) < 0:
        return "enemy_hit"
    if env.steps >= env.max_steps:
        return "timeout"
    return "goal_not_reached"


def _generate_family_scenario(
    family: str,
    rng: random.Random,
    *,
    split: str,
) -> tuple[dict[str, Any], dict[str, Any], list[int]]:
    difficulty = _difficulty_bin(rng, split)
    if family == "flat_run":
        return _flat_run(rng, difficulty)
    if family == "single_gap":
        return _single_gap(rng, difficulty)
    if family == "stair_climb":
        return _stair_climb(rng, difficulty)
    if family == "platform_chain":
        return _platform_chain(rng, difficulty)
    if family == "moving_bridge":
        return _moving_bridge(rng, difficulty)
    if family == "enemy_hop":
        return _enemy_hop(rng, difficulty)
    if family == "enemy_patrol":
        return _enemy_patrol(rng, difficulty)
    if family == "enemy_gap":
        return _enemy_gap(rng, difficulty)
    if family == "enemy_stomp":
        return _enemy_stomp(rng, difficulty)
    if family == "retreat_recovery":
        return _retreat_recovery(rng, difficulty)
    if family == "wait_timing":
        return _wait_timing(rng, difficulty)
    if family == "mixed_section":
        return _mixed_section(rng, difficulty)
    raise ValueError(f"unknown Block SMB Monte Carlo family {family!r}")


def _difficulty_bin(rng: random.Random, split: str) -> str:
    if split == "stress":
        return "hard"
    if split == "train":
        return rng.choices(("easy", "medium"), weights=(3, 1), k=1)[0]
    return rng.choices(("easy", "medium", "hard"), weights=(2, 2, 1), k=1)[0]


def _right_actions(max_steps: int = DEFAULT_BLOCK_SMB_MC_MAX_STEPS) -> list[int]:
    return [1] * max_steps


def _pad(actions: list[int], max_steps: int = DEFAULT_BLOCK_SMB_MC_MAX_STEPS) -> list[int]:
    if len(actions) >= max_steps:
        return actions[:max_steps]
    return actions + [actions[-1] if actions else 0] * (max_steps - len(actions))


def _flat_run(rng: random.Random, difficulty: str) -> tuple[dict[str, Any], dict[str, Any], list[int]]:
    goal_x = rng.randint(224, 232)
    coin_x = rng.randint(110, 150)
    scenario = {
        "world_width": 256,
        "mario": [20, 200],
        "platforms": [[0, 220, 256, 20]],
        "coins": [[coin_x, 200, 10, 10]],
        "goal": [goal_x, 200, 16, 20],
    }
    return scenario, {"goal_x": goal_x, "coin_x": coin_x, "difficulty_bin": difficulty}, _right_actions()


def _single_gap(rng: random.Random, difficulty: str) -> tuple[dict[str, Any], dict[str, Any], list[int]]:
    first_width = 100
    gap_width = {"easy": 44, "medium": 48, "hard": 50}[difficulty]
    landing_x = first_width + gap_width
    scenario = {
        "world_width": 256,
        "mario": [20, 200],
        "platforms": [[0, 220, first_width, 20], [landing_x, 220, 256 - landing_x, 20]],
        "coins": [[120, 160, 10, 10]],
        "goal": [220, 200, 16, 20],
    }
    actions = _pad([1] * 10 + [2] * 17 + [1])
    return scenario, {"gap_x": first_width, "gap_width": gap_width, "difficulty_bin": difficulty}, actions


def _stair_climb(rng: random.Random, difficulty: str) -> tuple[dict[str, Any], dict[str, Any], list[int]]:
    step_height = {"easy": 28, "medium": 30, "hard": 30}[difficulty]
    scenario = {
        "world_width": 256,
        "mario": [20, 200],
        "platforms": [
            [0, 220, 60, 20],
            [60, 190, 40, 50],
            [100, 160, 40, 80],
            [140, 130, 116, 110],
        ],
        "coins": [[75, 170, 10, 10], [115, 140, 10, 10]],
        "goal": [220, 110, 16, 20],
    }
    actions = _pad([2] * 8 + [1] * 2 + [2] * 6 + [1])
    return scenario, {"step_count": 3, "step_height": step_height, "difficulty_bin": difficulty}, actions


def _platform_chain(rng: random.Random, difficulty: str) -> tuple[dict[str, Any], dict[str, Any], list[int]]:
    scenario = {
        "world_width": 256,
        "mario": [20, 100],
        "platforms": [
            [0, 120, 40, 10],
            [70, 160, 40, 10],
            [140, 120, 40, 10],
            [200, 220, 56, 20],
        ],
        "coins": [[85, 140, 10, 10], [155, 100, 10, 10]],
        "goal": [230, 200, 16, 20],
    }
    actions = _pad([1] * 8 + [2] * 16 + [1])
    return scenario, {"platform_count": 4, "difficulty_bin": difficulty}, actions


def _moving_bridge(rng: random.Random, difficulty: str) -> tuple[dict[str, Any], dict[str, Any], list[int]]:
    speed = {"easy": 0.5, "medium": 0.7, "hard": 0.9}[difficulty]
    scenario = {
        "world_width": 256,
        "mario": [20, 200],
        "platforms": [
            [0, 220, 90, 20],
            {"x": 106, "y": 220, "w": 50, "h": 20, "moving": [96, 146, speed]},
            [172, 220, 84, 20],
        ],
        "coins": [[128, 190, 10, 10]],
        "goal": [230, 200, 16, 20],
    }
    actions = _pad([1] * 10 + [2] * 14 + [1] * 8 + [2] * 14 + [1])
    return scenario, {"platform_speed": speed, "difficulty_bin": difficulty}, actions


def _enemy_hop(rng: random.Random, difficulty: str) -> tuple[dict[str, Any], dict[str, Any], list[int]]:
    enemy_x = {"easy": 104, "medium": 106, "hard": 108}[difficulty]
    scenario = {
        "world_width": 256,
        "mario": [20, 200],
        "platforms": [[0, 220, 256, 20]],
        "enemies": [[enemy_x, 206, enemy_x, enemy_x, 0]],
        "coins": [[145, 190, 10, 10]],
        "goal": [230, 200, 16, 20],
    }
    actions = _pad([1] * 20 + [2] * 18 + [1])
    return scenario, {"enemy_x": enemy_x, "difficulty_bin": difficulty}, actions


def _enemy_patrol(rng: random.Random, difficulty: str) -> tuple[dict[str, Any], dict[str, Any], list[int]]:
    speed = {"easy": 0.5, "medium": 0.6, "hard": 0.7}[difficulty]
    scenario = {
        "world_width": 256,
        "mario": [20, 200],
        "platforms": [[0, 220, 256, 20]],
        "enemies": [
            {"x": 100, "y": 206, "patrol_min": 84, "patrol_max": 130, "speed": speed},
            {"x": 170, "y": 206, "patrol_min": 150, "patrol_max": 190, "speed": speed},
        ],
        "coins": [[130, 185, 10, 10], [200, 185, 10, 10]],
        "goal": [230, 200, 16, 20],
    }
    actions = _pad([1] * 12 + [2] * 18 + [1] * 18 + [2] * 18 + [1])
    return scenario, {"enemy_count": 2, "enemy_speed": speed, "difficulty_bin": difficulty}, actions


def _enemy_gap(rng: random.Random, difficulty: str) -> tuple[dict[str, Any], dict[str, Any], list[int]]:
    gap_width = {"easy": 46, "medium": 48, "hard": 50}[difficulty]
    first_width = 100
    landing_x = first_width + gap_width
    scenario = {
        "world_width": 256,
        "mario": [20, 200],
        "platforms": [[0, 220, first_width, 20], [landing_x, 220, 256 - landing_x, 20]],
        "enemies": [[178, 206, 168, 206, 0.4]],
        "coins": [[120, 160, 10, 10], [205, 185, 10, 10]],
        "goal": [230, 200, 16, 20],
    }
    actions = _pad([1] * 10 + [2] * 17 + [1] * 8 + [2] * 18 + [1])
    return scenario, {"gap_width": gap_width, "enemy_gap_offset": 178 - landing_x, "difficulty_bin": difficulty}, actions


def _enemy_stomp(rng: random.Random, difficulty: str) -> tuple[dict[str, Any], dict[str, Any], list[int]]:
    enemy_x = {"easy": 108, "medium": 110, "hard": 112}[difficulty]
    scenario = {
        "world_width": 256,
        "mario": [20, 200],
        "platforms": [[0, 220, 256, 20]],
        "enemies": [[enemy_x, 206, enemy_x, enemy_x, 0]],
        "coins": [[150, 185, 10, 10]],
        "goal": [230, 200, 16, 20],
    }
    actions = _pad([1] * 8 + [2] * 14 + [1])
    return scenario, {"enemy_x": enemy_x, "stomp_window": 14, "difficulty_bin": difficulty}, actions


def _retreat_recovery(rng: random.Random, difficulty: str) -> tuple[dict[str, Any], dict[str, Any], list[int]]:
    start_x = {"easy": 190, "medium": 200, "hard": 205}[difficulty]
    scenario = {
        "world_width": 256,
        "mario": [start_x, 200],
        "platforms": [[0, 220, 256, 20]],
        "coins": [[120, 200, 10, 10]],
        "goal": [35, 200, 16, 20],
    }
    return scenario, {"start_x": start_x, "goal_x": 35, "difficulty_bin": difficulty}, _pad([3])


def _wait_timing(rng: random.Random, difficulty: str) -> tuple[dict[str, Any], dict[str, Any], list[int]]:
    wait = {"easy": 18, "medium": 20, "hard": 22}[difficulty]
    speed = {"easy": 0.8, "medium": 1.0, "hard": 1.0}[difficulty]
    scenario = {
        "world_width": 256,
        "mario": [20, 200],
        "platforms": [
            [0, 220, 85, 20],
            {"x": 130, "y": 220, "w": 50, "h": 20, "moving": [90, 130, speed]},
            [180, 220, 76, 20],
        ],
        "coins": [[130, 190, 10, 10]],
        "goal": [230, 200, 16, 20],
    }
    actions = _pad([0] * wait + [1] * 20 + [2] * 16 + [1])
    return scenario, {"wait_window": wait, "platform_speed": speed, "difficulty_bin": difficulty}, actions


def _mixed_section(rng: random.Random, difficulty: str) -> tuple[dict[str, Any], dict[str, Any], list[int]]:
    # Conservative first version: a combined enemy-gap timing task with the
    # mixed-section family label. Harder multi-screen compositions should be
    # introduced under a new distribution ID.
    scenario, params, actions = _enemy_gap(rng, difficulty)
    params = {
        **params,
        "section_count": 2,
        "families": ["single_gap", "enemy_hop"],
        "difficulty_bin": difficulty,
    }
    return scenario, params, actions
