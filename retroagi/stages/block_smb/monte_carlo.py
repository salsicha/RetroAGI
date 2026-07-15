"""Versioned Monte Carlo scenario families for Block SMB."""

from __future__ import annotations

import copy
import hashlib
import random
from collections import Counter
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable, Mapping, Optional

import pygame

from .env import MarioScenarioEnv

BLOCK_SMB_MC_SCHEMA_VERSION = "block_smb_monte_carlo.v1"
DEFAULT_BLOCK_SMB_MC_DISTRIBUTION_ID = "block_smb_mc_v1"
BLOCK_SMB_MC_SPLITS = ("train", "validation", "test", "stress")
BLOCK_SMB_MC_DIFFICULTY_BINS = ("easy", "medium", "hard")
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
    "chained_obstacles",
    "chained_enemy_gauntlet",
    "full_smb_opening_proxy",
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
        "chained_obstacles": {
            "section_count": [3, 4],
            "world_width": [480, 544],
            "enemy_count": [1, 2],
            "pipe_count": [2, 2],
            "pipe_height": [38, 54],
        },
        "chained_enemy_gauntlet": {
            "section_count": [4, 5],
            "world_width": [512, 576],
            "enemy_count": [2, 3],
            "gap_count": [1, 1],
            "pipe_count": [1, 2],
        },
        "full_smb_opening_proxy": {
            "section_count": [4, 5],
            "world_width": [512, 576],
            "enemy_count": [2, 2],
            "pipe_count": [2, 3],
            "pipe_height": [38, 58],
        },
        "mixed_section": {
            "section_count": [4, 5],
            "families": [
                "enemy_hop",
                "single_gap",
                "enemy_patrol",
                "pipe_jump",
            ],
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
    difficulty: Optional[str] = None,
    family_weights: Optional[Mapping[str, float]] = None,
    validate_reachability: bool = True,
    max_rejections: int = 32,
    rejection_counter: Optional[Counter[str]] = None,
) -> BlockSMBScenarioSample:
    """Sample one replayable scenario from the versioned distribution.

    When ``rejection_counter`` is provided, every rejected attempt (including
    attempts preceding an eventual success) is tallied into it by reason.
    """

    if split not in BLOCK_SMB_MC_SPLITS:
        raise ValueError(f"split must be one of {BLOCK_SMB_MC_SPLITS}")
    if sample_index < 0:
        raise ValueError("sample_index must be non-negative")
    if difficulty is not None and difficulty not in BLOCK_SMB_MC_DIFFICULTY_BINS:
        raise ValueError(f"difficulty must be one of {BLOCK_SMB_MC_DIFFICULTY_BINS}")
    if max_rejections < 0:
        raise ValueError("max_rejections must be non-negative")
    specs = block_smb_monte_carlo_family_specs(distribution_id)
    rejected: Counter[str] = Counter()
    rejected_fingerprints: set[str] = set()
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
            difficulty=difficulty,
        )
        fingerprint = repr((selected_family, scenario, actions))
        if fingerprint in rejected_fingerprints:
            # Fail fast: a retry regenerated an already-rejected scenario, so
            # further identical attempts can never rescue this sample.
            rejected["duplicate_regeneration"] += 1
            break
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
            if rejection_counter is not None:
                rejection_counter.update(rejected)
            return sample
        rejected[str(reachability.get("rejection_reason", "unreachable"))] += 1
        rejected_fingerprints.add(fingerprint)
    if rejection_counter is not None:
        rejection_counter.update(rejected)
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
        sample = sample_block_smb_monte_carlo_scenario(
            distribution_id=distribution_id,
            split=split,
            seed=seed,
            sample_index=sample_index,
            family_weights=family_weights,
            validate_reachability=validate_reachability,
            max_rejections=max_rejections,
            rejection_counter=rejected_counts,
        )
        samples.append(sample)
    return BlockSMBMonteCarloSampleSet(
        schema_version=BLOCK_SMB_MC_SCHEMA_VERSION,
        distribution_id=distribution_id,
        split=split,
        seed=int(seed),
        samples=tuple(samples),
        rejected_counts=dict(rejected_counts),
    )


def sample_block_smb_monte_carlo_parameter_sweep(
    *,
    distribution_id: str = DEFAULT_BLOCK_SMB_MC_DISTRIBUTION_ID,
    split: str,
    seed: int,
    repeats_per_difficulty: int = 1,
    families: Optional[Iterable[str]] = None,
    validate_reachability: bool = True,
    max_rejections: int = 32,
) -> BlockSMBMonteCarloSampleSet:
    """Return a deterministic family x difficulty Monte Carlo sweep."""

    if split not in BLOCK_SMB_MC_SPLITS:
        raise ValueError(f"split must be one of {BLOCK_SMB_MC_SPLITS}")
    if repeats_per_difficulty <= 0:
        raise ValueError("repeats_per_difficulty must be positive")
    specs = block_smb_monte_carlo_family_specs(distribution_id)
    selected_families = tuple(str(family) for family in (families or BLOCK_SMB_MC_FAMILIES))
    if not selected_families:
        raise ValueError("families must be non-empty")
    unknown = sorted(set(selected_families) - set(specs))
    if unknown:
        choices = ", ".join(BLOCK_SMB_MC_FAMILIES)
        raise ValueError(f"unknown Block SMB Monte Carlo family {unknown!r}; expected {choices}")

    samples: list[BlockSMBScenarioSample] = []
    rejected_counts: Counter[str] = Counter()
    sample_index = 0
    for family in selected_families:
        for difficulty in BLOCK_SMB_MC_DIFFICULTY_BINS:
            for repeat in range(int(repeats_per_difficulty)):
                candidate = sample_block_smb_monte_carlo_scenario(
                    distribution_id=distribution_id,
                    split=split,
                    seed=seed,
                    sample_index=sample_index,
                    family=family,
                    difficulty=difficulty,
                    validate_reachability=validate_reachability,
                    max_rejections=max_rejections,
                    rejection_counter=rejected_counts,
                )
                sample = _with_sweep_metadata(candidate, repeat=repeat)
                samples.append(sample)
                sample_index += 1
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
        reason = None if reachable else _oracle_rejection_reason(env, last_info, max_steps)
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
    # Fail closed: without Monte Carlo validation evidence the gate is not met.
    mc_gate_met = False
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


def _with_sweep_metadata(
    sample: BlockSMBScenarioSample,
    *,
    repeat: int,
) -> BlockSMBScenarioSample:
    scenario_id = (
        f"{sample.scenario_id}.{sample.difficulty_bin}.sweep_r{int(repeat):02d}"
    )
    parameters = {
        **dict(sample.parameters),
        "parameter_sweep": True,
        "sweep_repeat": int(repeat),
    }
    constraints = {**dict(sample.constraints), "parameter_sweep": True}
    oracle = {
        **dict(sample.oracle),
        "action_source": f"{sample.distribution_id}:{sample.family}:sweep_oracle_v1",
    }
    reachability = dict(sample.reachability)
    scenario = _with_sample_metadata(
        sample.scenario,
        schema_version=sample.schema_version,
        distribution_id=sample.distribution_id,
        family=sample.family,
        split=sample.split,
        seed=sample.seed,
        sample_seed=sample.sample_seed,
        sample_index=sample.sample_index,
        scenario_id=scenario_id,
        parameters=parameters,
        constraints=constraints,
        oracle=oracle,
        reachability=reachability,
    )
    return BlockSMBScenarioSample(
        schema_version=sample.schema_version,
        distribution_id=sample.distribution_id,
        family=sample.family,
        split=sample.split,
        seed=sample.seed,
        sample_seed=sample.sample_seed,
        sample_index=sample.sample_index,
        scenario_id=scenario_id,
        parameters=parameters,
        constraints=constraints,
        oracle=oracle,
        reachability=reachability,
        scenario=scenario,
    )


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


def _oracle_rejection_reason(
    env: MarioScenarioEnv,
    info: Mapping[str, Any],
    max_steps: int,
) -> str:
    if env.mario["y"] > env.height:
        return "fall_death"
    terms = info.get("reward_terms", {}) if isinstance(info, Mapping) else {}
    if isinstance(terms, Mapping) and float(terms.get("enemy_hit", 0.0)) < 0:
        return "enemy_hit"
    # Compare against the oracle step budget, not the env's much larger cap.
    if env.steps >= max_steps:
        return "timeout"
    return "goal_not_reached"


def _generate_family_scenario(
    family: str,
    rng: random.Random,
    *,
    split: str,
    difficulty: Optional[str] = None,
) -> tuple[dict[str, Any], dict[str, Any], list[int]]:
    difficulty = difficulty or _difficulty_bin(rng, split)
    if difficulty not in BLOCK_SMB_MC_DIFFICULTY_BINS:
        raise ValueError(f"difficulty must be one of {BLOCK_SMB_MC_DIFFICULTY_BINS}")
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
    if family == "chained_obstacles":
        return _chained_obstacles(rng, difficulty)
    if family == "chained_enemy_gauntlet":
        return _chained_enemy_gauntlet(rng, difficulty)
    if family == "full_smb_opening_proxy":
        return _full_smb_opening_proxy(rng, difficulty)
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
    gap_width = {"easy": 44, "medium": 48, "hard": 50}[difficulty] + rng.randint(-2, 1)
    coin_x = rng.randint(112, 128)
    landing_x = first_width + gap_width
    scenario = {
        "world_width": 256,
        "mario": [20, 200],
        "platforms": [[0, 220, first_width, 20], [landing_x, 220, 256 - landing_x, 20]],
        "coins": [[coin_x, 160, 10, 10]],
        "goal": [220, 200, 16, 20],
    }
    actions = _pad([1] * 10 + [2] * 17 + [1])
    return scenario, {"gap_x": first_width, "gap_width": gap_width, "difficulty_bin": difficulty}, actions


def _stair_climb(rng: random.Random, difficulty: str) -> tuple[dict[str, Any], dict[str, Any], list[int]]:
    step_height = {"easy": 28, "medium": 30, "hard": 30}[difficulty]
    coin_a_x = rng.randint(70, 82)
    coin_b_x = rng.randint(110, 122)
    goal_x = rng.randint(216, 228)
    scenario = {
        "world_width": 256,
        "mario": [20, 200],
        "platforms": [
            [0, 220, 60, 20],
            [60, 190, 40, 50],
            [100, 160, 40, 80],
            [140, 130, 116, 110],
        ],
        "coins": [[coin_a_x, 170, 10, 10], [coin_b_x, 140, 10, 10]],
        "goal": [goal_x, 110, 16, 20],
    }
    actions = _pad([2] * 8 + [1] * 2 + [2] * 6 + [1])
    return (
        scenario,
        {"step_count": 3, "step_height": step_height, "goal_x": goal_x, "difficulty_bin": difficulty},
        actions,
    )


def _platform_chain(rng: random.Random, difficulty: str) -> tuple[dict[str, Any], dict[str, Any], list[int]]:
    coin_a_x = rng.randint(78, 96)
    coin_b_x = rng.randint(142, 168)
    goal_x = rng.randint(224, 234)
    scenario = {
        "world_width": 256,
        "mario": [20, 100],
        "platforms": [
            [0, 120, 40, 10],
            [70, 160, 40, 10],
            [140, 120, 40, 10],
            [200, 220, 56, 20],
        ],
        "coins": [[coin_a_x, 140, 10, 10], [coin_b_x, 100, 10, 10]],
        "goal": [goal_x, 200, 16, 20],
    }
    actions = _pad([1] * 8 + [2] * 16 + [1])
    return scenario, {"platform_count": 4, "goal_x": goal_x, "difficulty_bin": difficulty}, actions


def _moving_bridge(rng: random.Random, difficulty: str) -> tuple[dict[str, Any], dict[str, Any], list[int]]:
    speed_range = {"easy": (0.45, 0.55), "medium": (0.65, 0.75), "hard": (0.85, 0.92)}[difficulty]
    speed = round(rng.uniform(*speed_range), 3)
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
    enemy_x = {"easy": 104, "medium": 106, "hard": 108}[difficulty] + rng.randint(-2, 2)
    coin_x = rng.randint(140, 152)
    scenario = {
        "world_width": 256,
        "mario": [20, 200],
        "platforms": [[0, 220, 256, 20]],
        "enemies": [[enemy_x, 206, enemy_x, enemy_x, 0]],
        "coins": [[coin_x, 190, 10, 10]],
        "goal": [230, 200, 16, 20],
    }
    actions = _pad([1] * 20 + [2] * 18 + [1])
    return scenario, {"enemy_x": enemy_x, "difficulty_bin": difficulty}, actions


def _enemy_patrol(rng: random.Random, difficulty: str) -> tuple[dict[str, Any], dict[str, Any], list[int]]:
    speed = round(
        {"easy": 0.5, "medium": 0.6, "hard": 0.7}[difficulty] + rng.uniform(-0.05, 0.05), 3
    )
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
    gap_width = {"easy": 46, "medium": 48, "hard": 50}[difficulty] + rng.randint(-2, 1)
    enemy_speed = round(0.4 + rng.uniform(-0.05, 0.05), 3)
    first_width = 100
    landing_x = first_width + gap_width
    scenario = {
        "world_width": 256,
        "mario": [20, 200],
        "platforms": [[0, 220, first_width, 20], [landing_x, 220, 256 - landing_x, 20]],
        "enemies": [[178, 206, 168, 206, enemy_speed]],
        "coins": [[120, 160, 10, 10], [205, 185, 10, 10]],
        "goal": [230, 200, 16, 20],
    }
    actions = _pad([1] * 10 + [2] * 17 + [1] * 8 + [2] * 18 + [1])
    return scenario, {"gap_width": gap_width, "enemy_gap_offset": 178 - landing_x, "difficulty_bin": difficulty}, actions


def _enemy_stomp(rng: random.Random, difficulty: str) -> tuple[dict[str, Any], dict[str, Any], list[int]]:
    enemy_x = {"easy": 108, "medium": 110, "hard": 112}[difficulty] + rng.randint(-1, 1)
    coin_x = rng.randint(146, 156)
    scenario = {
        "world_width": 256,
        "mario": [20, 200],
        "platforms": [[0, 220, 256, 20]],
        "enemies": [[enemy_x, 206, enemy_x, enemy_x, 0]],
        "coins": [[coin_x, 185, 10, 10]],
        "goal": [230, 200, 16, 20],
    }
    actions = _pad([1] * 8 + [2] * 14 + [1])
    return scenario, {"enemy_x": enemy_x, "stomp_window": 14, "difficulty_bin": difficulty}, actions


def _retreat_recovery(rng: random.Random, difficulty: str) -> tuple[dict[str, Any], dict[str, Any], list[int]]:
    start_x = {"easy": 190, "medium": 200, "hard": 205}[difficulty] + rng.randint(-2, 3)
    coin_x = rng.randint(110, 130)
    scenario = {
        "world_width": 256,
        "mario": [start_x, 200],
        "platforms": [[0, 220, 256, 20]],
        "coins": [[coin_x, 200, 10, 10]],
        "goal": [35, 200, 16, 20],
    }
    return scenario, {"start_x": start_x, "goal_x": 35, "difficulty_bin": difficulty}, _pad([3])


def _wait_timing(rng: random.Random, difficulty: str) -> tuple[dict[str, Any], dict[str, Any], list[int]]:
    wait = {"easy": 18, "medium": 20, "hard": 22}[difficulty] + rng.randint(-1, 1)
    # The easy bin pairs a slow fractional bridge with a shorter travel range so
    # the bridge stays close enough to cross with the shared action script.
    if difficulty == "easy":
        speed = round(rng.uniform(0.78, 0.88), 3)
        move_min = 100
    else:
        speed = round(rng.uniform(0.95, 1.0), 3)
        move_min = 90
    scenario = {
        "world_width": 256,
        "mario": [20, 200],
        "platforms": [
            [0, 220, 85, 20],
            {"x": 130, "y": 220, "w": 50, "h": 20, "moving": [move_min, 130, speed]},
            [180, 220, 76, 20],
        ],
        "coins": [[130, 190, 10, 10]],
        "goal": [230, 200, 16, 20],
    }
    actions = _pad([0] * wait + [1] * 20 + [2] * 16 + [1])
    return (
        scenario,
        {
            "wait_window": wait,
            "platform_speed": speed,
            "platform_range": [move_min, 130],
            "difficulty_bin": difficulty,
        },
        actions,
    )


def _chained_obstacles(
    rng: random.Random,
    difficulty: str,
) -> tuple[dict[str, Any], dict[str, Any], list[int]]:
    enemy_x = {"easy": 94, "medium": 96, "hard": 98}[difficulty] + rng.randint(-2, 2)
    pipe_a_h = {"easy": 34, "medium": 38, "hard": 42}[difficulty] + rng.randint(-2, 2)
    pipe_b_h = {"easy": 48, "medium": 54, "hard": 58}[difficulty] + rng.randint(-2, 2)
    # Below ~0.255 the patrol phase collides with the shared oracle script, so
    # keep the jitter floor above that for the easy bin.
    second_enemy_speed = round(
        {"easy": 0.3, "medium": 0.4, "hard": 0.5}[difficulty] + rng.uniform(-0.02, 0.05), 3
    )
    scenario = {
        "world_width": 512,
        "mario": [20, 200],
        "platforms": [
            [0, 220, 512, 20],
            [180, 220 - pipe_a_h, 28, pipe_a_h],
            [318, 220 - pipe_b_h, 32, pipe_b_h],
        ],
        "enemies": [
            [enemy_x, 206, enemy_x, enemy_x, 0],
            {
                "x": 386,
                "y": 206,
                "patrol_min": 374,
                "patrol_max": 414,
                "speed": second_enemy_speed,
            },
        ],
        "coins": [
            [132, 190, 10, 10],
            [220, 176, 10, 10],
            [356, 156, 10, 10],
            [430, 190, 10, 10],
        ],
        "goal": [482, 200, 16, 20],
    }
    actions = _pad(
        [1] * 16
        + [2] * 18
        + [1] * 30
        + [2] * 20
        + [1] * 44
        + [2] * 20
        + [1] * 90
    )
    return (
        scenario,
        {
            "section_count": 4,
            "world_width": 512,
            "enemy_count": 2,
            "pipe_count": 2,
            "pipe_heights": [pipe_a_h, pipe_b_h],
            "difficulty_bin": difficulty,
        },
        actions,
    )


def _chained_enemy_gauntlet(
    rng: random.Random,
    difficulty: str,
) -> tuple[dict[str, Any], dict[str, Any], list[int]]:
    gap_width = {"easy": 48, "medium": 50, "hard": 52}[difficulty] + rng.randint(-1, 1)
    landing_x = 180 + gap_width
    pipe_h = {"easy": 40, "medium": 44, "hard": 48}[difficulty] + rng.randint(-2, 2)
    patrol_speed = round(
        {"easy": 0.3, "medium": 0.4, "hard": 0.5}[difficulty] + rng.uniform(-0.05, 0.05), 3
    )
    scenario = {
        "world_width": 544,
        "mario": [20, 200],
        "platforms": [
            [0, 220, 180, 20],
            [landing_x, 220, 544 - landing_x, 20],
            [370, 220 - pipe_h, 30, pipe_h],
        ],
        "enemies": [
            [96, 206, 96, 96, 0],
            {
                "x": 294,
                "y": 206,
                "patrol_min": 284,
                "patrol_max": 314,
                "speed": patrol_speed,
            },
        ],
        "coins": [
            [132, 190, 10, 10],
            [204, 170, 10, 10],
            [330, 190, 10, 10],
            [430, 190, 10, 10],
        ],
        "goal": [508, 200, 16, 20],
    }
    actions = _pad(
        [1] * 16
        + [2] * 18
        + [1] * 8
        + [2] * 20
        + [1] * 8
        + [2] * 20
        + [1] * 40
        + [2] * 20
        + [1] * 110
    )
    return (
        scenario,
        {
            "section_count": 5,
            "world_width": 544,
            "enemy_count": 2,
            "gap_count": 1,
            "gap_width": gap_width,
            "pipe_count": 1,
            "pipe_height": pipe_h,
            "difficulty_bin": difficulty,
        },
        actions,
    )


def _full_smb_opening_proxy(
    rng: random.Random,
    difficulty: str,
) -> tuple[dict[str, Any], dict[str, Any], list[int]]:
    # Mirrors the early Full SMB demands that were failing: first enemy, pipe,
    # taller pipe, and another enemy before the level can stabilize.
    scenario, params, actions = _chained_obstacles(rng, difficulty)
    scenario = copy.deepcopy(scenario)
    tall_pipe_h = {"easy": 50, "medium": 56, "hard": 60}[difficulty] + rng.randint(-2, 2)
    scenario["platforms"][2] = [318, 220 - tall_pipe_h, 32, tall_pipe_h]
    params = {
        **params,
        "section_count": 5,
        "families": ["enemy_hop", "pipe_jump", "tall_pipe_jump", "enemy_patrol"],
        "pipe_heights": [params["pipe_heights"][0], tall_pipe_h],
        "difficulty_bin": difficulty,
    }
    return scenario, params, actions


def _mixed_section(rng: random.Random, difficulty: str) -> tuple[dict[str, Any], dict[str, Any], list[int]]:
    scenario, params, actions = _chained_enemy_gauntlet(rng, difficulty)
    params = {
        **params,
        "families": ["enemy_hop", "single_gap", "enemy_patrol", "pipe_jump"],
        "difficulty_bin": difficulty,
    }
    return scenario, params, actions
