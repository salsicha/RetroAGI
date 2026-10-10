"""Versioned Monte Carlo scenario families for Block SMB."""

from __future__ import annotations

import copy
import hashlib
import random
from collections import Counter
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable, Mapping, Optional

import pygame

from retroagi.core.smb_physics import NES_JUMP_FRAMES

from .action_families import ACTION_FAMILIES, action_family_scenario
from .bridge_traversal import bridge_oracle
from .env import MarioScenarioEnv
from .hierarchy import FAMILY_PREREQUISITES, HIERARCHY_FAMILIES
from .skill_families import (
    ROUTE_SKILL_FAMILIES,
    SKILL_FAMILIES,
    WATCH_FRAMES,
    skill_family_scenario,
)
from .tactic_families import NEW_FAMILIES, TACTIC_FAMILIES, family_route, tactic_family_scenario
from .tactic_schedule import segment
from .transfer_failure_families import (
    TRANSFER_FAILURE_FAMILIES,
    TRANSFER_FAILURE_SCHEMAS,
    transfer_failure_scenario,
)

# Names the generated layouts in scenario ids and replay seeds.
BLOCK_SMB_MC_ID = "block_smb_monte_carlo"
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
    "tall_pipe_jump",
    "pipe_mount",
    "pit_leap",
    "stomp_mount",
    "platform_hop",
    "bridge_wait",
    "bridge_mount",
    "bridge_dismount",
    *TRANSFER_FAILURE_FAMILIES,
    *HIERARCHY_FAMILIES,
    *NEW_FAMILIES,
    *ACTION_FAMILIES,
    *SKILL_FAMILIES,
)
DEFAULT_BLOCK_SMB_MC_MAX_STEPS = 320
# Families that are advance all the way: their schedule is one advance segment
# toward the goal (tactic_schedule). Every other family states its own.
# Families whose schedule is one segment toward the goal, all the way.
ONE_SEGMENT_FAMILIES = frozenset(
    "flat_run single_gap stair_climb platform_chain enemy_hop enemy_patrol enemy_gap "
    "enemy_stomp retreat_recovery tall_pipe_jump pipe_mount pit_leap stomp_mount "
    "platform_hop stair_gap landing_enemy enemy_on_platform".split()
)
# Of those, the families that advance: Mario goes right, the level's way. In
# retreat_recovery the goal is behind him: going back to the left is
# retreating (teacher_tokens).
ADVANCE_FAMILIES = ONE_SEGMENT_FAMILIES - {"retreat_recovery"}
# Families whose moving platform is waited for, ridden or jumped to and from.
BRIDGE_SEGMENT_FAMILIES = frozenset(
    ("bridge_wait", "wait_timing", "moving_bridge", "bridge_mount", "bridge_dismount")
)


@dataclass(frozen=True)
class BlockSMBScenarioFamilySpec:
    """Schema entry describing one parameterized Block SMB family."""

    family: str
    parameter_schema: Mapping[str, Any]
    constraints: Mapping[str, Any]
    oracle: Mapping[str, Any]

    def __post_init__(self) -> None:
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

    split: str
    seed: int
    samples: tuple[BlockSMBScenarioSample, ...] = field(default_factory=tuple)
    rejected_counts: Mapping[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.split not in BLOCK_SMB_MC_SPLITS:
            raise ValueError(f"split must be one of {BLOCK_SMB_MC_SPLITS}")
        if any(sample.split != self.split for sample in self.samples):
            raise ValueError("all samples must belong to the sample-set split")

    @property
    def sample_count(self) -> int:
        return len(self.samples)

    def scenarios(self) -> list[tuple[str, dict]]:
        return [
            (sample.scenario_id, copy.deepcopy(dict(sample.scenario))) for sample in self.samples
        ]

    def manifest(self, *, include_scenarios: bool = False) -> dict[str, Any]:
        sample_records = [
            sample.to_dict() if include_scenarios else sample.metadata() for sample in self.samples
        ]
        return {
            "split": self.split,
            "seed": self.seed,
            "sample_count": self.sample_count,
            "samples": sample_records,
            "coverage": summarize_block_smb_monte_carlo_samples(self.samples),
            "rejected_counts": dict(self.rejected_counts),
            "rejected_sample_count": int(sum(self.rejected_counts.values())),
            "scenario_ids": [sample.scenario_id for sample in self.samples],
        }


def block_smb_monte_carlo_family_specs() -> dict[str, BlockSMBScenarioFamilySpec]:
    """Return the supported family schema."""

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
            "spawn_x": [16, 40],
            "enemy_distance": [52, 164],
            "enemy_speed": [0.0, 0.6],
            "goal": "stomp the enemy, recover, then reach the finish",
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
        "tall_pipe_jump": {
            "pipe_x": [180, 180],
            "pipe_width": [30, 30],
            "pipe_height": [56, 64],
            "goal_x": [266, 276],
        },
        "pipe_mount": {
            "pipe_x": [120, 120],
            "pipe_width": [30, 30],
            "pipe_height": [42, 64],
            "spawn_distance": [30, 30],
            "goal": "on the pipe top",
            "a_level_action": [2, 2],
        },
        "pit_leap": {
            "gap_width": [40, 66],
            "edge_x": [100, 100],
            "goal": "on the far ledge",
            "a_level_action": [2, 2],
        },
        "stomp_mount": {
            "enemy_distance": [52, 76],
            "enemy_speed": [0.0, 0.9],
            "enemy_initial_direction": [-1, 1],
            "frames_to_first_turn": [0, 12],
            "patrol_halfwidth": [0, 14],
            "goal": "land on the enemy itself; the goal rides the patrolling target",
            "a_level_action": [2, 2],
        },
        "platform_hop": {
            "pit_width": [68, 110],
            "platform_width": [24, 24],
            "platform_speed": [0.3, 0.3],
            "goal": "on the far ledge via the moving platform",
            "a_level_action": [2, 2],
        },
        "bridge_wait": {
            "initial_phase_frames": [12, 60],
            "platform_speed": [1.6, 2.4],
            "gap_width": [200, 200],
            "goal": "wait, board the bridge, and reach the far shore and finish",
            "a_level_action": [0, 0],
        },
    }
    schemas.update(
        {
            "single_gap": {"gap_x": [94, 108], "gap_width": [38, 57]},
            "stair_climb": {
                "step_count": [3, 3],
                "step_width": [36, 42],
                "step_height": [26, 35],
            },
            "platform_chain": {
                "platform_count": [4, 4],
                "gap_spacing": [22, 38],
                "platform_tops": [116, 220],
            },
            "enemy_hop": {"enemy_x": [94, 130], "enemy_count": [1, 1]},
            "enemy_patrol": {
                "enemy_count": [2, 2],
                "patrol_offset": [-8, 8],
                "initial_direction": [-1, 1],
                "enemy_speed": [0.45, 0.75],
            },
            "enemy_gap": {
                "gap_width": [44, 51],
                "enemy_gap_offset": [22, 42],
            },
            "retreat_recovery": {
                "start_x": [188, 208],
                "goal_x": [35, 35],
            },
        }
    )
    schemas.update(TRANSFER_FAILURE_SCHEMAS)
    for family in ("bridge_mount", "bridge_dismount"):
        schemas[family] = dict(
            platform_width=[56, 100],
            platform_speed=[0.6, 1.8],
            required_jump=True,
            single_jump=True,
            a_level_action=[0, 0],
            goal=(
                "stable moving-platform landing"
                if family == "bridge_mount"
                else "jump from moving platform to far shore"
            ),
        )
    for family in ("wait_timing", "moving_bridge"):
        schemas[family] = {k: v for k, v in schemas["bridge_wait"].items() if k != "a_level_action"}
    schemas["moving_bridge"]["spawn_x"] = [20, 60]
    schemas["chained_obstacles"].update(
        section_count=[4, 4], world_width=[512, 512], enemy_count=[2, 2], pipe_height=[32, 60]
    )
    schemas["chained_enemy_gauntlet"].update(
        section_count=[5, 5], world_width=[544, 544], enemy_count=[2, 2], pipe_count=[1, 1]
    )
    schemas["full_smb_opening_proxy"].update(
        section_count=[4, 4], world_width=[512, 512], pipe_count=[2, 2], pipe_height=[32, 62]
    )
    schemas["mixed_section"]["composition"] = ["enemy_gap_pipe", "enemy_two_pipes"]
    for family in ("bridge_wait", "moving_bridge", "wait_timing"):
        schemas[family].update(
            platform_speed=[0.5, 2.4],
            platform_width=[48, 100],
            gap_width=[95, 200],
            variant=["wide", "narrow"],
            initial_phase_frames=[0, 240],
        )
    schemas["retreat_recovery"].update(
        variant=["flat", "gap", "mount"],
        gap_width=[36, 56],
        mount_rise=[28, 52],
        goal_x=[35, 85],
    )
    for family in HIERARCHY_FAMILIES:
        schemas[family] = {
            "difficulty_bin": list(BLOCK_SMB_MC_DIFFICULTY_BINS),
            "prerequisites": list(FAMILY_PREREQUISITES[family]),
        }
    for family in TACTIC_FAMILIES:
        schemas[family] = {
            "difficulty_bin": list(BLOCK_SMB_MC_DIFFICULTY_BINS),
            "tactics": "explicit, per layout (tactic_schedule)",
        }
    for family in SKILL_FAMILIES:
        schemas[family] = {
            "difficulty_bin": list(BLOCK_SMB_MC_DIFFICULTY_BINS),
            "objective": (
                "reach one local platform destination under a supplied climb or descent tactic"
                if family in ROUTE_SKILL_FAMILIES
                else "land past the enemy without killing it"
            ),
        }
    for family in ACTION_FAMILIES:
        schemas[family] = {
            "difficulty_bin": list(BLOCK_SMB_MC_DIFFICULTY_BINS),
            "single_action": "one tactic throughout (action_families)",
        }
    return {
        family: BlockSMBScenarioFamilySpec(
            family=family,
            parameter_schema=schema,
            constraints={
                **base_constraints,
                "family": family,
                "max_gap_width": (
                    255
                    if family in ("bridge_mount", "bridge_dismount")
                    else (
                        200
                        if family
                        in (
                            "bridge_wait",
                            "wait_timing",
                            "moving_bridge",
                            "tactics_bridge_sequence",
                            "tactics_bridge_then_gap",
                        )
                        else (110 if family == "platform_hop" else 66)
                    )
                ),
                "minimum_landing_width": (
                    30
                    if family in ("pipe_mount", "tall_pipe_jump")
                    else (32 if family == "stair_gap" else (36 if family == "stair_climb" else 40))
                ),
            },
            oracle=oracle,
        )
        for family, schema in schemas.items()
    }


def stable_block_smb_monte_carlo_seed(
    split: str,
    seed: int,
    sample_index: int,
    *,
    attempt: int = 0,
) -> int:
    """Return a stable 63-bit seed for the replay tuple."""

    if split not in BLOCK_SMB_MC_SPLITS:
        raise ValueError(f"split must be one of {BLOCK_SMB_MC_SPLITS}")
    key = f"{BLOCK_SMB_MC_ID}|{split}|{int(seed)}|{int(sample_index)}|{int(attempt)}"
    digest = hashlib.sha256(key.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") & ((1 << 63) - 1)


def sample_block_smb_monte_carlo_scenario(
    *,
    split: str,
    seed: int,
    sample_index: int,
    family: Optional[str] = None,
    difficulty: Optional[str] = None,
    family_weights: Optional[Mapping[str, float]] = None,
    max_rejections: int = 32,
    rejection_counter: Optional[Counter[str]] = None,
    calibrate_deadline: bool = True,
) -> BlockSMBScenarioSample:
    """Sample one replayable layout whose route completes it.

    When ``rejection_counter`` is provided, every rejected attempt (including
    attempts preceding an eventual success) is tallied into it by reason.
    A speed-run layout's deadline is measured by replaying the spatial teacher
    through the vision model on the CPU (teacher_replay.calibrate_budget),
    minutes per layout; ``calibrate_deadline=False`` keeps the deadline from
    the teacher's button route, for uses that need no deadline (coverage).
    """

    if split not in BLOCK_SMB_MC_SPLITS:
        raise ValueError(f"split must be one of {BLOCK_SMB_MC_SPLITS}")
    if sample_index < 0:
        raise ValueError("sample_index must be non-negative")
    if difficulty is not None and difficulty not in BLOCK_SMB_MC_DIFFICULTY_BINS:
        raise ValueError(f"difficulty must be one of {BLOCK_SMB_MC_DIFFICULTY_BINS}")
    if max_rejections < 0:
        raise ValueError("max_rejections must be non-negative")
    specs = block_smb_monte_carlo_family_specs()
    rejected: Counter[str] = Counter()
    rejected_fingerprints: set[str] = set()
    for attempt in range(max_rejections + 1):
        sample_seed = stable_block_smb_monte_carlo_seed(
            split,
            seed,
            sample_index,
            attempt=attempt,
        )
        rng = ParameterDraws(random.Random(sample_seed))
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
            # Skip rechecking this layout, but keep drawing: finite parameter
            # spaces can repeat a rejected layout before producing a valid one.
            rejected["duplicate_regeneration"] += 1
            continue
        constraints = specs[selected_family].constraints
        scenario_id = _scenario_id(split, seed, sample_index, selected_family)
        route, action_source, reachability = _verified_route(selected_family, scenario, actions)
        if route is None:
            reachability = {"reachable": False, "rejection_reason": action_source}
        else:
            actions = route
        oracle = {
            "kind": "scripted_action_sequence",
            "actions": list(actions[: _frame_budget(scenario)]),
            "action_source": f"{selected_family}:{action_source}",
            "expected_completion_steps": reachability.get("completion_steps"),
            "expected_min_progress": reachability.get("max_progress"),
        }
        sample = BlockSMBScenarioSample(
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
            if calibrate_deadline and selected_family.startswith("speed_run_"):
                from .teacher_replay import calibrate_budget

                calibrate_budget(sample.scenario, selected_family)
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
        f"for {split}/{seed}/{sample_index} (family={family}, difficulty={difficulty}); "
        f"rejected {reasons}"
    )


def sample_block_smb_monte_carlo_parameter_sweep(
    *,
    split: str,
    seed: int,
    repeats_per_difficulty: int = 1,
    families: Optional[Iterable[str]] = None,
    max_rejections: int = 32,
    executor: Any = None,
    calibrate_deadlines: bool = True,
) -> BlockSMBMonteCarloSampleSet:
    """Return a deterministic family x difficulty Monte Carlo sweep.

    Layouts are independent, so an ``executor`` with an order-preserving
    ``map`` may generate them concurrently with identical results.
    ``calibrate_deadlines``: see sample_block_smb_monte_carlo_scenario.
    """

    if split not in BLOCK_SMB_MC_SPLITS:
        raise ValueError(f"split must be one of {BLOCK_SMB_MC_SPLITS}")
    if repeats_per_difficulty <= 0:
        raise ValueError("repeats_per_difficulty must be positive")
    specs = block_smb_monte_carlo_family_specs()
    selected_families = tuple(str(family) for family in (families or BLOCK_SMB_MC_FAMILIES))
    if not selected_families:
        raise ValueError("families must be non-empty")
    unknown = sorted(set(selected_families) - set(specs))
    if unknown:
        choices = ", ".join(BLOCK_SMB_MC_FAMILIES)
        raise ValueError(f"unknown Block SMB Monte Carlo family {unknown!r}; expected {choices}")

    specs = [
        (split, seed, family, difficulty, repeat, max_rejections, calibrate_deadlines)
        for family in selected_families
        for difficulty in BLOCK_SMB_MC_DIFFICULTY_BINS
        for repeat in range(int(repeats_per_difficulty))
    ]
    specs = [(index, *spec) for index, spec in enumerate(specs)]
    samples: list[BlockSMBScenarioSample] = []
    rejected_counts: Counter[str] = Counter()
    results = (
        map(_sweep_sample, specs)
        if executor is None
        else executor.map(_sweep_sample, specs, chunksize=4)
    )
    for sample, rejected in results:
        samples.append(sample)
        rejected_counts.update(rejected)
    return BlockSMBMonteCarloSampleSet(
        split=split,
        seed=int(seed),
        samples=tuple(samples),
        rejected_counts=dict(rejected_counts),
    )


def _sweep_sample(spec):
    sample_index, split, seed, family, difficulty, repeat, max_rejections, calibrate = spec
    rejected: Counter[str] = Counter()
    candidate = sample_block_smb_monte_carlo_scenario(
        split=split,
        seed=seed,
        sample_index=sample_index,
        family=family,
        difficulty=difficulty,
        max_rejections=max_rejections,
        rejection_counter=rejected,
        calibrate_deadline=calibrate,
    )
    return _with_sweep_metadata(candidate, repeat=repeat), rejected


# ── Every combination of a family's parameters ──────────────────────────────

# A full sweep takes every value of every drawn parameter: every whole number
# of an integer range, every option, every order, and the values of a
# fractional range in steps of UNIFORM_STEP (speeds, in pixels a frame), of a
# 0-1 draw in steps of RANDOM_STEP (it only ever decides a branch). Layouts
# sampled at random take their values from the same sets (ParameterDraws), so
# every held-out layout is one the sweep covers.
UNIFORM_STEP = 0.01
RANDOM_STEP = 0.05


def uniform_values(low: float, high: float) -> list[float]:
    """Every value of a fractional range, in steps of UNIFORM_STEP."""
    if high < low:
        low, high = high, low
    count = int(round((high - low) / UNIFORM_STEP))
    return [round(low + UNIFORM_STEP * i, 6) for i in range(count + 1)]


def random_values() -> list[float]:
    """Every value of a 0-1 draw, in steps of RANDOM_STEP."""
    count = int(round(1.0 / RANDOM_STEP))
    return [round(RANDOM_STEP * i, 6) for i in range(count)]


class ParameterDraws:
    """random.Random for the layout sampler, with fractional draws taking the
    sweep's values (uniform_values, random_values), so a layout sampled at
    random is always one a full sweep makes."""

    def __init__(self, rng: random.Random):
        self._rng = rng

    def __getattr__(self, name):
        return getattr(self._rng, name)

    def uniform(self, low, high):
        return self._rng.choice(uniform_values(low, high))

    def random(self):
        return self._rng.choice(random_values())


def combination_count(family: str, difficulty: str) -> int:
    """How many combinations a full sweep of a family makes at one difficulty,
    counted from its first combination's draws (exact when later draws don't
    depend on earlier values, which holds for every family's main draws)."""
    draws = CombinationDraws()
    _generate_family_scenario_raw(family, draws, split="train", difficulty=difficulty)
    count = 1
    for values in draws.counts:
        count *= values
    return count


class CombinationDraws:
    """Stands in for random.Random in a family's layout generator, to make every
    combination of every value of the parameters it draws.

    Each draw (randint, randrange, uniform, random, choice, choices, sample,
    shuffle) takes each of its values in turn: every whole number of its range,
    every option, every order; a fractional range in steps of UNIFORM_STEP, a
    0-1 draw in steps of RANDOM_STEP. ``path`` names the value of each draw in
    turn (the first, past its end). The draws made, and how many values each
    had, are kept, so next_path() gives the next combination; the number of
    draws may depend on earlier values, and every branch is followed.
    """

    def __init__(self, path=()):
        self.path = list(path)
        self.taken: list[int] = []
        self.counts: list[int] = []

    def _pick(self, values):
        k = len(self.taken)
        index = self.path[k] if k < len(self.path) else 0
        self.taken.append(index)
        self.counts.append(len(values))
        return values[index]

    def randint(self, low, high):
        return self._pick(list(range(low, high + 1)))

    def randrange(self, start, stop=None, step=1):
        if stop is None:
            start, stop = 0, start
        return self._pick(list(range(start, stop, step)))

    def uniform(self, low, high):
        return self._pick(uniform_values(low, high))

    def random(self):
        return self._pick(random_values())

    def choice(self, options):
        return self._pick(list(options))

    def choices(self, population, weights=None, *, cum_weights=None, k=1):
        return [self.choice(population) for _ in range(k)]

    def sample(self, population, k):
        """``k`` different items in order: every ordered pick."""
        left = list(population)
        return [left.pop(left.index(self.choice(left))) for _ in range(k)]

    def shuffle(self, items):
        """Every order of ``items`` (in place)."""
        for i in range(len(items) - 1, 0, -1):
            j = self._pick(list(range(i + 1)))
            items[i], items[j] = items[j], items[i]

    def next_path(self) -> Optional[list[int]]:
        """The next combination after the one just drawn (None after the last)."""
        k = len(self.taken) - 1
        while k >= 0 and self.taken[k] + 1 >= self.counts[k]:
            k -= 1
        return None if k < 0 else self.taken[:k] + [self.taken[k] + 1]


def combination_prefixes(
    family: str, difficulty: str, *, at_least: int = 12
) -> list[tuple[int, ...]]:
    """The first few draws' values, as prefixes that split a family's
    combinations into parts that can be made side by side: each prefix's
    combinations are those starting with it (block_smb_parameter_combinations
    with ``prefix``). Draws are added until there are at least ``at_least``
    prefixes or a combination has no more draws."""
    prefixes: list[tuple[int, ...]] = [()]
    while len(prefixes) < at_least:
        grown = []
        for prefix in prefixes:
            draws = CombinationDraws(prefix)
            _generate_family_scenario_raw(family, draws, split="train", difficulty=difficulty)
            if len(draws.counts) <= len(prefix):
                return prefixes  # a combination with no further draw
            grown += [(*prefix, i) for i in range(draws.counts[len(prefix)])]
        prefixes = grown
    return prefixes


def block_smb_parameter_combinations(
    family: str, difficulty: str, *, prefix: tuple = ()
) -> tuple[list[dict[str, Any]], int]:
    """Every layout a full sweep of a family's drawn parameters makes at one
    difficulty (CombinationDraws: every value of every draw, every
    combination), finished and route-verified as the sampler makes them; with
    ``prefix``, only the combinations whose first draws take those values.

    A layout made by two combinations is kept once; a combination whose layout
    has no verified route is left out. Returns the layouts and how many
    combinations were left out.
    """
    if difficulty not in BLOCK_SMB_MC_DIFFICULTY_BINS:
        raise ValueError(f"difficulty must be one of {BLOCK_SMB_MC_DIFFICULTY_BINS}")
    constraints = block_smb_monte_carlo_family_specs()[family].constraints
    layouts: list[dict[str, Any]] = []
    seen: set[str] = set()
    dropped = 0
    path: Optional[list[int]] = list(prefix)
    while path is not None:
        draws = CombinationDraws(path)
        scenario, parameters, actions = _generate_family_scenario(
            family, draws, split="train", difficulty=difficulty
        )
        path = draws.next_path()
        if path is not None and tuple(path[: len(prefix)]) != tuple(prefix):
            path = None  # past this prefix's combinations
        made = repr(scenario)
        if made in seen:
            continue
        seen.add(made)
        route, source, reachability = _verified_route(family, scenario, actions)
        if route is None:
            dropped += 1
            continue
        index = len(layouts)
        oracle = {
            "kind": "scripted_action_sequence",
            "actions": list(route[: _frame_budget(scenario)]),
            "action_source": f"{family}:{source}",
            "expected_completion_steps": reachability.get("completion_steps"),
            "expected_min_progress": reachability.get("max_progress"),
        }
        layouts.append(
            _with_sample_metadata(
                scenario,
                family=family,
                split="train",
                seed=0,
                sample_seed=index,
                sample_index=index,
                scenario_id=f"{BLOCK_SMB_MC_ID}.combination.{family}.{difficulty}.{index:06d}",
                parameters={
                    **dict(parameters),
                    "difficulty_bin": difficulty,
                    "combination": list(draws.taken),
                },
                constraints=constraints,
                oracle=oracle,
                reachability=reachability,
            )
        )
    return layouts, dropped


def validate_block_smb_monte_carlo_oracle(
    scenario: Mapping[str, Any],
    actions: Iterable[int],
    *,
    max_steps: Optional[int] = None,
) -> dict[str, Any]:
    """Run the scripted oracle and return reachability diagnostics (within the
    layout's frame budget unless ``max_steps`` is given)."""

    max_steps = _frame_budget(scenario) if max_steps is None else max_steps
    env = MarioScenarioEnv()
    total_return = 0.0
    completion_steps: int | None = None
    last_info: Mapping[str, Any] = {}
    reached_goal = False
    try:
        env.reset(scenario=copy.deepcopy(dict(scenario)), seed=0)
        for step_index, action in enumerate(list(actions)[:max_steps]):
            _observation, reward, terminated, truncated, info = env.step(int(action))
            last_info = info
            total_return += float(reward)
            if terminated or truncated:
                completion_steps = step_index + 1
                # Trust the env's own goal event. The env fires the goal reward
                # and terminates without a death when Mario reaches the goal.
                # The positional _goal_reached() re-check rebuilds an int rect
                # from Mario's float x, dropping sub-pixel goal touches (right
                # edge 230.5 vs goal left 230) and spuriously rejecting oracles
                # that actually complete the level.
                terms = info.get("reward_terms", {}) if isinstance(info, Mapping) else {}
                goal_credit = (
                    float(terms.get("goal", 0.0) or 0.0) if isinstance(terms, Mapping) else 0.0
                )
                if terminated and not bool(info.get("death", False)) and goal_credit > 0.0:
                    reached_goal = True
                break
        reachable = reached_goal or _goal_reached(env)
        if not reachable and completion_steps is None:
            completion_steps = max_steps
        max_progress = float(last_info.get("max_x_reached", env._max_x_reached))
        reason = None if reachable else _oracle_rejection_reason(env, last_info, max_steps)
        return {
            "reachable": bool(reachable),
            "stomp_completed": bool(env._stomp_credited),
            "bridge_boarded": bool(env._bridge_boarded),
            "bridge_crossed": bool(env._bridge_crossed),
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


def _scenario_id(split: str, seed: int, sample_index: int, family: str) -> str:
    return f"{BLOCK_SMB_MC_ID}.{split}.{int(seed)}.{int(sample_index):06d}.{family}"


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
        "block_smb_monte_carlo": {key: copy.deepcopy(value) for key, value in metadata.items()},
    }
    return enriched


def _with_sweep_metadata(
    sample: BlockSMBScenarioSample,
    *,
    repeat: int,
) -> BlockSMBScenarioSample:
    scenario_id = f"{sample.scenario_id}.{sample.difficulty_bin}.sweep_r{int(repeat):02d}"
    parameters = {
        **dict(sample.parameters),
        "parameter_sweep": True,
        "sweep_repeat": int(repeat),
    }
    constraints = {**dict(sample.constraints), "parameter_sweep": True}
    oracle = {
        **dict(sample.oracle),
        "action_source": f"{sample.family}:sweep_oracle",
    }
    reachability = dict(sample.reachability)
    scenario = _with_sample_metadata(
        sample.scenario,
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
    # The env's credited-goal event is authoritative. Under goal_on_stomp the
    # goal rect is only a tracking proxy riding the enemy: positional overlap
    # must never count, or an oracle that dies walking into the enemy would
    # validate as reachable.
    if getattr(env, "_goal_credited", False):
        return True
    if (
        getattr(env, "_goal_on_stomp", False)
        or getattr(env, "_require_stomp_before_goal", False)
        or getattr(env, "_require_bridge_before_goal", False)
        or getattr(env, "_goal_requires_support", False)
    ):
        return False
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


def _frame_budget(scenario) -> int:
    """Frames the teacher's route may take (the layout's own budget, if it has one)."""
    return int(scenario.get("frame_budget", DEFAULT_BLOCK_SMB_MC_MAX_STEPS))


def _family_tactics(family, scenario) -> list:
    """The layout's tactic segments: its own, or its family's (tactic_schedule)."""
    if "tactics" in scenario:
        return scenario["tactics"]
    goal = scenario["goal"]
    direction = -1 if goal[0] + goal[2] / 2 < scenario["mario"][0] else 1
    if family in SKILL_FAMILIES:
        return [
            segment(
                "advance" if direction > 0 else "retreat",
                direction,
                keep_alive=range(len(scenario.get("enemies", ()))),
            )
        ]
    if family in BRIDGE_SEGMENT_FAMILIES:
        return [segment("advance", direction, kind="bridge")]
    if family == "piranha_avoidance":
        return [segment("advance", 1, kind="plant", past_enemy=0), segment("advance", 1)]
    if family in ONE_SEGMENT_FAMILIES or family in ACTION_FAMILIES:
        return [segment("advance", direction)]
    raise ValueError(f"family {family!r} states no tactics")


def _generate_family_scenario(family, rng, *, split, difficulty=None):
    from .local_traversal import LOCAL_TRAVERSAL_FAMILIES, normalize_oracle_jumps

    scenario, params, actions = _generate_family_scenario_raw(
        family, rng, split=split, difficulty=difficulty
    )
    if family in LOCAL_TRAVERSAL_FAMILIES:
        scenario.setdefault("reward_goal_distance_shaping", 2.0)
        scenario.setdefault("goal_requires_support", True)
    _finish_layout(family, scenario, params)
    scenario["tactics"] = _family_tactics(family, scenario)
    if (
        family in TACTIC_FAMILIES
        or family in ACTION_FAMILIES
        or family in SKILL_FAMILIES
        or family in ("bridge_mount", "bridge_dismount")
    ):
        # The teacher's route under the layout's own tactics.
        actions = family_route(family, scenario)
        return scenario, params, actions or [0]
    if family in LOCAL_TRAVERSAL_FAMILIES and family not in (
        "pipe_mount",
        "pit_leap",
        "retreat_recovery",
    ):
        actions = _pad(normalize_oracle_jumps(scenario, actions))
    return scenario, params, actions


def _finish_layout(family, scenario, params):
    """Fit an authored layout to the NES player and its task.

    Generators place Mario's spawn for a 16-pixel-tall body; the NES small
    body is 12 tall, so its feet stay where they were authored.
    """
    scenario["mario"][1] += 4
    if family == "enemy_stomp":
        # Patrol limits are not observable. Short, invisible limits made
        # identical observed motion require incompatible jump holds; use the
        # visible floor span, so no turnaround happens mid-approach.
        for enemy in scenario["enemies"]:
            enemy[2], enemy[3] = 0, scenario["world_width"]
        params.update(patrol_halfwidth=None, enemy_motion="floor_span")
    if family in ("pit_leap", "platform_hop"):
        # The duration-isolation task begins at a real running takeoff: the
        # NES caps a jump initiated from rest at walking horizontal speed.
        scenario["mario_velocity"] = [2.5, 0.0]
    if family in ("enemy_stomp", "stomp_mount"):
        scenario["task_objective"] = "stomp"
    if family == "retreat_recovery":
        scenario["task_direction"] = -1


def _as_played(family, scenario):
    """A copy of an authored layout as the sampler will play it (after _finish_layout).

    Generators that search for their own scripted route must search this
    copy: the finished layout moves Mario's spawn and can change the task.
    """
    played = copy.deepcopy(scenario)
    _finish_layout(family, played, {})
    return played


def _verified_route(family, scenario, authored_actions):
    """The first route that completes the layout: (actions, source, reachability).

    Tries the authored route, then a local terrain search, then (for moving
    bridges) a bridge search. Returns (None, reason, None) when none completes.
    """
    from .local_traversal import terrain_oracle

    max_steps = _frame_budget(scenario)

    def reachable(actions):
        return validate_block_smb_monte_carlo_oracle(scenario, actions, max_steps=max_steps)

    if family in TACTIC_FAMILIES or family in ACTION_FAMILIES or family in SKILL_FAMILIES:
        # Only the teacher's route under the layout's tactics counts.
        result = reachable(authored_actions)
        if result["reachable"]:
            return list(authored_actions), "tactic_teacher", result
        return None, "no_verified_route", None
    candidates = [list(authored_actions)]
    if family == "platform_hop":
        # This family isolates duration selection at the initial state.
        # Never accept a fallback route that silently adds a run-up.
        routes = [[2] * hold + [1] * (max_steps - hold) for hold in NES_JUMP_FRAMES]
        routes = [route for route in routes if reachable(route)["reachable"]]
        if not routes:
            return None, "no_immediate_jump", None
        candidates = [routes[len(routes) // 2]]
    for source in ("authored", "local_search", "bridge_search"):
        if source == "local_search":
            candidates.append(terrain_oracle(scenario, max_steps=max_steps))
        elif source == "bridge_search":
            if family not in ("bridge_wait", "moving_bridge", "wait_timing"):
                continue
            candidates.append(bridge_oracle(scenario, max_steps=max_steps)[0])
        result = reachable(candidates[-1])
        if result["reachable"]:
            return candidates[-1], source, result
    return None, "no_verified_route", None


def _generate_family_scenario_raw(
    family: str,
    rng: random.Random,
    *,
    split: str,
    difficulty: Optional[str] = None,
) -> tuple[dict[str, Any], dict[str, Any], list[int]]:
    difficulty = difficulty or _difficulty_bin(rng, split)
    if difficulty not in BLOCK_SMB_MC_DIFFICULTY_BINS:
        raise ValueError(f"difficulty must be one of {BLOCK_SMB_MC_DIFFICULTY_BINS}")
    if family in TACTIC_FAMILIES:
        return tactic_family_scenario(family, rng, difficulty)
    if family in SKILL_FAMILIES:
        return skill_family_scenario(family, rng, difficulty)
    if family in ACTION_FAMILIES:
        return action_family_scenario(family, rng, difficulty)
    if family in TRANSFER_FAILURE_FAMILIES:
        return transfer_failure_scenario(family, rng, difficulty)
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
    if family == "tall_pipe_jump":
        return _tall_pipe_jump(rng, difficulty)
    if family == "pipe_mount":
        return _pipe_mount(rng, difficulty)
    if family == "pit_leap":
        return _pit_leap(rng, difficulty)
    if family == "stomp_mount":
        return _stomp_mount(rng, difficulty)
    if family in ("bridge_mount", "bridge_dismount"):
        from .bridge_curriculum import bridge_jump_scenario

        return bridge_jump_scenario(rng, difficulty, family)
    if family == "bridge_wait":
        return _bridge_wait(rng, difficulty)
    if family == "platform_hop":
        return _platform_hop(rng, difficulty)
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


def _flat_run(
    rng: random.Random, difficulty: str
) -> tuple[dict[str, Any], dict[str, Any], list[int]]:
    goal_x = rng.randint(224, 232)
    coin_x = rng.randint(110, 150)
    scenario = {
        "world_width": 256,
        "mario": [20, 200],
        "platforms": [[0, 220, 256, 20]],
        "coins": [[coin_x, 200, 10, 10]],
        "goal": [goal_x, 200, 16, 20],
    }
    return (
        scenario,
        {"goal_x": goal_x, "coin_x": coin_x, "difficulty_bin": difficulty},
        _right_actions(),
    )


def _single_gap(
    rng: random.Random, difficulty: str
) -> tuple[dict[str, Any], dict[str, Any], list[int]]:
    first_width = rng.randint(94, 108)
    gap_width = {"easy": 40, "medium": 48, "hard": 56}[difficulty] + rng.randint(-2, 1)
    coin_x = rng.randint(112, 128)
    landing_x = first_width + gap_width
    scenario = {
        "world_width": 256,
        "mario": [20, 200],
        "platforms": [[0, 220, first_width, 20], [landing_x, 220, 256 - landing_x, 20]],
        "coins": [[coin_x, 160, 10, 10]],
        "goal": [220, 200, 16, 20],
    }
    actions = _pad([1] * max(0, 10 + round((first_width - 100) / 3)) + [2] * 16 + [1])
    return (
        scenario,
        {"gap_x": first_width, "gap_width": gap_width, "difficulty_bin": difficulty},
        actions,
    )


def _stair_climb(
    rng: random.Random, difficulty: str
) -> tuple[dict[str, Any], dict[str, Any], list[int]]:
    step_height = {"easy": 26, "medium": 30, "hard": 34}[difficulty] + rng.randint(0, 1)
    step_width = rng.randint(36, 42)
    coin_a_x = rng.randint(70, 82)
    coin_b_x = rng.randint(110, 122)
    goal_x = rng.randint(216, 228)
    scenario = {
        "world_width": 256,
        "mario": [20, 200],
        "platforms": [
            [0, 220, 60, 20],
            [60, 220 - step_height, step_width, step_height + 20],
            [60 + step_width, 220 - 2 * step_height, step_width, 2 * step_height + 20],
            [
                60 + 2 * step_width,
                220 - 3 * step_height,
                196 - 2 * step_width,
                3 * step_height + 20,
            ],
        ],
        "coins": [[coin_a_x, 170, 10, 10], [coin_b_x, 140, 10, 10]],
        "goal": [goal_x, 200 - 3 * step_height, 16, 20],
    }
    # Retimed for the grounded spawn: liftoff now happens on the first
    # frame instead of after a settle, shifting every landing.
    actions = _pad([2] * 8 + [1] * 4 + [2] * 14 + [1] * 2 + [2] * 12 + [1])
    return (
        scenario,
        {
            "step_count": 3,
            "step_width": step_width,
            "step_height": step_height,
            "goal_x": goal_x,
            "difficulty_bin": difficulty,
        },
        actions,
    )


def _platform_chain(
    rng: random.Random, difficulty: str
) -> tuple[dict[str, Any], dict[str, Any], list[int]]:
    gap = {"easy": 24, "medium": 30, "hard": 36}[difficulty] + rng.randint(-2, 2)
    low_y = 160 + rng.randint(-4, 4)
    high_y = 120 + rng.randint(-4, 4)
    second_x, third_x = 40 + gap, 80 + 2 * gap
    shore_x = 120 + 3 * gap
    world_width = max(256, shore_x + 56)
    coin_a_x = rng.randint(78, 96)
    coin_b_x = rng.randint(142, 168)
    goal_x = world_width - rng.randint(22, 32)
    scenario = {
        "world_width": world_width,
        "mario": [20, 100],
        "platforms": [
            [0, 120, 40, 10],
            [second_x, low_y, 40, 10],
            [third_x, high_y, 40, 10],
            [shore_x, 220, world_width - shore_x, 20],
        ],
        "coins": [[coin_a_x, 140, 10, 10], [coin_b_x, 100, 10, 10]],
        "goal": [goal_x, 200, 16, 20],
    }
    actions = _pad([1] * 8 + [2] * 16 + [1])
    return (
        scenario,
        {
            "gap_spacing": gap,
            "platform_tops": [120, low_y, high_y, 220],
            "platform_count": 4,
            "goal_x": goal_x,
            "difficulty_bin": difficulty,
        },
        actions,
    )


def _moving_bridge(
    rng: random.Random, difficulty: str
) -> tuple[dict[str, Any], dict[str, Any], list[int]]:
    scenario, params, _ = _wait_timing(rng, difficulty)
    # Add a walking approach; the free policy must approach before waiting.
    scenario["mario"][0] = rng.randint(20, 60)
    actions, wait = bridge_oracle(scenario)
    params["spawn_x"] = scenario["mario"][0]
    params["required_wait"] = wait
    return scenario, params, _pad(actions)


def _enemy_hop(
    rng: random.Random, difficulty: str
) -> tuple[dict[str, Any], dict[str, Any], list[int]]:
    enemy_x = {"easy": 96, "medium": 112, "hard": 128}[difficulty] + rng.randint(-2, 2)
    coin_x = rng.randint(140, 152)
    scenario = {
        "world_width": 256,
        "mario": [20, 200],
        "platforms": [[0, 220, 256, 20]],
        "enemies": [[enemy_x, 206, enemy_x, enemy_x, 0]],
        "coins": [[coin_x, 190, 10, 10]],
        "goal": [230, 200, 16, 20],
    }
    actions = _pad([1] * max(0, 20 + round((enemy_x - 106) / 3)) + [2] * 16 + [1])
    return scenario, {"enemy_x": enemy_x, "difficulty_bin": difficulty}, actions


def _enemy_patrol(
    rng: random.Random, difficulty: str
) -> tuple[dict[str, Any], dict[str, Any], list[int]]:
    speed = round(
        {"easy": 0.5, "medium": 0.6, "hard": 0.7}[difficulty] + rng.uniform(-0.05, 0.05), 3
    )
    offset = rng.randint(-8, 8)
    direction = rng.choice((-1, 1))
    scenario = {
        "world_width": 256,
        "mario": [20, 200],
        "platforms": [[0, 220, 256, 20]],
        "enemies": [
            {
                "x": 100 + offset,
                "y": 206,
                "patrol_min": 84 + offset,
                "patrol_max": 130 + offset,
                "speed": speed,
                "direction": direction,
            },
            {
                "x": 170 - offset,
                "y": 206,
                "patrol_min": 150 - offset,
                "patrol_max": 190 - offset,
                "speed": speed,
                "direction": -direction,
            },
        ],
        "coins": [[130, 185, 10, 10], [200, 185, 10, 10]],
        "goal": [230, 200, 16, 20],
    }
    actions = _pad([1] * 12 + [2] * 18 + [1] * 18 + [2] * 18 + [1])
    return (
        scenario,
        {
            "patrol_offset": offset,
            "initial_direction": direction,
            "enemy_count": 2,
            "enemy_speed": speed,
            "difficulty_bin": difficulty,
        },
        actions,
    )


def _enemy_gap(
    rng: random.Random, difficulty: str
) -> tuple[dict[str, Any], dict[str, Any], list[int]]:
    gap_width = {"easy": 46, "medium": 48, "hard": 50}[difficulty] + rng.randint(-2, 1)
    enemy_speed = round(0.4 + rng.uniform(-0.05, 0.05), 3)
    first_width = rng.randint(96, 104)
    enemy_offset = {"easy": 40, "medium": 32, "hard": 24}[difficulty] + rng.randint(-2, 2)
    landing_x = first_width + gap_width
    scenario = {
        "world_width": 256,
        "mario": [20, 200],
        "platforms": [[0, 220, first_width, 20], [landing_x, 220, 256 - landing_x, 20]],
        "enemies": [
            [
                landing_x + enemy_offset,
                206,
                landing_x + enemy_offset - 10,
                landing_x + enemy_offset + 28,
                enemy_speed,
            ]
        ],
        "coins": [[120, 160, 10, 10], [205, 185, 10, 10]],
        "goal": [230, 200, 16, 20],
    }
    actions = _pad([1] * 10 + [2] * 17 + [1] * 8 + [2] * 18 + [1])
    return (
        scenario,
        {"gap_width": gap_width, "enemy_gap_offset": enemy_offset, "difficulty_bin": difficulty},
        actions,
    )


def _enemy_stomp(
    rng: random.Random, difficulty: str
) -> tuple[dict[str, Any], dict[str, Any], list[int]]:
    spawn_x = rng.randint(16, 40)
    low, high, speed, patrol = {
        "easy": (52, 72, 0.0, 0),
        "medium": (92, 116, 0.3, 8),
        "hard": (140, 164, 0.6, 12),
    }[difficulty]
    distance = rng.randint(low, high)
    enemy_x = spawn_x + distance
    direction = rng.choice((-1, 1)) if speed else 1
    scenario = {
        "world_width": 360,
        "mario": [spawn_x, 200],
        "platforms": [[0, 220, 360, 20]],
        "enemies": [[enemy_x, 206, enemy_x - patrol, enemy_x + patrol, speed, direction]],
        "coins": [],
        "goal": [334, 200, 16, 20],
        "require_stomp_before_goal": True,
        "reward_goal_distance_shaping": 2.0,
        # Watched before the first decision: the enemy's speed and direction
        # cannot be seen in one picture (skill_families.WATCH_FRAMES).
        "watch_frames": WATCH_FRAMES,
    }
    # A walk-then-jump guess; the sampler verifies it in the finished layout
    # and otherwise takes the local search's route. The engine requires the
    # stomp, so reaching the finish alone never verifies a route.
    actions = _pad([1] * max(0, round((distance - 70) / 3)) + [2] * 12 + [1])
    return (
        scenario,
        {
            "spawn_x": spawn_x,
            "enemy_x": enemy_x,
            "enemy_distance": distance,
            "enemy_speed": speed,
            "enemy_initial_direction": direction,
            "patrol_halfwidth": patrol,
            "difficulty_bin": difficulty,
        },
        actions,
    )


def _retreat_recovery(
    rng: random.Random, difficulty: str
) -> tuple[dict[str, Any], dict[str, Any], list[int]]:
    start_x = {"easy": 190, "medium": 200, "hard": 205}[difficulty] + rng.randint(-2, 3)
    coin_x = rng.randint(110, 130)
    scenario = {
        "world_width": 256,
        "reward_progress_per_pixel": 0.0,
        "reward_goal_distance_shaping": 8.0,
        "mario": [start_x, 200],
        "platforms": [[0, 220, 256, 20]],
        "coins": [[coin_x, 200, 10, 10]],
        "goal": [35, 200, 16, 20],
    }
    variant = rng.choice(("flat", "mount", "gap"))
    if variant == "mount":
        rise = {"easy": 30, "medium": 40, "hard": 50}[difficulty] + rng.randint(-2, 2)
        edge = rng.randint(146, 154)
        scenario["platforms"] = [[edge, 220, 256 - edge, 20], [70, 220 - rise, edge - 60, 10]]
        scenario["goal"] = [85, 200 - rise, 16, 20]
    elif variant == "gap":
        width = {"easy": 38, "medium": 46, "hard": 54}[difficulty] + rng.randint(-2, 2)
        edge = rng.randint(150, 158)
        scenario["platforms"] = [[edge, 220, 256 - edge, 20], [0, 220, edge - width, 20]]
    from .local_traversal import terrain_oracle

    actions = terrain_oracle(scenario)
    return (
        scenario,
        {
            "variant": variant,
            "start_x": start_x,
            "goal_x": scenario["goal"][0],
            "difficulty_bin": difficulty,
        },
        _pad(actions),
    )


def _wait_timing(
    rng: random.Random, difficulty: str
) -> tuple[dict[str, Any], dict[str, Any], list[int]]:
    # Unlike bridge_wait, both choosing to wait and leaving are learned.
    scenario, params, actions = _bridge_wait(rng, difficulty)
    params.pop("a_level_action")
    params.pop("a_level_action_scope")
    return scenario, params, actions


def _opening_wait(scenario: Mapping[str, Any]) -> Optional[int]:
    """Frames Mario must stand on the shore before walking boards the bridge."""
    from .bridge_traversal import bridge_safe_wait_frames

    env = MarioScenarioEnv()
    try:
        env.reset(scenario=scenario)
        waits = bridge_safe_wait_frames(env, horizon=96)
        return waits[0] if waits else None
    finally:
        env.close()


def _bridge_wait(
    rng: random.Random, difficulty: str
) -> tuple[dict[str, Any], dict[str, Any], list[int]]:
    low, high, min_speed, max_speed = {
        "easy": (12, 22, 2.0, 2.4),
        "medium": (30, 42, 1.8, 2.2),
        "hard": (48, 60, 1.6, 2.0),
    }[difficulty]
    target_wait = rng.randint(low, high)
    speed = round(rng.uniform(min_speed, max_speed), 3)
    scenario = {
        "world_width": 380,
        "mario": [60, 204],
        "platforms": [
            [0, 220, 85, 20],
            {
                "x": 245,
                "y": 220,
                "w": 100,
                "h": 20,
                "moving": [75, 245, speed],
                "direction": -1,
            },
            [285, 220, 95, 20],
        ],
        "coins": [],
        "goal": [350, 200, 16, 20],
        "require_bridge_before_goal": True,
        "reward_wait_survival": 0.05,
        "reward_goal_distance_shaping": 2.0,
    }
    variant = "wide"
    travel_high = 245
    if rng.random() < 0.4:
        variant = "narrow"
        width = rng.randint(48, 60)
        right_start = rng.randint(180, 216)
        speed = round(rng.uniform(0.5, 1.1), 3)
        travel_high = right_start - width + 10
        scenario["platforms"][1].update(w=width, moving=[75, travel_high, speed])
        scenario["platforms"][2] = [right_start, 220, 380 - right_start, 20]
    # Place the approaching bridge where Mario's opening wait under NES
    # walking is nearest the sampled wait for this difficulty.
    best = None
    for direction in (-1, 1):
        for x in range(75, travel_high + 1, 3):
            scenario["platforms"][1].update(x=x, direction=direction)
            wait = _opening_wait(scenario)
            if wait is not None and (best is None or abs(wait - target_wait) < best[0]):
                best = (abs(wait - target_wait), x, direction)
    _, initial_x, direction = best if best else (0, travel_high, -1)
    scenario["platforms"][1].update(x=initial_x, direction=direction)
    actions, wait = bridge_oracle(scenario)
    return (
        scenario,
        {
            "required_wait": wait,
            "initial_phase_frames": round((initial_x - 75) / speed),
            "platform_speed": speed,
            "platform_initial_x": initial_x,
            "gap_width": scenario["platforms"][2][0] - 85,
            "platform_width": scenario["platforms"][1]["w"],
            "variant": variant,
            "a_level_action": 0,
            "a_level_action_scope": "first_primitive",
            "difficulty_bin": difficulty,
        },
        _pad(actions),
    )


def _verified_isolation_hold(scenario):
    from .local_traversal import local_objective, safe_jump_holds

    env = MarioScenarioEnv()
    try:
        env.reset(scenario=scenario)
        valid = safe_jump_holds(env, local_objective(env), 1)
        return valid[len(valid) // 2] if valid else 16
    finally:
        env.close()


def _pipe_mount(
    rng: random.Random, difficulty: str
) -> tuple[dict[str, Any], dict[str, Any], list[int]]:
    # B-level isolation family: the A-level decision is given (a_level_action
    # forces RIGHT_JUMP during rollouts), the goal sits ON the pipe top, and a
    # goal-distance shaping reward gives a dense vertical-progress gradient.
    # Only the B-level jump parameters (hold duration) have to be learned, and
    # the required hold grows monotonically with the pipe height, so the
    # height -> duration mapping is identifiable from the C-stream state the
    # B transformer now receives.
    pipe_height = {
        "easy": rng.randint(42, 48),
        "medium": rng.randint(52, 58),
        # 65+ cannot be mounted by an immediate jump from the fixed spawn, and
        # forced-A rollouts jump immediately, so the hard band stops at 64.
        "hard": rng.randint(60, 64),
    }[difficulty]
    pipe_x, pipe_width = 120, 30
    spawn_distance = 30
    goal_x = pipe_x
    scenario = {
        "world_width": 256,
        "mario": [pipe_x - spawn_distance, 200],
        "platforms": [
            [0, 220, 256, 20],
            [pipe_x, 220 - pipe_height, pipe_width, pipe_height],
        ],
        "platform_kinds": ["ground", "pipe"],
        "coins": [],
        "goal": [goal_x, 220 - pipe_height - 20, pipe_width, 20],
        "reward_goal_distance_shaping": 2.0,
        "goal_requires_support": True,
        "single_jump_attempt": True,
        # No jump-energy tax: in a single-jump episode an over-hold misses
        # the target and fails outright, which is the real minimal-
        # sufficient-hold pressure. A per-frame tax made the 1-frame tap
        # the cheapest FAILURE and collapsed the duration head to the
        # floor bin before coaching could walk it anywhere.
    }
    oracle_hold = _verified_isolation_hold(scenario)
    actions = _pad([2] * oracle_hold + [1] * 30)
    return (
        scenario,
        {
            "pipe_x": pipe_x,
            "pipe_width": pipe_width,
            "pipe_height": pipe_height,
            "spawn_distance": spawn_distance,
            "goal_x": goal_x,
            # A-level intent handed to the rollout: force RIGHT_JUMP so only
            # the B-level executor parameters remain to be learned.
            "a_level_action": 2,
            # Single-jump scenario: the episode ends when the jump completes;
            # mounting the pipe (goal on its top) is credited on landing and
            # anything else is an immediate failure.
            "single_jump": True,
            "difficulty_bin": difficulty,
        },
        actions,
    )


def _pit_leap(
    rng: random.Random, difficulty: str
) -> tuple[dict[str, Any], dict[str, Any], list[int]]:
    # B-level isolation family: jump over a pit. The A-level decision is given
    # (RIGHT_JUMP forced) and the gap width bands make the required hold grow
    # monotonically (probed minima: 42px -> 8 frames, 54 -> 10, 66 -> 14), so
    # the width -> duration mapping is identifiable. Undershoot falls into the
    # pit, and the energy regulator makes minimal sufficient holds optimal.
    gap_width = {
        "easy": rng.randint(40, 46),
        "medium": rng.randint(52, 58),
        "hard": rng.randint(62, 66),
    }[difficulty]
    edge_x = 100
    # Single-jump scenario: the episode IS the one commanded jump and ends
    # the moment it completes. The goal spans the entire far ledge, so any
    # landing that cleared the pit is credited on the landing frame; an
    # undershoot back onto the near ledge ends the episode as a failure
    # instead of devolving into hop chains.
    far_x = edge_x + gap_width
    scenario = {
        "world_width": 320,
        "mario": [edge_x - 30, 200],
        "platforms": [
            [0, 220, edge_x, 20],
            [far_x, 220, 320 - far_x, 20],
        ],
        "coins": [],
        "goal": [far_x, 200, 320 - far_x - 4, 20],
        "reward_goal_distance_shaping": 2.0,
        "goal_requires_support": True,
        "single_jump_attempt": True,
    }
    oracle_hold = _verified_isolation_hold(scenario)
    actions = _pad([2] * oracle_hold + [1] * 40)
    return (
        scenario,
        {
            "gap_width": gap_width,
            "edge_x": edge_x,
            "a_level_action": 2,
            "single_jump": True,
            "difficulty_bin": difficulty,
        },
        actions,
    )


def _stomp_mount(
    rng: random.Random, difficulty: str
) -> tuple[dict[str, Any], dict[str, Any], list[int]]:
    # Single-jump interception. The moving tiers vary direction and phase,
    # with a first reversal while jump release can still change the arc.
    enemy_distance, patrol_halfwidth, enemy_speed = {
        "easy": (rng.randint(52, 60), 0, 0.0),
        "medium": (rng.randint(62, 68), 10, 0.6),
        "hard": (rng.randint(70, 76), 14, 0.9),
    }[difficulty]
    direction = rng.choice((-1, 1)) if enemy_speed else 1
    turn_frames = rng.randint(6, 12) if enemy_speed else 0
    enemy_x = 40 + enemy_distance
    to_boundary = enemy_speed * turn_frames
    if direction > 0:
        patrol_max = enemy_x + to_boundary
        patrol_min = patrol_max - 2 * patrol_halfwidth
    else:
        patrol_min = enemy_x - to_boundary
        patrol_max = patrol_min + 2 * patrol_halfwidth
    scenario = {
        "world_width": 340,
        "mario": [40, 200],
        "platforms": [[0, 220, 340, 20]],
        "enemies": [[enemy_x, 206, patrol_min, patrol_max, enemy_speed, direction]],
        "coins": [],
        "goal": [enemy_x - 2, 186, 16, 20],
        "goal_on_stomp": True,
        "reward_goal_distance_shaping": 2.0,
        "reward_progress_per_pixel": 0.0,
    }
    # Reachability remains an exact-physics check. Direction/phase variation
    # invalidates the old single time-indexed hold per tier. Pick a reachable
    # scripted demonstration; ordinary policy rollouts never execute it. Only
    # holds the jump executor can play (the NES jump menu) are tried, on the
    # layout as the sampler will play it.
    played = _as_played("stomp_mount", scenario)
    preferred = {"easy": 8, "medium": 10, "hard": 12}[difficulty]
    oracle_hold = preferred
    for hold in sorted(NES_JUMP_FRAMES, key=lambda h: (abs(h - preferred), h)):
        actions = _pad([2] * hold + [1] * 60)
        if validate_block_smb_monte_carlo_oracle(played, actions, max_steps=60)["reachable"]:
            oracle_hold = hold
            break
    actions = _pad([2] * oracle_hold + [1] * 60)
    return (
        scenario,
        {
            "enemy_distance": enemy_distance,
            "enemy_x": enemy_x,
            "enemy_speed": enemy_speed,
            "patrol_halfwidth": patrol_halfwidth,
            "enemy_initial_direction": direction,
            "frames_to_first_turn": turn_frames,
            "a_level_action": 2,
            "single_jump": True,
            "difficulty_bin": difficulty,
        },
        actions,
    )


def _platform_hop(
    rng: random.Random, difficulty: str
) -> tuple[dict[str, Any], dict[str, Any], list[int]]:
    # Single-jump teacher: land the one commanded jump ON a narrow platform
    # suspended over a pit. The platform is static and the goal sits on it,
    # so the episode is exactly one arc judged at its landing — riding a
    # MOVING platform is a jump-plus-wait composite and belongs to the
    # bridge families, not the single-jump foundation. Pit width bands make
    # the required hold grow monotonically; falling short or long is death.
    pit_width = {
        "easy": rng.randint(68, 78),
        "medium": rng.randint(88, 98),
        "hard": rng.randint(102, 110),
    }[difficulty]
    edge_x = 90
    platform_width = 24
    platform_x = edge_x + (pit_width - platform_width) // 2
    scenario = {
        "world_width": 380,
        "mario": [edge_x - 30, 200],
        "platforms": [
            [0, 220, edge_x, 20],
            [platform_x, 198, platform_width, 10],
            [edge_x + pit_width, 220, 380 - edge_x - pit_width, 20],
        ],
        "coins": [],
        "goal": [platform_x + 4, 178, 16, 20],
        "reward_goal_distance_shaping": 2.0,
        "goal_requires_support": True,
        "single_jump_attempt": True,
    }
    # A placeholder route: the sampler (_verified_route) always replaces it
    # with the middle reachable immediate jump on the NES jump menu, played
    # on the finished layout's running takeoff.
    oracle_hold = {"easy": 10, "medium": 12, "hard": 14}[difficulty]
    actions = _pad([2] * oracle_hold + [1] * 80)
    return (
        scenario,
        {
            "pit_width": pit_width,
            "platform_width": platform_width,
            "platform_x": platform_x,
            "a_level_action": 2,
            "single_jump": True,
            "difficulty_bin": difficulty,
        },
        actions,
    )


def _tall_pipe_jump(
    rng: random.Random, difficulty: str
) -> tuple[dict[str, Any], dict[str, Any], list[int]]:
    # A single tall pipe that must be jumped over/onto to reach the goal
    # beyond it. A spatial run ends at its destination before the separate
    # immediate jump, so this lesson must be reachable from rest. The old
    # 65-68px band relied on a continuous button-route run-up that the spatial
    # command contract cannot express. Match pipe_mount's supported ceiling.
    pipe_h = rng.randint(*{"easy": (56, 60), "medium": (60, 62), "hard": (63, 64)}[difficulty])
    pipe_x, pipe_w = 180, 30
    goal_x = rng.randint(266, 276)
    scenario = {
        "world_width": 320,
        "mario": [20, 200],
        "platforms": [
            [0, 220, 320, 20],
            [pipe_x, 220 - pipe_h, pipe_w, pipe_h],
        ],
        "platform_kinds": ["ground", "pipe"],
        "coins": [[pipe_x + 10, 220 - pipe_h - 20, 10, 10]],
        "goal": [goal_x, 200, 16, 20],
        "reward_goal_distance_shaping": 2.0,
    }
    actions = _pad([1] * 38 + [2] * 16 + [1] * 120)
    return (
        scenario,
        {
            "pipe_x": pipe_x,
            "pipe_width": pipe_w,
            "pipe_height": pipe_h,
            "goal_x": goal_x,
            "difficulty_bin": difficulty,
        },
        actions,
    )
