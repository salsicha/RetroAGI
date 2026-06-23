"""Game profile contracts for progressive-resolution stages."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from .actions import (
    SMB_ACTION_SPECS,
    ActionSpec,
    action_backend_id as _action_backend_id,
    action_button_vector as _action_button_vector,
)
from .rewards import RewardConfigSchema, RewardTermSpec
from .stage_resolution import STANDARD_STAGE_NAMES
from .synthetic import SyntheticDataSpec, SyntheticSplitSpec
from .tasks import GameTaskSchema, GameTaskSpec, TaskSuccessThreshold


@dataclass(frozen=True)
class AssetRequirement:
    """Asset source and provenance requirements for a game profile."""

    name: str
    required: bool
    local_path: str
    provenance: str
    license_notes: str

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("asset requirement name must be non-empty")
        if not self.local_path:
            raise ValueError(f"asset requirement {self.name!r} must define local_path")
        if self.required and not self.provenance:
            raise ValueError(f"required asset {self.name!r} must define provenance")
        if not self.license_notes:
            raise ValueError(f"asset requirement {self.name!r} must define license_notes")


@dataclass(frozen=True)
class StageLadderEntry:
    """One fidelity rung in a game's progressive-resolution ladder."""

    name: str
    stage_spec_name: str
    role: str
    required_artifacts: tuple[str, ...] = ()
    promotion_gate_summary: str = ""

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("stage ladder entry name must be non-empty")
        if not self.stage_spec_name:
            raise ValueError(f"stage ladder entry {self.name!r} must define stage_spec_name")
        if not self.role:
            raise ValueError(f"stage ladder entry {self.name!r} must define role")


@dataclass(frozen=True)
class BlockGameSpec:
    """Game-owned contract for a fast symbolic mid-fidelity simulator."""

    game_name: str
    name: str
    stage_name: str
    adapter: str
    environment: str
    physics: str
    observation_kind: str
    symbolic_state: tuple[str, ...]
    semantic_classes: tuple[str, ...]
    exact_label_sources: Mapping[str, str]
    fixed_scenarios: Mapping[str, str]
    procedural_scenario_generator: str
    reset_modes: tuple[str, ...]
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.game_name:
            raise ValueError("block game spec game_name must be non-empty")
        if not self.name:
            raise ValueError("block game spec name must be non-empty")
        if not self.stage_name:
            raise ValueError(f"block game spec {self.name!r} must define stage_name")
        if not self.adapter:
            raise ValueError(f"block game spec {self.name!r} must define adapter")
        if not self.environment:
            raise ValueError(f"block game spec {self.name!r} must define environment")
        if not self.physics:
            raise ValueError(f"block game spec {self.name!r} must define physics")
        if not self.observation_kind:
            raise ValueError(
                f"block game spec {self.name!r} must define observation_kind"
            )
        if not self.symbolic_state:
            raise ValueError(
                f"block game spec {self.name!r} must define symbolic_state"
            )
        if not self.semantic_classes:
            raise ValueError(
                f"block game spec {self.name!r} must define semantic_classes"
            )
        if not self.exact_label_sources:
            raise ValueError(
                f"block game spec {self.name!r} must define exact_label_sources"
            )
        if not self.fixed_scenarios:
            raise ValueError(
                f"block game spec {self.name!r} must define fixed_scenarios"
            )
        if not self.procedural_scenario_generator:
            raise ValueError(
                f"block game spec {self.name!r} must define procedural_scenario_generator"
            )
        if not self.reset_modes:
            raise ValueError(f"block game spec {self.name!r} must define reset_modes")

    @property
    def fixed_scenario_names(self) -> tuple[str, ...]:
        return tuple(self.fixed_scenarios)

    def fixed_scenario_source(self, name: str) -> str:
        try:
            return self.fixed_scenarios[name]
        except KeyError as exc:
            raise KeyError(
                f"unknown fixed block scenario {name!r} for {self.name!r}"
            ) from exc

    def exact_label_source(self, name: str) -> str:
        try:
            return self.exact_label_sources[name]
        except KeyError as exc:
            raise KeyError(
                f"unknown exact label source {name!r} for {self.name!r}"
            ) from exc


@dataclass(frozen=True)
class GameSpec:
    """Declarative game profile used by future multi-game stage registries."""

    name: str
    family: str
    action_space: tuple[ActionSpec, ...]
    observation_sources: tuple[str, ...]
    semantic_classes: tuple[str, ...]
    signal_schema: Mapping[str, str]
    reward_terms: Mapping[str, str]
    stage_ladder: tuple[StageLadderEntry, ...]
    emulator_backend: str
    asset_requirements: tuple[AssetRequirement, ...] = ()
    licensing: Mapping[str, str] = field(default_factory=dict)
    reward_schema: RewardConfigSchema | None = None
    task_schema: GameTaskSchema | None = None
    synthetic_data: tuple[SyntheticDataSpec, ...] = ()
    block_game: BlockGameSpec | None = None

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("game name must be non-empty")
        if not self.family:
            raise ValueError(f"game {self.name!r} must define family")
        if not self.action_space:
            raise ValueError(f"game {self.name!r} must define action_space")
        if not self.observation_sources:
            raise ValueError(f"game {self.name!r} must define observation_sources")
        if not self.semantic_classes:
            raise ValueError(f"game {self.name!r} must define semantic_classes")
        if not self.signal_schema:
            raise ValueError(f"game {self.name!r} must define signal_schema")
        if not self.reward_terms:
            raise ValueError(f"game {self.name!r} must define reward_terms")
        if not self.stage_ladder:
            raise ValueError(f"game {self.name!r} must define stage_ladder")
        if not self.emulator_backend:
            raise ValueError(f"game {self.name!r} must define emulator_backend")
        self._validate_actions()
        self._validate_stage_ladder()
        self._validate_reward_schema()
        self._validate_task_schema()
        self._validate_synthetic_data()
        self._validate_block_game()
        if not self.licensing:
            raise ValueError(f"game {self.name!r} must define licensing metadata")

    def _validate_actions(self) -> None:
        ids = [action.stable_id for action in self.action_space]
        expected = list(range(len(ids)))
        if ids != expected:
            raise ValueError(
                f"game {self.name!r} action stable IDs must be contiguous from zero; "
                f"got {ids}"
            )
        names = [action.name for action in self.action_space]
        if len(set(names)) != len(names):
            raise ValueError(f"game {self.name!r} action names must be unique")

    def _validate_stage_ladder(self) -> None:
        names = [stage.name for stage in self.stage_ladder]
        if len(set(names)) != len(names):
            raise ValueError(f"game {self.name!r} stage ladder names must be unique")
        unknown = sorted(set(names).difference(STANDARD_STAGE_NAMES))
        if unknown:
            raise ValueError(
                f"game {self.name!r} stage ladder uses non-standard names: {unknown}"
            )
        if self.stage_ladder[0].name != "synthetic":
            raise ValueError(f"game {self.name!r} stage ladder must start with synthetic")
        if self.stage_ladder[-1].name != "full":
            raise ValueError(f"game {self.name!r} stage ladder must end with full")

    def _validate_reward_schema(self) -> None:
        if self.reward_schema is None:
            return
        if self.reward_schema.game_name != self.name:
            raise ValueError(
                f"game {self.name!r} reward schema is for "
                f"{self.reward_schema.game_name!r}"
            )
        schema_terms = set(self.reward_schema.term_names)
        described_terms = set(self.reward_terms)
        missing = sorted(described_terms.difference(schema_terms))
        if missing:
            raise ValueError(
                f"game {self.name!r} reward schema is missing described terms: {missing}"
            )

    def _validate_task_schema(self) -> None:
        if self.task_schema is None:
            return
        if self.task_schema.game_name != self.name:
            raise ValueError(
                f"game {self.name!r} task schema is for {self.task_schema.game_name!r}"
            )
        stage_names = {stage.stage_spec_name for stage in self.stage_ladder}
        unsupported = sorted(
            {task.stage_name for task in self.task_schema.tasks}.difference(stage_names)
        )
        if unsupported:
            raise ValueError(
                f"game {self.name!r} task schema references unknown stages: {unsupported}"
            )

    def _validate_synthetic_data(self) -> None:
        names = [spec.name for spec in self.synthetic_data]
        if len(set(names)) != len(names):
            raise ValueError(f"game {self.name!r} synthetic data names must be unique")
        stage_names = {stage.stage_spec_name for stage in self.stage_ladder}
        for spec in self.synthetic_data:
            if spec.game_name != self.name:
                raise ValueError(
                    f"game {self.name!r} synthetic data {spec.name!r} is for "
                    f"{spec.game_name!r}"
                )
            if spec.stage_name not in stage_names:
                raise ValueError(
                    f"game {self.name!r} synthetic data {spec.name!r} references "
                    f"unknown stage {spec.stage_name!r}"
                )

    def _validate_block_game(self) -> None:
        if self.block_game is None:
            return
        if self.block_game.game_name != self.name:
            raise ValueError(
                f"game {self.name!r} block game spec {self.block_game.name!r} is for "
                f"{self.block_game.game_name!r}"
            )
        stage_names = {stage.stage_spec_name for stage in self.stage_ladder}
        if self.block_game.stage_name not in stage_names:
            raise ValueError(
                f"game {self.name!r} block game spec {self.block_game.name!r} "
                f"references unknown stage {self.block_game.stage_name!r}"
            )

    @property
    def action_count(self) -> int:
        return len(self.action_space)

    @property
    def stage_names(self) -> tuple[str, ...]:
        return tuple(stage.name for stage in self.stage_ladder)

    def action(self, value: int | str) -> ActionSpec:
        if isinstance(value, str):
            for action in self.action_space:
                if action.name == value:
                    return action
            raise KeyError(f"unknown action {value!r} for game {self.name!r}")
        return self.action_space[int(value)]

    def action_backend_id(self, value: int | str) -> int:
        """Return the stage-native discrete ID for a policy action."""

        return _action_backend_id(self.action(value))

    def action_button_vector(self, value: int | str, buttons: Sequence[str]):
        """Return a backend button vector for a policy action and button layout."""

        return _action_button_vector(self.action(value), buttons)

    def reward_config(self, values: Mapping[str, float] | None = None) -> dict[str, float]:
        """Return a validated reward-term config owned by this game profile."""

        if self.reward_schema is None:
            if values:
                raise ValueError(f"game {self.name!r} does not define a reward schema")
            return {}
        return self.reward_schema.validate(values)

    def task(self, name: str) -> GameTaskSpec:
        if self.task_schema is None:
            raise KeyError(f"game {self.name!r} does not define tasks")
        return self.task_schema.task(name)

    @property
    def fixed_tasks(self) -> tuple[GameTaskSpec, ...]:
        if self.task_schema is None:
            return ()
        return self.task_schema.fixed_tasks

    def synthetic_data_spec(self, name: str) -> SyntheticDataSpec:
        for spec in self.synthetic_data:
            if spec.name == name:
                return spec
        raise KeyError(f"unknown synthetic data spec {name!r} for game {self.name!r}")

    def block_game_spec(self) -> BlockGameSpec:
        if self.block_game is None:
            raise KeyError(f"game {self.name!r} does not define a block game spec")
        return self.block_game


SMB_REWARD_SCHEMA = RewardConfigSchema(
    game_name="smb",
    terms=(
        RewardTermSpec(
            name="progress",
            default=0.05,
            direction="positive",
            signal="progress",
            description="Forward progress reward per pixel",
        ),
        RewardTermSpec(
            name="coin",
            default=10.0,
            direction="positive",
            signal="collectibles.coins",
            description="Reward for collecting one coin",
        ),
        RewardTermSpec(
            name="enemy_stomp",
            default=5.0,
            direction="positive",
            signal="objectives.enemy_stomp",
            description="Reward for defeating an enemy safely",
        ),
        RewardTermSpec(
            name="goal",
            default=50.0,
            direction="positive",
            signal="completion",
            description="Terminal reward for reaching the goal",
        ),
        RewardTermSpec(
            name="fall_death",
            default=-10.0,
            direction="negative",
            signal="death",
            description="Penalty for falling below the viewport",
        ),
        RewardTermSpec(
            name="enemy_hit",
            default=-10.0,
            direction="negative",
            signal="objectives.enemy_hit",
            description="Penalty for unsafe enemy contact",
        ),
        RewardTermSpec(
            name="frame_penalty",
            default=-0.01,
            direction="negative",
            signal="time",
            description="Per-step time cost",
        ),
    ),
)


SMB_TASK_SCHEMA = GameTaskSchema(
    game_name="smb",
    tasks=(
        GameTaskSpec(
            name="level_1_flat.json",
            stage_name="block_smb",
            task_type="fixed",
            source="retroagi/stages/block_smb/scenarios/level_1_flat.json",
            reset_seed=101_001,
            curriculum_stage=1,
            success_threshold=TaskSuccessThreshold(
                min_success_rate=1.0,
                min_mean_return=55.0,
                min_episodes=3,
                max_steps=200,
                rationale=(
                    "Flat run: reach the goal reliably without relying on one "
                    "lucky rollout."
                ),
            ),
            description="Flat fixed Block SMB scenario",
        ),
        GameTaskSpec(
            name="level_2_gap.json",
            stage_name="block_smb",
            task_type="fixed",
            source="retroagi/stages/block_smb/scenarios/level_2_gap.json",
            reset_seed=101_002,
            curriculum_stage=2,
            success_threshold=TaskSuccessThreshold(
                min_success_rate=1.0,
                min_mean_return=55.0,
                min_episodes=3,
                max_steps=200,
                rationale=(
                    "Gap run: cross the gap and reach the goal reliably within "
                    "the time budget."
                ),
            ),
            description="Gap fixed Block SMB scenario",
        ),
        GameTaskSpec(
            name="level_3_stairs.json",
            stage_name="block_smb",
            task_type="fixed",
            source="retroagi/stages/block_smb/scenarios/level_3_stairs.json",
            reset_seed=101_003,
            curriculum_stage=3,
            success_threshold=TaskSuccessThreshold(
                min_success_rate=1.0,
                min_mean_return=55.0,
                min_episodes=3,
                max_steps=200,
                rationale=(
                    "Stair run: climb the stepped platforms and reach the "
                    "elevated goal."
                ),
            ),
            description="Stairs fixed Block SMB scenario",
        ),
        GameTaskSpec(
            name="level_4_platforms.json",
            stage_name="block_smb",
            task_type="fixed",
            source="retroagi/stages/block_smb/scenarios/level_4_platforms.json",
            reset_seed=101_004,
            curriculum_stage=4,
            success_threshold=TaskSuccessThreshold(
                min_success_rate=1.0,
                min_mean_return=55.0,
                min_episodes=3,
                max_steps=200,
                rationale=(
                    "Platform run: traverse separated platforms and reach the "
                    "final goal."
                ),
            ),
            description="Moving-platform fixed Block SMB scenario",
        ),
        GameTaskSpec(
            name="generated_block_smb",
            stage_name="block_smb",
            task_type="procedural",
            source="MarioScenarioEnv.generate_scenario",
            reset_seed=150_000,
            curriculum_stage=5,
            generation_seed=50_000,
            generation_config={
                "width_range": (256, 512),
                "gap_density": "curriculum",
                "enemy_density": "curriculum",
            },
            description="Procedural Block SMB generalization task template",
        ),
    ),
)


SMB_SYNTHETIC_DATA_SPECS = (
    SyntheticDataSpec(
        game_name="smb",
        name="synthetic_1d_concept",
        stage_name="synthetic_1d",
        observation_kind="procedural one-dimensional hierarchy tokens",
        target_kind="continuous controller targets and next-token hierarchy labels",
        generator="retroagi.stages.synthetic_1d.train.generate_dataset_splits",
        splits=(
            SyntheticSplitSpec("train", 1_000, 10_001),
            SyntheticSplitSpec("validation", 200, 20_001),
            SyntheticSplitSpec("test", 200, 30_001),
        ),
        shape_contract={
            "seq_len_a": 8,
            "ratio_ab": 2,
            "ratio_bc": 4,
            "vocab_size": 20,
            "tensors": ("xa", "ya", "xb", "yb", "xc", "yc"),
        },
        description=(
            "Cheap architecture-concept data used before SMB pixels, symbolic "
            "physics, or emulator frames."
        ),
        metadata={
            "purpose": "architecture validation",
            "pixel_free": True,
            "emulator_free": True,
        },
    ),
)


SMB_BLOCK_GAME_SPEC = BlockGameSpec(
    game_name="smb",
    name="block_smb",
    stage_name="block_smb",
    adapter="retroagi.stages.block_smb.adapter.BlockSMBStage",
    environment="retroagi.stages.block_smb.env.MarioScenarioEnv",
    physics="deterministic pygame-ce platformer physics",
    observation_kind="low-resolution RGB frame plus symbolic state_vec",
    symbolic_state=(
        "mario_position",
        "mario_velocity",
        "grounded",
        "facing",
        "camera_x",
        "max_x_reached",
        "nearest_coin",
        "nearest_enemy",
        "platform_below",
    ),
    semantic_classes=(
        "background",
        "mario",
        "platform",
        "coin",
        "goal",
        "enemy",
        "moving_platform",
    ),
    exact_label_sources={
        "semantics": "BlockVisionTransformer.semantic_targets",
        "position": "BlockVisionTransformer.position_target",
    },
    fixed_scenarios={
        "level_1_flat.json": "retroagi/stages/block_smb/scenarios/level_1_flat.json",
        "level_2_gap.json": "retroagi/stages/block_smb/scenarios/level_2_gap.json",
        "level_3_stairs.json": "retroagi/stages/block_smb/scenarios/level_3_stairs.json",
        "level_4_platforms.json": (
            "retroagi/stages/block_smb/scenarios/level_4_platforms.json"
        ),
    },
    procedural_scenario_generator="MarioScenarioEnv.generate_scenario",
    reset_modes=("fixed_scenario", "procedural_seed"),
    metadata={
        "fast_reset": True,
        "exact_semantic_labels": True,
        "procedural_scenarios": True,
        "simplified_physics": True,
    },
)


SMB_GAME_SPEC = GameSpec(
    name="smb",
    family="super_mario_bros",
    action_space=SMB_ACTION_SPECS,
    observation_sources=(
        "synthetic_1d_sequences",
        "block_smb_rgb_frames",
        "full_smb_emulator_rgb_frames",
        "full_smb_backend_variables",
    ),
    semantic_classes=(
        "sky",
        "ground",
        "brick",
        "question_block",
        "pipe",
        "coin",
        "goomba",
        "koopa",
        "mario",
        "mushroom",
        "hill",
        "cloud",
        "bush",
    ),
    signal_schema={
        "progress": "x position or normalized horizontal progress",
        "score": "game score when available",
        "coins": "coin count",
        "lives": "remaining lives",
        "completion": "level completion flag",
        "death": "death or fall flag",
        "timeout": "time-limit or max-step truncation",
    },
    reward_terms={
        "progress": "positive reward for forward movement",
        "coin": "positive reward for collection",
        "enemy_stomp": "positive reward for safe enemy defeat",
        "enemy_hit": "negative reward for unsafe collision",
        "fall_death": "negative terminal reward for death",
        "goal": "positive terminal reward for completion",
        "frame_penalty": "small per-step cost",
    },
    reward_schema=SMB_REWARD_SCHEMA,
    task_schema=SMB_TASK_SCHEMA,
    synthetic_data=SMB_SYNTHETIC_DATA_SPECS,
    block_game=SMB_BLOCK_GAME_SPEC,
    stage_ladder=(
        StageLadderEntry(
            name="synthetic",
            stage_spec_name="synthetic_1d",
            role="architecture validation before game-specific training",
        ),
        StageLadderEntry(
            name="block",
            stage_spec_name="block_smb",
            role="simplified synthetic SMB model training",
            required_artifacts=("data/block_vit/block_vit.pth", "data/block_smb/policy.pth"),
            promotion_gate_summary="fixed-scenario success thresholds",
        ),
        StageLadderEntry(
            name="full_asset_mock",
            stage_spec_name="full_smb",
            role="Full SMB ViT bootstrap on full-game assets in synthetic scenes",
            required_artifacts=("data/vit/full_smb_vit.pth",),
            promotion_gate_summary="held-out semantic and position metrics",
        ),
        StageLadderEntry(
            name="full",
            stage_spec_name="full_smb",
            role="full emulator inference validation and continued training",
            required_artifacts=(
                "data/vit/full_smb_vit.pth",
                "data/full_smb/transferred_policy.pth",
            ),
            promotion_gate_summary="inference, transfer, comparison, and training metrics",
        ),
    ),
    emulator_backend="stable-retro",
    asset_requirements=(
        AssetRequirement(
            name="smb_sprites",
            required=True,
            local_path="assets/sprites/",
            provenance=(
                "Extracted by scripts/vit/extract_sprites.py from documented "
                "SMB sprite sources"
            ),
            license_notes="Record upstream source and usage terms before committing assets",
        ),
        AssetRequirement(
            name="smb_rom",
            required=True,
            local_path="local stable-retro import",
            provenance="User-provided legally obtained Super Mario Bros NES ROM",
            license_notes=(
                "Do not commit ROM content; required only for real Full SMB emulator runs"
            ),
        ),
    ),
    licensing={
        "assets": "Document source, license, and redistribution status for each sprite source",
        "rom": "User-owned local content; never committed to the repository",
        "generated_data": "Generated datasets must record asset provenance in metadata",
    },
)


GAME_SPECS: Mapping[str, GameSpec] = {SMB_GAME_SPEC.name: SMB_GAME_SPEC}


def game_names() -> tuple[str, ...]:
    return tuple(sorted(GAME_SPECS))


def get_game_spec(name: str) -> GameSpec:
    try:
        return GAME_SPECS[name]
    except KeyError as exc:
        available = ", ".join(game_names())
        raise KeyError(f"unknown game {name!r}; available: {available}") from exc


def validate_game_spec(game: GameSpec) -> GameSpec:
    """Return ``game`` after dataclass construction-time validation."""

    return GameSpec(
        name=game.name,
        family=game.family,
        action_space=tuple(game.action_space),
        observation_sources=tuple(game.observation_sources),
        semantic_classes=tuple(game.semantic_classes),
        signal_schema=dict(game.signal_schema),
        reward_terms=dict(game.reward_terms),
        reward_schema=game.reward_schema,
        task_schema=game.task_schema,
        synthetic_data=tuple(game.synthetic_data),
        block_game=game.block_game,
        stage_ladder=tuple(game.stage_ladder),
        emulator_backend=game.emulator_backend,
        asset_requirements=tuple(game.asset_requirements),
        licensing=dict(game.licensing),
    )


def assert_stage_ladder(game: GameSpec, expected: Sequence[str]) -> None:
    """Raise a clear error when a game profile does not expose a required ladder."""

    expected_tuple = tuple(expected)
    if game.stage_names != expected_tuple:
        raise ValueError(
            f"game {game.name!r} stage ladder {game.stage_names!r} does not match "
            f"expected {expected_tuple!r}"
        )
