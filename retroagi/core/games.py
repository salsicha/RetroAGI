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
from .backends import BackendCapabilitySpec, GameBackendSpec
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
class AssetChecklistItem:
    """One required provenance or licensing check for game assets and datasets."""

    name: str
    target: str
    stage_names: tuple[str, ...]
    evidence: tuple[str, ...]
    policy: str
    required: bool = True
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("asset checklist item name must be non-empty")
        if not self.target:
            raise ValueError(
                f"asset checklist item {self.name!r} must define target"
            )
        if not self.stage_names:
            raise ValueError(
                f"asset checklist item {self.name!r} must define stage_names"
            )
        empty_stages = [stage for stage in self.stage_names if not stage]
        if empty_stages:
            raise ValueError(
                f"asset checklist item {self.name!r} stage_names must be non-empty"
            )
        if not self.evidence:
            raise ValueError(
                f"asset checklist item {self.name!r} must define evidence"
            )
        empty_evidence = [item for item in self.evidence if not item]
        if empty_evidence:
            raise ValueError(
                f"asset checklist item {self.name!r} evidence must be non-empty"
            )
        if not self.policy:
            raise ValueError(
                f"asset checklist item {self.name!r} must define policy"
            )

    def to_manifest(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "target": self.target,
            "stage_names": list(self.stage_names),
            "evidence": list(self.evidence),
            "policy": self.policy,
            "required": self.required,
            "metadata": dict(self.metadata),
        }


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
    backend: GameBackendSpec | None = None
    asset_requirements: tuple[AssetRequirement, ...] = ()
    asset_checklist: tuple[AssetChecklistItem, ...] = ()
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
        if not self.licensing:
            raise ValueError(f"game {self.name!r} must define licensing metadata")
        if self.backend is not None and self.backend.name != self.emulator_backend:
            raise ValueError(
                f"game {self.name!r} backend spec {self.backend.name!r} must "
                f"match emulator_backend {self.emulator_backend!r}"
            )
        self._validate_actions()
        self._validate_stage_ladder()
        self._validate_reward_schema()
        self._validate_task_schema()
        self._validate_synthetic_data()
        self._validate_block_game()
        self._validate_asset_checklist()

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

    def _validate_asset_checklist(self) -> None:
        needs_checklist = bool(self.asset_requirements or self.synthetic_data)
        if needs_checklist and not self.asset_checklist:
            raise ValueError(
                f"game {self.name!r} must define asset_checklist when assets "
                "or generated data are used"
            )

        names = [item.name for item in self.asset_checklist]
        if len(set(names)) != len(names):
            raise ValueError(
                f"game {self.name!r} asset checklist item names must be unique"
            )

        asset_names = {asset.name for asset in self.asset_requirements}
        synthetic_names = {spec.name for spec in self.synthetic_data}
        allowed_targets = asset_names.union(synthetic_names).union(self.licensing)
        stage_names = set(self.stage_names)
        for item in self.asset_checklist:
            if item.target not in allowed_targets:
                raise ValueError(
                    f"game {self.name!r} asset checklist item {item.name!r} "
                    f"references unknown target {item.target!r}"
                )
            unknown_stages = sorted(set(item.stage_names).difference(stage_names))
            if unknown_stages:
                raise ValueError(
                    f"game {self.name!r} asset checklist item {item.name!r} "
                    f"references unknown stages: {unknown_stages}"
                )

        required_targets = {
            item.target for item in self.asset_checklist if item.required
        }
        missing_assets = sorted(
            asset.name
            for asset in self.asset_requirements
            if asset.required and asset.name not in required_targets
        )
        if missing_assets:
            raise ValueError(
                f"game {self.name!r} asset checklist must cover required "
                f"assets: {missing_assets}"
            )
        if self.synthetic_data and not (
            "generated_data" in required_targets
            or required_targets.intersection(synthetic_names)
        ):
            raise ValueError(
                f"game {self.name!r} asset checklist must cover generated data"
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

    def asset_checklist_item(self, name: str) -> AssetChecklistItem:
        for item in self.asset_checklist:
            if item.name == name:
                return item
        raise KeyError(f"unknown asset checklist item {name!r} for game {self.name!r}")

    def block_game_spec(self) -> BlockGameSpec:
        if self.block_game is None:
            raise KeyError(f"game {self.name!r} does not define a block game spec")
        return self.block_game

    def backend_spec(self) -> GameBackendSpec:
        if self.backend is not None:
            return self.backend
        return GameBackendSpec(
            name=self.emulator_backend,
            provider_kind="custom",
            entrypoint="game profile",
            observation_api="stage-native observations",
            action_api="stage-native actions",
            metadata={"implicit": True},
        )


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
            description="Separated-platform fixed Block SMB scenario",
        ),
        GameTaskSpec(
            name="level_5_enemy_hop.json",
            stage_name="block_smb",
            task_type="fixed",
            source="retroagi/stages/block_smb/scenarios/level_5_enemy_hop.json",
            reset_seed=101_005,
            curriculum_stage=5,
            success_threshold=TaskSuccessThreshold(
                min_success_rate=1.0,
                min_mean_return=55.0,
                min_episodes=3,
                max_steps=200,
                rationale=(
                    "Enemy-hop run: clear the ground enemy and reach the goal "
                    "without unsafe contact."
                ),
            ),
            description="Single-enemy avoidance fixed Block SMB scenario",
        ),
        GameTaskSpec(
            name="level_6_enemy_patrol.json",
            stage_name="block_smb",
            task_type="fixed",
            source="retroagi/stages/block_smb/scenarios/level_6_enemy_patrol.json",
            reset_seed=101_006,
            curriculum_stage=6,
            success_threshold=TaskSuccessThreshold(
                min_success_rate=1.0,
                min_mean_return=55.0,
                min_episodes=3,
                max_steps=200,
                rationale=(
                    "Enemy-patrol run: time jumps over multiple patrolling "
                    "enemies and reach the goal."
                ),
            ),
            description="Patrolling-enemy avoidance fixed Block SMB scenario",
        ),
        GameTaskSpec(
            name="level_7_moving_bridge.json",
            stage_name="block_smb",
            task_type="fixed",
            source="retroagi/stages/block_smb/scenarios/level_7_moving_bridge.json",
            reset_seed=101_007,
            curriculum_stage=7,
            success_threshold=TaskSuccessThreshold(
                min_success_rate=1.0,
                min_mean_return=55.0,
                min_episodes=3,
                max_steps=200,
                rationale=(
                    "Moving-bridge run: use the moving platform as the gap "
                    "bridge and reach the goal."
                ),
            ),
            description="Moving-platform traversal fixed Block SMB scenario",
        ),
        GameTaskSpec(
            name="level_8_enemy_gap.json",
            stage_name="block_smb",
            task_type="fixed",
            source="retroagi/stages/block_smb/scenarios/level_8_enemy_gap.json",
            reset_seed=101_008,
            curriculum_stage=8,
            success_threshold=TaskSuccessThreshold(
                min_success_rate=1.0,
                min_mean_return=55.0,
                min_episodes=3,
                max_steps=200,
                rationale=(
                    "Enemy-gap run: combine a gap crossing with safe enemy "
                    "avoidance or stomp timing before the goal."
                ),
            ),
            description="Combined gap-and-enemy fixed Block SMB scenario",
        ),
        GameTaskSpec(
            name="level_9_enemy_stomp.json",
            stage_name="block_smb",
            task_type="fixed",
            source="retroagi/stages/block_smb/scenarios/level_9_enemy_stomp.json",
            reset_seed=101_009,
            curriculum_stage=9,
            success_threshold=TaskSuccessThreshold(
                min_success_rate=1.0,
                min_mean_return=55.0,
                min_episodes=3,
                max_steps=200,
                rationale=(
                    "Enemy-stomp run: clear a blocking ground enemy with safe "
                    "top contact and reach the goal."
                ),
            ),
            description="Enemy-stomp fixed Block SMB scenario",
        ),
        GameTaskSpec(
            name="generated_block_smb",
            stage_name="block_smb",
            task_type="procedural",
            source="MarioScenarioEnv.generate_scenario",
            reset_seed=150_000,
            curriculum_stage=10,
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
        "step_progress",
        "goal_delta",
        "support_edge",
        "next_platform_delta",
        "ground_ahead",
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
        "level_5_enemy_hop.json": (
            "retroagi/stages/block_smb/scenarios/level_5_enemy_hop.json"
        ),
        "level_6_enemy_patrol.json": (
            "retroagi/stages/block_smb/scenarios/level_6_enemy_patrol.json"
        ),
        "level_7_moving_bridge.json": (
            "retroagi/stages/block_smb/scenarios/level_7_moving_bridge.json"
        ),
        "level_8_enemy_gap.json": (
            "retroagi/stages/block_smb/scenarios/level_8_enemy_gap.json"
        ),
        "level_9_enemy_stomp.json": (
            "retroagi/stages/block_smb/scenarios/level_9_enemy_stomp.json"
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
    backend=GameBackendSpec(
        name="stable-retro",
        provider_kind="stable_retro",
        entrypoint="retro.make",
        observation_api="Gymnasium-style RGB observation",
        action_api="MultiBinary button vector from backend button names",
        reset_api="reset(seed=None)",
        step_api="step(button_vector) -> obs, reward, terminated, truncated, info",
        state_api="env.get_state/set_state or env.em.get_state/set_state",
        capabilities=BackendCapabilitySpec(
            reset_seed=True,
            save_load_state=True,
            frame_step=True,
            action_repeat=True,
            render=True,
            headless=True,
            gymnasium_step_api=True,
            legacy_gym_step_api=True,
        ),
        metadata={
            "game": "SuperMarioBros-Nes",
            "wrapper": "retroagi.core.backends.GymnasiumBackendAdapter",
        },
    ),
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
    asset_checklist=(
        AssetChecklistItem(
            name="smb_sprites_source_license",
            target="smb_sprites",
            stage_names=("full_asset_mock",),
            evidence=(
                "source_url_or_repository",
                "license_or_terms_summary",
                "redistribution_decision",
                "crop_coordinates_or_extraction_manifest",
                "local_path_manifest",
            ),
            policy=(
                "Record sprite source, license terms, redistribution decision, "
                "and extraction details before committing sprites or generated "
                "asset-mock datasets."
            ),
        ),
        AssetChecklistItem(
            name="smb_rom_local_only",
            target="smb_rom",
            stage_names=("full",),
            evidence=(
                "user_owned_rom_confirmation",
                "stable_retro_import_name",
                "local_checksum_record",
                "gitignore_or_artifact_exclusion",
            ),
            policy=(
                "ROM content must remain user-provided local content and must "
                "not be committed or bundled with generated artifacts."
            ),
        ),
        AssetChecklistItem(
            name="smb_generated_data_provenance",
            target="generated_data",
            stage_names=("synthetic", "block", "full_asset_mock"),
            evidence=(
                "generator_entrypoint",
                "resolved_config",
                "split_seeds",
                "source_asset_versions",
                "dataset_metadata",
            ),
            policy=(
                "Generated datasets must record generator code, config, seeds, "
                "source asset provenance, and redistribution status before they "
                "are committed or referenced by checkpoints."
            ),
        ),
    ),
    licensing={
        "assets": "Document source, license, and redistribution status for each sprite source",
        "rom": "User-owned local content; never committed to the repository",
        "generated_data": "Generated datasets must record asset provenance in metadata",
    },
)


PONG_ACTION_SPECS = (
    ActionSpec(
        name="noop",
        stable_id=0,
        release_all=True,
        backend_action_id=0,
        description="Hold the paddle steady",
    ),
    ActionSpec(
        name="up",
        stable_id=1,
        buttons=("UP",),
        backend_action_id=2,
        description="Move the controlled paddle upward",
    ),
    ActionSpec(
        name="down",
        stable_id=2,
        buttons=("DOWN",),
        backend_action_id=5,
        description="Move the controlled paddle downward",
    ),
)


PONG_REWARD_SCHEMA = RewardConfigSchema(
    game_name="pong",
    terms=(
        RewardTermSpec(
            name="paddle_alignment",
            default=0.02,
            direction="positive",
            signal="objectives.paddle_alignment",
            description="Reward for reducing paddle-to-ball vertical error",
        ),
        RewardTermSpec(
            name="rally_hit",
            default=1.0,
            direction="positive",
            signal="objectives.rally_hit",
            description="Reward for returning the ball",
        ),
        RewardTermSpec(
            name="score_point",
            default=5.0,
            direction="positive",
            signal="score_delta",
            description="Reward for scoring a point",
        ),
        RewardTermSpec(
            name="concede_point",
            default=-5.0,
            direction="negative",
            signal="objectives.concede_point",
            description="Penalty for missing the ball and conceding a point",
        ),
        RewardTermSpec(
            name="frame_penalty",
            default=-0.001,
            direction="negative",
            signal="time",
            description="Small per-frame cost to prefer decisive returns",
        ),
    ),
)


PONG_TASK_SCHEMA = GameTaskSchema(
    game_name="pong",
    tasks=(
        GameTaskSpec(
            name="centered_return",
            stage_name="pong_block",
            task_type="fixed",
            source="retroagi/stages/pong_block/scenarios/centered_return.json",
            reset_seed=202_001,
            curriculum_stage=1,
            success_threshold=TaskSuccessThreshold(
                min_success_rate=0.9,
                min_mean_return=1.0,
                min_episodes=5,
                max_steps=300,
                rationale=(
                    "Centered serve: return most rallies from a controlled "
                    "neutral start before adding angles."
                ),
            ),
            description="Block Pong fixed task with centered serves",
        ),
        GameTaskSpec(
            name="angled_return",
            stage_name="pong_block",
            task_type="fixed",
            source="retroagi/stages/pong_block/scenarios/angled_return.json",
            reset_seed=202_002,
            curriculum_stage=2,
            success_threshold=TaskSuccessThreshold(
                min_success_rate=0.8,
                min_mean_return=0.5,
                min_episodes=5,
                max_steps=360,
                rationale=(
                    "Angled serve: track vertical velocity changes and return "
                    "a majority of harder rallies."
                ),
            ),
            description="Block Pong fixed task with angled ball trajectories",
        ),
        GameTaskSpec(
            name="generated_pong_rallies",
            stage_name="pong_block",
            task_type="procedural",
            source="PongBlockEnv.generate_rally",
            reset_seed=202_100,
            curriculum_stage=3,
            generation_seed=252_000,
            generation_config={
                "ball_speed_range": (0.5, 1.25),
                "serve_angle_range": (-0.7, 0.7),
                "opponent_policy": "scripted_tracking",
            },
            description="Procedural Block Pong rally generalization template",
        ),
    ),
)


PONG_SYNTHETIC_DATA_SPECS = (
    SyntheticDataSpec(
        game_name="pong",
        name="pong_scalar_control",
        stage_name="synthetic_1d",
        observation_kind="scalar paddle, ball position, and velocity sequences",
        target_kind="paddle action labels and next-state trajectory targets",
        generator="retroagi.stages.synthetic_1d.train.generate_dataset_splits",
        splits=(
            SyntheticSplitSpec("train", 1_000, 212_001),
            SyntheticSplitSpec("validation", 200, 222_001),
            SyntheticSplitSpec("test", 200, 232_001),
        ),
        shape_contract={
            "seq_len_a": 8,
            "ratio_ab": 2,
            "ratio_bc": 4,
            "vocab_size": 20,
            "features": (
                "paddle_y",
                "ball_x",
                "ball_y",
                "ball_vx",
                "ball_vy",
            ),
            "targets": ("action_id", "next_paddle_y", "next_ball_y"),
        },
        description=(
            "Cheap Pong control data for architecture validation before "
            "raster observations or high-fidelity Gymnasium frames."
        ),
        metadata={
            "purpose": "non-SMB architecture validation",
            "pixel_free": True,
            "emulator_free": True,
        },
    ),
)


PONG_BLOCK_GAME_SPEC = BlockGameSpec(
    game_name="pong",
    name="pong_block",
    stage_name="pong_block",
    adapter="retroagi.stages.pong_block.adapter.PongBlockStage",
    environment="retroagi.stages.pong_block.env.PongBlockEnv",
    physics="deterministic 2D paddle-ball physics with scripted opponent",
    observation_kind="low-resolution paddle/ball raster plus symbolic state_vec",
    symbolic_state=(
        "paddle_y",
        "opponent_y",
        "ball_position",
        "ball_velocity",
        "rally_length",
        "score_delta",
        "serve_direction",
    ),
    semantic_classes=(
        "background",
        "agent_paddle",
        "opponent_paddle",
        "ball",
        "wall",
        "score_hud",
    ),
    exact_label_sources={
        "semantics": "PongBlockEnv.semantic_targets",
        "position": "PongBlockEnv.ball_and_paddle_targets",
    },
    fixed_scenarios={
        "centered_return": (
            "retroagi/stages/pong_block/scenarios/centered_return.json"
        ),
        "angled_return": (
            "retroagi/stages/pong_block/scenarios/angled_return.json"
        ),
    },
    procedural_scenario_generator="PongBlockEnv.generate_rally",
    reset_modes=("fixed_scenario", "procedural_seed"),
    metadata={
        "fast_reset": True,
        "exact_semantic_labels": True,
        "procedural_scenarios": True,
        "simplified_physics": True,
        "profile_status": "proof_of_concept",
    },
)


PONG_GAME_SPEC = GameSpec(
    name="pong",
    family="pong",
    action_space=PONG_ACTION_SPECS,
    observation_sources=(
        "pong_scalar_control_sequences",
        "pong_block_raster_frames",
        "pong_full_gymnasium_rgb_frames",
        "pong_backend_variables",
    ),
    semantic_classes=(
        "background",
        "agent_paddle",
        "opponent_paddle",
        "ball",
        "wall",
        "score_hud",
    ),
    signal_schema={
        "progress": "rally length or normalized ball return progress",
        "score_delta": "agent point differential for the current episode",
        "paddle_y": "controlled paddle vertical position",
        "ball_position": "ball x/y position",
        "ball_velocity": "ball x/y velocity",
        "completion": "task success or point scored",
        "death": "point conceded or terminal miss",
        "timeout": "max-step truncation",
    },
    reward_terms={
        "paddle_alignment": "positive shaping for tracking the ball",
        "rally_hit": "positive reward for returning the ball",
        "score_point": "positive reward for scoring",
        "concede_point": "negative reward for conceding",
        "frame_penalty": "small per-frame cost",
    },
    reward_schema=PONG_REWARD_SCHEMA,
    task_schema=PONG_TASK_SCHEMA,
    synthetic_data=PONG_SYNTHETIC_DATA_SPECS,
    block_game=PONG_BLOCK_GAME_SPEC,
    stage_ladder=(
        StageLadderEntry(
            name="synthetic",
            stage_spec_name="synthetic_1d",
            role="scalar paddle-ball control concept validation",
            required_artifacts=("data/pong/synthetic_policy.pth",),
            promotion_gate_summary="controller prediction and action-label metrics",
        ),
        StageLadderEntry(
            name="block",
            stage_spec_name="pong_block",
            role="simplified Pong paddle-ball training with exact labels",
            required_artifacts=(
                "data/pong_block/pong_block_vit.pth",
                "data/pong_block/policy.pth",
            ),
            promotion_gate_summary="fixed-rally return thresholds",
        ),
        StageLadderEntry(
            name="full",
            stage_spec_name="pong_full",
            role="Gymnasium Pong inference validation and continued training",
            required_artifacts=("data/pong/full_policy.pth",),
            promotion_gate_summary="full-frame inference and score differential",
        ),
    ),
    emulator_backend="gymnasium-pong",
    backend=GameBackendSpec(
        name="gymnasium-pong",
        provider_kind="gymnasium",
        entrypoint="gymnasium.make",
        observation_api="Gymnasium RGB frame or native observation",
        action_api="Discrete action ID from ActionSpec.backend_action_id",
        reset_api="reset(seed=None)",
        step_api="step(action_id) -> obs, reward, terminated, truncated, info",
        state_api="clone_state/restore_state when supplied by the environment",
        capabilities=BackendCapabilitySpec(
            reset_seed=True,
            save_load_state=False,
            frame_step=True,
            action_repeat=True,
            render=True,
            headless=True,
            gymnasium_step_api=True,
            legacy_gym_step_api=False,
        ),
        metadata={
            "env_id": "ALE/Pong-v5",
            "full_rung_status": "planned",
            "wrapper": "retroagi.core.backends.GymnasiumBackendAdapter",
        },
    ),
    asset_checklist=(
        AssetChecklistItem(
            name="pong_generated_data_provenance",
            target="generated_data",
            stage_names=("synthetic", "block"),
            evidence=(
                "generator_entrypoint",
                "resolved_config",
                "split_seeds",
                "simulator_version",
                "dataset_metadata",
            ),
            policy=(
                "Generated Pong datasets must record generator code, config, "
                "seeds, simulator version, and dataset metadata before they "
                "are committed or referenced by checkpoints."
            ),
        ),
    ),
    licensing={
        "assets": "No repository-managed Pong art assets are required for this POC",
        "generated_data": "Generated datasets must record simulator provenance",
        "emulator": (
            "Full Gymnasium/ALE Pong remains an optional local dependency and "
            "is not required for the profile-only proof of concept"
        ),
    },
)


GAME_SPECS: Mapping[str, GameSpec] = {
    PONG_GAME_SPEC.name: PONG_GAME_SPEC,
    SMB_GAME_SPEC.name: SMB_GAME_SPEC,
}


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
        backend=game.backend,
        asset_requirements=tuple(game.asset_requirements),
        asset_checklist=tuple(game.asset_checklist),
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
