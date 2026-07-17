"""Game plugin registry for multi-game progressive-resolution components."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

from .actions import SMB_ACTIONS
from .game_promotion import (
    GamePromotionGateSpec,
    PromotionArtifactGateSpec,
    PromotionMetricGateSpec,
)
from .games import PONG_GAME_SPEC, SMB_GAME_SPEC, GameSpec
from .perception import (
    PerceptionDatasetSourceSpec,
    PerceptionPipelineSpec,
    SemanticVocabularySpec,
)
from .rewards import RewardConfigSchema
from .stage_resolution import StageResolution, resolve_game_stage
from .tasks import TaskSuccessThreshold


@dataclass(frozen=True)
class GamePluginSpec:
    """Named component entrypoints owned by one game profile."""

    name: str
    game: GameSpec
    stage_adapters: Mapping[str, str]
    vision_encoders: Mapping[str, str]
    asset_pipelines: Mapping[str, str] = field(default_factory=dict)
    perception_pipelines: Mapping[str, PerceptionPipelineSpec] = field(default_factory=dict)
    signal_extractors: Mapping[str, str] = field(default_factory=dict)
    reward_schema: RewardConfigSchema | None = None
    success_thresholds: Mapping[str, TaskSuccessThreshold] = field(default_factory=dict)
    promotion_gates: Mapping[str, GamePromotionGateSpec] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("game plugin name must be non-empty")
        if self.name != self.game.name:
            raise ValueError(
                f"game plugin {self.name!r} must match game profile " f"{self.game.name!r}"
            )
        if not self.stage_adapters:
            raise ValueError(f"game plugin {self.name!r} must define stage_adapters")
        if not self.vision_encoders:
            raise ValueError(f"game plugin {self.name!r} must define vision_encoders")
        stage_names = set(self.game.stage_names)
        self._validate_named_components("stage adapter", self.stage_adapters, stage_names)
        self._validate_named_components("vision encoder", self.vision_encoders, stage_names)
        if self.reward_schema is not None and self.reward_schema.game_name != self.name:
            raise ValueError(
                f"game plugin {self.name!r} reward schema is for "
                f"{self.reward_schema.game_name!r}"
            )
        task_names = {task.name for task in self.game.fixed_tasks}
        unknown_thresholds = sorted(set(self.success_thresholds).difference(task_names))
        if unknown_thresholds:
            raise ValueError(
                f"game plugin {self.name!r} success thresholds reference "
                f"unknown fixed tasks: {unknown_thresholds}"
            )
        if self.asset_pipelines:
            self._validate_named_components(
                "asset pipeline",
                self.asset_pipelines,
                stage_names.union({"assets", "perception"}),
            )
        if self.signal_extractors:
            self._validate_named_components(
                "signal extractor",
                self.signal_extractors,
                stage_names,
            )
        self._validate_perception_pipelines()
        unknown_gate_names = sorted(
            name for name, gate in self.promotion_gates.items() if name != gate.rung_name
        )
        if unknown_gate_names:
            raise ValueError(
                f"game plugin {self.name!r} promotion gate keys must match "
                f"rung names: {unknown_gate_names}"
            )

    def _validate_named_components(
        self,
        kind: str,
        components: Mapping[str, str],
        allowed_names: set[str],
    ) -> None:
        unknown = sorted(set(components).difference(allowed_names))
        if unknown:
            raise ValueError(
                f"game plugin {self.name!r} {kind}s reference unknown names: {unknown}"
            )
        empty = sorted(name for name, entrypoint in components.items() if not entrypoint)
        if empty:
            raise ValueError(f"game plugin {self.name!r} {kind}s must define entrypoints: {empty}")

    def _validate_perception_pipelines(self) -> None:
        for name, pipeline in self.perception_pipelines.items():
            if name != pipeline.name:
                raise ValueError(
                    f"game plugin {self.name!r} perception pipeline keys must "
                    f"match pipeline names: {name!r} != {pipeline.name!r}"
                )
            if pipeline.game_name != self.name:
                raise ValueError(
                    f"game plugin {self.name!r} perception pipeline {name!r} "
                    f"is for {pipeline.game_name!r}"
                )
            try:
                self.resolve_stage(pipeline.stage_name)
            except KeyError as exc:
                raise ValueError(
                    f"game plugin {self.name!r} perception pipeline {name!r} "
                    f"references unknown stage {pipeline.stage_name!r}"
                ) from exc
            for source in pipeline.dataset_sources:
                for source_stage in source.stage_names:
                    try:
                        self.resolve_stage(source_stage)
                    except KeyError as exc:
                        raise ValueError(
                            f"game plugin {self.name!r} perception pipeline "
                            f"{name!r} source {source.name!r} references "
                            f"unknown stage {source_stage!r}"
                        ) from exc

    def stage_adapter(self, name: str) -> str:
        stage = self.resolve_stage(name)
        try:
            return self.stage_adapters[stage.name]
        except KeyError as exc:
            raise KeyError(
                f"unknown stage adapter {stage.name!r} for game plugin {self.name!r}"
            ) from exc

    def vision_encoder(self, name: str) -> str:
        stage = self.resolve_stage(name)
        try:
            return self.vision_encoders[stage.name]
        except KeyError as exc:
            raise KeyError(
                f"unknown vision encoder {stage.name!r} for game plugin {self.name!r}"
            ) from exc

    def resolve_stage(self, name: str) -> StageResolution:
        return resolve_game_stage(self.game, name)

    def asset_pipeline(self, name: str) -> str:
        try:
            return self.asset_pipelines[name]
        except KeyError as exc:
            raise KeyError(
                f"unknown asset pipeline {name!r} for game plugin {self.name!r}"
            ) from exc

    def perception_pipeline(self, name: str) -> PerceptionPipelineSpec:
        if name in self.perception_pipelines:
            return self.perception_pipelines[name]
        try:
            stage = self.resolve_stage(name)
        except KeyError as exc:
            raise KeyError(
                f"unknown perception pipeline {name!r} for game plugin {self.name!r}"
            ) from exc
        for pipeline in self.perception_pipelines.values():
            if pipeline.stage_name == stage.name:
                return pipeline
        raise KeyError(f"unknown perception pipeline {name!r} for game plugin {self.name!r}")

    def signal_extractor(self, name: str) -> str:
        stage = self.resolve_stage(name)
        try:
            return self.signal_extractors[stage.name]
        except KeyError as exc:
            raise KeyError(
                f"unknown signal extractor {stage.name!r} for game plugin {self.name!r}"
            ) from exc

    def reward_config(self, values: Mapping[str, float] | None = None) -> dict[str, float]:
        if self.reward_schema is None:
            return self.game.reward_config(values)
        return self.reward_schema.validate(values)

    def success_threshold(self, name: str) -> TaskSuccessThreshold:
        try:
            return self.success_thresholds[name]
        except KeyError as exc:
            raise KeyError(
                f"unknown success threshold {name!r} for game plugin {self.name!r}"
            ) from exc

    def promotion_gate(self, rung_name: str) -> GamePromotionGateSpec | None:
        return self.promotion_gates.get(rung_name)


class GamePluginRegistry:
    """Lookup table for game profiles and their component entrypoints."""

    def __init__(self, plugins: tuple[GamePluginSpec, ...]):
        if not plugins:
            raise ValueError("game plugin registry must define at least one plugin")
        names = [plugin.name for plugin in plugins]
        if len(set(names)) != len(names):
            raise ValueError("game plugin registry names must be unique")
        self._plugins = {plugin.name: plugin for plugin in plugins}

    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._plugins))

    def get(self, name: str) -> GamePluginSpec:
        try:
            return self._plugins[name]
        except KeyError as exc:
            available = ", ".join(self.names())
            raise KeyError(f"unknown game plugin {name!r}; available: {available}") from exc

    def game(self, name: str) -> GameSpec:
        return self.get(name).game

    def stage_adapter(self, game_name: str, stage_name: str) -> str:
        return self.get(game_name).stage_adapter(stage_name)

    def vision_encoder(self, game_name: str, stage_name: str) -> str:
        return self.get(game_name).vision_encoder(stage_name)

    def asset_pipeline(self, game_name: str, name: str) -> str:
        return self.get(game_name).asset_pipeline(name)

    def perception_pipeline(
        self,
        game_name: str,
        name: str,
    ) -> PerceptionPipelineSpec:
        return self.get(game_name).perception_pipeline(name)

    def reward_config(
        self,
        game_name: str,
        values: Mapping[str, float] | None = None,
    ) -> dict[str, float]:
        return self.get(game_name).reward_config(values)

    def success_threshold(self, game_name: str, task_name: str) -> TaskSuccessThreshold:
        return self.get(game_name).success_threshold(task_name)

    def signal_extractor(self, game_name: str, stage_name: str) -> str:
        return self.get(game_name).signal_extractor(stage_name)

    def promotion_gate(
        self,
        game_name: str,
        rung_name: str,
    ) -> GamePromotionGateSpec | None:
        return self.get(game_name).promotion_gate(rung_name)


def _artifact_gate(field: str) -> PromotionArtifactGateSpec:
    return PromotionArtifactGateSpec(
        field=field,
        reason=f"{field} must be written before this rung can promote",
    )


SMB_GAME_PLUGIN = GamePluginSpec(
    name="smb",
    game=SMB_GAME_SPEC,
    stage_adapters={
        "synthetic": "retroagi.stages.synthetic_1d.train",
        "block": "retroagi.stages.block_smb.adapter.BlockSMBStage",
        "full_asset_mock": "retroagi.stages.full_smb.vision",
        "full": "retroagi.stages.full_smb.adapter.FullSMBStage",
    },
    vision_encoders={
        "block": "retroagi.stages.block_smb.vision.BlockVisionTransformer",
        "full_asset_mock": "retroagi.stages.full_smb.vision.FullSMBVisionTransformer",
        "full": "retroagi.stages.full_smb.vision.FullSMBSegmentationVision",
    },
    asset_pipelines={
        "assets": "scripts.vit.extract_sprites",
        "perception": "scripts.vit.generate_dataset",
        "full_asset_mock": "retroagi.stages.full_smb.vision",
    },
    perception_pipelines={
        "block": PerceptionPipelineSpec(
            game_name="smb",
            name="block",
            stage_name="block",
            semantic_vocabulary=SemanticVocabularySpec(
                name="smb_block_synthetic",
                classes=(
                    "background",
                    "mario",
                    "platform",
                    "coin",
                    "goal",
                    "enemy",
                    "moving_platform",
                ),
                background_class="background",
                metadata={
                    "label_source": "exact simulator palette and symbolic state labels",
                },
            ),
            vision_encoder="retroagi.stages.block_smb.vision.BlockVisionTransformer",
            asset_extraction=(
                "retroagi.stages.block_smb.vision.BlockVisionTransformer." "semantic_targets"
            ),
            synthetic_frame_composition=("retroagi.stages.block_smb.env.MarioScenarioEnv"),
            checkpoint_path="data/block_vit/block_vit.pth",
            diagnostic_thresholds={
                "min_accuracy": 0.95,
                "min_foreground_accuracy": 0.90,
                "min_mean_iou": 0.70,
                "max_position_rmse": 0.06,
                "min_position_within_tolerance": 0.90,
                "position_tolerance": 0.05,
            },
            dataset_artifacts=("procedural Block SMB rollouts",),
            dataset_sources=(
                PerceptionDatasetSourceSpec(
                    name="block_exact_state_labels",
                    source_kind="emulator_state",
                    stage_names=("block",),
                    observation_source="block_smb_rgb_frames",
                    label_source=(
                        "MarioScenarioEnv symbolic state and "
                        "BlockVisionTransformer palette targets"
                    ),
                    entrypoint="scripts.vit.train_block_vit",
                    dataset_artifacts=("procedural Block SMB rollouts",),
                    metadata={"requires_asset_pipeline": False},
                ),
            ),
            metadata={
                "checkpoint_kind": "vision_encoder",
                "training_entrypoint": "scripts.vit.train_block_vit",
            },
        ),
        "full_asset_mock": PerceptionPipelineSpec(
            game_name="smb",
            name="full_asset_mock",
            stage_name="full_asset_mock",
            semantic_vocabulary=SemanticVocabularySpec(
                name="smb_full_asset_mock",
                classes=SMB_GAME_SPEC.semantic_classes,
                background_class="sky",
                metadata={
                    "label_source": "synthetic masks composed from full-game assets",
                },
            ),
            vision_encoder="retroagi.stages.full_smb.vision.FullSMBVisionTransformer",
            asset_extraction="scripts.vit.extract_sprites",
            synthetic_frame_composition="scripts.vit.generate_dataset",
            checkpoint_path="data/vit/full_smb_vit.pth",
            diagnostic_thresholds={
                "semantic_accuracy_threshold": 0.25,
                "foreground_accuracy_threshold": 0.10,
                "mean_iou_threshold": 0.05,
                "position_within_tolerance_threshold": 0.10,
                "position_tolerance": 0.25,
                "min_class_coverage": 13.0,
            },
            dataset_artifacts=(
                "assets/spritesheets/",
                "assets/sprites/",
                "data/vit/train.npz",
                "data/vit/val.npz",
            ),
            dataset_sources=(
                PerceptionDatasetSourceSpec(
                    name="full_asset_mock_synthetic_scenes",
                    source_kind="asset_synthetic",
                    stage_names=("full_asset_mock",),
                    observation_source="synthetic full-game asset RGB scenes",
                    label_source="synthetic masks from scripts.vit.generate_dataset",
                    entrypoint="scripts.vit.generate_dataset",
                    dataset_artifacts=(
                        "assets/spritesheets/",
                        "assets/sprites/",
                        "data/vit/train.npz",
                        "data/vit/val.npz",
                    ),
                    metadata={"requires_asset_pipeline": True},
                ),
            ),
            metadata={
                "checkpoint_kind": "vision_encoder",
                "training_entrypoint": "scripts.vit.train_vit",
            },
        ),
    },
    signal_extractors={
        "full": "retroagi.stages.full_smb.adapter.FullSMBSignalExtractor",
    },
    reward_schema=SMB_GAME_SPEC.reward_schema,
    success_thresholds={
        task.name: task.success_threshold
        for task in SMB_GAME_SPEC.fixed_tasks
        if task.success_threshold is not None
    },
    promotion_gates={
        "interface-smoke": GamePromotionGateSpec(
            rung_name="interface-smoke",
            failure_reason="SMB interface smoke exceeded the promotion budget",
        ),
        "synthetic-concept": GamePromotionGateSpec(
            rung_name="synthetic-concept",
            metric_gates=(
                PromotionMetricGateSpec(
                    metric="controller_mse",
                    operator="<=",
                    threshold_key="controller_mse_threshold",
                    reason=(
                        "Synthetic concept controller MSE must stay within the "
                        "selected SMB promotion budget"
                    ),
                ),
            ),
            artifact_gates=(
                _artifact_gate("summary_path"),
                _artifact_gate("checkpoint_path"),
            ),
            failure_reason="SMB synthetic concept gate failed",
        ),
        "block-smb-smoke": GamePromotionGateSpec(
            rung_name="block-smb-smoke",
            metric_gates=(
                PromotionMetricGateSpec(
                    metric="eval_success_rate",
                    operator=">=",
                    threshold_key="success_rate_threshold",
                    reason=(
                        "Block SMB policy success rate must meet the selected "
                        "promotion threshold"
                    ),
                ),
                PromotionMetricGateSpec(
                    metric="gradient_norm",
                    operator="finite",
                    reason="Block SMB training must report a finite gradient norm",
                ),
            ),
            artifact_gates=(
                _artifact_gate("summary_path"),
                _artifact_gate("checkpoint_path"),
                _artifact_gate("log_path"),
            ),
            failure_reason="SMB block smoke gate failed",
        ),
        "full-smb-asset-mock-perception": GamePromotionGateSpec(
            rung_name="full-smb-asset-mock-perception",
            metric_gates=(
                PromotionMetricGateSpec(
                    metric="accuracy",
                    operator=">=",
                    threshold_key="semantic_accuracy_threshold",
                    reason="Full SMB asset-mock semantic accuracy is below threshold",
                ),
                PromotionMetricGateSpec(
                    metric="foreground_accuracy",
                    operator=">=",
                    threshold_key="foreground_accuracy_threshold",
                    reason="Full SMB foreground accuracy is below threshold",
                ),
                PromotionMetricGateSpec(
                    metric="mean_iou",
                    operator=">=",
                    threshold_key="mean_iou_threshold",
                    reason="Full SMB asset-mock mean IoU is below threshold",
                ),
                PromotionMetricGateSpec(
                    metric="position_within_tolerance",
                    operator=">=",
                    threshold_key="position_within_tolerance_threshold",
                    reason="Full SMB position predictions are below threshold",
                ),
                PromotionMetricGateSpec(
                    metric="class_coverage",
                    operator=">=",
                    threshold=13.0,
                    reason="Full SMB asset-mock validation must cover every class",
                ),
                PromotionMetricGateSpec(
                    metric="position_rmse",
                    operator="finite",
                    reason="Full SMB position RMSE must be finite",
                ),
                PromotionMetricGateSpec(
                    metric="position_tolerance",
                    operator="finite",
                    reason="Full SMB position tolerance must be finite",
                ),
            ),
            artifact_gates=(
                _artifact_gate("full_smb_vision_checkpoint_path"),
                _artifact_gate("summary_path"),
            ),
            failure_reason="SMB Full SMB asset-mock perception gate failed",
        ),
        "full-smb-transfer-smoke": GamePromotionGateSpec(
            rung_name="full-smb-transfer-smoke",
            metric_gates=(
                PromotionMetricGateSpec(
                    metric="deterministic_action",
                    operator=">=",
                    threshold=0.0,
                    reason="Full SMB inference action must be non-negative",
                    name="deterministic_action_min",
                ),
                PromotionMetricGateSpec(
                    metric="deterministic_action",
                    operator="<=",
                    threshold=float(len(SMB_ACTIONS) - 1),
                    reason="Full SMB inference action must fit the SMB action space",
                    name="deterministic_action_max",
                ),
                PromotionMetricGateSpec(
                    metric="continued_global_step",
                    operator=">",
                    threshold=0.0,
                    reason="Full SMB continued training must advance global_step",
                ),
                PromotionMetricGateSpec(
                    metric="controller_transfer_key_count",
                    operator=">",
                    threshold=0.0,
                    reason="Controller transfer must copy at least one tensor",
                ),
                PromotionMetricGateSpec(
                    metric="controller_transfer_max_abs_delta",
                    operator="==",
                    threshold=0.0,
                    reason="Controller transfer must preserve copied tensors exactly",
                ),
                PromotionMetricGateSpec(
                    metric="controller_adaptation_key_count",
                    operator=">",
                    threshold=0.0,
                    reason="Continued Full SMB training must expose controller tensors",
                ),
                PromotionMetricGateSpec(
                    metric="controller_adaptation_changed_tensors",
                    operator=">",
                    threshold=0.0,
                    reason="Continued Full SMB training must update controller tensors",
                ),
            ),
            artifact_gates=(
                _artifact_gate("source_block_checkpoint_path"),
                _artifact_gate("full_smb_vision_checkpoint_path"),
                _artifact_gate("transfer_checkpoint_path"),
                _artifact_gate("continued_checkpoint_path"),
                _artifact_gate("summary_path"),
            ),
            failure_reason="SMB Full SMB transfer smoke gate failed",
        ),
    },
)


PONG_GAME_PLUGIN = GamePluginSpec(
    name="pong",
    game=PONG_GAME_SPEC,
    stage_adapters={
        "synthetic": "retroagi.stages.synthetic_1d.train",
        "block": "retroagi.stages.pong_block.adapter.PongBlockStage",
        "full": "retroagi.stages.pong_full.adapter.PongFullStage",
    },
    vision_encoders={
        "synthetic": "retroagi.core.LinearVisionEncoder",
        "block": "retroagi.stages.pong_block.vision.PongBlockVisionEncoder",
        "full": "retroagi.stages.pong_full.vision.PongFullVisionEncoder",
    },
    perception_pipelines={
        "block": PerceptionPipelineSpec(
            game_name="pong",
            name="block",
            stage_name="block",
            semantic_vocabulary=SemanticVocabularySpec(
                name="pong_block_synthetic",
                classes=PONG_GAME_SPEC.block_game_spec().semantic_classes,
                background_class="background",
                metadata={
                    "label_source": "exact paddle-ball simulator state labels",
                },
            ),
            vision_encoder="retroagi.stages.pong_block.vision.PongBlockVisionEncoder",
            asset_extraction=None,
            synthetic_frame_composition="retroagi.stages.pong_block.env.PongBlockEnv",
            checkpoint_path="data/pong_block/pong_block_vit.pth",
            diagnostic_thresholds={
                "min_accuracy": 0.95,
                "min_mean_iou": 0.75,
                "max_position_rmse": 0.04,
                "min_position_within_tolerance": 0.95,
                "position_tolerance": 0.04,
            },
            dataset_artifacts=("procedural Pong block rollouts",),
            dataset_sources=(
                PerceptionDatasetSourceSpec(
                    name="pong_block_exact_state_labels",
                    source_kind="emulator_state",
                    stage_names=("block",),
                    observation_source="pong_block_raster_frames",
                    label_source="PongBlockEnv symbolic state and semantic targets",
                    entrypoint="retroagi.stages.pong_block.env.PongBlockEnv",
                    dataset_artifacts=("procedural Pong block rollouts",),
                    metadata={"requires_asset_pipeline": False},
                ),
            ),
            metadata={
                "checkpoint_kind": "vision_encoder",
                "profile_status": "proof_of_concept",
            },
        ),
        "full": PerceptionPipelineSpec(
            game_name="pong",
            name="full",
            stage_name="full",
            semantic_vocabulary=SemanticVocabularySpec(
                name="pong_full_self_supervised",
                classes=PONG_GAME_SPEC.semantic_classes,
                background_class="background",
                metadata={
                    "label_source": "future self-supervised or manual frame labels",
                },
            ),
            vision_encoder="retroagi.stages.pong_full.vision.PongFullVisionEncoder",
            asset_extraction=None,
            synthetic_frame_composition=None,
            checkpoint_path="data/pong/full_vit.pth",
            diagnostic_thresholds={
                "min_temporal_consistency": 0.8,
                "max_position_drift": 0.08,
                "min_manual_label_accuracy": 0.90,
            },
            dataset_artifacts=("future Gymnasium Pong frame rollouts",),
            dataset_sources=(
                PerceptionDatasetSourceSpec(
                    name="pong_full_contrastive_rollouts",
                    source_kind="self_supervised",
                    stage_names=("full",),
                    observation_source="Gymnasium Pong RGB frame pairs",
                    label_source="temporal contrastive objective",
                    entrypoint="retroagi.stages.pong_full.vision",
                    dataset_artifacts=("future Gymnasium Pong frame rollouts",),
                    metadata={"requires_asset_pipeline": False},
                ),
            ),
            metadata={
                "checkpoint_kind": "vision_encoder",
                "profile_status": "planned_full_rung",
            },
        ),
    },
    signal_extractors={
        "block": "retroagi.core.PongSignalExtractor",
        "full": "retroagi.core.PongSignalExtractor",
    },
    reward_schema=PONG_GAME_SPEC.reward_schema,
    success_thresholds={
        task.name: task.success_threshold
        for task in PONG_GAME_SPEC.fixed_tasks
        if task.success_threshold is not None
    },
    promotion_gates={
        "synthetic-concept": GamePromotionGateSpec(
            rung_name="synthetic-concept",
            metric_gates=(
                PromotionMetricGateSpec(
                    metric="controller_mse",
                    operator="<=",
                    threshold_key="controller_mse_threshold",
                    reason="Pong synthetic scalar control must meet the budgeted MSE",
                ),
            ),
            artifact_gates=(
                _artifact_gate("summary_path"),
                _artifact_gate("checkpoint_path"),
            ),
            failure_reason="Pong synthetic concept gate failed",
        ),
        "block-smb-smoke": GamePromotionGateSpec(
            rung_name="block-smb-smoke",
            metric_gates=(
                PromotionMetricGateSpec(
                    metric="eval_success_rate",
                    operator=">=",
                    threshold_key="success_rate_threshold",
                    reason=(
                        "Pong block policy return rate must meet the selected "
                        "promotion threshold"
                    ),
                ),
            ),
            artifact_gates=(
                _artifact_gate("summary_path"),
                _artifact_gate("checkpoint_path"),
            ),
            failure_reason="Pong block smoke gate failed",
        ),
    },
)


GAME_PLUGIN_REGISTRY = GamePluginRegistry((PONG_GAME_PLUGIN, SMB_GAME_PLUGIN))


def game_plugin_names() -> tuple[str, ...]:
    return GAME_PLUGIN_REGISTRY.names()


def get_game_plugin(name: str) -> GamePluginSpec:
    return GAME_PLUGIN_REGISTRY.get(name)
