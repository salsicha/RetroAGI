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
from .games import GameSpec, SMB_GAME_SPEC
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
    reward_schema: RewardConfigSchema | None = None
    success_thresholds: Mapping[str, TaskSuccessThreshold] = field(default_factory=dict)
    promotion_gates: Mapping[str, GamePromotionGateSpec] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("game plugin name must be non-empty")
        if self.name != self.game.name:
            raise ValueError(
                f"game plugin {self.name!r} must match game profile "
                f"{self.game.name!r}"
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
        unknown_gate_names = sorted(
            name
            for name, gate in self.promotion_gates.items()
            if name != gate.rung_name
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
            raise ValueError(
                f"game plugin {self.name!r} {kind}s must define entrypoints: {empty}"
            )

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
            raise KeyError(
                f"unknown game plugin {name!r}; available: {available}"
            ) from exc

    def game(self, name: str) -> GameSpec:
        return self.get(name).game

    def stage_adapter(self, game_name: str, stage_name: str) -> str:
        return self.get(game_name).stage_adapter(stage_name)

    def vision_encoder(self, game_name: str, stage_name: str) -> str:
        return self.get(game_name).vision_encoder(stage_name)

    def asset_pipeline(self, game_name: str, name: str) -> str:
        return self.get(game_name).asset_pipeline(name)

    def reward_config(
        self,
        game_name: str,
        values: Mapping[str, float] | None = None,
    ) -> dict[str, float]:
        return self.get(game_name).reward_config(values)

    def success_threshold(self, game_name: str, task_name: str) -> TaskSuccessThreshold:
        return self.get(game_name).success_threshold(task_name)

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


GAME_PLUGIN_REGISTRY = GamePluginRegistry((SMB_GAME_PLUGIN,))


def game_plugin_names() -> tuple[str, ...]:
    return GAME_PLUGIN_REGISTRY.names()


def get_game_plugin(name: str) -> GamePluginSpec:
    return GAME_PLUGIN_REGISTRY.get(name)
