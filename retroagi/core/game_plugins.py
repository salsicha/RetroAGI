"""Game plugin registry for multi-game progressive-resolution components."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

from .games import GameSpec, SMB_GAME_SPEC
from .rewards import RewardConfigSchema
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
        try:
            return self.stage_adapters[name]
        except KeyError as exc:
            raise KeyError(
                f"unknown stage adapter {name!r} for game plugin {self.name!r}"
            ) from exc

    def vision_encoder(self, name: str) -> str:
        try:
            return self.vision_encoders[name]
        except KeyError as exc:
            raise KeyError(
                f"unknown vision encoder {name!r} for game plugin {self.name!r}"
            ) from exc

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
)


GAME_PLUGIN_REGISTRY = GamePluginRegistry((SMB_GAME_PLUGIN,))


def game_plugin_names() -> tuple[str, ...]:
    return GAME_PLUGIN_REGISTRY.names()


def get_game_plugin(name: str) -> GamePluginSpec:
    return GAME_PLUGIN_REGISTRY.get(name)
