"""Game-owned visual perception pipeline contracts."""

from __future__ import annotations

from dataclasses import dataclass, field
from math import isfinite
from typing import Any, Mapping


@dataclass(frozen=True)
class SemanticVocabularySpec:
    """Stable semantic class ordering for one perception pipeline."""

    name: str
    classes: tuple[str, ...]
    background_class: str | None = None
    ignored_classes: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("semantic vocabulary name must be non-empty")
        if not self.classes:
            raise ValueError(f"semantic vocabulary {self.name!r} must define classes")
        if any(not class_name for class_name in self.classes):
            raise ValueError(
                f"semantic vocabulary {self.name!r} class names must be non-empty"
            )
        if len(set(self.classes)) != len(self.classes):
            raise ValueError(
                f"semantic vocabulary {self.name!r} class names must be unique"
            )
        if self.background_class is not None and self.background_class not in self.classes:
            raise ValueError(
                f"semantic vocabulary {self.name!r} background class "
                f"{self.background_class!r} is not in classes"
            )
        unknown_ignored = sorted(set(self.ignored_classes).difference(self.classes))
        if unknown_ignored:
            raise ValueError(
                f"semantic vocabulary {self.name!r} ignored classes are unknown: "
                f"{unknown_ignored}"
            )

    def class_index(self, class_name: str) -> int:
        try:
            return self.classes.index(class_name)
        except ValueError as exc:
            raise KeyError(
                f"unknown semantic class {class_name!r} in vocabulary {self.name!r}"
            ) from exc

    def to_manifest(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "classes": list(self.classes),
            "background_class": self.background_class,
            "ignored_classes": list(self.ignored_classes),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class PerceptionPipelineSpec:
    """Game-profile contract for one visual perception training path."""

    game_name: str
    name: str
    stage_name: str
    semantic_vocabulary: SemanticVocabularySpec
    vision_encoder: str
    asset_extraction: str
    synthetic_frame_composition: str
    checkpoint_path: str
    diagnostic_thresholds: Mapping[str, float]
    dataset_artifacts: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for field_name in (
            "game_name",
            "name",
            "stage_name",
            "vision_encoder",
            "asset_extraction",
            "synthetic_frame_composition",
            "checkpoint_path",
        ):
            if not getattr(self, field_name):
                raise ValueError(f"perception pipeline {field_name} must be non-empty")
        if not self.diagnostic_thresholds:
            raise ValueError(
                f"perception pipeline {self.name!r} must define diagnostic thresholds"
            )
        invalid_thresholds = [
            name
            for name, value in self.diagnostic_thresholds.items()
            if isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not isfinite(float(value))
        ]
        if invalid_thresholds:
            raise ValueError(
                f"perception pipeline {self.name!r} diagnostic thresholds must be "
                f"finite numbers: {invalid_thresholds}"
            )
        empty_artifacts = [
            artifact for artifact in self.dataset_artifacts if not artifact
        ]
        if empty_artifacts:
            raise ValueError(
                f"perception pipeline {self.name!r} dataset artifacts must be non-empty"
            )

    @property
    def semantic_classes(self) -> tuple[str, ...]:
        return self.semantic_vocabulary.classes

    def to_manifest(self) -> dict[str, Any]:
        return {
            "game_name": self.game_name,
            "name": self.name,
            "stage_name": self.stage_name,
            "semantic_vocabulary": self.semantic_vocabulary.to_manifest(),
            "vision_encoder": self.vision_encoder,
            "asset_extraction": self.asset_extraction,
            "synthetic_frame_composition": self.synthetic_frame_composition,
            "checkpoint_path": self.checkpoint_path,
            "diagnostic_thresholds": {
                name: float(value)
                for name, value in self.diagnostic_thresholds.items()
            },
            "dataset_artifacts": list(self.dataset_artifacts),
            "metadata": dict(self.metadata),
        }
