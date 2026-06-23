"""Game profile contracts for progressive-resolution stages."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence

from .actions import SMB_ACTION_SPECS, ActionSpec


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
        if self.stage_ladder[0].name != "synthetic":
            raise ValueError(f"game {self.name!r} stage ladder must start with synthetic")
        if self.stage_ladder[-1].name != "full":
            raise ValueError(f"game {self.name!r} stage ladder must end with full")

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
