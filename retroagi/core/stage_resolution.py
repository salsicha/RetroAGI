"""Game-neutral progressive-resolution stage naming."""

from __future__ import annotations

from dataclasses import dataclass


CANONICAL_STAGE_NAMES = ("synthetic", "block", "full")
OPTIONAL_STAGE_NAMES = ("symbolic", "tile", "sprite", "emulator", "full_asset_mock")
STANDARD_STAGE_NAMES = CANONICAL_STAGE_NAMES + OPTIONAL_STAGE_NAMES

STAGE_NAME_ALIASES = {
    "synthetic": "synthetic",
    "synthetic-1d": "synthetic",
    "block": "block",
    "block-smb": "block",
    "full": "full",
    "full-smb": "full",
    "symbolic": "symbolic",
    "tile": "tile",
    "sprite": "sprite",
    "emulator": "emulator",
    "full-asset-mock": "full_asset_mock",
    "asset-mock": "full_asset_mock",
}


@dataclass(frozen=True)
class StageResolution:
    """Resolved game ladder entry for one game-neutral stage name."""

    name: str
    stage_spec_name: str
    role: str


def normalize_stage_name(value: str) -> str:
    """Return the game-neutral stage name for a user or legacy stage token."""

    key = value.strip().lower().replace("_", "-")
    try:
        return STAGE_NAME_ALIASES[key]
    except KeyError as exc:
        choices = ", ".join(stage_name_choices())
        raise ValueError(f"unknown stage {value!r}; expected one of: {choices}") from exc


def stage_name_choices() -> tuple[str, ...]:
    """Return accepted user-facing stage tokens."""

    return tuple(sorted(STAGE_NAME_ALIASES))


def is_standard_stage_name(value: str) -> bool:
    """Return whether ``value`` resolves to a declared standard rung name."""

    try:
        return normalize_stage_name(value) in STANDARD_STAGE_NAMES
    except ValueError:
        return False


def resolve_game_stage(game, value: str) -> StageResolution:
    """Resolve a stage token against one game's declared stage ladder."""

    name = normalize_stage_name(value)
    for stage in game.stage_ladder:
        if stage.name == name:
            return StageResolution(
                name=stage.name,
                stage_spec_name=stage.stage_spec_name,
                role=stage.role,
            )
    available = ", ".join(game.stage_names)
    raise KeyError(
        f"game {game.name!r} does not define stage {name!r}; available: {available}"
    )
