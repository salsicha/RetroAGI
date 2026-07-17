"""Game-neutral signal contracts for stage adapters."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol


@dataclass(frozen=True)
class GameSignals:
    """Common game-progress signals extracted from backend-native info."""

    position: tuple[float, float] | None = None
    progress: float | None = None
    score: int | None = None
    health: float | None = None
    lives: int | None = None
    inventory: Mapping[str, Any] = field(default_factory=dict)
    collectibles: Mapping[str, int] = field(default_factory=dict)
    completion: bool = False
    death: bool = False
    timeout: bool = False
    terminated: bool = False
    truncated: bool = False
    objectives: Mapping[str, Any] = field(default_factory=dict)
    termination_reason: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "position": self.position,
            "progress": self.progress,
            "score": self.score,
            "health": self.health,
            "lives": self.lives,
            "inventory": dict(self.inventory),
            "collectibles": dict(self.collectibles),
            "completion": self.completion,
            "death": self.death,
            "timeout": self.timeout,
            "terminated": self.terminated,
            "truncated": self.truncated,
            "objectives": dict(self.objectives),
            "termination_reason": self.termination_reason,
        }


class GameSignalExtractor(Protocol):
    """Extract game-neutral signals from a stage backend info mapping."""

    game_name: str

    def extract(
        self,
        info: Mapping[str, Any],
        *,
        terminated: bool,
        truncated: bool,
    ) -> GameSignals:
        """Return normalized game signals for one backend transition."""


@dataclass(frozen=True)
class PongSignals(GameSignals):
    """Normalized signals for Pong-style paddle-ball environments."""

    paddle_y: float | None = None
    opponent_y: float | None = None
    ball_position: tuple[float, float] | None = None
    ball_velocity: tuple[float, float] | None = None
    score_delta: int | None = None
    rally_length: int | None = None

    def as_dict(self) -> dict[str, Any]:
        data = super().as_dict()
        data.update(
            {
                "paddle_y": self.paddle_y,
                "opponent_y": self.opponent_y,
                "ball_position": self.ball_position,
                "ball_velocity": self.ball_velocity,
                "score_delta": self.score_delta,
                "rally_length": self.rally_length,
            }
        )
        return data


class PongSignalExtractor(GameSignalExtractor):
    """Extract game-neutral signals from Pong backend info mappings."""

    game_name = "pong"

    def extract(
        self,
        info: Mapping[str, Any],
        *,
        terminated: bool,
        truncated: bool,
    ) -> PongSignals:
        return extract_pong_signals(info, terminated=terminated, truncated=truncated)


def extract_pong_signals(
    info: Mapping[str, Any], *, terminated: bool, truncated: bool
) -> PongSignals:
    """Extract Pong variables into the shared signal schema."""

    reason = _string_value(info, ("termination_reason", "reason", "done_reason"))
    ball_position = _pair_value(info, "ball_position", ("ball_x", "x"), ("ball_y", "y"))
    ball_velocity = _pair_value(
        info,
        "ball_velocity",
        ("ball_vx", "vx"),
        ("ball_vy", "vy"),
    )
    paddle_y = _float_value(info, ("paddle_y", "agent_paddle_y", "player_y"))
    opponent_y = _float_value(info, ("opponent_y", "enemy_paddle_y"))
    score_delta = _int_value(info, ("score_delta", "reward_score_delta"))
    rally_length = _int_value(info, ("rally_length", "rally", "hits"))
    alignment = _float_value(info, ("paddle_alignment", "alignment"))
    if alignment is None and paddle_y is not None and ball_position is not None:
        alignment = 1.0 - min(abs(paddle_y - ball_position[1]), 1.0)

    completion = (
        _bool_value(info, ("completion", "point_scored", "scored"), default=False)
        or (score_delta is not None and score_delta > 0)
        or _reason_matches(reason, ("score", "scored", "win", "point"))
    )
    death = (
        _bool_value(info, ("death", "miss", "conceded"), default=False)
        or (score_delta is not None and score_delta < 0)
        or _reason_matches(reason, ("miss", "concede", "loss", "lost"))
    )
    timeout = bool(truncated) or _bool_value(
        info, ("timeout", "TimeLimit.truncated"), default=False
    )
    objectives = {
        "paddle_alignment": alignment,
        "rally_hit": _bool_value(info, ("rally_hit", "hit"), default=False),
        "concede_point": death,
    }
    objectives = {key: value for key, value in objectives.items() if value is not None}
    return PongSignals(
        position=ball_position,
        progress=float(rally_length) if rally_length is not None else None,
        score=score_delta,
        paddle_y=paddle_y,
        opponent_y=opponent_y,
        ball_position=ball_position,
        ball_velocity=ball_velocity,
        score_delta=score_delta,
        rally_length=rally_length,
        completion=completion,
        death=death,
        timeout=timeout,
        terminated=bool(terminated),
        truncated=bool(truncated),
        objectives=objectives,
        termination_reason=reason,
    )


def _float_value(info: Mapping[str, Any], keys: tuple[str, ...]) -> float | None:
    for key in keys:
        value = info.get(key)
        if value is not None:
            try:
                return float(value)
            except (TypeError, ValueError):
                return None
    return None


def _int_value(info: Mapping[str, Any], keys: tuple[str, ...]) -> int | None:
    value = _float_value(info, keys)
    if value is None:
        return None
    return int(value)


def _bool_value(info: Mapping[str, Any], keys: tuple[str, ...], *, default: bool) -> bool:
    for key in keys:
        value = info.get(key)
        if value is not None:
            return bool(value)
    return default


def _pair_value(
    info: Mapping[str, Any],
    direct_key: str,
    x_keys: tuple[str, ...],
    y_keys: tuple[str, ...],
) -> tuple[float, float] | None:
    direct = info.get(direct_key)
    if isinstance(direct, Mapping):
        x = _float_value(direct, x_keys)
        y = _float_value(direct, y_keys)
        if x is not None and y is not None:
            return (x, y)
    elif direct is not None:
        try:
            values = list(direct)
        except TypeError:
            values = []
        if len(values) >= 2:
            try:
                return (float(values[0]), float(values[1]))
            except (TypeError, ValueError):
                return None

    x = _float_value(info, x_keys)
    y = _float_value(info, y_keys)
    if x is not None and y is not None:
        return (x, y)
    return None


def _string_value(info: Mapping[str, Any], keys: tuple[str, ...]) -> str | None:
    for key in keys:
        value = info.get(key)
        if value is not None:
            return str(value)
    return None


def _reason_matches(reason: str | None, needles: tuple[str, ...]) -> bool:
    if reason is None:
        return False
    lowered = reason.lower()
    return any(needle in lowered for needle in needles)
