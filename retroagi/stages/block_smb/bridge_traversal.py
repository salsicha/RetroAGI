"""Collision-footprint walking windows for flat moving-bridge crossings.

The predictor uses the engine's acceleration, platform reversal, integer
rectangles, and carry rules. It certifies continuous support for walking;
actual boarding/landing remains authoritative for policy jumps.
"""

from dataclasses import dataclass, replace
from typing import Any


@dataclass
class BridgeWalkState:
    x: float
    vx: float
    width: int
    bridge_x: float
    bridge_width: int
    direction: int
    speed: float
    low: float
    high: float
    left_end: int
    right_start: int
    accel: float
    decel: float
    skid_decel: float
    max_speed: float

    def support(self) -> str | None:
        left = int(self.x)
        right = left + self.width
        support = "left" if left < self.left_end else None
        if right > round(self.bridge_x) and left < round(self.bridge_x) + self.bridge_width:
            support = "bridge"
        if right > self.right_start:
            support = "right"
        return support

    def stably_boarded(self) -> bool:
        return (
            self.support() == "bridge"
            and self.x >= round(self.bridge_x) + 4
            and self.x + self.width <= round(self.bridge_x) + self.bridge_width - 4
        )

    def advance(self, walking: bool) -> str | None:
        if walking:
            self.vx = min(
                self.max_speed, self.vx + (self.skid_decel if self.vx < 0 else self.accel)
            )
        else:
            self.vx = (
                max(0.0, self.vx - self.decel) if self.vx > 0 else min(0.0, self.vx + self.decel)
            )
        old = round(self.bridge_x)
        self.bridge_x += self.speed * self.direction
        if self.bridge_x <= self.low:
            self.bridge_x, self.direction = self.low, 1
        elif self.bridge_x >= self.high:
            self.bridge_x, self.direction = self.high, -1
        self.x += self.vx
        support = self.support()
        if support == "bridge":
            self.x += round(self.bridge_x) - old
        return support

    def can_walk_to(self, target: str) -> bool:
        probe = replace(self)
        for _ in range(48):
            support = probe.advance(True)
            if support is None:
                return False
            if support == target and (target != "bridge" or probe.stably_boarded()):
                return True
        return False


def bridge_walk_state(env) -> BridgeWalkState | None:
    if not env.mario.get("on_ground"):
        return None
    bridge = next((p for p in env.platforms if p.get("moving")), None)
    if bridge is None:
        return None
    floor_y = bridge["rect"].top
    if abs(env.mario["y"] + env.mario["h"] - floor_y) > 1e-4:
        return None
    floors = [p["rect"] for p in env.platforms if not p.get("moving") and p["rect"].top == floor_y]
    left = [r for r in floors if r.left < bridge["move_min"]]
    right = [r for r in floors if r.left > bridge["move_min"]]
    if not left or not right:
        return None
    return BridgeWalkState(
        env.mario["x"],
        env.mario["vx"],
        env.mario["w"],
        bridge["move_x"],
        bridge["rect"].w,
        bridge["move_dir"],
        bridge["move_speed"],
        bridge["move_min"],
        bridge["move_max"],
        max(r.right for r in left),
        min(r.left for r in right),
        env.accel,
        env.decel,
        env.skid_decel,
        env.max_walk_speed,
    )


def bridge_safe_wait_frames(env, horizon: int = 64) -> list[int]:
    """Safe numbers of NOOP frames before walking to the next support."""
    state = bridge_walk_state(env)
    if state is None or state.support() == "right":
        return []
    target = (
        "right"
        if state.support() == "bridge" and state.x >= round(state.bridge_x) + 4
        else "bridge"
    )
    safe = []
    for wait in range(horizon + 1):
        if state.can_walk_to(target):
            safe.append(wait)
        if state.advance(False) is None:
            break
    return safe


def bridge_phase(env, opening_wait: bool) -> str:
    if env._bridge_crossed:
        return "finish"
    state = bridge_walk_state(env)
    if state is not None and state.support() == "right":
        return "exit"
    if state is not None and state.support() == "bridge" and state.x >= round(state.bridge_x) + 4:
        return "exit" if 1 in bridge_safe_wait_frames(env) else "ride"
    return "wait" if opening_wait else "board"


def bridge_completion_metrics(
    episodes: int,
    departures: int,
    safe: int,
    boards: int,
    crosses: int,
    finishes: int,
    events: int = 0,
    timers: int = 0,
) -> dict[str, Any]:
    return {
        "episodes": episodes,
        "opening_departures": departures,
        "safe_departures": safe,
        "boardings": boards,
        "crossings": crosses,
        "finishes_after_boarding": finishes,
        "event_releases": events,
        "timer_releases": timers,
        "safe_departure_rate": safe / departures if departures else None,
        "boarding_rate": boards / episodes if episodes else None,
        "crossing_after_boarding_rate": crosses / boards if boards else None,
        "finish_after_boarding_rate": finishes / boards if boards else None,
    }


def bridge_oracle(scenario: dict, max_steps: int = 240) -> tuple[list[int], int]:
    """Validate wait, board, ride, and exit with real physics for generation."""
    from .env import MarioScenarioEnv

    env = MarioScenarioEnv()
    actions = []
    opening_wait = 0
    phase = "wait"
    try:
        env.reset(scenario=scenario)
        for frame in range(max_steps):
            if phase in ("wait", "ride"):
                ready = 1 in bridge_safe_wait_frames(env)
                action = 0
                if ready and (phase == "ride" or frame >= 3):
                    if phase == "wait":
                        opening_wait = frame + 1
                    phase = "board" if phase == "wait" else "exit"
            else:
                action = 1
            _, _, done, truncated, _ = env.step(action)
            actions.append(action)
            state = bridge_walk_state(env)
            if phase == "board" and state is not None and state.stably_boarded():
                phase = "ride"
            if done or truncated:
                break
        return actions, opening_wait
    finally:
        env.close()
