"""Local objectives and completion diagnostics for the tall-pipe composite."""

from dataclasses import dataclass
from typing import Any, Mapping

from .env import MarioScenarioEnv
from .monte_carlo import block_smb_monte_carlo_metadata

TALL_PIPE_MIN_TRAINING_STEPS = 160
ENEMY_STOMP_MIN_TRAINING_STEPS = 160


def is_tall_pipe_scenario(scenario: Mapping[str, Any] | None) -> bool:
    return bool(
        scenario is not None
        and block_smb_monte_carlo_metadata(scenario).get("family") == "tall_pipe_jump"
    )


def training_rollout_steps(requested: int, scenario: Mapping[str, Any] | None) -> int:
    """Leave time to finish and recover; evaluation budgets remain explicit.

    The tall-pipe oracle needs 82–86 frames. A 60-frame training episode
    cannot reach the finish at any speed. Apply the floor to old replay
    scenarios too, without depending on newly generated metadata. Composite
    enemy stomps also need time for the approach, bounce, and finish.
    """
    if is_tall_pipe_scenario(scenario):
        return max(requested, TALL_PIPE_MIN_TRAINING_STEPS)
    if scenario is not None and (
        scenario.get("require_stomp_before_goal")
        or block_smb_monte_carlo_metadata(scenario).get("family") == "enemy_stomp"
    ):
        return max(requested, ENEMY_STOMP_MIN_TRAINING_STEPS)
    if scenario is not None and (
        scenario.get("require_bridge_before_goal")
        or block_smb_monte_carlo_metadata(scenario).get("family") == "bridge_wait"
    ):
        return max(requested, 240)
    return requested


@dataclass
class TallPipeTraversal:
    left: float
    top: float
    width: float
    mounted: bool = False
    phase: str = "mount"

    @classmethod
    def from_stage(cls, scenario, env: MarioScenarioEnv) -> "TallPipeTraversal | None":
        if not is_tall_pipe_scenario(scenario):
            return None
        parameters = block_smb_monte_carlo_metadata(scenario)["parameters"]
        pipe = next(p["rect"] for p in env.platforms if p["rect"].x == parameters["pipe_x"])
        return cls(float(pipe.left), float(pipe.top), float(pipe.w))

    def observe(self, env: MarioScenarioEnv) -> bool:
        """Credit actual support on the pipe, never a jump's generic landing flag."""
        mario = env.mario
        contact = bool(
            mario["on_ground"]
            and abs(mario["y"] + mario["h"] - self.top) < 1e-4
            and mario["x"] + mario["w"] > self.left
            and mario["x"] < self.left + self.width
        )
        self.mounted |= contact
        if contact or mario["x"] >= self.left + self.width:
            self.phase = "finish"
        elif mario["x"] + mario["w"] <= self.left and mario["y"] + mario["h"] > self.top:
            # A retreat off the approach side needs a new mounting attempt.
            self.phase = "mount"
        return contact

    def target(self, env: MarioScenarioEnv) -> tuple[float, float]:
        """Pipe-top center and horizontal landing tolerance, including Mario's width."""
        return self.left + self.width / 2.0, (self.width + env.mario["w"]) / 2.0


def pipe_completion_metrics(episodes: int, mounts: int, finishes: int) -> dict[str, Any]:
    """Separate support on the pipe from subsequent episode completion."""
    return {
        "episodes": episodes,
        "mount_successes": mounts,
        "finish_after_mount_successes": finishes,
        "mount_success_rate": mounts / episodes if episodes else None,
        "finish_after_mount_success_rate": finishes / mounts if mounts else None,
    }
