"""Local execution feedback from successive pictures, never simulator state.

Skill supplies a destination directly. Predictive ground and flight controllers
select the buttons; no learned action network or duration proposal is needed.
SpatialFeedback is the motion estimate (smb_estimator.MotionEstimator) with
the destination in progress and its flight or ground move.
"""

from collections import deque
from dataclasses import dataclass, field

from . import smb_trajectory
from .actions import SMBAction
from .smb_estimator import MotionEstimator, moving_support
from .smb_ground_control import GroundMove
from .smb_trajectory import plan_flight


def moving_support_offset(scene):
    support = moving_support(scene)
    if support is None:
        return None
    x0, _, x1, _ = scene.mario.box
    return (x0 + x1) / 2 - support[0]


@dataclass
class SpatialFeedback(MotionEstimator):
    destination: object = None
    start_displacement: float = 0.0
    flight: object = None
    status: str = "idle"
    travel: object = None
    start_feet: float = 0.0
    observed_frames: int = 0
    positions: deque = field(default_factory=lambda: deque(maxlen=24))
    # Why the last action ended (the agent's reason: "landed", "done",
    # "threat", ...), reported to the skill.
    ended: str | None = None

    @property
    def stalled(self):
        return (
            self.destination is not None
            and self.destination.mode != "hold"
            and len(self.positions) == self.positions.maxlen
            and all(
                max(p[k] for p in self.positions) - min(p[k] for p in self.positions) <= 2
                for k in (0, 1)
            )
        )

    def report(self, scene):
        """Measured command outcome for skill, never a judgement of goal safety."""
        from .tokens import EXECUTION_WIDTH

        if self.destination is None or scene.mario.box is None:
            return [0.0] * EXECUTION_WIDTH
        box = scene.mario.box
        dx = self.displacement - self.start_displacement
        dy = box[3] - self.start_feet
        rx, ry = self.destination.x - dx, self.destination.y - dy
        if self.destination.mode == "hold":
            rx = ry = 0.0
        elif self.flight is not None:
            rx = self.flight.goal[0] - (box[0] + box[2]) / 2
            ry = self.flight.goal[1] - box[3]
        elif self.travel is not None and self.travel.target is not None:
            rx = self.travel.target.box[0] + self.travel.target_offset - (box[0] + box[2]) / 2
        return [
            1.0,
            float(self.stalled),
            float(abs(rx) <= 4 and abs(ry) <= 4),
            min(self.observed_frames / 192, 1.0),
            dx / 256,
            dy / 240,
            rx / 256,
            ry / 240,
            float(self.ended == "threat"),
        ]

    def _lost(self) -> None:
        self.positions.clear()

    def _speed_corrected(self, scene) -> None:
        if self.flight is not None and smb_trajectory.REPLAN_IN_FLIGHT:
            # Flight owns a copy of the physical state; correcting
            # only this tracker leaves its next prediction stale.
            self.flight.motion.x_speed = self.motion.x_speed
            self.flight.motion.moving = self.motion.moving
        elif self.flight is not None and not self.flight.done:
            # An open-loop flight planned with the wrong speed (a
            # takeoff before it was seen) plans its rest again.
            self.flight.replan(scene, self.motion)

    def _observed(self, scene, shift) -> None:
        if self.flight is not None and shift is not None:
            self.flight.shift(shift)
        if self.destination is not None:
            self.observed_frames += 1
            self.positions.append((self.displacement, scene.mario.box[3]))

    def begin(self, destination, scene, proposed=None, aim=None):
        """The plan for ``destination``. ``aim`` (deployed play): the pointer
        of what the skill aimed at; a run then ends inside the still thing it
        names (smb_aim)."""
        from .smb_aim import aimed_thing
        from .smb_executor import HOLD_GROUND, ActionPlan

        self.flight = None
        self.travel = None
        self.status = "unplanned"
        previous_destination = self.destination
        self.destination = destination
        self.start_displacement = self.displacement
        self.start_feet = scene.mario.box[3] if scene.mario.box else 0.0
        self.observed_frames = 0
        if (
            destination is None
            or destination.mode == "hold"
            or previous_destination is None
            or previous_destination.mode == "hold"
        ):
            self.positions.clear()
        if destination is None:
            if proposed is None:
                raise ValueError("the executor needs a destination")
            return proposed  # Exact teacher playback only.
        if proposed is None and destination.mode == "hold":
            self.status = "holding"
            return ActionPlan(
                HOLD_GROUND,
                max(1, destination.frames),
                wait=destination.frames > 0,
                threat=destination.threat,
            )
        thing = aimed_thing(scene, aim) if aim is not None and scene.mario.box is not None else None
        middle = (scene.mario.box[0] + scene.mario.box[2]) / 2 if scene.mario.box is not None else 0.0
        if proposed is None and destination.mode == "run":
            if scene.mario.box is None:
                self.status = "lost_observation"
                return ActionPlan(HOLD_GROUND, 1)
            self.travel = GroundMove(
                self,
                destination.x,
                self.displacement,
                scene.mario.box[3],
                destination.y,
                carry=destination.carry,
                aim=None if thing is None else thing.shifted(-middle),
            )
            return ActionPlan(0, 1)
        if destination is not None and destination.mode == "jump" and scene.mario.box is None:
            self.status = "lost_observation"
            return ActionPlan(HOLD_GROUND, 1)
        grounded_jump = bool(
            destination is not None
            and destination.mode == "jump"
            and scene.mario.box is not None
            and scene.mario.support != "air"
            and scene.mario.on_something
        )
        if (
            destination is not None
            and scene.mario.support == "air"
            and (
                proposed is None
                or proposed.action in (SMBAction.RIGHT_JUMP, SMBAction.LEFT_JUMP, SMBAction.JUMP)
            )
        ):
            # A second action cannot silently extend the preceding jump hold.
            # Airborne steering releases A; only a grounded command starts a jump.
            coast = (
                (1 if destination.x > 0 else 3 if destination.x < 0 else 0)
                if proposed is None
                else {
                    SMBAction.RIGHT_JUMP: SMBAction.RIGHT,
                    SMBAction.LEFT_JUMP: SMBAction.LEFT,
                    SMBAction.JUMP: SMBAction.NOOP,
                }[proposed.action]
            )
            return ActionPlan(coast, proposed.frames if proposed is not None else 1)
        if grounded_jump:
            # Use the motion estimate already available; a jump request must
            # not insert waiting frames to collect additional observations.
            speed = (
                self.speed
                if proposed is not None and self.speed is not None
                else self.motion.x_speed / 16
            )
            self.flight = plan_flight(
                scene,
                self.tracks,
                speed,
                destination,
                proposed,
                motion=self.motion if proposed is None else None,
            )
            self.status = "attempting"
            action = {-1: 4, 0: 5, 1: 2}[self.flight.direction]
            return ActionPlan(action, max(1, self.flight.hold))
        if proposed is None:
            self.status = "observing_support"
            return ActionPlan(HOLD_GROUND, 1)
        return proposed

    def arrived(self):
        if self.travel is not None:
            return self.travel.done
        target = self.destination
        if target is None or target.mode != "run" or not self.tracked or target.x == 0:
            return False
        # A run ends with Mario standing at its destination: one that takes
        # him off a ledge onto a lower floor is still on its way while he
        # falls (his landing ends it), as is a steer in the air.
        mario = self.previous.mario if self.previous is not None else None
        if mario is None or not (mario.on_something and mario.support != "air"):
            return False
        distance = self.displacement - self.start_displacement
        return distance * (1 if target.x > 0 else -1) >= abs(target.x)
