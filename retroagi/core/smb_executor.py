"""Timed button actions and a closed-loop, platform-relative hold controller.

Skill sends a spatial destination to SpatialFeedback, which builds a predictive
ground or flight controller. This object owns that maneuver and emits its buttons.
ActionPlan is an internal control record, also used for exact teacher playback.

Ordinary timed plans finish after their frames or an observed landing.
A visually checked Flight owns the whole jump, including its coast after A
is released; its frame count is only the proposed button-hold duration.
The layers choose again after landing or an execution failure/timeout.
HOLD_GROUND instead
reads the current vision scene every frame and emits left, right or no buttons
to preserve the initial horizontal offset on the supporting platform. The
anchor persists across consecutive hold plans. No simulator state is read.
"""

from dataclasses import dataclass, field
from typing import Optional

from .actions import SMB_JUMP_ACTIONS, SMBAction
from .smb_pixel_types import VISIBLE_COLUMNS
from .smb_scene_labels import SceneObservation, Surface

# Controller actions are not emulator buttons. Only press() translates these
# into the six shared SMBAction values.
HOLD_GROUND = 6
EXECUTOR_ACTIONS = (*tuple(SMBAction), HOLD_GROUND)

# Supported button-hold durations for physical control and teacher playback.
FRAME_COUNTS = tuple(range(1, 33))


@dataclass
class ActionPlan:
    """An internal button action and hold duration, never a learned layer output."""

    action: int
    frames: int

    def __post_init__(self):
        if self.action not in EXECUTOR_ACTIONS:
            raise ValueError(f"unknown executor action {self.action!r}")
        self.action = int(self.action)
        if self.frames not in FRAME_COUNTS:
            raise ValueError(f"{self.frames} frames is outside 1 to {FRAME_COUNTS[-1]}")


@dataclass
class GroundHold:
    """Track a visual support and correct relative position and drift.

    Surfaces have no identity in vision, so match by geometry between frames.
    Missing/ambiguous support releases buttons without acquiring a new anchor.
    """

    surface: Optional[Surface] = None
    offset: Optional[float] = None
    previous: Optional[float] = None
    width: float = 0.0
    right_edge: bool = False

    @staticmethod
    def clipped(surface: Surface) -> bool:
        return surface.x0 <= VISIBLE_COLUMNS[0] or surface.x1 >= VISIBLE_COLUMNS[1]

    def reference(self, surface: Surface) -> float:
        """Use a visible edge and the remembered width when the other is clipped."""
        self.width = max(self.width, surface.x1 - surface.x0)
        if self.right_edge:
            return surface.x0 + self.width if surface.x1 >= VISIBLE_COLUMNS[1] else surface.x1
        return surface.x1 - self.width if surface.x0 <= VISIBLE_COLUMNS[0] else surface.x0

    def press(self, scene: Optional[SceneObservation]) -> int:
        if scene is None or scene.mario.box is None:
            self.previous = None
            return int(SMBAction.NOOP)
        x0, _, x1, feet = scene.mario.box
        center = (x0 + x1) / 2
        # Compact moving-platform boxes preserve the platform edges even where
        # Mario covers part of the surface. Fall back to reported surfaces.
        moving = [Surface(b[0], b[2], b[1], True) for b in scene.moving_platforms]
        surfaces = [s for s in scene.surfaces if not s.moving or not moving] + moving
        # A surface clipped at BOTH edges provides no horizontal landmark.
        # Its viewport edges do not move when the camera scrolls. Treating one
        # as a world anchor makes the controller accelerate to chase the scroll.
        surfaces = [
            s for s in surfaces if not (s.x0 <= VISIBLE_COLUMNS[0] and s.x1 >= VISIBLE_COLUMNS[1])
        ]
        if self.surface is None:
            candidates = [
                s
                for s in surfaces
                if s.x0 < x1
                and s.x1 > x0
                and abs(s.top - feet) <= 4
                and s.moving == (scene.mario.support == "moving_platform")
            ]
            if scene.mario.support == "air" or not candidates:
                return int(SMBAction.NOOP)
            support = min(candidates, key=lambda s: abs(s.top - feet))
            self.right_edge = support.x0 <= VISIBLE_COLUMNS[0]
            self.offset = center - self.reference(support)
        else:
            old = self.surface
            candidates = [
                s
                for s in surfaces
                if s.moving == old.moving
                and (
                    abs((s.x1 - s.x0) - (old.x1 - old.x0)) <= 8
                    or self.clipped(s)
                    or self.clipped(old)
                )
                and abs(s.x0 - old.x0) <= 48
                and abs(s.top - old.top) <= 24
            ]
            candidates.sort(key=lambda s: abs(s.x0 - old.x0) + abs(s.top - old.top))
            if not candidates or (
                len(candidates) > 1
                and abs(candidates[0].x0 - old.x0) + abs(candidates[0].top - old.top)
                == abs(candidates[1].x0 - old.x0) + abs(candidates[1].top - old.top)
            ):
                self.previous = None
                return int(SMBAction.NOOP)
            support = candidates[0]
        self.surface = support
        current = center - self.reference(support)
        velocity = 0.0 if self.previous is None else current - self.previous
        self.previous = current
        if scene.mario.support == "air" or abs(support.top - feet) > 4:
            self.previous = None
            return int(SMBAction.NOOP)
        # Anticipate two frames of drift so corrections brake before crossing
        # the anchor. A pixel of tolerance avoids reacting to sprite rounding.
        correction = self.offset - current - 2.0 * velocity
        if abs(correction) <= 1.0:
            return int(SMBAction.NOOP)
        return int(SMBAction.RIGHT if correction > 0 else SMBAction.LEFT)


@dataclass
class SMBExecutor:
    """Plays one ActionPlan (optionally a whole checked flight) at a time. Each frame:

    1. if it has pressed all of the plan's frames, ``finished`` is true and the
       agent ends the plan (``end("done")``); if the agent saw Mario land, it
       ends the plan (``end("landed")``);
    2. when idle, the agent starts the next plan;
    3. ``press()`` gives the button action for this frame.
    """

    plan: Optional[ActionPlan] = None
    pressed: int = 0
    history: list = field(default_factory=list)  # (plan, frames pressed, why it ended)
    hold: GroundHold = field(default_factory=GroundHold)
    last_button: Optional[int] = None
    flight: object = None
    travel: object = None

    @property
    def idle(self) -> bool:
        return self.plan is None

    @property
    def finished(self) -> bool:
        """The timed plan or checked flight has finished."""
        if self.flight is not None:
            return self.flight.done
        if self.travel is not None:
            return self.travel.done
        return self.plan is not None and self.pressed >= self.plan.frames

    @property
    def reconsider(self) -> bool:
        """Waiting must expose one-frame departure windows to the policy.

        Keep the support anchor when another hold follows; only its scheduling
        interval ends. A learned 32-frame hold must not hide a bridge or plant
        transition for the whole interval.
        """
        return self.plan is not None and self.plan.action == HOLD_GROUND and self.pressed > 0

    def start(self, plan: ActionPlan, flight=None, travel=None) -> None:
        if not self.idle:
            raise RuntimeError("the executor is still playing an action")
        self.plan = plan
        self.flight = flight
        self.travel = travel
        self.pressed = 0
        if plan.action != HOLD_GROUND:
            self.hold = GroundHold()

    def end(self, reason: str) -> None:
        """End the current plan ("done" or "landed")."""
        if self.plan is not None:
            self.history.append((self.plan, self.pressed, reason))
            self.plan = None
            self.flight = None
            self.travel = None

    def press(self, scene: Optional[SceneObservation] = None) -> int:
        """The button action for this frame."""
        if self.plan is None:
            raise RuntimeError("no action to play: start a plan first")
        if self.finished:
            raise RuntimeError("the plan has pressed all of its frames: end it first")
        if self.flight is not None and self.pressed == 0 and self.last_button in SMB_JUMP_ACTIONS:
            # Do not advance the flight model until the actual takeoff edge.
            button = {2: 1, 4: 3, 5: 0}[self.plan.action]
            self.last_button = button
            return button
        button = (
            self.flight.press(scene)
            if self.flight is not None
            else self.travel.press(scene)
            if self.travel is not None
            else self.hold.press(scene)
            if self.plan.action == HOLD_GROUND
            else self.plan.action
        )
        if (
            self.pressed == 0
            and button in SMB_JUMP_ACTIONS
            and self.last_button in SMB_JUMP_ACTIONS
        ):
            # A new jump needs a physical release edge, including when landing
            # interrupts a held jump. This release is not one of its A frames.
            button = {
                SMBAction.RIGHT_JUMP: SMBAction.RIGHT,
                SMBAction.LEFT_JUMP: SMBAction.LEFT,
                SMBAction.JUMP: SMBAction.NOOP,
            }[button]
            self.last_button = int(button)
            return int(button)
        self.pressed += 1
        self.last_button = int(button)
        return int(button)
