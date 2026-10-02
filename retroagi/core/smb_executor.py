"""The executor: presses one button action for a number of frames.

The action layer gives it two numbers, and nothing else reaches it:

- which button action to press: nothing, right, right + jump, left,
  left + jump or jump;
- for how many frames: any whole number from 1 to 32.

It presses that button action on each of those frames. The action is over
when it has pressed all of them, or earlier when the agent tells it Mario has
landed (smb_agent watches the vision transformer's report for that); then the
layers choose the next action. A jump is the jump button held for its frames;
when they are pressed the layers choose again, even with Mario in the air,
and his landing ends whatever action is running then.
"""

from dataclasses import dataclass, field
from typing import Optional

from .actions import SMBAction

# The frame counts the action layer chooses from, for every action.
FRAME_COUNTS = tuple(range(1, 33))


@dataclass
class ActionPlan:
    """One decision of the action layer: a button action and a frame count."""

    action: int
    frames: int

    def __post_init__(self):
        self.action = int(SMBAction(self.action))
        if self.frames not in FRAME_COUNTS:
            raise ValueError(f"{self.frames} frames is outside 1 to {FRAME_COUNTS[-1]}")


@dataclass
class SMBExecutor:
    """Plays one ActionPlan at a time. Each frame:

    1. if it has pressed all of the plan's frames, ``finished`` is true and the
       agent ends the plan (``end("done")``); if the agent saw Mario land, it
       ends the plan (``end("landed")``);
    2. when idle, the agent starts the next plan;
    3. ``press()`` gives the button action for this frame.
    """

    plan: Optional[ActionPlan] = None
    pressed: int = 0
    history: list = field(default_factory=list)  # (plan, frames pressed, why it ended)

    @property
    def idle(self) -> bool:
        return self.plan is None

    @property
    def finished(self) -> bool:
        """The plan has pressed all of its frames."""
        return self.plan is not None and self.pressed >= self.plan.frames

    def start(self, plan: ActionPlan) -> None:
        if not self.idle:
            raise RuntimeError("the executor is still playing an action")
        self.plan = plan
        self.pressed = 0

    def end(self, reason: str) -> None:
        """End the current plan ("done" or "landed")."""
        if self.plan is not None:
            self.history.append((self.plan, self.pressed, reason))
            self.plan = None

    def press(self) -> int:
        """The button action for this frame."""
        if self.plan is None:
            raise RuntimeError("no action to play: start a plan first")
        if self.finished:
            raise RuntimeError("the plan has pressed all of its frames: end it first")
        self.pressed += 1
        return self.plan.action
