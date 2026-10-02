"""The executor: plays one chosen action, and says when it has ended.

The action layer chooses an action (SMBAction) and a frame count only when the
previous action has ended. The executor then turns them into one button
press per frame:

- walk and wait actions (right, left, nothing) last their frame count, or,
  when started in the air (after a stomp bounce, say), until Mario lands;
- a jump action (right jump, left jump, jump) holds the jump button for its
  frame count, keeps the direction after release, and ends when Mario lands.
  The NES starts a jump only on a fresh press, so if the jump button is still
  down from the previous action, the jump first releases it for one frame.

An action is interrupted, ending early, by things the vision transformer
reports in the current picture, and nothing else:

- Mario's drawn box touches an enemy that is not defeated;
- Mario is not drawn at all;
- during a walk or wait, Mario stops standing (he walked off an edge).

The executor sees only SceneObservations: never the game.
"""

from dataclasses import dataclass, field
from typing import Optional

from .actions import SMB_JUMP_ACTIONS, SMBAction, smb_jump_release_action
from .smb_physics import NES_JUMP_FRAMES
from .smb_scene_labels import SceneObservation

# Frame counts the action layer chooses from: jump holds (the NES menu) and
# the lengths of walks and waits.
JUMP_FRAMES = NES_JUMP_FRAMES
STEADY_FRAMES = (1, 2, 3, 4, 6, 8, 10, 12, 16, 20, 24, 32, 40, 48, 64, 96)
# A jump that has not landed after this many frames ends anyway.
MAX_JUMP_FRAMES = 150
# The picture shows the game one frame late, so a jump that has not left the
# ground this many frames after its hold never will (a ceiling, say).
TAKEOFF_GRACE = 4
# Pixels between boxes that still count as touching.
TOUCH = 1


def frame_menu(action: int) -> tuple[int, ...]:
    """The frame counts available for an action."""
    return JUMP_FRAMES if SMBAction(action) in SMB_JUMP_ACTIONS else STEADY_FRAMES


def touching(a, b, margin: int = TOUCH) -> bool:
    """Whether two boxes overlap or are at most ``margin`` pixels apart."""
    return (
        a[0] - margin < b[2]
        and b[0] - margin < a[2]
        and a[1] - margin < b[3]
        and b[1] - margin < a[3]
    )


def enemy_contact(scene: SceneObservation) -> bool:
    if scene.mario.box is None:
        return False
    return any(
        enemy.kind != "defeated" and touching(scene.mario.box, enemy.box) for enemy in scene.enemies
    )


@dataclass
class ActionPlan:
    """One decision of the action layer."""

    action: int
    frames: int

    def __post_init__(self):
        self.action = int(SMBAction(self.action))
        if self.frames not in frame_menu(self.action):
            raise ValueError(
                f"{self.frames} frames is not on the menu for {SMBAction(self.action).name}"
            )


@dataclass
class SMBExecutor:
    """Plays one ActionPlan at a time. Each frame, in this order:

    1. ``ended(scene)``: has the plan ended, judged from this frame's picture?
       Returns the reason ("done", "landed", "timeout", "enemy_contact",
       "mario_missing", "left_ground") or None. When it ends, the layers
       decide again and ``start`` the next plan.
    2. ``press()``: the button action (SMBAction) to send to the game this frame.
    """

    plan: Optional[ActionPlan] = None
    pressed: int = 0
    airborne: bool = False
    started_standing: bool = False
    repress: bool = False  # the jump button must be released for one frame first
    last_button: int = int(SMBAction.NOOP)
    history: list = field(default_factory=list)

    @property
    def idle(self) -> bool:
        return self.plan is None

    def start(self, plan: ActionPlan, scene: SceneObservation) -> None:
        if not self.idle:
            raise RuntimeError("the executor is still playing an action")
        self.plan = plan
        self.pressed = 0
        self.airborne = False
        self.started_standing = scene.mario.support != "air"
        self.repress = (
            SMBAction(plan.action) in SMB_JUMP_ACTIONS
            and SMBAction(self.last_button) in SMB_JUMP_ACTIONS
        )

    def reset(self) -> None:
        self.plan = None
        self.pressed = 0
        self.airborne = False
        self.repress = False
        self.last_button = int(SMBAction.NOOP)

    @property
    def jumping(self) -> bool:
        return self.plan is not None and SMBAction(self.plan.action) in SMB_JUMP_ACTIONS

    def ended(self, scene: SceneObservation) -> Optional[str]:
        """Why the plan ends at this frame's picture (and end it), or None."""
        if self.plan is None:
            return None
        reason = None
        if self.pressed > 0:
            if scene.mario.box is None:
                reason = "mario_missing"
            elif enemy_contact(scene):
                reason = "enemy_contact"
            elif self.jumping:
                if scene.mario.support == "air":
                    self.airborne = True
                elif self.airborne or self.pressed >= self.plan.frames + TAKEOFF_GRACE:
                    reason = "landed"
                if reason is None and self.pressed >= MAX_JUMP_FRAMES:
                    reason = "timeout"
            elif self.started_standing and scene.mario.support == "air":
                reason = "left_ground"
            elif not self.started_standing and scene.mario.support != "air":
                reason = "landed"
            elif self.pressed >= self.plan.frames:
                reason = "done"
        if reason is not None:
            self.history.append((self.plan, self.pressed, reason))
            self.plan = None
        return reason

    def press(self) -> int:
        """The button action for this frame (a jump is released after its hold)."""
        if self.plan is None:
            raise RuntimeError("no action to play: start a plan first")
        action = SMBAction(self.plan.action)
        if self.repress:
            # One frame with the jump button up, so the NES sees a fresh press.
            self.repress = False
            self.last_button = int(smb_jump_release_action(action))
            return self.last_button
        if self.jumping and self.pressed >= self.plan.frames:
            action = smb_jump_release_action(action)
        self.pressed += 1
        self.last_button = int(action)
        return self.last_button
