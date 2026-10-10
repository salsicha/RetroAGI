"""From an action to the executor, and keeping it on its target
(docs/action-predictor-controller.md, sections 2 to 4).

The skill chooses an action (a verb aimed at an object it sees); the action
predictor gives where it ends when it succeeds. That end, from Mario's feet,
is the executor's destination, in the verb's mode. When the action's object
is a tracked enemy or moving platform, the destination is bound to it:
Mario's predicted end measured from where the object is predicted to be
then. Each frame the target tracker checks the object against its forecast;
when it has left it (a walker turned, a platform reversed), the destination
moves with the object's new forecast: a jump plans the rest of its flight
again, a run moves where it stops. The executor itself models nothing in
the world.
"""

from dataclasses import dataclass
from typing import Optional

from .action_tokens import ActionToken
from .smb_observer import packed_lists
from .tokens import SKILL_X, SKILL_Y, SkillToken


def destination_of(action: ActionToken, outcome: dict) -> SkillToken:
    """The executor's command for an action with its predicted end
    (``outcome``: "dx", "dy" from Mario's feet)."""
    if action.verb == "hold":
        return SkillToken("hold", 0, 0)
    x = max(SKILL_X[0], min(SKILL_X[-1], round(outcome["dx"])))
    y = max(SKILL_Y[0], min(SKILL_Y[-1], round(outcome["dy"])))
    return SkillToken(action.mode, x, y)


@dataclass
class ActionTarget:
    """An action bound to its object's visual identity."""

    action: ActionToken
    identity: int
    # Mario's predicted end from the object's predicted point (its middle
    # and top), pixels.
    offset: tuple
    # The tracker's frame on which the action is predicted to end.
    end_frame: int


def _identity(scene, spatial, target) -> int:
    """The visual identity of a reported enemy or moving platform (0: none,
    or another kind of object, which does not move)."""
    name, slot = target
    if name not in ("enemies", "moving_platforms"):
        return 0
    items = packed_lists(scene)[name]
    if slot >= len(items):
        return 0
    box, kind = (
        (items[slot].box, items[slot].kind) if name == "enemies" else (items[slot], "platform")
    )
    matches = [t for t in spatial.tracks.tracks if t.kind == kind and t.box == box]
    return matches[0].identity if len(matches) == 1 else 0


def bind(action: ActionToken, outcome: dict, scene, spatial, frame: int) -> Optional[ActionTarget]:
    """The action bound to its moving object, or None (hold, a static
    object, or no predicted object point). ``frame``: the tracker's frame."""
    if action.target is None or "object_dx" not in outcome:
        return None
    identity = _identity(scene, spatial, action.target)
    if not identity:
        return None
    offset = (outcome["dx"] - outcome["object_dx"], outcome["dy"] - outcome["object_dy"])
    return ActionTarget(action, identity, offset, frame + max(1, round(outcome["frames"])))


def retarget(target: ActionTarget, scene, spatial, tracker) -> Optional[tuple]:
    """When the tracker (ObjectTracker) reports the bound object has left its
    forecast: the destination's new place from Mario's feet now (pixels),
    else None."""
    if scene.mario.box is None or not tracker.changed(target.identity):
        return None
    remaining = max(1, target.end_frame - tracker.frame)
    where = tracker.where(target.identity, remaining)
    if where is None:
        return None
    box = scene.mario.box
    centre = (box[0] + box[2]) / 2 + spatial.camera_position
    return (
        where[0] + target.offset[0] - centre,
        where[1] + target.offset[1] - box[3],
    )


def apply(destination: tuple, scene, spatial) -> None:
    """Move the running maneuver's destination (from Mario's feet now): a
    flight plans the rest of its jump again (Flight.replan), a run moves
    where it stops."""
    box = scene.mario.box
    flight, travel = spatial.flight, spatial.travel
    if flight is not None and not flight.done:
        flight.goal = ((box[0] + box[2]) / 2 + destination[0], box[3] + destination[1])
        flight.replan(scene, spatial.motion)
    elif travel is not None and not travel.done:
        travel.distance = (spatial.displacement - travel.start) + destination[0]
        travel.target = None
