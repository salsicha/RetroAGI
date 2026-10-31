"""The still thing a run is aimed at, found again in every picture.

The skill points at one of the things vision reports (a surface, a pipe, a
block); the world model's predicted end becomes the executor's destination.
Without the teacher (deployed play) a run drives to the thing itself (user,
2026-10-31: "make executor drive to object the skill aimed at"): it ends
inside it. Each picture the thing is found again, so an end that comes into
view as Mario approaches (a floor ending at a pit past the picture's edge)
bounds the run. Jumps keep the plan made at takeoff: landing on the aimed
thing's top inside its ends and steering toward it in the air was tried and
did worse in 1-1 (Mario landed on the small blocks the skill aimed at and
slid off beside the first Goomba); a jump's landing height on a still thing
is vision's (smb_agent.landing_height). Moving platforms are followed by the
executor's tracks (GroundMove).

Ends are a thing's real ends: pieces vision splits a floor into at seams are
joined, the cuts where enemies stand are not ends (fixed_surfaces), and an
end at the picture's edge is open (the thing may go on past it)."""

import math
from dataclasses import dataclass
from typing import Optional

from .smb_observer import packed_lists
from .smb_pixel_types import SCREEN_SHAPE
from .smb_scene_labels import BlockView, SceneObservation, Surface, fixed_surfaces

# How far inside a thing's ends a move onto it puts Mario's middle (pixels;
# his body is 10 wide).
AIM_INSET = 8
# Pieces this close together (pixels apart, tops within SEAM_ROWS) are one
# thing: vision splits a floor at seams (1-1's first floor came as 8-59 and
# 63-248, their tops a row apart).
SEAM_GAP, SEAM_ROWS = 8, 2
# A thing is found again where it overlaps where it was, give or take this.
REFIND_SLACK = 8


@dataclass
class Aim:
    """A still thing's ends (screen pixels; open ends -inf / inf), its top row,
    and the span of it vision last showed (finite: where it is found again)."""

    left: float
    right: float
    top: float
    seen_left: float
    seen_right: float

    def shifted(self, dx: float) -> "Aim":
        return Aim(self.left + dx, self.right + dx, self.top, self.seen_left + dx, self.seen_right + dx)

    def place(self, x: float) -> float:
        """The nearest place to ``x`` (screen) for Mario's middle on the thing:
        at least AIM_INSET inside its ends, or its middle when it is narrower."""
        if self.right - self.left >= 2 * AIM_INSET:
            return min(self.right - AIM_INSET, max(self.left + AIM_INSET, x))
        return (self.left + self.right) / 2


def _still_pieces(scene: SceneObservation) -> list:
    """(left, right, top) of every still thing vision shows: surfaces (not
    cut at enemies, not moving), pipes and blocks."""
    pieces = [(s.x0, s.x1, s.top) for s in fixed_surfaces(scene) if not s.moving]
    pieces += [(p[0], p[2], p[1]) for p in scene.pipes]
    pieces += [(b.box[0], b.box[2], b.box[1]) for b in scene.blocks]
    return pieces


def _joined(scene: SceneObservation, left: float, right: float, top: float) -> Aim:
    """The span grown over every still piece at the same top that meets it
    (within SEAM_GAP); ends at the picture's edges open."""
    pieces = [p for p in _still_pieces(scene) if abs(p[2] - top) <= SEAM_ROWS]
    grown = True
    while grown:
        grown = False
        for x0, x1, _ in pieces:
            if x0 < left and x1 >= left - SEAM_GAP:
                left, grown = x0, True
            if x1 > right and x0 <= right + SEAM_GAP:
                right, grown = x1, True
    width = SCREEN_SHAPE[1]
    return Aim(
        -math.inf if left <= SEAM_GAP else left,
        math.inf if right >= width - SEAM_GAP else right,
        top,
        left,
        right,
    )


def aimed_thing(scene: SceneObservation, target) -> Optional[Aim]:
    """The still thing at ``target`` (a pointer: (list, slot) into the scene's
    lists as the policy read them, smb_observer.packed_lists): a surface that
    does not move, a pipe or a block; None for anything else (an enemy, a
    moving platform or surface), or an empty slot."""
    if target is None:
        return None
    name, slot = target
    if name not in ("surfaces", "pipes", "blocks"):
        return None
    things = packed_lists(scene).get(name, ())
    if not 0 <= slot < len(things):
        return None
    thing = things[slot]
    if isinstance(thing, Surface):
        if thing.moving:
            return None
        left, right, top = thing.x0, thing.x1, thing.top
    else:
        left, top, right, _ = thing.box if isinstance(thing, BlockView) else thing
    return _joined(scene, left, right, top)


def refound(aim: Aim, scene: Optional[SceneObservation]) -> Aim:
    """``aim`` (already moved with the picture's scroll) as this picture shows
    it: the still pieces at its top overlapping it, joined; an open end that
    has come into view closes. Unchanged where vision shows none of it."""
    if scene is None:
        return aim
    left = aim.seen_left - REFIND_SLACK
    right = aim.seen_right + REFIND_SLACK
    found = [
        p
        for p in _still_pieces(scene)
        if abs(p[2] - aim.top) <= SEAM_ROWS and p[1] >= left and p[0] <= right
    ]
    if not found:
        return aim
    return _joined(scene, min(p[0] for p in found), max(p[1] for p in found), aim.top)
