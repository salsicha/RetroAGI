"""The things vision reports in a picture, as lists of fixed greatest
length: when a list overflows, the things nearest Mario are kept
(packed_lists). A pointer (list, slot) names one reported thing.
"""

from typing import Sequence

from .pixel_types import SCREEN_SHAPE
from .scene_labels import Box, SceneObservation

# How many of each thing the policy is told about (data/scene_slot_survey.json).
SCENE_SLOTS = {
    "enemies": 6,
    "coins": 16,
    "power_ups": 2,
    "moving_platforms": 3,
    "pipes": 4,
    "blocks": 10,
    "surfaces": 14,
    "gaps": 5,
}


def _centre(box: Box) -> tuple[float, float]:
    return (box[0] + box[2]) / 2, (box[1] + box[3]) / 2


def _reference(scene: SceneObservation) -> tuple[float, float, float]:
    """Mario's centre x, centre y and feet row, or the screen's centre without him."""
    if scene.mario.box is None:
        return SCREEN_SHAPE[1] / 2, SCREEN_SHAPE[0] / 2, SCREEN_SHAPE[0] / 2
    x, y = _centre(scene.mario.box)
    return x, y, float(scene.mario.box[3])


def _nearest(items: Sequence, limit: int, distance) -> list:
    """At most ``limit`` items: the nearest by ``distance``, back in their original order."""
    if len(items) <= limit:
        return list(items)
    keep = sorted(range(len(items)), key=lambda i: distance(items[i]))[:limit]
    return [items[i] for i in sorted(keep)]


def _box_distance(mx: float, my: float):
    return lambda box: abs(_centre(box)[0] - mx) + abs(_centre(box)[1] - my)


def packed_lists(scene: SceneObservation) -> dict[str, list]:
    """Each list as the C row holds it: at most SCENE_SLOTS items, nearest Mario kept.

    Slot i of a list in the C row is item i here, so a pointer (list, slot)
    names one reported thing.
    """
    mx, my, feet = _reference(scene)
    near = _box_distance(mx, my)
    lists = {
        "enemies": _nearest(scene.enemies, SCENE_SLOTS["enemies"], lambda e: near(e.box)),
        "blocks": _nearest(scene.blocks, SCENE_SLOTS["blocks"], lambda b: near(b.box)),
        "surfaces": _nearest(
            scene.surfaces,
            SCENE_SLOTS["surfaces"],
            lambda s: max(0.0, s.x0 - mx, mx - s.x1) + abs(s.top - feet),
        ),
        "gaps": _nearest(scene.gaps, SCENE_SLOTS["gaps"], lambda g: max(0.0, g.x0 - mx, mx - g.x1)),
    }
    for name in ("coins", "power_ups", "moving_platforms", "pipes"):
        lists[name] = _nearest(getattr(scene, name), SCENE_SLOTS[name], near)
    return lists
