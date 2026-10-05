"""The one reader that turns pictures into everything a policy receives.

A policy observes the game only through its game's vision transformer.
VisionObserver takes screens (and nothing else: no game, no memory, no
simulator) and returns, per screen, the SceneObservation the vision
transformer reports and the policy's three inputs packed from it:

- src_c: one row of numbers in named spans (C_SPANS): Mario (present, drawn
  box, facing, standing / in the air / on a moving platform), then fixed-length
  lists of enemies, coins, power-ups, moving platforms, pipes, blocks,
  surfaces and gaps, each slot "present" plus the thing's position relative
  to Mario. There is no speed, no motion and nothing from earlier frames:
  anything over time is the policy's own memory.
- src_a, src_b: a code for each of 8 (A) and 16 (B) vertical screen bands,
  the most important thing drawn in the band (COLUMN_CODES).

Both games use this module unchanged, so a Block SMB policy reads Full SMB
exactly as it read Block SMB. List lengths (SCENE_SLOTS) come from the slot
survey (scripts/vision/survey_scene_slots.py): the larger of the two games'
99.9th percentile of things on screen at once. When a list overflows, the
things nearest Mario are kept.
"""

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch

from .smb_pixel_types import SCREEN_SHAPE, visible_window
from .smb_scene_labels import ENEMY_KINDS, SUPPORTS, Box, SceneObservation

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
# Numbers per slot: present, then the position relative to Mario, then extras.
_SLOT_WIDTH = {
    "enemies": 5 + len(ENEMY_KINDS),
    "coins": 5,
    "power_ups": 5,
    "moving_platforms": 5,
    "pipes": 5,
    "blocks": 6,
    "surfaces": 5,
    "gaps": 3,
}
MARIO_WIDTH = 6 + len(SUPPORTS)  # present, box (4), facing right, support one-hot
# The policy's three inputs (StageSpec): A has 8 bands, B 16, and C is 16 x 22.
SEQ_LEN_A, SEQ_LEN_B, RATIO_BC = 8, 16, 22
SEQ_LEN_C = SEQ_LEN_B * RATIO_BC


def _spans() -> dict[str, tuple[int, int]]:
    spans, offset = {}, 0
    for name, width in (
        ("c_mario", MARIO_WIDTH),
        *((f"c_{name}", SCENE_SLOTS[name] * _SLOT_WIDTH[name]) for name in SCENE_SLOTS),
    ):
        spans[name] = (offset, offset + width)
        offset += width
    if offset > SEQ_LEN_C:
        raise ValueError(f"the scene needs {offset} numbers but C holds {SEQ_LEN_C}")
    spans["c_reserved"] = (offset, SEQ_LEN_C)
    return spans


C_SPANS = _spans()

# Column codes for src_a / src_b, most important first: a band's code is the
# first of these drawn in it. 0 means nothing (or no Mario to compare with).
COLUMN_CODES = (
    "nothing",
    "mario",
    "walker",
    "plant",
    "other_enemy",
    "gap",
    "moving_platform",
    "power_up",
    "coin",
    "high_step_up",
    "step_up",
    "level",
    "step_down",
    "block_overhead",
    "defeated_enemy",
)
CODE = {name: index for index, name in enumerate(COLUMN_CODES)}
# Surfaces this many pixels above Mario's feet are a high step (beyond a
# small hop); within LEVEL_BAND of his feet they are level with him.
HIGH_STEP = 36
LEVEL_BAND = 4


@dataclass(frozen=True)
class PolicyInput:
    """Everything a policy receives for one screen; nothing about the game itself."""

    src_a: torch.Tensor  # [8] long column codes
    src_b: torch.Tensor  # [16] long column codes
    src_c: torch.Tensor  # [SEQ_LEN_C] float, laid out by C_SPANS
    scene: SceneObservation


def observation_layout() -> dict:
    """The layout of the policy's inputs, stored with checkpoints and compared on load."""
    return {
        "version": 2,
        "scene_slots": dict(SCENE_SLOTS),
        "c_spans": {name: list(span) for name, span in C_SPANS.items()},
        "column_codes": list(COLUMN_CODES),
        "enemy_kinds": list(ENEMY_KINDS),
        "supports": list(SUPPORTS),
        "seq_len": [SEQ_LEN_A, SEQ_LEN_B, SEQ_LEN_C],
    }


# ── Packing ───────────────────────────────────────────────────────────────────


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


def _relative_box(box: Box, mx: float, my: float) -> list[float]:
    width, height = SCREEN_SHAPE[1], SCREEN_SHAPE[0]
    return [
        (box[0] - mx) / width,
        (box[1] - my) / height,
        (box[2] - mx) / width,
        (box[3] - my) / height,
    ]


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


def pack_c(scene: SceneObservation) -> np.ndarray:
    """The C row for one scene (C_SPANS)."""
    row = np.zeros(SEQ_LEN_C, np.float32)
    mx, my, feet = _reference(scene)
    width, height = SCREEN_SHAPE[1], SCREEN_SHAPE[0]

    start = C_SPANS["c_mario"][0]
    if scene.mario.box is not None:
        x0, y0, x1, y1 = scene.mario.box
        row[start : start + 5] = (1.0, x0 / width, y0 / height, x1 / width, y1 / height)
    row[start + 5] = float(scene.mario.facing_right)
    row[start + 6 + SUPPORTS.index(scene.mario.support)] = 1.0

    def fill(name, entries):
        begin, _ = C_SPANS[f"c_{name}"]
        slot = _SLOT_WIDTH[name]
        for index, values in enumerate(entries):
            row[begin + index * slot : begin + (index + 1) * slot] = values

    lists = packed_lists(scene)
    fill(
        "enemies",
        [
            [
                1.0,
                *_relative_box(e.box, mx, my),
                *np.eye(len(ENEMY_KINDS))[ENEMY_KINDS.index(e.kind)],
            ]
            for e in lists["enemies"]
        ],
    )
    for name in ("coins", "power_ups", "moving_platforms", "pipes"):
        fill(name, [[1.0, *_relative_box(box, mx, my)] for box in lists[name]])
    fill(
        "blocks",
        [
            [1.0, *_relative_box(b.box, mx, my), float(b.kind == "question_block")]
            for b in lists["blocks"]
        ],
    )
    fill(
        "surfaces",
        [
            [
                1.0,
                (s.x0 - mx) / width,
                (s.x1 - mx) / width,
                (s.top - feet) / height,
                float(s.moving),
            ]
            for s in lists["surfaces"]
        ],
    )
    fill("gaps", [[1.0, (g.x0 - mx) / width, (g.x1 - mx) / width] for g in lists["gaps"]])
    return row


def column_codes(scene: SceneObservation, bands: int) -> np.ndarray:
    """The most important thing drawn in each of ``bands`` equal vertical screen bands."""
    width = SCREEN_SHAPE[1]
    band_width = width / bands
    found = [[] for _ in range(bands)]

    def mark(x0: float, x1: float, code: str) -> None:
        first = max(0, int(x0 // band_width))
        last = min(bands - 1, int((max(x0, x1 - 1)) // band_width))
        for band in range(first, last + 1):
            found[band].append(CODE[code])

    _, _, feet = _reference(scene)
    if scene.mario.box is not None:
        mark(scene.mario.box[0], scene.mario.box[2], "mario")
    for enemy in scene.enemies:
        code = {"walker": "walker", "plant": "plant", "other": "other_enemy"}.get(
            enemy.kind, "defeated_enemy"
        )
        mark(enemy.box[0], enemy.box[2], code)
    for gap in scene.gaps:
        mark(gap.x0, gap.x1, "gap")
    for box in scene.moving_platforms:
        mark(box[0], box[2], "moving_platform")
    for box in scene.power_ups:
        mark(box[0], box[2], "power_up")
    for box in scene.coins:
        mark(box[0], box[2], "coin")
    for surface in scene.surfaces:
        rise = feet - surface.top
        if scene.mario.box is None or abs(rise) <= LEVEL_BAND:
            code = "level"
        elif rise > HIGH_STEP:
            code = "high_step_up"
        elif rise > 0:
            code = "step_up"
        else:
            code = "step_down"
        mark(surface.x0, surface.x1, code)
    for block in scene.blocks:
        if scene.mario.box is not None and block.box[3] <= scene.mario.box[1]:
            mark(block.box[0], block.box[2], "block_overhead")
    return np.array([min(codes) if codes else CODE["nothing"] for codes in found], np.int64)


def policy_input(scene: SceneObservation) -> PolicyInput:
    """A scene packed into the policy's three inputs."""
    return PolicyInput(
        src_a=torch.from_numpy(column_codes(scene, SEQ_LEN_A)),
        src_b=torch.from_numpy(column_codes(scene, SEQ_LEN_B)),
        src_c=torch.from_numpy(pack_c(scene)),
        scene=scene,
    )


class VisionObserver:
    """A game's scene vision transformer, and the packing every policy shares.

    observe() takes pictures, only pictures: [N, 240, 256, 3] uint8 screens (a
    single [240, 256, 3] screen is one picture). Like every picture the vision
    transformers learned from, only the part the NES shows is used
    (smb_pixel_types.visible_window).
    """

    def __init__(self, vision):
        self.vision = vision
        self.vision.eval()

    @torch.no_grad()
    def observe(self, screens) -> list[SceneObservation]:
        screens = np.asarray(screens)
        if screens.ndim == 3:
            screens = screens[None]
        if screens.shape[1:] != (*SCREEN_SHAPE, 3) or screens.dtype != np.uint8:
            raise ValueError(
                f"observe takes uint8 screens [N, 240, 256, 3], got {screens.shape} {screens.dtype}"
            )
        return self.vision.scene(np.stack([visible_window(screen) for screen in screens]))

    def policy_inputs(self, screens) -> list[PolicyInput]:
        return [policy_input(scene) for scene in self.observe(screens)]
