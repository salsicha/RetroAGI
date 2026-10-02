"""The object types both SMB vision models give every pixel, and what follows.

Both SMB vision models (one per-pixel vision transformer class, with Block SMB
and Full SMB weights) label every pixel of the 256x240 screen with one of these
types. Labels for training come from each game's own truth: the practice game's
drawing, and the NES game's memory. Everything the decision-making part takes
from vision is computed from a label image by the functions here, so the two
models cannot drift apart in meaning.
"""

import numpy as np
from scipy import ndimage

PIXEL_TYPES = (
    "background",
    "mario",
    "ground",
    "brick",
    "question_block",
    "pipe",
    "coin",
    "enemy",
    "moving_platform",
    "power_up",
)
TYPE_ID = {name: index for index, name in enumerate(PIXEL_TYPES)}
SCREEN_SHAPE = (240, 256)
# The part of the 256x240 screen the NES shows (8 pixels are cut from every
# edge). Both games are seen and labelled only through this window: outside
# it, pictures and labels repeat the window's edge (visible_window).
VISIBLE_ROWS = (8, 232)
VISIBLE_COLUMNS = (8, 248)
# Surfaces Mario can stand on.
SOLID_TYPES = tuple(TYPE_ID[name] for name in ("ground", "brick", "question_block", "pipe"))
STANDING_TYPES = SOLID_TYPES + (TYPE_ID["moving_platform"],)


def visible_window(array):
    """A full-screen picture or label map with everything outside the NES window
    replaced by the window's nearest edge pixel (no stretching)."""
    array = np.asarray(array)
    (top, bottom), (left, right) = VISIBLE_ROWS, VISIBLE_COLUMNS
    inner = array[top:bottom, left:right]
    padding = [(top, SCREEN_SHAPE[0] - bottom), (left, SCREEN_SHAPE[1] - right)]
    padding += [(0, 0)] * (array.ndim - 2)
    return np.pad(inner, padding, mode="edge")


def mario_region(labels):
    """Pixels of the largest connected group labelled Mario, or None."""
    groups, count = ndimage.label(np.asarray(labels) == TYPE_ID["mario"])
    if not count:
        return None
    sizes = np.bincount(groups.ravel())[1:]
    return groups == int(np.argmax(sizes)) + 1


def mario_position(labels):
    """Centre of Mario's largest pixel group as (x, y) in pixels, or None."""
    region = mario_region(labels)
    if region is None:
        return None
    rows, columns = np.nonzero(region)
    return float(columns.mean()), float(rows.mean())


def mario_standing(labels, *, reach=2):
    """Whether a standable pixel lies within `reach` pixels below Mario's feet."""
    region = mario_region(labels)
    if region is None:
        return False
    rows, columns = np.nonzero(region)
    bottom = int(rows.max())
    feet = columns[rows >= bottom - 1]
    below = np.asarray(labels)[bottom + 1 : bottom + 1 + reach, feet.min() : feet.max() + 1]
    return bool(np.isin(below, STANDING_TYPES).any())


def type_shares(labels):
    """Fraction of the screen covered by each type, in PIXEL_TYPES order."""
    counts = np.bincount(np.asarray(labels).ravel(), minlength=len(PIXEL_TYPES))
    return counts[: len(PIXEL_TYPES)] / max(int(counts.sum()), 1)


def compare(predicted, truth):
    """Per-type agreement between a predicted and a true label image.

    For each type: the share of true pixels found ("found") and the share of
    pixels given that type which really are that type ("correct").
    """
    predicted, truth = np.asarray(predicted), np.asarray(truth)
    result = {"pixels_correct": float((predicted == truth).mean())}
    for index, name in enumerate(PIXEL_TYPES):
        true_pixels, given_pixels = truth == index, predicted == index
        both = int((true_pixels & given_pixels).sum())
        result[name] = {
            "true_pixels": int(true_pixels.sum()),
            "found": both / int(true_pixels.sum()) if true_pixels.any() else None,
            "correct": both / int(given_pixels.sum()) if given_pixels.any() else None,
        }
    return result


def vision_output(pixel_logits):
    """The common vision output from per-pixel type scores [B, types, 240, 256].

    Both SMB vision models end here. Per-square type scores (the average pixel
    probability in each 16x16 square, as a log) feed the column labels and
    screen shares; Mario's position and whether he stands come from the
    per-pixel labels by the rules above. A frame without Mario reports
    position (0, 0) and `mario_found` False.
    """
    import torch
    import torch.nn.functional as F

    from retroagi.core.interfaces import VisionOutput

    if tuple(pixel_logits.shape[-2:]) != SCREEN_SHAPE or pixel_logits.shape[1] != len(PIXEL_TYPES):
        raise ValueError("pixel_logits must be [batch, types, 240, 256]")
    probabilities = pixel_logits.float().softmax(1)
    squares = F.avg_pool2d(probabilities, 16)
    # max(1).indices equals argmax(1) (first of equal maxima) and is several
    # times faster on one CPU thread, where policy rollouts run.
    labels = probabilities.max(1).indices.to(torch.uint8)
    positions, support, found = [], [], []
    for frame in labels.cpu().numpy():
        position = mario_position(frame)
        found.append(position is not None)
        x, y = position if position is not None else (0.0, 0.0)
        positions.append((x / (SCREEN_SHAPE[1] - 1), y / (SCREEN_SHAPE[0] - 1)))
        if not mario_standing(frame):
            support.append(0)
        else:
            region = mario_region(frame)
            rows, columns = np.nonzero(region)
            below = frame[rows.max() + 1 : rows.max() + 3, columns.min() : columns.max() + 1]
            on_lift = bool((below == TYPE_ID["moving_platform"]).any())
            support.append(2 if on_lift else 1)
    device = pixel_logits.device
    support_ids = torch.tensor(support, device=device)
    support_logits = torch.full((len(support), 3), -4.0, device=device)
    support_logits.scatter_(1, support_ids.unsqueeze(1), 4.0)
    semantic_logits = squares.clamp_min(1e-6).log()
    return VisionOutput(
        position=torch.tensor(positions, dtype=torch.float32, device=device),
        semantic_logits=semantic_logits,
        semantic_ids=semantic_logits.argmax(1),
        tokens=squares.flatten(2).transpose(1, 2),
        metadata={
            "semantic_classes": PIXEL_TYPES,
            "support_classes": ("air", "ground", "platform"),
            "pixel_labels": labels,
            "mario_found": found,
        },
        support_logits=support_logits,
        support_ids=support_ids,
    )
