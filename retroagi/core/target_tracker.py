"""The target tracker: where an object the vision reports will be some frames
from now (docs/action-predictor-controller.md, section 3).

The predictive controller has no model of the world; it asks this tracker
where its target (a walking enemy, a moving platform) will be when Mario gets
there, and the tracker tells it when the world departs from what it said.

One small recurrent network (a GRU) is shared by every tracked enemy and
moving platform and stepped once per frame for each. Per frame and object it
reads, from vision only:

- whether the object is seen, how it moved since it was last seen and on
  average over its last SMOOTH_FRAMES sightings (pixels per frame, the
  camera's scroll removed);
- its kind (walker, plant, other enemy, moving platform);
- its surroundings: how far its middle is from each end of the surface it
  stands on (a walker turns at a ledge or a wall);
- Mario's position relative to it.

Its recurrent state carries what one picture cannot show: direction, speed,
and where a plant or platform is in its cycle. It outputs, for each of
TRACKER_HORIZONS frames ahead, the object's displacement from where it is
now (x, y), an uncertainty for each, and the chance that it is still seen.
The displacement is a correction to steady motion (its average movement
continued), so an untrained tracker predicts steady motion.
"""

import math
from collections import defaultdict, deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .smb_observer import SCENE_SLOTS, packed_lists
from .smb_pixel_types import VISIBLE_COLUMNS

TRACKER_HORIZONS = (1, 2, 4, 8, 16, 32, 64, 96)
OBJECT_SLOTS = SCENE_SLOTS["enemies"] + SCENE_SLOTS["moving_platforms"]
OBJECT_KINDS = ("walker", "plant", "other", "platform")
# One object at one frame, as play records it (object_rows): its visual
# identity (0: no object), kind (index in OBJECT_KINDS), box in world pixels,
# the ends of the surface it stands on (world x; "supported" 0 when none),
# and Mario's middle and feet (world pixels).
OBJECT_ROW = (
    "identity",
    "kind",
    "x0",
    "y0",
    "x1",
    "y1",
    "support_left",
    "support_right",
    "supported",
    "mario_x",
    "mario_feet",
)
ROW = {name: i for i, name in enumerate(OBJECT_ROW)}
# The steady motion a forecast corrects: an object's movement per frame
# averaged over its last SMOOTH_FRAMES sightings. Boxes move by whole pixels,
# so a walker at 0.3 pixels per frame moves 0 or 1 between two frames: its
# last movement continued for 64 frames was 21 pixels off on average, the
# average over 16 sightings 3.
SMOOTH_FRAMES = 16
# The tracker's input per object and frame (track_inputs).
TRACK_INPUT = 1 + 2 + 2 + len(OBJECT_KINDS) + 3 + 2


def _visible(box) -> bool:
    """Whole inside the window the vision sees (a clipped box moves its middle
    without the object moving)."""
    return VISIBLE_COLUMNS[0] < box[0] and box[2] < VISIBLE_COLUMNS[1]


def object_rows(scene, spatial) -> np.ndarray:
    """[OBJECT_SLOTS, len(OBJECT_ROW)]: the enemies' slots, then the moving
    platforms', each with its visual identity from ``spatial``'s tracks
    (SpatialFeedback) and its position in world pixels."""
    rows = np.zeros((OBJECT_SLOTS, len(OBJECT_ROW)), np.float32)
    if scene.mario.box is None:
        return rows
    camera = spatial.camera_position
    lists = packed_lists(scene)
    tracks = spatial.tracks.tracks
    mario = scene.mario.box
    objects = [
        (slot, e.box, e.kind) for slot, e in enumerate(lists["enemies"]) if e.kind != "defeated"
    ]
    objects += [
        (SCENE_SLOTS["enemies"] + slot, box, "platform")
        for slot, box in enumerate(lists["moving_platforms"])
    ]
    for slot, box, kind in objects:
        matches = [t for t in tracks if t.kind == kind and t.box == box]
        if len(matches) != 1 or not _visible(box):
            continue
        row = rows[slot]
        row[ROW["identity"]] = matches[0].identity
        row[ROW["kind"]] = OBJECT_KINDS.index(kind)
        row[ROW["x0"] : ROW["y1"] + 1] = (box[0] + camera, box[1], box[2] + camera, box[3])
        middle = (box[0] + box[2]) / 2
        under = [
            s for s in scene.surfaces if abs(s.top - box[3]) <= 3 and s.x0 - 2 <= middle <= s.x1 + 2
        ]
        if kind != "platform" and under:
            row[ROW["support_left"]] = under[0].x0 + camera
            row[ROW["support_right"]] = under[0].x1 + camera
            row[ROW["supported"]] = 1.0
        row[ROW["mario_x"]] = (mario[0] + mario[2]) / 2 + camera
        row[ROW["mario_feet"]] = mario[3]
    return rows


def _point(row):
    """An object's reference point: its middle and its top (world pixels)."""
    return np.array(((row[ROW["x0"]] + row[ROW["x1"]]) / 2, row[ROW["y0"]]), np.float32)


def _move(row, previous, gap):
    """Movement per frame since the object was last seen (pixels)."""
    return (_point(row) - _point(previous)) / gap


def track_inputs(row, previous, gap: int, average=None) -> np.ndarray:
    """[TRACK_INPUT] for one object at one frame: ``row`` (None: not seen this
    frame), ``previous`` its row when last seen (None: never), ``gap`` the
    frames since then, ``average`` its average movement per frame (None:
    unknown)."""
    out = np.zeros(TRACK_INPUT, np.float32)
    if row is None:
        return out
    out[0] = 1.0
    if previous is not None and gap > 0:
        out[1:3] = _move(row, previous, gap) / 2
    if average is not None:
        out[3:5] = np.asarray(average) / 2
    kind = int(row[ROW["kind"]])
    out[5 + kind] = 1.0
    k = 5 + len(OBJECT_KINDS)
    if row[ROW["supported"]]:
        middle = (row[ROW["x0"]] + row[ROW["x1"]]) / 2
        out[k] = 1.0
        out[k + 1] = min(4.0, max(0.0, middle - row[ROW["support_left"]]) / 64)
        out[k + 2] = min(4.0, max(0.0, row[ROW["support_right"]] - middle) / 64)
    out[k + 3] = (row[ROW["mario_x"]] - (row[ROW["x0"]] + row[ROW["x1"]]) / 2) / 128
    out[k + 4] = (row[ROW["mario_feet"]] - row[ROW["y1"]]) / 128
    return out


class TargetTracker(nn.Module):
    def __init__(self, width: int = 48):
        super().__init__()
        self.width = width
        self.cell = nn.GRU(TRACK_INPUT, width, batch_first=True)
        self.head = nn.Linear(width, len(TRACKER_HORIZONS) * 5)
        self.register_buffer(
            "horizons", torch.tensor(TRACKER_HORIZONS, dtype=torch.float32), persistent=False
        )
        # Untrained: steady motion, an uncertainty growing with the horizon
        # (about a pixel 4 frames ahead, 8 pixels 64 ahead), and most likely
        # still seen.
        nn.init.zeros_(self.head.weight)
        with torch.no_grad():
            bias = self.head.bias.view(len(TRACKER_HORIZONS), 5)
            bias.zero_()
            bias[:, 2:4] = math.log(0.25)
            bias[:, 4] = 2.0

    def run(self, inputs, hidden=None):
        """inputs [N, T, TRACK_INPUT] -> (states [N, T, width], last [1, N, width])."""
        return self.cell(inputs, hidden)

    def forecast(self, states, velocity):
        """From states [..., width] and each object's average movement velocity
        [..., 2] (pixels per frame): {"displacement", "sigma"} [...,
        horizons, 2] in pixels and "visible" [..., horizons] logits."""
        raw = self.head(states).view(*states.shape[:-1], len(TRACKER_HORIZONS), 5)
        h = self.horizons.view(*([1] * (states.dim() - 1)), -1, 1)
        steady = velocity.unsqueeze(-2) * h
        spread = 2.0 + h / 2
        return {
            "displacement": steady + raw[..., :2] * spread,
            "sigma": raw[..., 2:4].clamp(-4, 3).exp() * spread,
            "visible": raw[..., 4],
        }


def tracker_loss(predicted, target, seen, known):
    """predicted: TargetTracker.forecast; ``target`` [..., horizons, 2]
    displacement where the object is seen then; ``seen`` [..., horizons]
    bool; ``known`` [..., horizons]: whether that frame is within the episode
    (whether the object is seen then is known). Returns (loss, stats with
    the mean error in pixels at each horizon)."""
    loss = F.binary_cross_entropy_with_logits(predicted["visible"][known], seen[known].float())
    stats = {}
    if seen.any():
        mean, sigma = predicted["displacement"][seen], predicted["sigma"][seen]
        error = mean - target[seen]
        loss = loss + (0.5 * (error / sigma).square() + sigma.log()).mean()
        pixels = (predicted["displacement"] - target).detach().norm(dim=-1)
        for k, h in enumerate(TRACKER_HORIZONS):
            mask = seen[..., k]
            if mask.any():
                stats[f"error_{h}"] = float(pixels[..., k][mask].mean())
    return loss, stats


def interpolate(horizons, values, frames: float):
    """A forecast's value ``frames`` ahead, read linearly between its horizons
    (``values`` [horizons, ...]; before the first, from no displacement)."""
    points = (0, *horizons)
    if frames >= points[-1]:
        return values[-1]
    k = next(i for i in range(1, len(points)) if frames <= points[i])
    low = values[k - 2] if k >= 2 else values[0] * 0
    w = (frames - points[k - 1]) / (points[k] - points[k - 1])
    return low * (1 - w) + values[k - 1] * w


class ObjectTracker:
    """The tracker at play: one recurrent state per visual identity, stepped
    every frame; where an object will be, and whether it has departed from
    what was forecast (then the controller is sent the updated target)."""

    # How many frames back the forecast checked against what is seen now was
    # made, and how many of its uncertainties away the object must be.
    CHECK_LAG = 4
    CHECK_SIGMAS = 3.0

    def __init__(self, model: TargetTracker, device="cpu"):
        self.model = model.to(device).eval()
        self.device = device
        self.hidden: dict = {}
        self.last: dict = {}  # identity -> (row, frame) when last seen
        self.velocity: dict = defaultdict(lambda: np.zeros(2, np.float32))
        self.moves: dict = defaultdict(lambda: deque(maxlen=SMOOTH_FRAMES))
        self.forecasts: dict = defaultdict(lambda: deque(maxlen=self.CHECK_LAG + 1))
        self.frame = 0

    @torch.no_grad()
    def observe(self, rows: np.ndarray) -> None:
        """One frame's object_rows."""
        seen = {int(r[ROW["identity"]]): r for r in rows if r[ROW["identity"]]}
        identities = sorted(set(seen) | set(self.hidden))
        if not identities:
            self.frame += 1
            return
        inputs, moving = [], set()
        for i in identities:
            row = seen.get(i)
            previous, when = self.last.get(i, (None, self.frame))
            average = None
            if row is not None and previous is not None and self.frame > when:
                self.moves[i].append(_move(row, previous, self.frame - when))
                average = np.mean(self.moves[i], axis=0)
                self.velocity[i] = average.astype(np.float32)
                moving.add(i)
            inputs.append(track_inputs(row, previous, self.frame - when, average))
        hidden = torch.stack(
            [
                self.hidden.get(i, torch.zeros(self.model.width, device=self.device))
                for i in identities
            ]
        )
        x = torch.tensor(np.stack(inputs), device=self.device).unsqueeze(1)
        states, last = self.model.run(x, hidden.unsqueeze(0))
        velocity = torch.tensor(
            np.stack([self.velocity[i] for i in identities]), device=self.device
        )
        out = self.model.forecast(states[:, 0], velocity)
        for k, i in enumerate(identities):
            self.hidden[i] = last[0, k]
            if i in seen:
                self.last[i] = (seen[i], self.frame)
                where = _point(seen[i])
                self.forecasts[i].append(
                    (
                        self.frame,
                        where,
                        out["displacement"][k].cpu().numpy(),
                        out["sigma"][k].cpu().numpy(),
                        i in moving,
                    )
                )
        # Objects unseen for the longest horizon are forgotten.
        for i in [
            i for i, (_, when) in self.last.items() if self.frame - when > TRACKER_HORIZONS[-1]
        ]:
            for store in (self.hidden, self.last, self.velocity, self.moves, self.forecasts):
                store.pop(i, None)
        self.frame += 1

    def where(self, identity: int, frames: float):
        """(x, y, sigma) in world pixels: where the object (its middle and top)
        will be ``frames`` after the last observed frame; None if unknown."""
        if not self.forecasts.get(identity):
            return None
        _, now, displacement, sigma, _ = self.forecasts[identity][-1]
        d = interpolate(TRACKER_HORIZONS, displacement, frames)
        s = interpolate(TRACKER_HORIZONS, sigma, frames)
        return float(now[0] + d[0]), float(now[1] + d[1]), float(np.linalg.norm(s))

    def changed(self, identity: int) -> bool:
        """Whether the object is now outside what was forecast CHECK_LAG
        frames ago (a walker turned, a platform reversed). A forecast made
        before its movement was seen (at its first sighting) is not checked."""
        history = self.forecasts.get(identity)
        if not history or len(history) <= self.CHECK_LAG:
            return False
        made, then, displacement, sigma, moving = history[0]
        if not moving:
            return False
        seen_frame, now, _, _, _ = history[-1]
        lag = seen_frame - made
        expected = then + interpolate(TRACKER_HORIZONS, displacement, lag)
        spread = interpolate(TRACKER_HORIZONS, sigma, lag)
        error = np.abs(now - expected)
        return bool((error > self.CHECK_SIGMAS * np.maximum(spread, 0.5)).any())


def track_examples(objects: np.ndarray):
    """Training sequences from one episode's object rows [T, OBJECT_SLOTS,
    len(OBJECT_ROW)]: for each visual identity, from its first to its last
    sighting, (inputs [L, TRACK_INPUT], average velocity [L, 2], targets [L,
    horizons, 2], seen [L, horizons], known [L, horizons], valid [L]):
    ``known`` marks the horizons within the episode, ``valid`` the frames it
    is seen (a forecast is made from those)."""
    frames = len(objects)
    sightings = defaultdict(dict)
    for t in range(frames):
        for row in objects[t]:
            if row[ROW["identity"]]:
                sightings[int(row[ROW["identity"]])][t] = row
    out = []
    for seen_at in sightings.values():
        first, last = min(seen_at), max(seen_at)
        length = last - first + 1
        inputs = np.zeros((length, TRACK_INPUT), np.float32)
        velocity = np.zeros((length, 2), np.float32)
        targets = np.zeros((length, len(TRACKER_HORIZONS), 2), np.float32)
        seen = np.zeros((length, len(TRACKER_HORIZONS)), bool)
        known = np.zeros((length, len(TRACKER_HORIZONS)), bool)
        valid = np.zeros(length, bool)
        previous, when, moving = None, first, np.zeros(2, np.float32)
        moves = deque(maxlen=SMOOTH_FRAMES)
        for k, t in enumerate(range(first, last + 1)):
            row = seen_at.get(t)
            if row is None:
                velocity[k] = moving
                continue
            average = None
            if previous is not None:
                moves.append(_move(row, previous, t - when))
                average = np.mean(moves, axis=0)
                moving = average.astype(np.float32)
            inputs[k] = track_inputs(row, previous, t - when, average)
            velocity[k] = moving
            valid[k] = True
            here = _point(row)
            for j, h in enumerate(TRACKER_HORIZONS):
                later = seen_at.get(t + h)
                known[k, j] = t + h < frames
                if later is not None:
                    targets[k, j] = _point(later) - here
                    seen[k, j] = True
            previous, when = row, t
        out.append((inputs, velocity, targets, seen, known, valid))
    return out


def steady_error(examples) -> dict:
    """The steady-motion forecast's mean error (pixels) at each horizon: the
    baseline the tracker must beat."""
    totals = defaultdict(list)
    for _, velocity, targets, seen, _, valid in examples:
        for j, h in enumerate(TRACKER_HORIZONS):
            mask = valid & seen[:, j]
            if mask.any():
                error = np.linalg.norm(velocity[mask] * h - targets[mask, j], axis=-1)
                totals[h].extend(error.tolist())
    return {h: float(np.mean(v)) for h, v in totals.items()}
