"""The collision trackers: for each enemy vision reports, every frame, whether
Mario and it will touch within HORIZON frames, when, where, and as a stomp
or a hit (user, 2026-10-15/16). They are trained on labels made from what
really happened (core/collision_labels) and are not used while the skills
and tactics train; the executor will end a wait or a walk and aim a stomp by
them in play without the teacher.

Each enemy's inputs per frame (frame_features), from the executor and vision
only:

- the executor's first-order kinematics of it (core/smb_kinematics: where
  it is from Mario's feet, its velocity, how many sightings the velocity
  rests on), how long it has been out of sight, and what vision shows of
  it (kind, size, the surface it walks on);
- the executor's kinematics of Mario (his speed, his feet's movement,
  whether he stands);
- the action Mario is carrying out (ActionState): its verb, where it sends
  him from where he is now, how long it has run, whether it keeps his
  speed, whether this enemy is its object.

Where an action sends Mario is what the executor was asked, except for an
action aimed at an enemy: there it is the enemy's point when the action
began. The teacher aimed those with the enemy's true future, which no input
may carry; a hold sends him nowhere (its length is the teacher's truth).

The enemies followed are those on the list of tracked things
(core/smb_tracked_objects: those seen now and those out of sight, kept where
they vanished; user, 2026-10-21), ROWS rows, each known by its entry's key.

One GRU (shared by every enemy, a state per entry) reads the first
OBSERVATION_WIDTH inputs every frame the enemy is on the list: what
one picture cannot show (a plant's phase, how long it has been hidden,
where a walker turns). A head reads its state
with all the frame's inputs; the action enters there only, so asking about
another action needs no more GRU steps. It gives CLASSES logits (a stomp or
a hit in each of BINS two-frame bins over HORIZON frames, or none) and the
contact place (from Mario's feet now) with its uncertainty.
"""

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

from .action_tokens import VERBS, executed_pointer, pointer_target
from .collision_labels import HORIZON, KINDS
from .layered_policy import POINTER_START
from .smb_kinematics import MARIO_STATE, TRACKED
from .smb_tracked_objects import CAPACITY, COLUMN as LIST_COLUMN, record_objects
from .tokens import SKILL_MODES

BINS = HORIZON // 2
NONE = 2 * BINS  # the class "no contact within HORIZON"
CLASSES = NONE + 1
# Each input's unit (pixels and pixels a frame are scaled to about 1).
POSITION_SCALE = 128.0
OBSERVATION_WIDTH = len(TRACKED) + len(MARIO_STATE)
# The enemies followed each frame: the rows of the list of tracked things.
ROWS = CAPACITY
ACTION_WIDTH = len(VERBS) + 1 + 2 + 1 + 1 + 1
FEATURES = OBSERVATION_WIDTH + ACTION_WIDTH
# The place's unit while learning (pixels).
PLACE_SCALE = 16.0
# Its uncertainty is kept between half a pixel and 256 pixels: a place
# predicted exactly (standing Mario's contacts are where he stands) would
# otherwise drive it to nothing and the loss to minus infinity.
LOG_SIGMA = (math.log(0.5 / PLACE_SCALE), math.log(256 / PLACE_SCALE))


@dataclass
class ActionState:
    """The action Mario is carrying out, as the trackers read it."""

    verb: int = -1  # VERBS index; -1: none (watched frames, an unnamed command)
    destination: tuple = (0.0, 0.0)  # from his feet at the start (pixels)
    start_position: tuple = (0.0, 0.0)  # his world displacement and feet row then
    start_frame: int = 0
    carry: bool = False
    target: int = 0  # the key of its enemy's entry on the list (0: none)


def action_state(verb, pointer, command, sprites, slot_keys, position, frame) -> ActionState:
    """The ActionState of an action starting now: ``verb`` and ``pointer``
    (VERBS index and pointer slot; -1 unknown), the executed ``command``
    (tokens.command_row: mode, x, y, carry, frames, threat; mode -1 none),
    this frame's sprite kinematics, the key of each sprite slot's entry
    (smb_tracked_objects.slot_keys), Mario's position."""
    mode = int(command[0])
    carry = bool(command[3]) if mode >= 0 else False
    if pointer >= 0:
        name, slot = pointer_target(pointer + POINTER_START)
        if name == "enemies":
            # The enemy's point as seen now (smb_kinematics: from his feet);
            # nowhere when it is not tracked (never the command: aimed with
            # its true future).
            if sprites[slot, 0] <= 0:
                return ActionState(verb, (0.0, 0.0), tuple(position), frame, carry, 0)
            destination = (float(sprites[slot, 1]), float(sprites[slot, 2]))
            return ActionState(
                verb, destination, tuple(position), frame, carry, int(slot_keys[slot])
            )
    destination = (0.0, 0.0)
    if mode in (SKILL_MODES.index("run"), SKILL_MODES.index("jump")):
        destination = (float(command[1]), float(command[2]))  # a hold: nowhere
    return ActionState(verb, destination, tuple(position), frame, carry, 0)


def tracker_rows(rows) -> np.ndarray:
    """[ROWS, len(TRACKED)]: the enemies the trackers follow, from the list's
    rows (smb_tracked_objects.TrackedObjects.read); a moving platform's row
    is empty (the trackers follow the enemies)."""
    rows = np.asarray(rows, np.float32)
    state = rows[:, : len(TRACKED)].copy()
    state[rows[:, LIST_COLUMN["platform"]] > 0.5] = 0.0
    return state


def frame_features(state, keys, mario, position, frame, action: ActionState):
    """[ROWS, FEATURES] for one frame: each followed enemy (tracker_rows:
    ``state`` [ROWS, len(TRACKED)], ``keys`` [ROWS]), the executor's
    kinematics of Mario, his position (world displacement, feet row) and the
    action in progress."""
    out = np.zeros((ROWS, FEATURES), np.float32)
    enemy = np.asarray(state, np.float32).copy()
    enemy[:, 1:3] /= POSITION_SCALE
    out[:, : len(TRACKED)] = enemy
    k = len(TRACKED)
    out[:, k : k + len(MARIO_STATE)] = mario
    k += len(MARIO_STATE)
    out[:, k + (action.verb if action.verb >= 0 else len(VERBS))] = 1.0
    k += len(VERBS) + 1
    moved = (position[0] - action.start_position[0], position[1] - action.start_position[1])
    out[:, k] = (action.destination[0] - moved[0]) / POSITION_SCALE
    out[:, k + 1] = (action.destination[1] - moved[1]) / POSITION_SCALE
    out[:, k + 2] = min(frame - action.start_frame, HORIZON) / HORIZON
    out[:, k + 3] = float(action.carry)
    out[:, k + 4] = (np.asarray(keys) == action.target) & (action.target > 0)
    present = enemy[:, 0] > 0
    out[~present] = 0.0
    return out


def record_features(record) -> tuple:
    """For a recorded episode (EpisodeTask.record_truth): ([T, ROWS,
    FEATURES], the list frame by frame: smb_tracked_objects.record_objects):
    the inputs the trackers had at each frame, the action in progress being
    the one carried out from each decision (the teacher's choice as named
    where it played)."""
    frames = len(record.sprite_kinematics)
    starts = {int(f): k for k, f in enumerate(record.decision_frames)}
    sprites = record.sprite_kinematics.astype(np.float32)
    mario = record.mario_kinematics.astype(np.float32)
    objects = record_objects(record)
    out = np.zeros((frames, ROWS, FEATURES), np.float32)
    action = ActionState()
    for t in range(frames):
        k = starts.get(t)
        if k is not None and record.action_verb is not None:
            verb = int(record.action_verb[k])
            action = action_state(
                verb,
                executed_pointer(verb, int(record.action_pointer[k])),
                record.commands[k],
                sprites[t],
                objects["slot_keys"][t],
                record.mario_position[t],
                t,
            )
        out[t] = frame_features(
            tracker_rows(objects["rows"][t]),
            objects["keys"][t],
            mario[t],
            record.mario_position[t],
            t,
            action,
        )
    return out, objects


class CollisionTracker(nn.Module):
    """The GRU over an enemy's observations and the head (module notes)."""

    def __init__(self, width: int = 64):
        super().__init__()
        self.width = width
        self.gru = nn.GRU(OBSERVATION_WIDTH, width, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(width + FEATURES, 128), nn.GELU(), nn.Linear(128, CLASSES + 4)
        )
        last = self.head[-1]
        nn.init.zeros_(last.weight)
        nn.init.zeros_(last.bias)
        with torch.no_grad():
            # Untrained: no contact (95% against the 96 contact classes).
            last.bias[NONE] = math.log(2 * BINS) + 3.0
            last.bias[CLASSES + 2 :] = math.log(2.0)  # place uncertainty, 2 x 16 pixels

    def observe(self, observations, hidden=None):
        """GRU states [N, L, width] after each observation [N, L, OBSERVATION_WIDTH]."""
        states, _ = self.gru(observations, hidden)
        return states

    def step(self, observation, hidden):
        """One frame per enemy: observation [B, OBSERVATION_WIDTH], hidden [B, width]."""
        _, out = self.gru(observation.unsqueeze(1), hidden.unsqueeze(0))
        return out[0]

    def answer(self, states, features) -> dict:
        """{"logits" [..., CLASSES], "place" [..., 2] (pixels), "log_sigma"
        [..., 2] (in PLACE_SCALE units)}."""
        out = self.head(torch.cat((states, features), -1))
        return {
            "logits": out[..., :CLASSES],
            "place": out[..., CLASSES : CLASSES + 2] * PLACE_SCALE,
            "log_sigma": out[..., CLASSES + 2 :].clamp(*LOG_SIGMA),
        }


def class_of(frames: int, kind: int) -> int:
    """The class of a contact ``frames`` after now (1..HORIZON), KINDS ``kind``."""
    return (frames - 1) // 2 + (BINS if KINDS[kind] == "hit" else 0)


def allowed_classes(frames, kind, censor) -> np.ndarray:
    """[N, CLASSES] bool: the classes each label allows: its contact's; none
    (known to HORIZON); or, censored from offset c, every contact from c on
    and none."""
    frames, kind, censor = (np.asarray(a) for a in (frames, kind, censor))
    out = np.zeros((len(frames), CLASSES), bool)
    seen = frames > 0
    rows = np.flatnonzero(seen)
    out[rows, [class_of(int(f), int(k)) for f, k in zip(frames[seen], kind[seen])]] = True
    known = ~seen & (censor == 0)
    out[known, NONE] = True
    for i in np.flatnonzero(~seen & (censor > 0)):
        first = (int(censor[i]) - 1) // 2
        out[i, first:BINS] = True
        out[i, BINS + first : NONE] = True
        out[i, NONE] = True
    return out


def collision_loss(out: dict, allowed, place, seen):
    """Minus the log-probability of the classes each label allows, and, for
    seen contacts, the place: squared error in PLACE_SCALE units plus a
    Gaussian fit of the uncertainty (the error detached, so a wide one
    cannot excuse a poor place). ``allowed`` [N, CLASSES] bool, ``place``
    [N, 2] pixels, ``seen`` [N] bool. Returns (loss, stats)."""
    logits = out["logits"]
    total = torch.logsumexp(logits, -1)
    kept = torch.logsumexp(logits.masked_fill(~allowed, float("-inf")), -1)
    classes = (total - kept).mean()
    stats = {"classes": float(classes)}
    loss = classes
    if seen.any():
        error = (out["place"][seen] - place[seen]) / PLACE_SCALE
        log_sigma = out["log_sigma"][seen]
        fit = (error.detach() ** 2 / (2 * torch.exp(2 * log_sigma)) + log_sigma).mean()
        squared = (error**2).mean()
        loss = loss + squared + fit
        stats["place_px"] = float(error.abs().mean() * PLACE_SCALE)
    return loss, stats


@dataclass
class Collision:
    """One enemy's estimate: the chance of any contact within HORIZON, of a
    hit and of a stomp by each two-frame bin (cumulative), the median frame
    of a contact given one, its place (pixels from Mario's feet now) and its
    uncertainty."""

    contact: float
    hit_by: np.ndarray
    stomp_by: np.ndarray
    frames: float
    place: tuple
    sigma: tuple

    def hit_within(self, frames: float) -> float:
        if frames < 1:
            return 0.0
        return float(self.hit_by[min(BINS - 1, int(frames - 1) // 2)])

    def stomp_within(self, frames: float) -> float:
        if frames < 1:
            return 0.0
        return float(self.stomp_by[min(BINS - 1, int(frames - 1) // 2)])


def collision_of(out: dict) -> Collision:
    """A Collision from one enemy's head output (unbatched)."""
    p = torch.softmax(out["logits"], -1).detach().cpu().numpy()
    stomp, hit = p[:BINS], p[BINS:NONE]
    contact = float(stomp.sum() + hit.sum())
    by_bin = stomp + hit
    median = BINS - 1
    if contact > 0:
        median = int(np.searchsorted(np.cumsum(by_bin), contact / 2))
    return Collision(
        contact=contact,
        hit_by=np.cumsum(hit),
        stomp_by=np.cumsum(stomp),
        frames=2.0 * median + 1.5,
        place=tuple(out["place"].detach().cpu().numpy().tolist()),
        sigma=tuple((torch.exp(out["log_sigma"]) * PLACE_SCALE).detach().cpu().numpy().tolist()),
    )


class CollisionTrackers:
    """The trackers at play: a GRU state per entry of the list (its key),
    stepped each frame its enemy is on the list (forgotten after HORIZON
    frames off it)."""

    def __init__(self, model: CollisionTracker, device="cpu"):
        self.model = model.eval()
        self.device = device
        self.hidden: dict = {}
        self.features: dict = {}
        self.seen: dict = {}
        self.frame = 0

    @torch.no_grad()
    def observe(self, features: np.ndarray, keys) -> None:
        """One frame: frame_features and each row's key."""
        self.frame += 1
        present = [
            (slot, int(keys[slot]))
            for slot in range(ROWS)
            if keys[slot] and features[slot, 0] > 0
        ]
        if present:
            ids = [i for _, i in present]
            rows = torch.as_tensor(
                np.stack([features[s, :OBSERVATION_WIDTH] for s, _ in present]), device=self.device
            )
            hidden = torch.stack(
                [
                    self.hidden.get(i, torch.zeros(self.model.width, device=self.device))
                    for i in ids
                ]
            )
            stepped = self.model.step(rows, hidden)
            for (slot, i), h in zip(present, stepped):
                self.hidden[i] = h
                self.features[i] = features[slot]
                self.seen[i] = self.frame
        for i in [i for i, last in self.seen.items() if self.frame - last > HORIZON]:
            for store in (self.hidden, self.features, self.seen):
                store.pop(i, None)

    def refresh(self, features: np.ndarray, keys) -> None:
        """This frame's inputs again after a new action began (no GRU step:
        the action enters the head only)."""
        for slot in range(ROWS):
            i = int(keys[slot])
            if i in self.features and features[slot, 0] > 0:
                self.features[i] = features[slot]

    @torch.no_grad()
    def estimate(self, identity: int, features: Optional[np.ndarray] = None):
        """The Collision for the entry ``identity`` (its key; None if not
        followed), with this frame's inputs or ``features`` (another
        action's, say)."""
        if identity not in self.hidden:
            return None
        row = self.features[identity] if features is None else features
        out = self.model.answer(
            self.hidden[identity], torch.as_tensor(row, dtype=torch.float32, device=self.device)
        )
        return collision_of(out)
