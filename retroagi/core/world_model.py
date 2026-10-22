"""The world model: the policy's LSTM, run once at the start of each action,
after the skill picks it, with that action as input; it predicts the
action's outcome (user, 2026-10-15/16). It is not used while the skills and
tactics train (the teacher carries their actions out there); it learns from
recorded outcomes: each executed action's measured one (core/action_outcomes)
and, at each decision, the other actions the teacher tried
(stages/block_smb/teacher_actions.action_rows), evaluated from the state
before the action without advancing it.

What it reads at an action's start:

- the scene encoder's tokens (trained with the skill; frozen here), but for
  the enemies' and moving platforms': those come from the list of tracked
  things (core/smb_tracked_objects; user, 2026-10-21) instead, one token a
  row: where each is now or where it vanished, its velocity, how long it
  has been out of sight, its kind, type and size (with_objects), and the
  row it keeps, so it is the same token from action to action; out of
  sight and off the screen too;
- Mario's kinematics (core/smb_kinematics), added to his token (one picture
  cannot show motion);
- the last execution feedback (how the previous action went);
- the tactic and the strategy switch the action was chosen under (user,
  2026-10-18: the same action is carried out differently under each);
- the chosen action: its verb and the token of where it ends (Mario's for a
  wait; the list's row for an enemy or a moving platform: object_pointers);
- its own state, carried from action to action.

What it predicts (OUTCOME_FIELDS, pixels from Mario's feet at the start and
frames): where his feet end, where the object ends (a moving one), how many
frames the action takes (for a wait, how long), with an uncertainty for
each, and the chance it succeeds. Its readout is the action predictor's
(which it replaces): a query made of the action and the state attends over
the scene.
"""

import hashlib
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .action_tokens import VERBS, pointer_index
from .smb_kinematics import MARIO_STATE, SPRITE_SLOTS
from .smb_observer import SCENE_SLOTS
from .smb_tracked_objects import CAPACITY, COLUMN as OBJECT_COLUMN, FIELDS as OBJECT_FIELDS
from .tokens import EXECUTION_WIDTH, STRATEGY_WIDTH, TACTIC_WIDTH

OUTCOME_FIELDS = ("dx", "dy", "object_dx", "object_dy", "frames")
OUTCOME_WIDTH = len(OUTCOME_FIELDS)
# Each field's unit while learning (pixels, pixels, ..., frames).
OUTCOME_SCALE = (64.0, 64.0, 64.0, 64.0, 32.0)
# One action as numbers: its verb (-1: none, padding) and the scene token its
# pointer names (MARIO_TOKEN for a wait), its outcome, whether it has a
# moving object, whether Mario survived it and did what it says ("success"),
# whether the lesson was won by it, the room it kept from every enemy
# (pixels, at most 64), whether it was the teacher's choice, whether it was
# the action actually carried out (else one the teacher tried), and whether
# its frames are only a lower bound (a one-frame teacher wait not played
# out).
ACTION_RECORD = (
    "verb",
    "pointer",
    *OUTCOME_FIELDS,
    "has_object",
    "success",
    "won",
    "room",
    "chosen",
    "executed",
    "frames_censored",
)
COLUMN = {name: i for i, name in enumerate(ACTION_RECORD)}
# Actions recorded per decision (the teacher's choice first; then every
# distinct action its jump table tried).
ACTIONS_PER_DECISION = 16
# The scene token a wait points at: Mario's.
MARIO_TOKEN = 1
# The scene tokens of the sprite slots smb_kinematics orders (enemies, then
# moving platforms).
SPRITE_TOKENS = [pointer_index(("enemies", i)) for i in range(SCENE_SLOTS["enemies"])] + [
    pointer_index(("moving_platforms", i)) for i in range(SCENE_SLOTS["moving_platforms"])
]
assert len(SPRITE_TOKENS) == SPRITE_SLOTS
# What an action was chosen under: the tactic (one-hot) and the strategy
# switch (tokens.encode_tactic, encode_strategy), side by side.
CONTEXT_WIDTH = TACTIC_WIDTH + STRATEGY_WIDTH
# The kinematics added to a sprite's token: its velocity and how many
# sightings it rests on (smb_kinematics.SPRITE_STATE "vx", "vy", "seen").
SPRITE_MOTION = slice(3, 6)


def add_motion(tokens, sprites, mario, sprite_layer, mario_layer):
    """Scene tokens [B, T, width] with the executor's first-order kinematics
    (smb_kinematics: each sprite slot's [B, SPRITE_SLOTS, 6], Mario's [B, 3])
    read by ``sprite_layer`` (the velocity and sightings) and ``mario_layer``
    and added to the sprites' tokens and Mario's: what one picture cannot
    show, estimated from what vision reported over the frames before. The
    world model and the decision layers read motion this way."""
    tokens = tokens.clone()
    index = torch.as_tensor(SPRITE_TOKENS, device=tokens.device)
    motion = sprite_layer(sprites[..., SPRITE_MOTION]) * sprites[..., :1]
    tokens[:, index] = tokens[:, index] + motion
    tokens[:, MARIO_TOKEN] = tokens[:, MARIO_TOKEN] + mario_layer(mario)
    return tokens


# A list row's numbers as the world model reads them: places in this unit.
OBJECT_POSITION_SCALE = 128.0


def object_pointers(pointers, slot_keys, keys, token_count: int):
    """Each action's pointer (scene token indices, [N]) with one at an
    enemy's or a moving platform's scene token replaced by its row on the
    list of tracked things (``token_count`` + row; with_objects): the sprite
    slots' keys ``slot_keys`` [N, SPRITE_SLOTS] and the list's ``keys`` [N,
    CAPACITY] say which row. One whose thing is not on the list keeps its
    scene token."""
    pointers = np.asarray(pointers, np.int64).copy()
    for n, pointer in enumerate(pointers):
        if pointer in SPRITE_TOKENS:
            key = int(slot_keys[n][SPRITE_TOKENS.index(int(pointer))])
            rows = np.flatnonzero(np.asarray(keys[n]) == key) if key else ()
            if len(rows):
                pointers[n] = token_count + int(rows[0])
    return pointers


def action_record(action, outcome: dict, chosen: bool) -> np.ndarray:
    """[len(ACTION_RECORD)] numbers for an action the teacher tried and its
    outcome (teacher_actions.action_outcome)."""
    row = np.zeros(len(ACTION_RECORD), np.float32)
    row[COLUMN["verb"]] = VERBS.index(action.verb)
    row[COLUMN["pointer"]] = MARIO_TOKEN if action.target is None else pointer_index(action.target)
    for name in ("dx", "dy", "frames"):
        row[COLUMN[name]] = outcome[name]
    if "object_dx" in outcome:
        row[COLUMN["object_dx"]] = outcome["object_dx"]
        row[COLUMN["object_dy"]] = outcome["object_dy"]
        row[COLUMN["has_object"]] = 1.0
    row[COLUMN["success"]] = float(outcome["success"])
    row[COLUMN["won"]] = float(outcome["won"])
    row[COLUMN["room"]] = outcome["room"]
    row[COLUMN["chosen"]] = float(chosen)
    # A one-frame wait of the teacher's is one it did not play out: its
    # length is at least that.
    row[COLUMN["frames_censored"]] = float(action.verb == "wait" and outcome["frames"] <= 1)
    return row


def empty_records() -> np.ndarray:
    """[ACTIONS_PER_DECISION, len(ACTION_RECORD)] padding: no actions."""
    rows = np.zeros((ACTIONS_PER_DECISION, len(ACTION_RECORD)), np.float32)
    rows[:, COLUMN["verb"]] = -1
    return rows


def scene_digest(policy) -> str:
    """A fingerprint of what the world model reads from a policy (or its
    state dict): its scene encoder's weights. A world model is trained on one
    encoder; it must stay as it was wherever the world model is used."""
    digest = hashlib.sha256()
    state = policy if isinstance(policy, dict) else policy.state_dict()
    for name, value in sorted(state.items()):
        if name.startswith("scene."):
            digest.update(name.encode())
            digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


class WorldModel(nn.Module):
    """The LSTM and its outcome readout (module notes)."""

    def __init__(self, width: int = 96, memory_width: int = 128, heads: int = 4):
        super().__init__()
        self.width = memory_width
        self.verb = nn.Embedding(len(VERBS), width)
        self.mario_motion = nn.Linear(len(MARIO_STATE), width)
        self.feedback = nn.Linear(EXECUTION_WIDTH, width)
        self.context = nn.Linear(CONTEXT_WIDTH, width)
        for layer in (self.mario_motion, self.feedback, self.context):
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)
        # The list of tracked things: each row's numbers, and the row itself
        # (the same thing keeps its row while it is on the list).
        self.objects = nn.Linear(len(OBJECT_FIELDS), width)
        self.object_row = nn.Parameter(torch.zeros(CAPACITY, width))
        self.cell = nn.LSTMCell(width, memory_width)
        self.state = nn.Linear(memory_width, width)
        self.query_norm = nn.LayerNorm(width)
        self.key_norm = nn.LayerNorm(width)
        self.attention = nn.MultiheadAttention(width, heads, batch_first=True)
        self.mlp = nn.Sequential(
            nn.LayerNorm(width), nn.Linear(width, 2 * width), nn.GELU(), nn.Linear(2 * width, width)
        )
        self.out = nn.Linear(width, 2 * OUTCOME_WIDTH + 1)
        self.register_buffer("scale", torch.tensor(OUTCOME_SCALE), persistent=False)
        # Untrained: no movement, half a unit of uncertainty (32 pixels, 16
        # frames), 32 frames long, an even chance of success.
        nn.init.zeros_(self.out.weight)
        with torch.no_grad():
            self.out.bias.zero_()
            self.out.bias[OUTCOME_WIDTH - 1] = math.log(math.expm1(31 / 32))
            self.out.bias[OUTCOME_WIDTH : 2 * OUTCOME_WIDTH] = math.log(0.5)

    def initial(self, batch: int, device) -> tuple:
        zeros = torch.zeros(batch, self.width, device=device)
        return zeros, zeros.clone()

    def with_objects(self, tokens, present, objects, mario):
        """The tokens the world model reads: the scene's ``tokens`` [B, T,
        width] (``present`` [B, T]) but for vision's enemy and moving
        platform tokens, then one token per row of the list of tracked
        things (``objects`` [B, CAPACITY, len(OBJECT_FIELDS)]:
        TrackedObjects.read); Mario's kinematics ([B, 3]) added to his token.
        Returns (tokens [B, T + CAPACITY, width], present)."""
        tokens = tokens.clone()
        tokens[:, MARIO_TOKEN] = tokens[:, MARIO_TOKEN] + self.mario_motion(mario)
        present = present.clone()
        present[:, SPRITE_TOKENS] = False
        numbers = objects.clone()
        numbers[..., OBJECT_COLUMN["x"]] /= OBJECT_POSITION_SCALE
        numbers[..., OBJECT_COLUMN["y"]] /= OBJECT_POSITION_SCALE
        listed = numbers[..., OBJECT_COLUMN["present"]] > 0.5
        rows = (self.objects(numbers) + self.object_row) * listed[..., None]
        return torch.cat((tokens, rows), 1), torch.cat((present, listed), 1)

    def _query(self, tokens, verbs, pointers, rows, context):
        query = self.verb(verbs) + tokens[rows, pointers]
        return query if context is None else query + self.context(context)

    def advance(self, tokens, feedback, verbs, pointers, state, rows=None, context=None):
        """One LSTM step for N actions from ``state`` ([N, memory width] each
        of hidden and cell): the scene's summary token, the action's verb and
        object token, what it was chosen under (``context`` [N,
        CONTEXT_WIDTH]: the tactic and the strategy switch), the feedback.
        ``rows`` [N]: each action's scene (default: one each). Returns the
        state after the step."""
        rows = torch.arange(len(verbs), device=verbs.device) if rows is None else rows
        query = self._query(tokens, verbs, pointers, rows, context)
        step = tokens[rows, 0] + query + self.feedback(feedback[rows])
        return self.cell(step, state)

    def readout(self, tokens, present, hidden, verbs, pointers, rows=None, context=None) -> dict:
        """Each action's outcome from its LSTM ``hidden`` state: {"mean",
        "sigma"} [N, OUTCOME_WIDTH] in pixels and frames, and "success" [N]
        logits."""
        rows = torch.arange(len(verbs), device=verbs.device) if rows is None else rows
        query = (self._query(tokens, verbs, pointers, rows, context) + self.state(hidden)).unsqueeze(1)
        keys = self.key_norm(tokens)[rows]
        attended, _ = self.attention(
            self.query_norm(query), keys, keys, key_padding_mask=~present[rows]
        )
        hidden = query + attended
        hidden = (hidden + self.mlp(hidden))[:, 0]
        raw = self.out(hidden)
        mean = raw[:, :OUTCOME_WIDTH] * self.scale
        frames = 1 + F.softplus(raw[:, OUTCOME_WIDTH - 1]) * self.scale[-1]
        mean = torch.cat((mean[:, :-1], frames[:, None]), dim=-1)
        sigma = raw[:, OUTCOME_WIDTH : 2 * OUTCOME_WIDTH].clamp(-6, 2).exp() * self.scale
        return {"mean": mean, "sigma": sigma, "success": raw[:, -1]}

    def predict(self, tokens, present, objects, mario, feedback, verbs, pointers, state, context=None):
        """At play: the outcome of each copy's chosen action and its state
        after the step (one step per action); ``pointers`` index the tokens
        with_objects makes (object_pointers)."""
        tokens, present = self.with_objects(tokens, present, objects, mario)
        state = self.advance(tokens, feedback, verbs, pointers, state, context=context)
        return self.readout(tokens, present, state[0], verbs, pointers, context=context), state


def outcome_loss(predicted, records):
    """The loss on actions ``records`` [N, len(ACTION_RECORD)]: whether each
    was survived and did what it says (all), and how the survived ones ended
    (an object's place only where the action has a moving one): the squared
    error of each field in its unit, and its uncertainty as a normal
    likelihood around the predicted value held fixed (fitting the
    uncertainty alone would let a wide one excuse a poor value). Frames known
    only as a lower bound count only when the prediction is below it.
    Returns (loss, stats with mean errors in pixels and frames)."""
    success = records[:, COLUMN["success"]] > 0.5
    loss = F.binary_cross_entropy_with_logits(predicted["success"], success.float())
    stats = {
        "actions": int(len(records)),
        "success_share": float(success.float().mean()),
        "success_accuracy": float(((predicted["success"] > 0) == success).float().mean()),
    }
    if success.any():
        rows = records[success]
        target = rows[:, [COLUMN[name] for name in OUTCOME_FIELDS]]
        mean, sigma = predicted["mean"][success], predicted["sigma"][success]
        scale = mean.new_tensor(OUTCOME_SCALE)
        fit = ((mean - target) / scale).square()
        spread = 0.5 * ((mean.detach() - target) / sigma).square() + (sigma / scale).log()
        each = fit + spread
        weight = torch.ones_like(each)
        weight[:, 2:4] = rows[:, COLUMN["has_object"], None]
        bound = rows[:, COLUMN["frames_censored"]] > 0.5
        # A lower bound: only a prediction below it is wrong.
        weight[:, 4] = torch.where(bound, (mean[:, 4] < target[:, 4]).float(), weight[:, 4])
        loss = loss + (each * weight).sum() / weight.sum().clamp_min(1)
        error = (mean - target).detach().abs()
        stats["end_error_pixels"] = float(error[:, :2].mean())
        exact = ~bound
        if exact.any():
            stats["frames_error"] = float(error[exact][:, 4].mean())
        has = weight[:, 2] > 0
        if has.any():
            stats["object_error_pixels"] = float(error[has][:, 2:4].mean())
    return loss, stats
