"""The action predictor: told an action (core/action_tokens), it predicts how
the action ends when it succeeds, and the chance that it does.

It reads only what the skill reads (vision): the scene encoder's tokens at
the decision, the scene memory's state, and the memory's forecast of where
each enemy will be when the next action ends. Its query is the action: the
verb, the chosen object's scene token (and that enemy's forecast), and the
memory; it attends over the whole scene, so other threats count too.

Outputs, in pixels from Mario's feet at the decision (x right, y down):

- where his feet end ("dx", "dy");
- where the action's object is then ("object_dx", "object_dy": an enemy's
  middle and top, a surface's left edge and top);
- how many frames the action takes ("frames");
- an uncertainty for each, and the chance that Mario survives it.

It is trained on the teacher's trials (stages/block_smb/teacher_actions): for
every action the teacher tried, its best safe version
(docs/action-predictor-controller.md, section 2).
"""

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .action_tokens import VERBS, pointer_index
from .layered_policy import ENEMY_TOKENS, FORECAST_WIDTH, PolicySettings
from .smb_observer import SCENE_SLOTS

OUTCOME_FIELDS = ("dx", "dy", "object_dx", "object_dy", "frames")
OUTCOME_WIDTH = len(OUTCOME_FIELDS)
# Each field's unit while learning (pixels, pixels, ..., frames).
OUTCOME_SCALE = (64.0, 64.0, 64.0, 64.0, 32.0)
# One tried action as numbers: its verb (-1: none, padding) and the scene
# token its pointer names (MARIO_TOKEN for hold), its outcome, whether it has
# an object, whether Mario survived and won, the room it kept from every
# threat (pixels, at most 64), and whether it was the teacher's choice.
ACTION_RECORD = ("verb", "pointer", *OUTCOME_FIELDS, "has_object", "safe", "won", "room", "chosen")
COLUMN = {name: i for i, name in enumerate(ACTION_RECORD)}
# Actions recorded per decision (the teacher's choice first; then every
# distinct action its jump table tried).
ACTIONS_PER_DECISION = 16
# The scene token a hold points at: Mario's.
MARIO_TOKEN = 1


def action_record(action, outcome: dict, chosen: bool) -> np.ndarray:
    """[len(ACTION_RECORD)] numbers for an action and its outcome
    (teacher_actions.action_outcome)."""
    row = np.zeros(len(ACTION_RECORD), np.float32)
    row[COLUMN["verb"]] = VERBS.index(action.verb)
    row[COLUMN["pointer"]] = MARIO_TOKEN if action.target is None else pointer_index(action.target)
    for name in ("dx", "dy", "frames"):
        row[COLUMN[name]] = outcome[name]
    if "object_dx" in outcome:
        row[COLUMN["object_dx"]] = outcome["object_dx"]
        row[COLUMN["object_dy"]] = outcome["object_dy"]
        row[COLUMN["has_object"]] = 1.0
    row[COLUMN["safe"]] = float(outcome["safe"])
    row[COLUMN["won"]] = float(outcome["won"])
    row[COLUMN["room"]] = outcome["room"]
    row[COLUMN["chosen"]] = float(chosen)
    return row


def empty_records() -> np.ndarray:
    """[ACTIONS_PER_DECISION, len(ACTION_RECORD)] padding: no actions."""
    rows = np.zeros((ACTIONS_PER_DECISION, len(ACTION_RECORD)), np.float32)
    rows[:, COLUMN["verb"]] = -1
    return rows


class ActionPredictor(nn.Module):
    def __init__(self, settings: PolicySettings = PolicySettings()):
        super().__init__()
        width = settings.width
        self.verb = nn.Embedding(len(VERBS), width)
        self.memory = nn.Linear(settings.memory_width, width)
        self.forecast = nn.Linear(FORECAST_WIDTH, width)
        self.query_norm = nn.LayerNorm(width)
        self.key_norm = nn.LayerNorm(width)
        self.attention = nn.MultiheadAttention(width, settings.heads, batch_first=True)
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

    def forward(self, scene_tokens, present, memory, enemy_forecast, rows, verbs, pointers):
        """Predict the outcome of each of N actions.

        ``scene_tokens`` [B, T, width] and ``present`` [B, T]: the encoded
        scenes at B decisions; ``memory`` [B, memory width]: the scene memory
        there; ``enemy_forecast`` [B, enemy slots, FORECAST_WIDTH]
        (layered_policy.forecast_features) or None; ``rows`` [N]: each
        action's decision; ``verbs`` [N]; ``pointers`` [N]: scene token
        indices. Returns {"mean", "sigma"} [N, OUTCOME_WIDTH] in pixels and
        frames, and "success" [N] logits.
        """
        query = self.verb(verbs) + scene_tokens[rows, pointers] + self.memory(memory[rows])
        if enemy_forecast is not None:
            enemy = (pointers >= ENEMY_TOKENS.start) & (pointers < ENEMY_TOKENS.stop)
            slot = (pointers - ENEMY_TOKENS.start).clamp(0, SCENE_SLOTS["enemies"] - 1)
            query = query + self.forecast(enemy_forecast[rows, slot]) * enemy[:, None]
        query = query.unsqueeze(1)
        keys = self.key_norm(scene_tokens)[rows]
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


def outcome_loss(predicted, records):
    """The predictor's loss on actions ``records`` [N, len(ACTION_RECORD)]:
    whether each was survived (all), and how the survived ones ended (an
    object's place only where the action has one): the squared error of each
    field in its unit, and its uncertainty as a normal likelihood around the
    predicted value held fixed (fitting the uncertainty alone would let a
    wide one excuse a poor value: the first predictor under-fitted its own
    training data by 20 pixels). Returns (loss, stats with mean errors in
    pixels and frames)."""
    safe = records[:, COLUMN["safe"]] > 0.5
    loss = F.binary_cross_entropy_with_logits(predicted["success"], safe.float())
    stats = {
        "actions": int(len(records)),
        "safe_share": float(safe.float().mean()),
        "success_accuracy": float(((predicted["success"] > 0) == safe).float().mean()),
    }
    if safe.any():
        target = records[safe][:, [COLUMN[name] for name in OUTCOME_FIELDS]]
        mean, sigma = predicted["mean"][safe], predicted["sigma"][safe]
        scale = mean.new_tensor(OUTCOME_SCALE)
        fit = ((mean - target) / scale).square()
        spread = 0.5 * ((mean.detach() - target) / sigma).square() + (sigma / scale).log()
        each = fit + spread
        weight = torch.ones_like(each)
        weight[:, 2:4] = records[safe][:, COLUMN["has_object"], None]
        loss = loss + (each * weight).sum() / weight.sum().clamp_min(1)
        error = (mean - target).detach().abs()
        stats["end_error_pixels"] = float(error[:, :2].mean())
        stats["frames_error"] = float(error[:, 4].mean())
        has = weight[:, 2] > 0
        if has.any():
            stats["object_error_pixels"] = float(error[has][:, 2:4].mean())
    return loss, stats
