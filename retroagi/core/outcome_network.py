"""The outcome network (docs/choose-and-control-plan.md phase 4): from the
state, one pass predicts the outcome of every candidate choice at once.

It reads the state as tokens (smb_choices.state_tokens): Mario, the
surfaces and the enemies on screen, x relative to Mario. A small
transformer lets every token see the others; then a head on each surface's
token gives its three landings (near, middle, far), a head on each enemy's
token its stomp and its two passes (beyond, back), and a head on Mario's
token the choices about no thing (smb_choices.GENERAL: the three runs, the
two waits). Each choice's outcome is PREDICTIONS: the chance Mario dies
during it or soon after (outcome_sampling's window), the chance the
executor finds it infeasible, the chance it reaches the flag, the chance it
leaves Mario stuck (no forward choice from there gets him on), and the
progress right (pixels), frames and points it brings.
"""

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from .smb_choices import ENEMY_SLOTS, ENEMY_WIDTH, GENERAL, MARIO_WIDTH, SURFACE_SLOTS, SURFACE_WIDTH

PREDICTIONS = ("death", "infeasible", "finished", "stuck", "progress", "frames", "points")
# The logits (chances) and the amounts, and the scale the amounts are
# learned at (pixels, frames, points).
CHANCES, AMOUNTS = (0, 1, 2, 3), (4, 5, 6)
SCALES = (64.0, 64.0, 5.0)
# Inputs are scaled to about one: x by the screen's width, y by its height,
# speeds by 4 pixels a frame.
X_SCALE, Y_SCALE, SPEED_SCALE = 256.0, 240.0, 4.0


@dataclass
class OutcomeConfig:
    width: int = 96
    layers: int = 3
    heads: int = 4


def _scaled(mario, surfaces, enemies):
    """The tokens' numbers scaled to about one (tensors [B, ...])."""
    m = mario.clone()
    m[..., [0, 2]] /= X_SCALE
    m[..., [1, 3]] /= Y_SCALE
    m[..., [4, 5]] /= SPEED_SCALE
    s = surfaces.clone()
    s[..., [0, 2]] /= X_SCALE
    s[..., [1, 3]] /= Y_SCALE
    e = enemies.clone()
    e[..., [0, 2]] /= X_SCALE
    e[..., [1, 3]] /= Y_SCALE
    e[..., [4, 5]] /= SPEED_SCALE
    return m, s, e


class OutcomeNetwork(nn.Module):
    def __init__(self, config: OutcomeConfig = OutcomeConfig()):
        super().__init__()
        self.config = config
        w = config.width
        self.mario_in = nn.Linear(MARIO_WIDTH, w)
        self.surface_in = nn.Linear(SURFACE_WIDTH, w)
        self.enemy_in = nn.Linear(ENEMY_WIDTH, w)
        self.kind = nn.Embedding(3, w)  # Mario, surface, enemy
        layer = nn.TransformerEncoderLayer(w, config.heads, 4 * w, dropout=0.0, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, config.layers)
        n = len(PREDICTIONS)
        self.land = nn.Linear(w, 3 * n)
        self.enemy = nn.Linear(w, 3 * n)
        self.general = nn.Linear(w, len(GENERAL) * n)

    def forward(self, mario, surfaces, enemies):
        """Tokens [B, ...] -> (land [B, SURFACE_SLOTS, 3, P], enemy
        [B, ENEMY_SLOTS, 3, P], general [B, len(GENERAL), P]), P =
        PREDICTIONS (chances as logits, amounts at SCALES)."""
        m, s, e = _scaled(mario, surfaces, enemies)
        b = m.shape[0]
        tokens = torch.cat([
            self.mario_in(m)[:, None] + self.kind.weight[0],
            self.surface_in(s) + self.kind.weight[1],
            self.enemy_in(e) + self.kind.weight[2],
        ], dim=1)
        absent = torch.cat([
            torch.zeros(b, 1, dtype=torch.bool, device=m.device),
            surfaces[..., 4:].sum(-1) == 0,
            enemies.abs().sum(-1) == 0,
        ], dim=1)
        h = self.encoder(tokens, src_key_padding_mask=absent)
        n = len(PREDICTIONS)
        land = self.land(h[:, 1:1 + SURFACE_SLOTS]).view(b, SURFACE_SLOTS, 3, n)
        enemy = self.enemy(h[:, 1 + SURFACE_SLOTS:]).view(b, ENEMY_SLOTS, 3, n)
        general = self.general(h[:, 0]).view(b, len(GENERAL), n)
        return land, enemy, general


def pick(outputs, kind, slot, modifier):
    """The predictions [N, P] of the choices (kind, slot, modifier) [N],
    one per row of ``outputs`` (forward's, for those rows' states)."""
    land, enemy, general = outputs
    rows = torch.arange(kind.shape[0], device=kind.device)
    general_index = torch.full_like(kind, 0)
    for g, (k, m) in enumerate(GENERAL):
        general_index = torch.where((kind == k) & (modifier == m), torch.full_like(kind, g), general_index)
    slot0 = slot.clamp(min=0)
    by_kind = torch.where(
        (kind == 0)[:, None], land[rows, slot0.clamp(max=SURFACE_SLOTS - 1), modifier.clamp(max=2)],
        torch.where(
            (kind <= 2)[:, None],
            enemy[rows, slot0.clamp(max=ENEMY_SLOTS - 1), torch.where(kind == 1, 0, 1 + modifier).clamp(max=2)],
            general[rows, general_index],
        ),
    )
    return by_kind


def predict(model: OutcomeNetwork, tokens, choices, device) -> np.ndarray:
    """For one state's tokens (state_tokens' first three) and its choices
    [(kind, slot, modifier)]: each choice's PREDICTIONS as chances and
    amounts in their own units."""
    mario, surfaces, enemies = (torch.as_tensor(t, device=device)[None] for t in tokens)
    with torch.no_grad():
        outputs = model(mario, surfaces, enemies)
        c = torch.as_tensor(np.array(choices, np.int64).reshape(-1, 3), device=device)
        outputs = tuple(o.expand(len(c), *o.shape[1:]) for o in outputs)
        p = pick(outputs, c[:, 0], c[:, 1], c[:, 2])
        out = p.clone()
        out[:, list(CHANCES)] = torch.sigmoid(p[:, list(CHANCES)])
        out[:, list(AMOUNTS)] = p[:, list(AMOUNTS)] * torch.tensor(SCALES, device=device)
    return out.cpu().numpy()
