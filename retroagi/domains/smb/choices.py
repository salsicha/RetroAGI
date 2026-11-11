"""The choices the chooser picks from and the state as the outcome network
reads it (docs/choose-and-control-plan.md phases 3 and 4), the same in
Block SMB and the real game.

A choice is a verb, the thing it is about and a modifier, carried out by the
executor (smb_controller.Controller):
- "land" on a surface on screen, near, in the middle or far along it;
- "stomp" an enemy on screen, or "pass" it, beyond it (modifier 0) or back
  on Mario's side (1);
- "run" RUN_SHORT or RUN_LONG pixels forward keeping speed, or RUN_BACK back;
- "wait" WAITS frames;
- "bump" a block over Mario's reach from below (a question block gives a coin
  or a mushroom; big Mario breaks a brick);
- "collect" an item on screen (a mushroom).
candidates lists them for a state; state_tokens turns the state into the
network's tokens and says which slot each thing took.
"""

import numpy as np

from .controller import Choice, State
from .planner import ENEMY_KINDS, Goal

KINDS = ("land", "stomp", "pass", "run", "wait", "bump", "collect")
PREFERS = ("near", "middle", "far")
RUN_SHORT, RUN_LONG, RUN_BACK = 32, 96, 48
WAITS = (8, 24)
# The choices about no thing, in the order the network's general head gives
# them: (kind, modifier).
GENERAL = ((3, 0), (3, 1), (3, 2), (4, 0), (4, 1))
# How far ahead and behind Mario a surface is a landing candidate (pixels),
# and at most how many of each kind of thing are candidates (the nearest).
AHEAD, BEHIND = 176, 64
MAX_SURFACES, MAX_ENEMIES, MAX_BUMPS = 8, 4, 4
# A block is bumped from below when its bottom is at most this far above
# Mario's feet (pixels: a big jump).
BUMP_REACH = 96
# Token sizes: at most this many surfaces, enemies and items per state.
SURFACE_SLOTS, ENEMY_SLOTS, ITEM_SLOTS = 24, 8, 2
SURFACE_KINDS = ("ground", "pipe", "block", "question", "moving", "edge")
MARIO_WIDTH, SURFACE_WIDTH, ENEMY_WIDTH, ITEM_WIDTH = 7, 4 + len(SURFACE_KINDS), 7 + len(ENEMY_KINDS), 6


def state_tokens(state: State) -> tuple:
    """(mario [MARIO_WIDTH], surfaces [SURFACE_SLOTS, SURFACE_WIDTH], enemies
    [ENEMY_SLOTS, ENEMY_WIDTH], items [ITEM_SLOTS, ITEM_WIDTH], slot of each
    thing's ref): x relative to Mario's left side, y as on screen (his
    height says whether he is big); Mario's speeds (pixels a frame) and
    whether he stands; each surface's kind; each enemy's velocity, whether
    it can be stomped and its kind (ENEMY_KINDS); each item's velocity. The nearest things
    take the slots; unused slots are zeros."""
    body = state.body
    x = body[0]
    mario = np.array([0.0, body[1], body[2] - x, body[3], state.motion.x_speed / 16,
                      state.motion.y_speed + state.motion.y_force / 256, float(state.grounded)], np.float32)
    surfaces = np.zeros((SURFACE_SLOTS, SURFACE_WIDTH), np.float32)
    enemies = np.zeros((ENEMY_SLOTS, ENEMY_WIDTH), np.float32)
    slots = {}
    near = sorted(state.solids, key=lambda s: max(s.x0 - body[2], body[0] - s.x1, 0.0))
    for k, s in enumerate(near[:SURFACE_SLOTS]):
        surfaces[k, :4] = (s.x0 - x, s.top, s.x1 - x, s.bottom)
        surfaces[k, 4 + SURFACE_KINDS.index(s.kind)] = 1.0
        slots[s.ref] = k
    near = sorted(state.enemies, key=lambda e: abs(e.box[0] - x))
    for k, e in enumerate(near[:ENEMY_SLOTS]):
        enemies[k, :7] = (e.box[0] - x, e.box[1], e.box[2] - x, e.box[3], e.vx, e.vy, float(e.stompable))
        enemies[k, 7 + ENEMY_KINDS.index(e.kind)] = 1.0
        slots[e.ref] = k
    items = np.zeros((ITEM_SLOTS, ITEM_WIDTH), np.float32)
    for k, i in enumerate(sorted(state.items, key=lambda i: abs(i.box[0] - x))[:ITEM_SLOTS]):
        items[k] = (i.box[0] - x, i.box[1], i.box[2] - x, i.box[3], i.vx, i.vy)
        slots[i.ref] = k
    return mario, surfaces, enemies, items, slots


def candidates(state: State) -> list:
    """Every candidate choice from ``state``: (Choice, kind, the ref of its
    thing or None, modifier)."""
    body = state.body
    mx = (body[0] + body[2]) / 2
    out = []
    reach = [s for s in state.solids if s.kind in ("ground", "pipe", "block", "question", "moving")
             and s.x1 > mx - BEHIND and s.x0 < mx + AHEAD]
    reach.sort(key=lambda s: max(s.x0 - mx, mx - s.x1, 0.0))
    for s in reach[:MAX_SURFACES]:
        for m, prefer in enumerate(PREFERS):
            out.append((Choice("jump", Goal("land", s.ref, s.x0, s.x1, prefer=prefer, top=s.top)), 0, s.ref, m))
    for e in sorted(state.enemies, key=lambda e: abs((e.box[0] + e.box[2]) / 2 - mx))[:MAX_ENEMIES]:
        beyond = 1 if (e.box[0] + e.box[2]) / 2 >= mx else -1
        out.append((Choice("jump", Goal("stomp", e.ref)), 1, e.ref, 0))
        out.append((Choice("jump", Goal("pass", e.ref, side=beyond)), 2, e.ref, 0))
        out.append((Choice("jump", Goal("pass", e.ref, side=-beyond)), 2, e.ref, 1))
    out.append((Choice("run", x=mx + RUN_SHORT, keep_speed=True), 3, None, 0))
    out.append((Choice("run", x=mx + RUN_LONG, keep_speed=True), 3, None, 1))
    out.append((Choice("run", x=mx - RUN_BACK), 3, None, 2))
    for m, frames in enumerate(WAITS):
        out.append((Choice("wait", frames=frames), 4, None, m))
    above = [s for s in state.solids if s.kind in ("block", "question") and s.bottom <= body[1]
             and s.bottom >= body[3] - BUMP_REACH and s.x1 > mx - BEHIND and s.x0 < mx + AHEAD]
    above.sort(key=lambda s: abs((s.x0 + s.x1) / 2 - mx))
    for s in above[:MAX_BUMPS]:
        out.append((Choice("jump", Goal("bump", s.ref, s.x0, s.x1, top=s.bottom)), 5, s.ref, 0))
    for i in sorted(state.items, key=lambda i: abs((i.box[0] + i.box[2]) / 2 - mx))[:ITEM_SLOTS]:
        out.append((Choice("jump", Goal("collect", i.ref)), 6, i.ref, 0))
    return out
