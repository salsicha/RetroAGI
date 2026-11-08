"""The choices the chooser picks from and the state as the outcome network
reads it (docs/choose-and-control-plan.md phases 3 and 4), the same in
Block SMB and the real game.

A choice is a verb, the thing it is about and a modifier, carried out by the
executor (smb_controller.Controller):
- "land" on a surface on screen, near, in the middle or far along it;
- "stomp" an enemy on screen, or "pass" it, beyond it (modifier 0) or back
  on Mario's side (1);
- "run" RUN_SHORT or RUN_LONG pixels forward keeping speed, or RUN_BACK back;
- "wait" WAITS frames.
candidates lists them for a state; state_tokens turns the state into the
network's tokens and says which slot each thing took.
"""

import numpy as np

from .smb_controller import Choice, State
from .smb_planner import Goal

KINDS = ("land", "stomp", "pass", "run", "wait")
PREFERS = ("near", "middle", "far")
RUN_SHORT, RUN_LONG, RUN_BACK = 32, 96, 48
WAITS = (8, 24)
# The choices about no thing, in the order the network's general head gives
# them: (kind, modifier).
GENERAL = ((3, 0), (3, 1), (3, 2), (4, 0), (4, 1))
# How far ahead and behind Mario a surface is a landing candidate (pixels),
# and at most how many of each kind of thing are candidates (the nearest).
AHEAD, BEHIND = 176, 64
MAX_SURFACES, MAX_ENEMIES = 8, 4
# Token sizes: at most this many surfaces and enemies per state.
SURFACE_SLOTS, ENEMY_SLOTS = 24, 8
SURFACE_KINDS = ("ground", "pipe", "block", "moving", "edge")
MARIO_WIDTH, SURFACE_WIDTH, ENEMY_WIDTH = 7, 4 + len(SURFACE_KINDS), 7


def state_tokens(state: State) -> tuple:
    """(mario [MARIO_WIDTH], surfaces [SURFACE_SLOTS, SURFACE_WIDTH], enemies
    [ENEMY_SLOTS, ENEMY_WIDTH], slot of each thing's ref): x relative to
    Mario's left side, y as on screen; Mario's speeds (pixels a frame) and
    whether he stands; each surface's kind; each enemy's velocity and
    whether it can be stomped. The nearest things take the slots; unused
    slots are zeros."""
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
        enemies[k] = (e.box[0] - x, e.box[1], e.box[2] - x, e.box[3], e.vx, e.vy, float(e.stompable))
        slots[e.ref] = k
    return mario, surfaces, enemies, slots


def candidates(state: State) -> list:
    """Every candidate choice from ``state``: (Choice, kind, the ref of its
    thing or None, modifier)."""
    body = state.body
    mx = (body[0] + body[2]) / 2
    out = []
    reach = [s for s in state.solids if s.kind in ("ground", "pipe", "block", "moving")
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
    return out
