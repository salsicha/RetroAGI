"""The choices the chooser picks from and the state as the outcome network
reads it (docs/choose-and-control-plan.md phases 3 and 4), the same in
Block SMB and the real game: Super Mario Bros' choice space (framework/
interfaces.ChoiceSpace).

A choice is a verb, the thing it is about and a modifier, carried out by the
controller (controller.Controller):
- "land" on a surface on screen, near, in the middle or far along it;
- "stomp" an enemy on screen, or "pass" it, beyond it or back on Mario's
  side;
- "run" RUN_SHORT or RUN_LONG pixels forward keeping speed, or RUN_BACK back;
- "wait" WAITS frames;
- "bump" a block over Mario's reach from below (a question block gives a coin
  or a mushroom; big Mario breaks a brick);
- "collect" an item on screen (a mushroom).
Every jump and run comes in each of the controller's ways of carrying it
out (controller.CARE: bold, usual, careful), so that how carefully is
chosen, by the outcome network's predictions, as the choice itself is
(user, 2026-11-12). A modifier is variant * len(CARE) + care, the variant
being the placement, the side or the run (0 where a choice has none).
SPACE.candidates lists them for a state; SPACE.tokens turns the state into
the network's tokens and says which slot each thing took; the rest of
SPACE tells the chooser what it needs to remember what it tried where.
"""

import numpy as np

from retroagi.framework.interfaces import Candidate
from retroagi.framework.outcome_network import ChoiceSpec, Kind, TokenGroup

from .controller import CARE, Choice, State
from .planner import ENEMY_KINDS, Goal

KINDS = ("land", "stomp", "pass", "run", "wait", "bump", "collect")
PREFERS = ("near", "middle", "far")
RUN_SHORT, RUN_LONG, RUN_BACK = 32, 96, 48
WAITS = (8, 24)
# The ways of carrying a choice out (controller.CARE), each a variant.
CARES = len(CARE)
# The choices about no thing, in the order the network's general head gives
# them: (kind, modifier): the three runs in each care, then the two waits.
GENERAL = tuple((3, m) for m in range(3 * CARES)) + ((4, 0), (4, 1))
# How far ahead and behind Mario a surface is a landing candidate (pixels),
# and at most how many of each kind of thing are candidates (the nearest).
AHEAD, BEHIND = 176, 64
MAX_SURFACES, MAX_ENEMIES, MAX_BUMPS = 8, 4, 4
# A block is bumped from below when its bottom is at most this far above
# Mario's feet (pixels: a big jump).
BUMP_REACH = 96
# Token sizes: at most this many surfaces, enemies and items per state.
SURFACE_SLOTS, ENEMY_SLOTS, ITEM_SLOTS = 24, 8, 2
SURFACE_KINDS = ("ground", "pipe", "block", "question", "moving", "edge", "spring")
MARIO_WIDTH, SURFACE_WIDTH, ENEMY_WIDTH, ITEM_WIDTH = 7, 4 + len(SURFACE_KINDS), 7 + len(ENEMY_KINDS), 6
# The network's inputs are scaled to about one: x by the screen's width, y
# by its height, speeds by 4 pixels a frame.
X, Y, SPEED = 256.0, 240.0, 4.0
# The token groups (Mario's own token gives the choices about no thing:
# GENERAL) and the kinds of choice, in KINDS' order: each kind's variants on
# its group's heads (a surface's: the three landings, then the bump; an
# enemy's: the stomp, then the two passes).
SPEC = ChoiceSpec(
    groups=(
        TokenGroup("mario", 0, MARIO_WIDTH, len(GENERAL), (X, Y, X, Y, SPEED, SPEED, 1.0)),
        TokenGroup("surfaces", SURFACE_SLOTS, SURFACE_WIDTH, 4 * CARES, (X, Y, X, Y) + (1.0,) * len(SURFACE_KINDS),
                   present=(4, SURFACE_WIDTH)),
        TokenGroup("enemies", ENEMY_SLOTS, ENEMY_WIDTH, 3 * CARES,
                   (X, Y, X, Y, SPEED, SPEED, 1.0) + (1.0,) * len(ENEMY_KINDS), present=(0, ENEMY_WIDTH)),
        TokenGroup("items", ITEM_SLOTS, ITEM_WIDTH, CARES, (X, Y, X, Y, SPEED, SPEED), present=(0, ITEM_WIDTH)),
    ),
    kinds=(
        Kind("land", 1, 0, 3 * CARES),
        Kind("stomp", 2, 0, CARES),
        Kind("pass", 2, CARES, 2 * CARES),
        Kind("run", 0, 0, 3 * CARES),
        Kind("wait", 0, 3 * CARES, 2),
        Kind("bump", 1, 3 * CARES, CARES),
        Kind("collect", 3, 0, CARES),
    ),
)
# The chooser's memory of what it tried where: a spot is SPOT pixels square;
# NEW_GROUND pixels beyond Mario's furthest so far everything is forgotten.
SPOT = 8
NEW_GROUND = 16


def tokens(state: State) -> tuple:
    """([mario [MARIO_WIDTH], surfaces [SURFACE_SLOTS, SURFACE_WIDTH], enemies
    [ENEMY_SLOTS, ENEMY_WIDTH], items [ITEM_SLOTS, ITEM_WIDTH]], slot of each
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
    return [mario, surfaces, enemies, items], slots


def candidates(state: State) -> list:
    """Every candidate choice from ``state`` (framework Candidates: the
    Choice, its kind, the ref of its thing or None, its modifier)."""
    body = state.body
    mx = (body[0] + body[2]) / 2
    out = []
    reach = [s for s in state.solids if s.kind in ("ground", "pipe", "block", "question", "moving", "spring")
             and s.x1 > mx - BEHIND and s.x0 < mx + AHEAD]
    reach.sort(key=lambda s: max(s.x0 - mx, mx - s.x1, 0.0))

    def each_care(kind, ref, variant, **choice):
        for care in range(CARES):
            out.append(Candidate(Choice(**choice, care=care), kind, ref, variant * CARES + care))

    for s in reach[:MAX_SURFACES]:
        for m, prefer in enumerate(PREFERS):
            each_care(0, s.ref, m, verb="jump", goal=Goal("land", s.ref, s.x0, s.x1, prefer=prefer, top=s.top))
    for e in sorted(state.enemies, key=lambda e: abs((e.box[0] + e.box[2]) / 2 - mx))[:MAX_ENEMIES]:
        beyond = 1 if (e.box[0] + e.box[2]) / 2 >= mx else -1
        each_care(1, e.ref, 0, verb="jump", goal=Goal("stomp", e.ref))
        each_care(2, e.ref, 0, verb="jump", goal=Goal("pass", e.ref, side=beyond))
        each_care(2, e.ref, 1, verb="jump", goal=Goal("pass", e.ref, side=-beyond))
    each_care(3, None, 0, verb="run", x=mx + RUN_SHORT, keep_speed=True)
    each_care(3, None, 1, verb="run", x=mx + RUN_LONG, keep_speed=True)
    each_care(3, None, 2, verb="run", x=mx - RUN_BACK)
    for m, frames in enumerate(WAITS):
        out.append(Candidate(Choice("wait", frames=frames), 4, None, m))
    above = [s for s in state.solids if s.kind in ("block", "question") and s.bottom <= body[1]
             and s.bottom >= body[3] - BUMP_REACH and s.x1 > mx - BEHIND and s.x0 < mx + AHEAD]
    above.sort(key=lambda s: abs((s.x0 + s.x1) / 2 - mx))
    for s in above[:MAX_BUMPS]:
        each_care(5, s.ref, 0, verb="jump", goal=Goal("bump", s.ref, s.x0, s.x1, top=s.bottom))
    for i in sorted(state.items, key=lambda i: abs((i.box[0] + i.box[2]) / 2 - mx))[:ITEM_SLOTS]:
        each_care(6, i.ref, 0, verb="jump", goal=Goal("collect", i.ref))
    return out


class Choices:
    """Super Mario Bros' choice space (framework/interfaces.ChoiceSpace)."""

    spec = SPEC
    new_ground = NEW_GROUND
    candidates = staticmethod(candidates)
    tokens = staticmethod(tokens)

    @staticmethod
    def place(state: State) -> tuple:
        """(Mario's x, the spot he stands at)."""
        x = state.body[0]
        return x, (round(x / SPOT), round(state.body[3] / SPOT))

    @staticmethod
    def signature(candidate: Candidate, state: State) -> tuple:
        """What a choice is, the same from one decision to the next: its
        kind and modifier, and for a landing where it lands (its slot moves
        as the camera scrolls)."""
        goal = candidate.choice.goal
        if goal is not None and goal.kind in ("land", "bump"):
            return candidate.kind, candidate.modifier, round(goal.x0 / SPOT)
        return candidate.kind, candidate.modifier

    @staticmethod
    def waits(candidate: Candidate) -> bool:
        return candidate.choice.verb == "wait"

    @staticmethod
    def still(candidate: Candidate) -> bool:
        """About no enemy (a landing, a run, a bump): whether the controller
        can carry it out from a spot does not change as enemies move."""
        return KINDS[candidate.kind] in ("land", "run", "bump")

    @staticmethod
    def changing(state: State) -> bool:
        return bool(state.enemies)


SPACE = Choices()
