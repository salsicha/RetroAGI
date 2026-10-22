"""Each executed action's measured outcome, from a recorded episode's truth:
what the world model learns for the actions actually carried out (user,
2026-10-16: executed actions and the teacher's tried ones).

Per decision k (from its frame f to the next decision's, or the episode's
end), in EXECUTED fields:

- verb, pointer: the action carried out (EpisodeRecord.action_verb /
  action_pointer; -1 unknown);
- dx, dy: where the middle of Mario's feet ended, from where it was at f;
- object_dx, object_dy: where the action's object ended (a moving one: an
  enemy's middle and top, at its death if it died during the action; a
  moving platform's left edge and top), from Mario's feet at f;
  has_object: 1 for those;
- frames: how long the action lasted;
- success: Mario lived through it and it did what it says (one ending on an
  enemy killed it; any other did not stall);
- won: the episode was won as it ended;
- room: the closest Mario came to a live enemy during it (box gap, pixels,
  at most 64).
"""

import numpy as np

from .action_tokens import executed_pointer, pointer_target
from .layered_policy import POINTER_START
from .smb_observer import SCENE_SLOTS

EXECUTED = (
    "verb",
    "pointer",
    "dx",
    "dy",
    "object_dx",
    "object_dy",
    "frames",
    "has_object",
    "success",
    "won",
    "room",
)
COLUMN = {name: i for i, name in enumerate(EXECUTED)}
ROOM = 64.0
ALIVE, DEAD = 1, 2


def _feet(box):
    return np.array(((box[0] + box[2]) / 2, box[3]), np.float32)


def _gap(a, b) -> float:
    return float(max(b[0] - a[2], a[0] - b[2], b[1] - a[3], a[1] - b[3], 0))


def executed_outcomes(record, no_progress_code: int) -> np.ndarray:
    """[D, len(EXECUTED)] float32 for a record with truth (record_truth);
    ``no_progress_code``: the "no_progress" end reason's code."""
    truth = record.truth
    mario = truth["mario"].astype(np.float32)
    enemies = truth["enemies"].astype(np.float32)
    state = truth["enemy_state"]
    platforms = truth["platforms"].astype(np.float32)
    last = len(mario) - 1
    starts = [int(f) for f in record.decision_frames]
    out = np.zeros((len(starts), len(EXECUTED)), np.float32)
    died = record.end == "death"
    for k, (f, e) in enumerate(zip(starts, starts[1:] + [last])):
        verb = int(record.action_verb[k])
        pointer = executed_pointer(verb, int(record.action_pointer[k]))
        row = out[k]
        row[COLUMN["verb"]], row[COLUMN["pointer"]] = verb, pointer
        origin = _feet(mario[f])
        row[COLUMN["dx"] : COLUMN["dy"] + 1] = _feet(mario[e]) - origin
        row[COLUMN["frames"]] = e - f
        target, kind = None, None
        if pointer >= 0:
            name, slot = pointer_target(pointer + POINTER_START)
            if name == "enemies":
                target = int(record.slot_match[f, slot])
                kind = "enemy"
            elif name == "moving_platforms":
                target = int(record.slot_match[f, SCENE_SLOTS["enemies"] + slot])
                kind = "platform"
        killed = False
        if target is not None and target >= 0:
            if kind == "enemy":
                deaths = np.flatnonzero(state[f + 1 : e + 1, target] == DEAD)
                killed = len(deaths) > 0 and state[f, target] == ALIVE
                at = f + 1 + int(deaths[0]) if killed else e
                box = enemies[at, target]
                point = np.array(((box[0] + box[2]) / 2, box[1]), np.float32)
            else:
                box = platforms[e, target]
                point = np.array((box[0], box[1]), np.float32)
            row[COLUMN["object_dx"] : COLUMN["object_dy"] + 1] = point - origin
            row[COLUMN["has_object"]] = 1.0
        lived = not (died and e == last)
        if kind == "enemy":
            did = killed
        else:
            did = e == last or int(record.ended[e]) != no_progress_code
        row[COLUMN["success"]] = float(lived and did)
        row[COLUMN["won"]] = float(record.won and e == last)
        room = ROOM
        for t in range(f, e + 1):
            for i in np.flatnonzero(state[t] == ALIVE):
                room = min(room, _gap(mario[t], enemies[t, i]))
        row[COLUMN["room"]] = room
    return out
