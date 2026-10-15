"""When and where each enemy and Mario will collide: the Block SMB simulator's
ground truth, standing in for the target trackers' predictions until they are
trained (user, 2026-10-15). Training only: nothing here reaches a policy.

For each live enemy, the first contact with Mario within HORIZON frames if he
follows a given path (his body per frame, in world pixels: standing where he
is, by default) and then stays at its end:

- how many frames until they touch;
- where: the middle of Mario's feet then (world pixels);
- how: a stomp (his feet coming down onto its top, as the simulator decides
  a stomp: smb_collision.stomp_contact) or a hit (any other touch, which
  kills him);
- how fast the enemy comes toward Mario (pixels a frame; negative: away).

The enemies move by their own rule (controller_teacher._enemy_paths): they
do not react to Mario, so their paths are the truth whatever he does.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

from retroagi.core.smb_collision import stomp_contact

HORIZON = 96


@dataclass(frozen=True)
class Collision:
    enemy: int  # the simulator's enemy index
    frames: int  # frames until contact (1: the next frame)
    x: float  # where: the middle of Mario's feet then (world pixels)
    y: float
    kind: str  # "stomp" or "hit"
    closing: float  # the enemy's speed toward Mario now (pixels a frame)


def _touch(a, b) -> bool:
    return a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]


def collisions(env, path: Optional[Sequence[tuple]] = None, horizon: int = HORIZON) -> list:
    """Every live enemy's first contact with Mario (Collision), soonest
    first; an enemy he would not touch within ``horizon`` frames is left out.
    ``path``: Mario's body (x0, y0, x1, y1) on each coming frame, the first
    being the next frame; after it he stays at its last box. None: he stands
    where he is."""
    from .controller_teacher import _body, _enemy_paths

    now = _body(env)
    path = list(path) if path else []
    paths = _enemy_paths(env, horizon)
    middle = (now[0] + now[2]) / 2
    found = []
    for index, boxes in paths.items():
        enemy = env.enemies[index]
        ahead = (boxes[0][0] + boxes[0][2]) / 2 - middle
        step = (boxes[1][0] - boxes[0][0]) if len(boxes) > 1 else 0.0
        closing = -step if ahead > 0 else step
        before = now
        for t in range(1, horizon + 1):
            body = path[t - 1] if t - 1 < len(path) else (path[-1] if path else now)
            box = boxes[min(t, len(boxes) - 1)]
            if _touch(body, box):
                vy = body[3] - before[3]
                stomp = enemy.get("stompable", True) and stomp_contact(body, box, vy)
                found.append(
                    Collision(
                        index,
                        t,
                        (body[0] + body[2]) / 2,
                        body[3],
                        "stomp" if stomp else "hit",
                        closing,
                    )
                )
                break
            before = body
    return sorted(found, key=lambda c: (c.frames, c.enemy))
