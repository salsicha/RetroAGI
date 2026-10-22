"""Ending a wait or a walk when a coming enemy needs a move (user,
2026-10-15/16): in play without the teacher, the executor ends Mario's wait
or ground move when the collision trackers say an enemy will hit him within
the lead a jump over it needs at its closing speed, plus THREAT_SLACK frames;
the skill then picks again (and learned from the teacher to jump). Its end is
reported as "threat" (the execution feedback "collision_due"). In training
the teacher's truth ends waits instead (teacher_tokens.whole_wait).

The lead is the executor's own: from its jump model (smb_trajectory.
hold_paths, open loop from where Mario stands), the smallest gap between him
and an enemy closing at that speed from which some jump still passes over it
and lands beyond it, in frames of the enemy's approach. The faster the enemy
comes, the sooner he must act.
"""

from functools import lru_cache
import numpy as np

# Frames a jump may be started late and still pass: the teacher's (controller_
# teacher.TRIGGER_SLACK, 1 + its largest tolerated delay).
THREAT_SLACK = 5
# With no jump that passes at any gap: a hit due within this many frames.
THREAT_HORIZON = 48
# An enemy the executor does not see closing in (just seen, say) but the
# trackers expect to hit: the lead at this closing speed (pixels a frame).
NOMINAL_CLOSING = 0.5
# The gaps (pixels between Mario's front and the enemy's near side) searched.
GAPS = range(0, 129)


def _passes(path, box, gap, closing, width, height, direction) -> bool:
    """Whether a jump ``path`` (Mario's box each frame from ``box``, his box
    at the start, standing) clears an enemy ``width`` x ``height`` standing on
    his floor ``gap`` pixels ahead of his front and coming at ``closing``
    pixels a frame, and lands beyond it."""
    feet = box[3]
    for t, (x0, _, x1, y1) in enumerate(path, start=1):
        if direction > 0:
            e0 = box[2] + gap - closing * t
            e1 = e0 + width
        else:
            e1 = box[0] - gap + closing * t
            e0 = e1 - width
        if x0 < e1 and e0 < x1 and y1 > feet - height:
            return False
        if t > 4 and y1 >= feet:  # back down at his feet's height
            return x0 >= e1 if direction > 0 else x1 <= e0
    return False


@lru_cache(maxsize=4096)
def _lead(closing, width, height, mario_width, mario_height, speed, direction):
    from .smb_physics import NESPlayerMotion
    from .smb_scene_labels import MarioView, SceneObservation
    from .smb_trajectory import hold_paths

    box = (0.0, 0.0, float(mario_width), float(mario_height))
    scene = SceneObservation(
        mario=MarioView(box=box, facing_right=direction > 0, support="ground", on_something=True),
        enemies=(),
        surfaces=(),
        gaps=(),
    )
    paths = hold_paths(scene, NESPlayerMotion(x_speed=speed * direction), box, direction)
    for gap in GAPS:
        if any(_passes(p, box, gap, closing, width, height, direction) for p in paths):
            return gap / closing
    return None


def jump_lead(closing, width, height, mario_size, speed, direction):
    """Frames before contact at which a jump over an enemy ``width`` x
    ``height`` pixels coming at ``closing`` pixels a frame (Mario's own
    approach included) must start at the latest: Mario ``mario_size`` (width,
    height), moving at ``speed`` pixels a frame toward it (``direction``).
    None when no jump passes it."""
    if closing <= 0.05:
        return None
    return _lead(
        round(float(closing), 2),
        int(round(width)),
        int(round(height)),
        int(round(mario_size[0])),
        int(round(mario_size[1])),
        int(round(max(0.0, speed) * 16)),
        int(direction),
    )


class ThreatWatch:
    """Whether a coming enemy needs a move now (module notes)."""

    def due(self, trackers, kinematics, scene, objects=()) -> bool:
        """``trackers``: the copy's CollisionTrackers; ``kinematics``: its
        (sprites, mario, identities, context) this frame; ``scene``: what
        vision shows (only a standing Mario is checked); ``objects``: the
        list of tracked things as read this frame (rows, keys, slot keys):
        every enemy on it is checked, seen or out of sight."""
        box = scene.mario.box
        grounded = box is not None and scene.mario.on_something and scene.mario.support != "air"
        if not grounded or trackers is None or not kinematics or not objects:
            return False
        from .smb_kinematics import TRACKED
        from .target_tracker import tracker_rows

        mario = kinematics[1]
        state, followed = tracker_rows(objects[0]), objects[1]
        x, vx = TRACKED.index("x"), TRACKED.index("vx")
        width, height = TRACKED.index("width"), TRACKED.index("height")
        for slot in range(len(state)):
            identity = int(followed[slot])
            if not identity or state[slot, 0] <= 0:
                continue
            estimate = trackers.estimate(identity)
            if estimate is None:
                continue
            side = 1 if state[slot, x] > 0 else -1
            closing = -state[slot, vx] * side + mario[0] * side
            if closing <= 0.05:
                closing = NOMINAL_CLOSING
            lead = jump_lead(
                closing,
                state[slot, width] * 16,
                state[slot, height] * 16,
                (box[2] - box[0], box[3] - box[1]),
                mario[0] * side,
                side,
            )
            within = THREAT_HORIZON if lead is None else lead + THREAT_SLACK
            if estimate.hit_within(within) >= 0.5:
                return True
        return False
