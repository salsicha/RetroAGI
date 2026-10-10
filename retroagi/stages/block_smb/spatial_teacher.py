"""Training-only spatial waypoints for static gap crossings.

Button-route takeoff positions assume uninterrupted acceleration. A spatial run
ends with braking, so its waypoint must leave a jump reachable from low speed.
These labels belong to the teacher; the runtime executor still attempts whatever
destination the skill requests. The landing given here is a proposal: the
teacher labels the version of the jump that lands farthest from the far
platform's edges (controller_teacher._with_margin).
"""

from retroagi.core.smb_coaching import training_target
from retroagi.core.tokens import SkillToken


def gap_destination(env):
    """Approach one supported takeoff point, then request the far-side landing."""
    m = env.mario
    support = m.get("_platform")
    if (
        not m["on_ground"]
        or support is None
        or support.get("moving")
        or getattr(env, "_action_jump_direction", 0)
    ):
        return None
    target = training_target(env)
    if target.kind not in ("gap", "mount") or target.platform_index is None:
        return None
    landing = env.platforms[target.platform_index]
    if landing.get("moving") or any(not e["dead"] for e in env.enemies):
        return None
    source, far = support["rect"], landing["rect"]
    direction = target.direction
    gap = far.left - source.right if direction > 0 else source.left - far.right
    if far.top < source.top or gap <= 0 or min(source.width, far.width) < m["w"] + 4:
        return None
    half = m["w"] / 2
    # Keep the whole body supported with two pixels of edge margin. Unlike a
    # button-route prefix, this waypoint does not depend on accumulated speed.
    inset = half + 2
    takeoff = source.right - inset if direction > 0 else source.left + inset
    arrival = far.left + inset if direction > 0 else far.right - inset
    center = m["x"] + half
    remaining = (takeoff - center) * direction
    if remaining > 3:
        return SkillToken("run", max(-128, min(128, round(takeoff - center))), 0)
    if abs(arrival - center) > 256:
        return None
    return SkillToken("jump", round(arrival - center), round(far.top - source.top))
