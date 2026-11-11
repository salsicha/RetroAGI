"""Discrete stomp contact and visual collision profiles used by local control."""


def walker_body(box):
    """Recognize the compact 10x10 walker sprite, whose last four rows are feet.

    Do not apply this profile to differently sized NES sprites or clipped boxes.
    Scene observations remain drawn rectangles; this conversion is controller
    knowledge, not hidden collision state supplied by the environment.
    """
    x0, y0, x1, y1 = box
    if abs(x1 - x0 - 10) < 1e-6 and abs(y1 - y0 - 10) < 1e-6:
        return (x0, y0, x1, y1 - 4)
    return box


def stomp_contact(body, enemy, vy):
    """Match integer collision rectangles and post-terrain vertical velocity."""
    x, y, right, bottom = body
    ex, ey, eright, ebottom = enemy
    x, y, w, h = int(x), int(y), int(right - x), int(bottom - y)
    ex, ey, ew, eh = int(ex), int(ey), int(eright - ex), int(ebottom - ey)
    return (
        x < ex + ew
        and x + w > ex
        and y < ey + eh
        and y + h > ey
        and vy > 0
        and y + h - vy <= ey + eh // 2
    )
