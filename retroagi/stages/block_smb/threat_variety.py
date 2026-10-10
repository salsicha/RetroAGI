"""Variety for the families that teach jumping over or past a pit or an enemy.

Every such threat is followed by another, a pit or an enemy, at a varied
distance. The landing is then bounded, and the teacher, which keeps Mario as
far as it can from every threat before, during and after a jump
(controller_teacher), teaches a landing that keeps away from both, not the
longest jump. Mario also starts at varied distances from the first threat,
right next to it included, as he can find himself in Full SMB.

Enemies patrol exactly the platform they stand on: an edge they turn at is
always one the vision can see.
"""

ENEMY_HEIGHT = 14
ENEMY_WIDTH = 10
# A following enemy's speed (0: it stands) and the direction it walks.
ENEMY_SPEEDS = (0.0, 0.0, 0.3, 0.45, 0.6)
# The narrowest piece of floor left past a cut pit.
REST = 32


def follow_up(
    rng, scenario, index, edge, side, *, window=(24, 72), pit=(8, 36), kinds=("pit", "enemy")
):
    """Bound the landing on platform ``index`` past the first threat.

    ``edge``: the x where the landing area starts (the far edge of the pit, or
    just past the enemy); ``side``: 1 when the landing lies to the right of
    it, -1 to the left. A pit of ``pit`` pixels is cut into the platform, or an
    enemy is placed on it, ``window`` pixels past ``edge``. Returns (kind, the
    landing window's width, the pit's width or the enemy's speed).
    """
    platforms = scenario["platforms"]
    x, top, width, height = platforms[index]
    right = x + width
    kind = rng.choice(kinds)
    room = rng.randint(*window)
    gap = rng.randint(*pit)
    space = (right - edge if side > 0 else edge - x) - gap - REST
    if kind == "pit" and space >= max(12, room // 2):
        # Keep a piece of floor past the pit (it reaches the world's end).
        room = min(room, space)
        if side > 0:
            cut = edge + room
            platforms[index] = [x, top, cut - x, height]
            platforms.append([cut + gap, top, right - cut - gap, height])
        else:
            cut = edge - room
            platforms[index] = [cut, top, right - cut, height]
            platforms.append([x, top, cut - gap - x, height])
        fit_patrols(scenario)
        return "pit", room, gap
    # An enemy (also where no pit fits).
    speed = rng.choice(ENEMY_SPEEDS)
    direction = rng.choice((-1, 1))
    enemy_x = edge + room if side > 0 else edge - room - ENEMY_WIDTH
    enemy_x = max(x, min(right - ENEMY_WIDTH, enemy_x))
    scenario.setdefault("enemies", []).append(
        [enemy_x, top - ENEMY_HEIGHT, x, right - ENEMY_WIDTH, speed, direction]
    )
    fit_patrols(scenario)
    return "enemy", room, speed


def fit_patrols(scenario):
    """Make every enemy patrol exactly the platform it stands on. The limits
    are for the enemy's left edge, so it turns with its body still on the
    platform."""
    for enemy in scenario.get("enemies", []):
        feet = enemy[1] + ENEMY_HEIGHT
        for x, top, width, _ in scenario["platforms"]:
            if top == feet and x <= enemy[0] and enemy[0] + ENEMY_WIDTH <= x + width:
                enemy[2], enemy[3] = x, x + width - ENEMY_WIDTH
                break


def start_distance(rng, reach):
    """How far Mario's front starts from a threat: often right next to it (0
    to 8 pixels), otherwise anywhere up to ``reach`` pixels."""
    if rng.random() < 0.4:
        return rng.randint(0, 8)
    return rng.randint(0, max(8, reach))


# How far ahead a jump from a standstill lands at most (the middle of Mario's
# feet, pixels), for a landing ``rise`` pixels higher (negative: lower), by the
# executor's motion model: 50 on level ground, about 0.36 less per pixel up and
# 0.35 more per pixel down.
def standing_reach(rise):
    return 50 - 0.36 * rise if rise > 0 else 50 - 0.35 * rise


# The share of layouts whose best landing a jump from a standstill can reach.
# Where it cannot, the best jump is the longest, whose label is nearly the same
# distance in every layout; most layouts keep the landing within reach.
REACHABLE = 0.75


def reachable_span(rng, low, high, ceiling):
    """A size between ``low`` and ``high``, kept at most ``ceiling`` (so that
    the best landing is within a standstill jump's reach) in about REACHABLE
    of layouts; ``high`` bounds it otherwise, and when ``ceiling`` is below
    ``low``."""
    if ceiling >= low and rng.random() < REACHABLE:
        return rng.randint(low, min(high, int(ceiling)))
    return rng.randint(low, high)
