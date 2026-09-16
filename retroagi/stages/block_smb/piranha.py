"""Pipe-bound, non-stompable hazards for generated avoidance practice.

This is a Block-physics proxy, not a reproduction of NES plant timing. Only
the exposed part collides or appears in observations; phase is teacher-only.
"""


def position_plant(enemy):
    rise = enemy["rise_frames"]
    exposed = enemy["exposed_frames"]
    hidden = enemy["hidden_frames"]
    phase = enemy["plant_tick"] % (2 * rise + exposed + hidden)
    if phase < rise:
        fraction = phase / rise
    elif phase < rise + exposed:
        fraction = 1.0
    elif phase < 2 * rise + exposed:
        fraction = (2 * rise + exposed - phase) / rise
    else:
        fraction = 0.0
    enemy["h"] = round(enemy["plant_height"] * fraction)
    enemy["y"] = enemy["pipe_top"] - enemy["h"]


def parse_plant(spec):
    rise = int(spec.get("rise_frames", 16))
    exposed = int(spec.get("exposed_frames", 48))
    hidden = int(spec.get("hidden_frames", 32))
    height = int(spec.get("plant_height", 24))
    width = int(spec.get("w", 12))
    if min(rise, exposed, hidden, height, width) <= 0:
        raise ValueError("Plant dimensions and phase durations must be positive")
    x = float(spec["x"])
    enemy = dict(
        kind="piranha_plant",
        stompable=False,
        x=x,
        y=0,
        w=width,
        h=height,
        pipe_top=float(spec["pipe_top"]),
        plant_height=height,
        rise_frames=rise,
        exposed_frames=exposed,
        hidden_frames=hidden,
        plant_tick=int(spec.get("phase", 0)),
        vx=0.0,
        vy=0.0,
        speed=0.0,
        direction=1,
        patrol_min=x,
        patrol_max=x,
        edge_aware=False,
        on_ground=False,
        dead=False,
    )
    position_plant(enemy)
    return enemy
