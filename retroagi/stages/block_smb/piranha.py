"""Piranha plants: pipe-bound, non-stompable, as the game's.

Each plant rises, stays up, sinks and stays down by its own cycle, and, as
the NES's plants do, does not come up while Mario's middle is within
WAITS_NEAR pixels of its own: it stays down until he is farther (user,
2026-10-21: every Block SMB plant follows the real game's rule; a Full SMB
plant stayed down 6,202 frames while Mario stood by its pipe). The part out
of the pipe is drawn; only the game's small collision box in it collides
(position_plant); cycle phase never enters policy inputs.
"""

# The NES plant does not rise while Mario is within 0x21 pixels of it.
WAITS_NEAR = 33
# The game's plant (game memory, 5-1): 16 wide and 24 tall, it rises out of
# its pipe at half a pixel a frame (RISE_FRAMES), stays up EXPOSED_FRAMES,
# sinks and stays down HIDDEN_FRAMES. It collides by a HIT_W x HIT_H box
# HIT_LEFT in from its left side and ending HIT_BELOW above its bottom
# (fully out: 4 to 10 pixels above the pipe's top), not its whole body.
PLANT_W, PLANT_H = 16, 24
RISE_FRAMES, EXPOSED_FRAMES, HIDDEN_FRAMES = 48, 64, 64
HIT_LEFT, HIT_W, HIT_H, HIT_BELOW = 3, 10, 6, 4


def position_plant(enemy):
    """The plant's sprite (``sprite_x``, ``exposed`` pixels of it above the
    pipe's top) by its cycle, and its collision box (``x``, ``y``, ``w``,
    ``h``) in it; ``conservative_envelope``: held fully out."""
    if enemy.get("conservative_envelope"):
        fraction = 1.0
    else:
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
    enemy["exposed"] = round(enemy["plant_height"] * fraction)
    if enemy.get("hit") == "body":
        # The retired curriculum's lesson plant: its whole part out of the
        # pipe collides (a tall one is a wall to wait for, which the game's
        # plant never is).
        enemy.update(x=enemy["sprite_x"], y=enemy["pipe_top"] - enemy["exposed"], w=enemy["sprite_w"],
                     h=enemy["exposed"])
        return
    bottom = enemy["pipe_top"] - enemy["exposed"] + enemy["plant_height"]
    enemy.update(x=enemy["sprite_x"] + HIT_LEFT, y=bottom - HIT_BELOW - HIT_H, w=HIT_W, h=HIT_H)


def plant_sprite(enemy) -> tuple:
    """(x, y, w, h) of the part of the plant out of its pipe (h 0: hidden)."""
    return (enemy["sprite_x"], enemy["pipe_top"] - enemy["exposed"], enemy["sprite_w"], enemy["exposed"])


def stays_down_near(enemy, mario_middle: float) -> bool:
    """Whether a plant at the end of its time down stays down this frame:
    Mario's middle is within its ``waits_near`` of its own."""
    near = enemy.get("waits_near", WAITS_NEAR)
    cycle = 2 * enemy["rise_frames"] + enemy["exposed_frames"] + enemy["hidden_frames"]
    last_down = enemy["plant_tick"] % cycle == cycle - 1
    return last_down and abs(enemy["sprite_x"] + enemy["sprite_w"] / 2 - mario_middle) <= near


def parse_plant(spec):
    """A plant from ``spec``: x (its sprite's left side), pipe_top, and
    optionally its cycle (rise_frames, exposed_frames, hidden_frames),
    plant_height, w (sprite width), phase and hit ("body": its whole
    exposed part collides, the retired curriculum's lesson plant; else the
    game's box)."""
    rise = int(spec.get("rise_frames", RISE_FRAMES))
    exposed = int(spec.get("exposed_frames", EXPOSED_FRAMES))
    hidden = int(spec.get("hidden_frames", HIDDEN_FRAMES))
    height = int(spec.get("plant_height", PLANT_H))
    width = int(spec.get("w", PLANT_W))
    if min(rise, exposed, hidden, height, width) <= 0:
        raise ValueError("Plant dimensions and phase durations must be positive")
    x = float(spec["x"])
    enemy = dict(
        kind="piranha_plant",
        stompable=False,
        sprite_x=x,
        sprite_w=width,
        **({"hit": "body"} if spec.get("hit") == "body" else {}),
        x=x,
        y=0,
        w=HIT_W,
        h=HIT_H,
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
        timed_crossing=bool(spec.get("timed_crossing", False)),
        # It does not come up while Mario's middle is this near its own
        # (stays_down_near).
        waits_near=float(WAITS_NEAR),
    )
    position_plant(enemy)
    return enemy
