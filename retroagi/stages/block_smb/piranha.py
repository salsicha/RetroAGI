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
    plant_height, w (sprite width) and phase."""
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


def has_plants(env):
    return any(e.get("kind") == "piranha_plant" for e in env.enemies)


def runner_crossing_frame(scenario, plant, *, limit=240):
    """Frames for a runner who never stops to reach the plant's centre line.

    Jumping does not raise horizontal speed, so no route that never waits
    crosses the plant earlier; the pipe itself is left out of this probe.
    """
    import copy

    from .env import MarioScenarioEnv

    flat = copy.deepcopy(
        {k: v for k, v in scenario.items() if k not in ("enemies", "platform_kinds", "tactics")}
    )
    flat["platforms"] = [scenario["platforms"][0]]
    env = MarioScenarioEnv()
    try:
        env.reset(scenario=flat)
        env.render = lambda: None
        centre = float(plant["x"]) + float(plant.get("w", 12)) / 2.0
        frames = 0
        while env.mario["x"] + env.mario["w"] / 2.0 < centre and frames < limit:
            env.step(1)
            frames += 1
        return frames
    finally:
        env.close()


def exposed_arrival_phase(scenario, rng, *, lead=8, lag=16):
    """A cycle phase that holds the plant fully raised while a runner crosses.

    The plant is fully exposed from `lead` frames before the earliest possible
    no-wait crossing until `lag` frames after it, so every timed layout
    requires waiting for an observed retraction.
    """
    plant = scenario["enemies"][0]
    rise = int(plant["rise_frames"])
    exposed = int(plant["exposed_frames"])
    period = 2 * rise + exposed + int(plant["hidden_frames"])
    if exposed < lead + lag:
        raise ValueError("Exposure is too short to cover a runner's crossing")
    crossing = runner_crossing_frame(scenario, plant)
    cycle = rise + lead + rng.randrange(exposed - lead - lag + 1)
    return (cycle - crossing) % period


def freeze_plant_envelopes(env):
    """Training probe only: certify against the plant's full collision envelope.

    The maximum height is fixed by pipe width in generated family layouts.
    Never condition labels on hidden cycle phase. The caller owns a full
    snapshot and restores it after the probe.
    """
    for enemy in env.enemies:
        if enemy.get("kind") == "piranha_plant":
            enemy["conservative_envelope"] = True
            position_plant(enemy)


def conservative_suffix(env, *, max_frames=320, release_state=None, variant=0):
    """The teacher's route with every plant held at its full height; it must
    then win with the plants as they are."""
    from .env_state import restore_env_state, snapshot_env_state
    from .policy_recovery import _coached_suffix, route_wins

    saved = snapshot_env_state(env)
    try:
        freeze_plant_envelopes(env)
        actions = _coached_suffix(
            env,
            max_frames=max_frames,
            release_state=release_state,
            hold_variant=variant,
            robust_takeoff=True,
        )
        restore_env_state(env, saved)
        return actions if actions is not None and route_wins(env, actions) else None
    finally:
        restore_env_state(env, saved)


def plant_oracle(scenario, max_steps=320, *, variant=0):
    from .env import MarioScenarioEnv

    env = MarioScenarioEnv()
    try:
        env.reset(scenario=scenario)
        env.render = lambda: None
        from .piranha_tactics import timed_plant, timed_suffix

        if timed_plant(env) is not None:
            return timed_suffix(env, max_frames=max_steps, variant=variant) or []
        return conservative_suffix(env, max_frames=max_steps, variant=variant) or []
    finally:
        env.close()
