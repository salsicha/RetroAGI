"""Pipe-bound, non-stompable hazards for generated avoidance practice.

Each plant rises, stays up, sinks and stays down by its own cycle, and, as
the NES's plants do, does not come up while Mario's middle is within
WAITS_NEAR pixels of its own: it stays down until he is farther (user,
2026-10-21: every Block SMB plant follows the real game's rule; a Full SMB
plant stayed down 6,202 frames while Mario stood by its pipe). Only the
exposed part collides or appears in observations; cycle phase never enters
policy inputs or duration certification.
"""

# The NES plant does not rise while Mario is within 0x21 pixels of it.
WAITS_NEAR = 33


def position_plant(enemy):
    if enemy.get("conservative_envelope"):
        enemy["h"] = enemy["plant_height"]
        enemy["y"] = enemy["pipe_top"] - enemy["h"]
        return
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


def stays_down_near(enemy, mario_middle: float) -> bool:
    """Whether a plant at the end of its time down stays down this frame:
    Mario's middle is within its ``waits_near`` of its own."""
    near = enemy.get("waits_near", WAITS_NEAR)
    cycle = 2 * enemy["rise_frames"] + enemy["exposed_frames"] + enemy["hidden_frames"]
    last_down = enemy["plant_tick"] % cycle == cycle - 1
    return last_down and abs(enemy["x"] + enemy["w"] / 2 - mario_middle) <= near


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
