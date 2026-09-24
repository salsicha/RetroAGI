"""Pipe-bound, non-stompable hazards for generated avoidance practice.

This is a Block-physics proxy, not a reproduction of NES plant timing. Only
the exposed part collides or appears in observations; cycle phase never enters policy inputs or duration certification.
"""


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
    )
    position_plant(enemy)
    return enemy


def has_plants(env):
    return any(e.get("kind") == "piranha_plant" for e in env.enemies)


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
    from .geometry_expert import restore_env_state, snapshot_env_state
    from .policy_recovery import _coached_suffix
    from .primitive_execution import teacher_route_reachable

    saved = snapshot_env_state(env)
    try:
        freeze_plant_envelopes(env)
        actions = _coached_suffix(
            env, max_frames=max_frames, release_state=release_state, hold_variant=variant
        )
        restore_env_state(env, saved)
        if actions is not None and teacher_route_reachable(
            env, actions, release_state=release_state
        ):
            return actions
        return None
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
