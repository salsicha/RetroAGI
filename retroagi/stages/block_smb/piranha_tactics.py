"""Training-only temporal plant teacher using observed plant motion.

No live cycle phase or remaining timer is used to choose a departure. The
teacher certifies against forecasts built from what has been seen: a descent
continues at its measured speed, and a retraction is followed by the shortest
hidden interval and fastest emergence in the timed family. A standing NES jump
cannot clear the timed pipe, so a certified departure may be a run-up that
starts while the plant is still descending. Otherwise the teacher stops in the
staging window without overshooting it.
"""

MIN_HIDDEN_FRAMES = 48
# Certified departures are searched within this distance of the pipe.
DEPARTURE_REACH = 65
# The timed family's plant height and rise/retraction durations.
TIMED_PLANT_HEIGHT = 80
TIMED_RISE_FRAMES = range(12, 21)
# A run-up needs only one safe continuation; long holds cross the pipe.
RUN_ON_FRAMES = range(2, 42, 2)
RUN_ON_HOLDS = (32, 26, 20, 14)
# Faster than NES running, so pruning by it never drops a feasible crossing.
RUNNER_TOP_SPEED = 3.0


def hold_menu(env):
    from retroagi.core.smb_physics import NES_JUMP_FRAMES

    return NES_JUMP_FRAMES


def timed_plant(env):
    return next((e for e in env.enemies if e.get("timed_crossing")), None)


def fresh_retraction(history):
    # An initially empty pipe has unknown phase. Require an observed
    # disappearance, not merely invisibility or time since episode reset.
    # The probe certifies against the remaining minimum hidden interval, so
    # any age inside that interval may be probed.
    return history is not None and history[0] == 0 and 0 < history[5] * 64 < MIN_HIDDEN_FRAMES


def plant_pipe(env, plant):
    return next(p["rect"] for p in env.platforms if p["rect"].top == plant["pipe_top"])


def staging_x(env, plant=None):
    return float(plant_pipe(env, plant or timed_plant(env)).left - 32)


def settle_x(env, action):
    """Where Mario comes to rest after one frame of `action` and then coasting."""
    from .env_state import restore_env_state, snapshot_env_state

    saved = snapshot_env_state(env)
    original_render = env.__dict__.get("render")
    env.render = lambda: None
    try:
        env.step(action)
        for _ in range(40):
            if abs(env.mario["vx"]) < 1e-9:
                break
            env.step(0)
        return env.mario["x"]
    finally:
        restore_env_state(env, saved)
        if original_render is None:
            del env.__dict__["render"]
        else:
            env.render = original_render


def approach_choice(env, target):
    """Walk to the staging window and brake early enough to stop inside it."""
    x, vx = env.mario["x"], env.mario["vx"]
    if x > target + 4:
        if vx > 0.5:
            return "hold_area", 0
        return ("retreat", 3) if settle_x(env, 3) >= target - 2 else ("hold_area", 0)
    if x < target - 2 or vx < -0.5:
        return ("advance", 1) if settle_x(env, 1) <= target + 4 else ("hold_area", 0)
    return "hold_area", 0


def plant_forecasts(env, history):
    """(rise, tick) cycle models consistent with the observed timed plant.

    A raised plant that is not descending has no bounded retraction, so
    nothing can be certified against it.
    """
    plant = timed_plant(env)
    if plant is None or history is None:
        return []
    if fresh_retraction(history):
        rise = min(TIMED_RISE_FRAMES)
        # One extra frame covers rounding at disappearance.
        return [(rise, 2 * rise + 64 + int(round(history[5] * 64)) + 1)]
    if not (history[0] == 1 and history[2] == 1 and history[1] > 0):
        return []
    drop = float(history[1]) * 8
    fraction = min(1.0, plant["h"] / TIMED_PLANT_HEIGHT)
    return [
        (rise, round(2 * rise + 64 - fraction * rise))
        for rise in TIMED_RISE_FRAMES
        if abs(TIMED_PLANT_HEIGHT / rise - drop) <= 1
    ]


def _crosses_alive(env, model, run, hold):
    """Run, jump for `hold` frames, and land past the plant under one forecast."""
    rise, tick = model
    probe = timed_plant(env)
    probe.pop("conservative_envelope", None)
    probe.update(
        rise_frames=rise,
        exposed_frames=64,
        hidden_frames=MIN_HIDDEN_FRAMES,
        plant_height=TIMED_PLANT_HEIGHT,
        plant_tick=tick,
    )
    airborne = False
    for frame in range(run + 96):
        action = 2 if run <= frame < run + hold else 1
        _, _, done, truncated, info = env.step(action)
        airborne |= frame >= run and not env.mario["on_ground"]
        if info["death"] or truncated:
            return False
        if airborne and env.mario["on_ground"]:
            if env.mario["x"] < probe["x"] + probe["w"]:
                return False
            return not any(env.step(1)[4]["death"] for _ in range(2))
        if done:
            return False
    return False


def _certified_departures(env, history, candidates):
    """Candidate (run, hold) departures that are safe under every forecast."""
    from .env_state import restore_env_state, snapshot_env_state

    plant = timed_plant(env)
    if plant is None or not env.mario["on_ground"]:
        return
    if env.mario["x"] >= plant["x"] + plant["w"]:
        return
    models = plant_forecasts(env, history)
    if not models:
        return
    saved = snapshot_env_state(env)
    original_render = env.render
    env.render = lambda: None
    try:
        for run, hold in candidates:
            safe = True
            for model in models:
                restore_env_state(env, saved)
                if not _crosses_alive(env, model, run, hold):
                    safe = False
                    break
            if safe:
                yield run, hold
    finally:
        restore_env_state(env, saved)
        env.render = original_render


def timed_safe_holds(env, history, direction=1):
    """Holds that cross safely when the jump starts now."""
    if direction != 1:
        return []
    candidates = [(0, hold) for hold in hold_menu(env)]
    return [hold for _, hold in _certified_departures(env, history, candidates)]


def timed_run_on(env, history):
    """Whether running on and jumping later crosses safely under every forecast."""
    plant = timed_plant(env)
    models = plant_forecasts(env, history)
    if plant is None or not models:
        return False
    # Even at top speed Mario must pass the plant before it is fully raised.
    passing = (plant["x"] + plant["w"] - env.mario["x"]) / RUNNER_TOP_SPEED
    raised = min(2 * rise + 64 + MIN_HIDDEN_FRAMES + rise - tick for rise, tick in models)
    if passing > raised:
        return False
    candidates = ((run, hold) for run in RUN_ON_FRAMES for hold in RUN_ON_HOLDS)
    return next(_certified_departures(env, history, candidates), None) is not None


def tactical_choice(env, history):
    """Return stance, motor action and certified holds at a decision state."""
    plant = timed_plant(env)
    if plant is None or env.mario["x"] >= plant["x"] + plant["w"]:
        return "advance", 1, []
    if plant_pipe(env, plant).left - env.mario["x"] - env.mario["w"] < DEPARTURE_REACH:
        valid = timed_safe_holds(env, history)
        if valid:
            return "advance", 2, valid
        if timed_run_on(env, history):
            return "advance", 1, []
    stance, action = approach_choice(env, staging_x(env, plant))
    return stance, action, []


def timed_suffix(env, *, max_frames=320, release_state=None, variant=0, observation_history=None):
    """The teacher's route through a timed plant (policy_recovery._coached_suffix)."""
    from .policy_recovery import _coached_suffix

    return _coached_suffix(
        env,
        max_frames=max_frames,
        release_state=release_state,
        hold_variant=variant,
        observation_history=observation_history,
    )
