"""Training-only temporal plant teacher using observed plant motion.

No live cycle phase or remaining timer is used to choose a departure. The
teacher certifies against forecasts built from what has been seen: a descent
continues at its measured speed, and a retraction is followed by the shortest
hidden interval and fastest emergence in the timed family. A standing NES jump
cannot clear the timed pipe, so a certified departure may be a run-up that
starts while the plant is still descending. Otherwise the teacher stops in the
staging window without overshooting it.
"""

from retroagi.core.models import TACTIC_STANCES
from retroagi.core.smb_enemy_history import EnemyObservationHistory

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
    from .geometry_expert import restore_env_state, snapshot_env_state

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
    from .geometry_expert import restore_env_state, snapshot_env_state

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


def tactic_label(env, history, action=None):
    """Label decisions only; other families and committed arcs are masked."""
    if not any(e.get("kind") == "piranha_plant" for e in env.enemies):
        return -1
    if not env.mario["on_ground"]:
        return -1
    if timed_plant(env) is not None:
        stance, _, _ = tactical_choice(env, history)
    elif action is not None:
        stance = "hold_area" if action == 0 else "retreat" if action in (3, 4) else "advance"
    else:
        return -1
    return TACTIC_STANCES.index(stance)


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


def _plant(env):
    return next((e for e in env.enemies if e.get("kind") == "piranha_plant"), None)


def _overrun_prefix(margin):
    """Run at full speed past the staging window without braking."""

    def prefix(env):
        target = staging_x(env)
        while env.mario["x"] <= target + margin:
            yield 1

    return prefix


def _wall_prefix(env):
    """Run into the pipe's side, as a learner that never stops does."""
    pipe = plant_pipe(env, _plant(env))
    # Positions are whole pixels, so NES walking from rest leaves x unchanged
    # for several frames; only a sustained stop means something blocks Mario.
    stalled = 0
    while env.mario["x"] + env.mario["w"] < pipe.left - 1 and stalled < 8:
        before = env.mario["x"]
        yield 1
        stalled = stalled + 1 if env.mario["x"] <= before else 0


def _spawn_hop_prefix(hold):
    """Open with a jump from spawn, then keep running until it lands."""

    def prefix(env):
        for _ in range(hold):
            yield 2
        airborne = not env.mario["on_ground"]
        while not (airborne and env.mario["on_ground"]):
            yield 1
            airborne |= not env.mario["on_ground"]

    return prefix


def _arrival_route(sample, prefix, *, max_frames=320):
    """Replay an unsupervised arrival prefix; the teacher labels the rest."""
    from dataclasses import replace

    from .env import MarioScenarioEnv
    from .piranha import conservative_suffix
    from .primitive_execution import JumpReleaseState, teacher_route_reachable

    env = MarioScenarioEnv()
    try:
        env.reset(scenario=sample.scenario)
        env.render = lambda: None
        if _plant(env) is None:
            return None
        history = EnemyObservationHistory()
        release = JumpReleaseState()
        actions = []
        for action in prefix(env):
            history.observe(env, env.steps)
            _, _, done, truncated, info = env.step(action)
            release.observe(env, action, info)
            actions.append(action)
            if done or truncated or info["death"] or len(actions) > 160:
                return None
        if not actions:
            return None
        remaining = max_frames - len(actions)
        if timed_plant(env) is not None:
            suffix = timed_suffix(
                env, observation_history=history, release_state=release, max_frames=remaining
            )
        else:
            suffix = conservative_suffix(env, max_frames=remaining, release_state=release)
        if not suffix:
            return None
        start = len(actions)
        actions += suffix
        env.reset(scenario=sample.scenario)
        if not teacher_route_reachable(env, actions):
            return None
        return replace(
            sample,
            oracle={**sample.oracle, "actions": actions, "supervision_start_frame": start},
        )
    finally:
        env.close()


def overshoot_demonstration(sample, *, margin=12):
    """A timed route that runs past the staging window and is then corrected.

    Teacher routes brake early and never overshoot, so retreat would otherwise
    be taught only by sparse policy-recovery rows, which the demonstration
    sampler upweights to a full action group. The overshooting prefix is
    replayed without supervision; the teacher labels only the correction.
    """
    if not any(
        isinstance(e, dict) and e.get("timed_crossing") for e in sample.scenario.get("enemies", ())
    ):
        return None
    return _arrival_route(sample, _overrun_prefix(margin))


# The pipe wall stops a runner 21 px past the staging window (_wall_prefix).
ARRIVAL_OVERRUN_MARGINS = (0, 8, 16)
# Short, medium and full spawn hops from the NES hold menu; route replay maps
# a hold to its nearest menu entry, so off-menu holds would not replay.
ARRIVAL_HOP_HOLDS = (14, 22, 32)


def arrival_demonstrations(sample, index=0):
    """Teacher corrections from the states a learner actually arrives in.

    The teacher walks up and stops at the staging window, so its routes never
    visit what a learner reaches: running at full speed past that window,
    pressed against the pipe, or landing from a hop taken at spawn. The
    stance head learned to wait only in the teacher's states. Each arrival
    prefix is replayed without supervision; the teacher labels the correction
    (brake, back up, wait for an observed retraction, or re-approach).
    """
    timed = any(
        isinstance(e, dict) and e.get("timed_crossing") for e in sample.scenario.get("enemies", ())
    )
    # Layout difficulty cycles every three indices; cycle hops across groups
    # of three so each difficulty sees every hold instead of a fixed one.
    hop = ARRIVAL_HOP_HOLDS[(index // 3) % len(ARRIVAL_HOP_HOLDS)]
    prefixes = [_wall_prefix, _spawn_hop_prefix(hop)]
    if timed:
        prefixes += [_overrun_prefix(margin) for margin in ARRIVAL_OVERRUN_MARGINS]
    routes = (_arrival_route(sample, prefix) for prefix in prefixes)
    return [route for route in routes if route is not None]
