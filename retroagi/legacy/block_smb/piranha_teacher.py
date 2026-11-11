"""The retired curriculum's plant lessons: where a runner crosses a plant,
plants held at their full height for a teacher's probe, and the teacher's
route past one (piranha.py has the plant itself).
"""

from retroagi.domains.smb.sim.piranha import position_plant


def has_plants(env):
    return any(e.get("kind") == "piranha_plant" for e in env.enemies)


def runner_crossing_frame(scenario, plant, *, limit=240):
    """Frames for a runner who never stops to reach the plant's centre line.

    Jumping does not raise horizontal speed, so no route that never waits
    crosses the plant earlier; the pipe itself is left out of this probe.
    """
    import copy

    from retroagi.domains.smb.sim.env import MarioScenarioEnv

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
    from retroagi.domains.smb.sim.env_state import restore_env_state, snapshot_env_state

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
    from retroagi.domains.smb.sim.env import MarioScenarioEnv

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
