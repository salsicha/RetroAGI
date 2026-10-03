"""Jump-on and jump-off prerequisites for moving-bridge traversal."""


def bridge_jump_scenario(rng, difficulty, family):
    width = rng.randint(*{"easy": (88, 100), "medium": (72, 84), "hard": (56, 68)}[difficulty])
    speed = round(
        rng.uniform(*{"easy": (0.8, 1.1), "medium": (1.0, 1.4), "hard": (1.3, 1.8)}[difficulty]), 3
    )
    mount = family == "bridge_mount"
    shore = 330 + rng.randint(-10, 10)
    low = 105 if mount else 95
    high = shore - width - 24
    initial = high if mount else low + rng.randint(0, 12)
    spawn = rng.randint(72, 75) if mount else initial + width - rng.randint(20, 28)
    scenario = dict(
        world_width=440,
        mario=[spawn, 204],
        platforms=[
            [0, 220, 85, 20],
            dict(
                x=initial,
                y=220,
                w=width,
                h=20,
                moving=[low, high, speed],
                direction=-1 if mount else 1,
            ),
            [shore, 220, 440 - shore, 20],
        ],
        goal=[420, 200, 16, 20],
        bridge_jump_task="mount" if mount else "dismount",
        task_objective=family,
        require_bridge_before_goal=True,
        single_jump_attempt=True,
        # A rider is paid goal-distance shaping by the bridge's own motion,
        # and walking toward the leading edge pays for ~50 frames before the
        # fatal drop — a wrong local optimum. The dismount lesson is a single
        # jump; goal credit and the coaching carry the signal.
        reward_goal_distance_shaping=2.0 if mount else 0.0,
        reward_wait_survival=0.0,
    )
    params = dict(
        platform_width=width,
        platform_speed=speed,
        spawn_x=spawn,
        initial_x=initial,
        right_shore=shore,
        required_jump=True,
        single_jump=True,
        # Give the opening WAIT observation, then let the policy reconsider
        # every frame as the moving target enters and leaves jump range.
        a_level_action=0,
        a_level_action_scope="first_primitive",
        difficulty_bin=difficulty,
    )
    # The route is the teacher's, played once the layout's tactics are set.
    return scenario, params, []


def bridge_takeoff_window(env, certify):
    """Holds certified now and after one more waiting frame, for a waiting jump.

    `certify` lists the holds certified in the current state. The probe
    restores the full environment state.
    """
    from .env_state import restore_env_state, snapshot_env_state

    now = certify() if env.mario["on_ground"] else []
    if not now:
        return now, []
    snapshot = snapshot_env_state(env)
    original_render = env.__dict__.get("render")
    env.render = lambda: None
    try:
        _, _, done, truncated, info = env.step(0)
        waiting = env.mario["on_ground"] and not (done or truncated or info["death"])
        later = certify() if waiting else []
    finally:
        restore_env_state(env, snapshot)
        if original_render is None:
            del env.__dict__["render"]
        else:
            env.render = original_render
    return now, later


def bridge_jump_allowed(now, later, robust=None):
    """Whether a waiting jump should launch now.

    A departure window opens with only the longest hold certified while the
    target is still far, and widens as the bridge approaches. Launching on
    that thin opening edge leaves a frame or two of timing margin, so wait
    while fewer than `robust` holds are certified and the set is not
    shrinking; launch once it is robust, starts to narrow, or is closing.
    """
    from .local_traversal import ROBUST_TAKEOFF_HOLDS

    robust = ROBUST_TAKEOFF_HOLDS if robust is None else robust
    return bool(now) and (not later or len(now) >= robust or len(later) < len(now))
