"""Jump-on and jump-off prerequisites for moving-bridge traversal."""

from types import SimpleNamespace

BRIDGE_JUMP_FAMILIES = ("bridge_mount", "bridge_dismount")
BRIDGE_FAMILIES = (*BRIDGE_JUMP_FAMILIES, "bridge_wait", "moving_bridge", "wait_timing")


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
        mario=[spawn, 208],
        physics_profile="nes_land_v1",
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
        reward_goal_distance_shaping=2.0,
        reward_wait_survival=0.0,
    )
    params = dict(
        platform_width=width,
        platform_speed=speed,
        spawn_x=spawn,
        initial_x=initial,
        right_shore=shore,
        required_jump=True,
        difficulty_bin=difficulty,
        family_revision=1,
    )
    return scenario, params, bridge_jump_oracle(scenario)


def bridge_jump_choice(model, env, *, variant=0):
    from retroagi.core.smb_coaching import interior_index, safe_jump_indices

    valid = safe_jump_indices(model, env, 2)
    if len(valid) >= 3 + variant % 3:
        index = valid[variant % len(valid)] if variant else interior_index(valid)
        return 2, index, valid
    if abs(env.mario["vx"]) > 1 / 16:
        return (3 if env.mario["vx"] > 0 else 1), 0, list(range(16))
    return 0, 0, [0]


def bridge_jump_oracle(scenario):
    from retroagi.core.smb_coaching import physical_batch
    from retroagi.core.smb_learning import primitive, runtime
    from retroagi.core.smb_runtime import make_smb_executor

    from .env import MarioScenarioEnv

    env = MarioScenarioEnv()
    model = SimpleNamespace(smb_runtime_contract=runtime())
    executor = make_smb_executor(model)
    actions = []
    try:
        env.reset(scenario=scenario)
        env.render = lambda: None
        for _ in range(320):
            batch = physical_batch(env)
            # All bridge waits are reconsidered every frame, as in pixel playback.
            batch.metadata["smb_geometry"]["scene"] = env
            committed = executor.prepare(batch)
            if committed is None and env.mario["on_ground"]:
                action, index, _ = bridge_jump_choice(model, env)
            else:
                action = committed if committed is not None else 1
                index = 0
            execution = executor.execute(action, batch=batch, motor_primitives=primitive(index))
            _, _, done, truncated, _ = env.step(execution.action)
            actions.append(int(execution.action))
            if done or truncated:
                break
        return actions
    finally:
        env.close()
