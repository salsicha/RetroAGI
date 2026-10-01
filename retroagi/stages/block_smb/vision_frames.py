"""Frames the Block vision transformer learns from and is measured on.

Monte Carlo family layouts are played with their teacher routes, with teacher
routes perturbed by held random actions, after a random wait, and with
persistent random actions; procedurally generated levels are played with
random actions. Mario is therefore seen standing, running, jumping, skidding,
falling, dying, stomping, riding lifts and pushing against walls. Each kept
frame carries its exact labels from MarioScenarioEnv.render_labels().
"""

import random
from dataclasses import dataclass
from typing import Any, Iterator, Optional

import numpy as np

from .env import MarioScenarioEnv
from .monte_carlo import (
    DEFAULT_BLOCK_SMB_MC_MAX_STEPS,
    sample_block_smb_monte_carlo_parameter_sweep,
)

FAMILY_ROUTES = ("teacher", "perturbed", "delayed", "random")
GENERATED_LEVEL = "generated_level"
# Action weights for random play: mostly forward, with jumps, waits and reversals.
RANDOM_ACTION_WEIGHTS = (12, 30, 25, 12, 8, 13)


@dataclass(frozen=True)
class VisionFrame:
    image: np.ndarray  # [240, 256, 3] uint8, what the policy sees
    labels: np.ndarray  # [240, 256] uint8 smb_pixel_types.TYPE_ID
    on_ground: bool  # the simulator's own flag
    enemy_rects: tuple  # (x, y, w, h) screen box of each drawn enemy
    family: str
    route: str


@dataclass(frozen=True)
class FamilyLayout:
    family: str
    difficulty: str
    scenario: dict
    teacher_actions: tuple


def family_layouts(split: str, seed: int, repeats: int, executor: Any = None) -> list:
    """Every family at every difficulty, ``repeats`` layouts each, with teacher routes."""
    sweep = sample_block_smb_monte_carlo_parameter_sweep(
        split=split, seed=seed, repeats_per_difficulty=repeats, executor=executor
    )
    return [
        FamilyLayout(
            family=sample.family,
            difficulty=sample.difficulty_bin,
            scenario=dict(sample.scenario),
            teacher_actions=tuple(int(a) for a in sample.oracle["actions"]),
        )
        for sample in sweep.samples
    ]


def generated_level(rng: random.Random) -> dict:
    return MarioScenarioEnv.generate_scenario(
        num_screens=rng.randint(1, 3),
        enemy_density=rng.uniform(0.25, 0.9),
        moving_platform_chance=rng.uniform(0.1, 0.5),
        seed=rng.randrange(1 << 30),
    )


def random_actions(rng: random.Random) -> Iterator[int]:
    """Persistent random play: each action is held for 1-20 frames."""
    while True:
        action = rng.choices(range(6), weights=RANDOM_ACTION_WEIGHTS)[0]
        yield from [action] * rng.randint(1, 20)


def route_actions(route: str, teacher: tuple, rng: random.Random) -> Iterator[int]:
    if route == "random" or not teacher:
        yield from random_actions(rng)
        return
    if route == "delayed":
        # Waiting (or drifting) first shows standing frames and shifts the
        # route's timing against moving enemies and lifts.
        yield from [rng.choice((0, 0, 3, 5))] * rng.randint(1, 48)
    padded = list(teacher) + [teacher[-1]] * DEFAULT_BLOCK_SMB_MC_MAX_STEPS
    noise = random_actions(rng)
    index = 0
    while True:
        if route == "perturbed" and rng.random() < 0.08:
            for _ in range(rng.randint(1, 12)):
                yield next(noise)
                index += 1
            continue
        yield padded[min(index, len(padded) - 1)]
        index += 1


def play(
    env: MarioScenarioEnv,
    scenario: dict,
    actions: Iterator[int],
    rng: random.Random,
    *,
    family: str,
    route: str,
    keep: float,
    max_steps: int = DEFAULT_BLOCK_SMB_MC_MAX_STEPS,
) -> Iterator[VisionFrame]:
    """Play one episode, keeping each frame with probability ``keep`` and the last."""
    env.reset(scenario=scenario, seed=rng.randrange(1 << 30))
    for step in range(max_steps + 1):
        done = False
        if step:
            _obs, _reward, terminated, truncated, _info = env.step(next(actions))
            done = terminated or truncated or step == max_steps
        if done or rng.random() < keep:
            yield VisionFrame(
                image=env.render(),
                labels=env.render_labels(),
                on_ground=bool(env.mario["on_ground"]),
                enemy_rects=tuple(tuple(r) for r in env.enemy_screen_rects()),
                family=family,
                route=route,
            )
        if done:
            return


def layout_frames(
    layout: FamilyLayout,
    route: str,
    rng: random.Random,
    *,
    keep: float,
    env: Optional[MarioScenarioEnv] = None,
) -> Iterator[VisionFrame]:
    own = env is None
    env = env or MarioScenarioEnv()
    try:
        yield from play(
            env,
            layout.scenario,
            route_actions(route, layout.teacher_actions, rng),
            rng,
            family=layout.family,
            route=route,
            keep=keep,
        )
    finally:
        if own:
            env.close()


def held_out_frames(layouts: list, seed: int, *, keep: float = 0.2) -> Iterator[VisionFrame]:
    """Each layout played with its teacher route and with a perturbed teacher route."""
    rng = random.Random(seed)
    env = MarioScenarioEnv()
    try:
        for layout in layouts:
            for route in ("teacher", "perturbed"):
                yield from layout_frames(layout, route, rng, keep=keep, env=env)
    finally:
        env.close()


def frame_stream(
    layouts: list,
    seed: int,
    *,
    keep: float = 0.25,
    generated_share: float = 0.15,
) -> Iterator[VisionFrame]:
    """An endless, seeded stream of frames from random layouts and routes."""
    rng = random.Random(seed)
    env = MarioScenarioEnv()
    try:
        while True:
            if rng.random() < generated_share:
                yield from play(
                    env,
                    generated_level(rng),
                    random_actions(rng),
                    rng,
                    family=GENERATED_LEVEL,
                    route="random",
                    keep=keep,
                )
                continue
            layout = rng.choice(layouts)
            route = rng.choices(FAMILY_ROUTES, weights=(2, 3, 2, 2))[0]
            yield from layout_frames(layout, route, rng, keep=keep, env=env)
    finally:
        env.close()
