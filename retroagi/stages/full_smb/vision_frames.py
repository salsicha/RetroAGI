"""Real Super Mario Bros frames the Full SMB vision transformer learns from and is measured on.

Each level's saved start is played by a simple random player: it mostly runs
right, jumps for random lengths, and now and then stops or walks back. It is
rewound a few seconds after each death so it reaches the whole level, so Mario
is seen standing, running, jumping, falling, dying, stomping, riding lifts and
pushing against walls. Every ``every``-th frame is labelled from game memory
(pixel_labels.label_frame); a frame memory does not fully explain is counted
by its reason and skipped, never used.
"""

from collections import Counter, deque
from typing import Iterator, Optional

import numpy as np

from .pixel_labels import LabelledFrame, Unexplained, label_frame

LEVELS = (
    "Level1-1",
    "Level1-4",
    "Level2-1",
    "Level2-1-clouds",
    "Level3-1",
    "Level4-1",
    "Level5-1",
    "Level6-1",
    "Level7-1",
    "Level8-1",
)
# Levels kept out of training and used only to measure the model.
TEST_LEVELS = ("Level1-1", "Level5-1")
TRAIN_LEVELS = tuple(level for level in LEVELS if level not in TEST_LEVELS)

WORLD, LEVEL = 0x75F, 0x75C  # [WorldNumber], [LevelNumber]
ENGINE_ROUTINE, PLAYER_SCREEN_ROW = 0x0E, 0xB5  # [GameEngineSubroutine], [Player_Y_HighPos]
REWIND_EVERY = 30  # frames between kept states for rewinding after a death


def _dying(ram) -> bool:
    routine = int(ram[ENGINE_ROUTINE])
    return routine in (6, 11) or (int(ram[PLAYER_SCREEN_ROW]) >= 2 and routine == 8)


def played_frames(level: str, *, frames: int, seed: int, every: int) -> Iterator[tuple]:
    """Play ``level`` for up to ``frames`` frames; yield (frame, state before, state after)
    for every ``every``-th frame, until the level is finished."""
    import retro

    rng = np.random.default_rng(seed)
    env = retro.make("SuperMarioBros-Nes", state=level, inttype=retro.data.Integrations.STABLE)
    try:
        buttons = list(env.buttons)
        env.reset()
        start = (int(env.get_ram()[WORLD]), int(env.get_ram()[LEVEL]))
        history = deque(maxlen=40)
        jump = back = wait = deaths = 0
        for t in range(frames):
            if t % REWIND_EVERY == 0:
                history.append(env.em.get_state())
            if jump == 0 and rng.random() < 0.07:
                jump = int(rng.integers(1, 33))
            if back == wait == 0 and rng.random() < 0.01:
                if rng.random() < 0.5:
                    back = int(rng.integers(5, 25))
                else:
                    wait = int(rng.integers(5, 25))
            action = np.zeros(len(buttons), dtype=np.int8)
            action[buttons.index("B")] = int(rng.random() < 0.8)
            if back:
                action[buttons.index("LEFT")], back = 1, back - 1
            elif wait:
                wait -= 1
            else:
                action[buttons.index("RIGHT")] = 1
            if jump:
                action[buttons.index("A")], jump = 1, jump - 1
            if t % every == 0:
                before = env.em.get_state()
                frame, *_ = env.step(action)
                yield frame, before, env.em.get_state()
            else:
                env.step(action)
            ram = env.get_ram()
            if _dying(ram):
                deaths += 1
                steps = min(len(history), 2 + deaths)
                env.em.set_state(history[-steps])
                for _ in range(steps - 1):
                    history.pop()
                jump = back = wait = 0
            elif t % 300 == 0:
                deaths = max(0, deaths - 1)
            if (int(ram[WORLD]), int(ram[LEVEL])) != start:
                break
    finally:
        env.close()


def labelled_frames(
    level: str,
    *,
    frames: int,
    seed: int,
    every: int,
    refusals: Optional[Counter] = None,
) -> Iterator[LabelledFrame]:
    """Exactly labelled frames from playing ``level``; refused frames are counted by reason."""
    for frame, before, after in played_frames(level, frames=frames, seed=seed, every=every):
        try:
            yield label_frame(frame, before, after, family=level)
        except Unexplained as reason:
            if refusals is not None:
                refusals[str(reason)] += 1
