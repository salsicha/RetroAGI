"""Full SMB for the four-layer agent: the game gives screens, the agent gives buttons.

FullSMBGame plays one saved level start in the emulator. reset() and
step(action) return the screen, padded back to the 256x240 NES picture like
every SMB picture: all the agent receives. Each step also returns a
FullSMBOutcome read from game memory (how far into the level Mario is,
whether he is dying, whether he has left the level). The outcome is for
scoring and training rewards only and never reaches the agent.

Button actions are the shared SMB actions (core.actions.SMBAction); the run
button is held while moving right, as in Block SMB's motion model.
"""

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from retroagi.core.actions import SMBAction, full_smb_action
from retroagi.core.smb_scene import canonical_rgb

from .vision_frames import _dying

WORLD, LEVEL = 0x75F, 0x75C  # [WorldNumber], [LevelNumber]
# Levels kept out of Full SMB policy training (the strategy layer) to test it.
# The vision transformer learns from every level (vision_frames.LEVELS).
POLICY_TEST_LEVELS = ("Level1-1", "Level5-1")
PAGE, X_ON_PAGE = 0x6D, 0x86  # [Player_PageLoc], [Player_X_Position]
RUN_ACTIONS = (SMBAction.RIGHT, SMBAction.RIGHT_JUMP)


@dataclass(frozen=True)
class FullSMBOutcome:
    """Read from game memory, for scoring only."""

    level_x: int  # Mario's distance into the level, in pixels
    dying: bool
    finished: bool  # the level has changed: the flagpole or a pipe to the next level


class FullSMBGame:
    """One saved level start of Super Mario Bros in the emulator."""

    def __init__(self, level: str):
        import retro

        self.level = level
        self.env = retro.make(
            "SuperMarioBros-Nes", state=level, inttype=retro.data.Integrations.STABLE
        )
        self.buttons = list(self.env.buttons)
        self.start: tuple[int, int] = (0, 0)

    def _screen(self, observation) -> np.ndarray:
        return np.ascontiguousarray(canonical_rgb(np.asarray(observation)[..., :3]))

    def outcome(self) -> FullSMBOutcome:
        ram = self.env.get_ram()
        return FullSMBOutcome(
            level_x=int(ram[PAGE]) * 256 + int(ram[X_ON_PAGE]),
            dying=_dying(ram),
            finished=(int(ram[WORLD]), int(ram[LEVEL])) != self.start,
        )

    def reset(self) -> np.ndarray:
        result = self.env.reset()
        observation = result[0] if isinstance(result, tuple) else result
        ram = self.env.get_ram()
        self.start = (int(ram[WORLD]), int(ram[LEVEL]))
        return self._screen(observation)

    def step(self, action: int) -> tuple[np.ndarray, FullSMBOutcome]:
        action = SMBAction(action)
        buttons = full_smb_action(action, self.buttons).astype(np.int8)
        if action in RUN_ACTIONS:
            buttons[self.buttons.index("B")] = 1
        observation, *_ = self.env.step(buttons)
        return self._screen(observation), self.outcome()

    def close(self) -> None:
        self.env.close()


@dataclass(frozen=True)
class FullSMBRun:
    level: str
    frames: int
    furthest_x: int
    died: bool
    finished: bool


def play_full_levels(agents, levels: Sequence[str], frames: int) -> list[FullSMBRun]:
    """Each level played once from its start by one copy of ``agents``
    (core.smb_agent.SMBAgents with at least len(levels) copies), side by side,
    until Mario dies, leaves the level or ``frames`` frames pass."""
    games = [FullSMBGame(level) for level in levels]
    screens = [game.reset() for game in games]
    for copy in range(len(games)):
        agents.reset(copy)
    furthest = [0] * len(games)
    ended: dict[int, FullSMBRun] = {}
    try:
        for frame in range(frames):
            playing = [k for k in range(len(games)) if k not in ended]
            if not playing:
                break
            steps = agents.act([screens[k] for k in playing], playing)
            for k, step in zip(playing, steps):
                screens[k], outcome = games[k].step(step.button)
                furthest[k] = max(furthest[k], outcome.level_x)
                if outcome.dying or outcome.finished or frame == frames - 1:
                    ended[k] = FullSMBRun(
                        levels[k], frame + 1, furthest[k], outcome.dying, outcome.finished
                    )
    finally:
        for game in games:
            game.close()
    return [ended[k] for k in range(len(games))]
