"""Full SMB: the game gives screens, the agent gives buttons.

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
from typing import Optional

import numpy as np

from retroagi.domains.smb.actions import SMBAction, full_smb_action

from .pixel_labels import canonical_rgb
from .vision_frames import ENGINE_ROUTINE, _dying

WORLD, LEVEL = 0x75F, 0x75C  # [WorldNumber], [LevelNumber]
# Levels kept out of any Full SMB policy training, to test the agent on.
# The vision transformer learns from every level (vision_frames.LEVELS).
POLICY_TEST_LEVELS = ("Level1-1", "Level5-1")
PAGE, X_ON_PAGE = 0x6D, 0x86  # [Player_PageLoc], [Player_X_Position]
RUN_ACTIONS = (SMBAction.RIGHT, SMBAction.RIGHT_JUMP)
# [GameEngineSubroutine] while Mario is played (not entering, dying or
# between levels).
PLAYING = 8
# [GameEngineSubroutine] while Mario slides down the flagpole, and while the
# level ends after it (his walk into the castle, the timer counted into the
# score): the level is done. The level number changes only after all that,
# hundreds of frames on (user, 2026-10-29: a 1-1 playthrough that took the
# flag was logged as stalled at the castle).
FLAGPOLE_SLIDE, END_OF_LEVEL = 4, 5


def level_finished(ram, start) -> bool:
    """Whether the level is done: Mario on the flagpole or past it, or in
    another level than ``start`` (world, level) (a pipe to the next one)."""
    routine = int(ram[ENGINE_ROUTINE])
    return routine in (FLAGPOLE_SLIDE, END_OF_LEVEL) or (int(ram[WORLD]), int(ram[LEVEL])) != tuple(start)


@dataclass(frozen=True)
class FullSMBOutcome:
    """Read from game memory, for scoring only."""

    level_x: int  # Mario's distance into the level, in pixels
    dying: bool
    finished: bool  # the flagpole taken, or the level changed (a pipe to the next one)


class FullSMBGame:
    """One saved level start of Super Mario Bros in the emulator, or (with
    ``start_state``) a level start captured in play (next_level_state)."""

    def __init__(self, level: str, start_state: Optional[bytes] = None):
        import retro

        self.level = level
        self.start_state = start_state
        # A captured start has no saved state of its own: the game's default
        # is loaded, then replaced at reset.
        self.env = retro.make(
            "SuperMarioBros-Nes",
            state=level if start_state is None else retro.State.DEFAULT,
            inttype=retro.data.Integrations.STABLE,
            # No viewer window: retro's default draws one every step (7% of a
            # frame's time) that nobody looks at.
            render_mode=None,
        )
        self.buttons = list(self.env.buttons)
        self.start: tuple[int, int] = (0, 0)
        self.raw: Optional[np.ndarray] = None  # the emulator's last picture, unpadded

    def _screen(self, observation) -> np.ndarray:
        return np.ascontiguousarray(canonical_rgb(np.asarray(observation)[..., :3]))

    def outcome(self) -> FullSMBOutcome:
        ram = self.env.get_ram()
        return FullSMBOutcome(
            level_x=int(ram[PAGE]) * 256 + int(ram[X_ON_PAGE]),
            dying=_dying(ram),
            finished=level_finished(ram, self.start),
        )

    def reset(self) -> np.ndarray:
        result = self.env.reset()
        observation = result[0] if isinstance(result, tuple) else result
        if self.start_state is not None:
            self.env.em.set_state(self.start_state)
            observation, *_ = self.env.step(np.zeros(len(self.buttons), np.int8))
        self.raw = np.asarray(observation)
        ram = self.env.get_ram()
        self.start = (int(ram[WORLD]), int(ram[LEVEL]))
        return self._screen(observation)

    def step(self, action: int) -> tuple[np.ndarray, FullSMBOutcome]:
        action = SMBAction(action)
        buttons = full_smb_action(action, self.buttons).astype(np.int8)
        if action in RUN_ACTIONS:
            buttons[self.buttons.index("B")] = 1
        observation, *_ = self.env.step(buttons)
        self.raw = np.asarray(observation)
        return self._screen(observation), self.outcome()

    def next_level_state(self, limit: int = 2000) -> Optional[tuple]:
        """After a level is finished: (the next level's name, "Level<world>-
        <level>", and the emulator's state once Mario is played in it, its
        intro and entry passed with no buttons pressed): the start a later
        playthrough of it begins from. None if that does not come within
        ``limit`` frames."""
        idle = np.zeros(len(self.buttons), np.int8)
        for _ in range(limit):
            ram = self.env.get_ram()
            here = (int(ram[WORLD]), int(ram[LEVEL]))
            if here != self.start and int(ram[ENGINE_ROUTINE]) == PLAYING:
                return f"Level{here[0] + 1}-{here[1] + 1}", self.env.em.get_state()
            self.env.step(idle)
        return None

    def close(self) -> None:
        self.env.close()
