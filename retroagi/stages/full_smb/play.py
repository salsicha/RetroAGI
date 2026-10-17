"""Full SMB for the layered agent: the game gives screens, the agent gives buttons.

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
from typing import Optional, Sequence

import numpy as np

from retroagi.core.actions import SMBAction, full_smb_action

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


@dataclass(frozen=True)
class FullSMBOutcome:
    """Read from game memory, for scoring only."""

    level_x: int  # Mario's distance into the level, in pixels
    dying: bool
    finished: bool  # the level has changed: the flagpole or a pipe to the next level


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
            finished=(int(ram[WORLD]), int(ram[LEVEL])) != self.start,
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


@dataclass(frozen=True)
class FullSMBRun:
    level: str
    frames: int
    furthest_x: int
    died: bool
    finished: bool


def play_full_levels(agents, levels: Sequence[str], frames: int) -> list[FullSMBRun]:
    """Each level played once from its start by copy 0 of ``agents``
    (core.smb_agent.SMBAgents), one after another (the emulator allows one
    game a process), until Mario dies, leaves the level or ``frames`` frames
    pass."""
    runs = []
    for level in levels:
        game = FullSMBGame(level)
        try:
            screen = game.reset()
            agents.reset(0)
            furthest = 0
            for frame in range(frames):
                step = agents.act([screen], [0])[0]
                screen, outcome = game.step(step.button)
                furthest = max(furthest, outcome.level_x)
                if outcome.dying or outcome.finished or frame == frames - 1:
                    runs.append(
                        FullSMBRun(level, frame + 1, furthest, outcome.dying, outcome.finished)
                    )
                    break
        finally:
            game.close()
    return runs
