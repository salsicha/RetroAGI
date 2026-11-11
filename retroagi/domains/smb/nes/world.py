"""The real game as a world (framework/interfaces.World): one level, or a
saved state of it, in the emulator (game.FullSMBGame). Played and
measured, never learned on.

What a vision-only agent may read is ``screen`` (the picture after the last
step). The rest (the game's memory through ``game``, progress, score) is
for the game-memory state source, for scoring and for building benches.
"""

from typing import Optional

from retroagi.framework.interfaces import Step

from .game import FullSMBGame
from .pixel_labels import PLAYER_FLOAT_STATE
from .truth import true_frame


class NESWorld:
    def __init__(self, level: str, start_state: Optional[bytes] = None):
        self.level = level
        self.game = FullSMBGame(level, start_state)
        self.screen = None
        self.progress = 0  # Mario's distance into the level (pixels)

    def reset(self) -> None:
        self.screen = self.game.reset()
        self.progress = self.game.outcome().level_x

    def step(self, action: int) -> Step:
        self.screen, outcome = self.game.step(action)
        self.progress = outcome.level_x
        return Step(over=outcome.dying or outcome.finished, died=outcome.dying, finished=outcome.finished)

    @property
    def settled(self) -> bool:
        """Mario stands on something."""
        return int(self.game.env.get_ram()[PLAYER_FLOAT_STATE]) == 0

    @property
    def score(self) -> int:
        return true_frame(self.game.env.get_ram(), self.game.start)["score"]

    def close(self) -> None:
        self.game.close()
