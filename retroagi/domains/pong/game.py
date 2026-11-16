"""The real Pong: the Atari 2600 cartridge in stable-retro's emulator
(game "Pong-Atari2600"; the cartridge image is imported once with
``python -m retro.import``). The agent plays the right-hand paddle.

reset() and step(action) return the screen. memory() reads what the game
keeps (the ball's place, both paddles, the hidden paddle knob, the score):
for measuring how the game behaves and for scoring only, never for a policy
(docs/requirements.md).
"""

from dataclasses import dataclass

import numpy as np

GAME = "Pong-Atari2600"
# The actions: hold nothing, push the knob up, push it down.
STAY, UP, DOWN = 0, 1, 2
ACTIONS = (STAY, UP, DOWN)
# Memory addresses (the 128 bytes of the console's memory start at 0x80).
BALL_X, BALL_Y, OUR_PADDLE, THEIR_PADDLE, THEIR_SCORE, OUR_SCORE, KNOB = 0xB1, 0xB6, 0xBC, 0xB2, 0x8D, 0x8E, 0x84


@dataclass(frozen=True)
class PongMemory:
    """What the game keeps, in its own units: the ball's left and top (0 while
    it is not in play), each paddle's top, the knob reading (shown every
    other frame; the other frames show the second player's), the scores."""

    ball_x: int
    ball_y: int
    our_paddle: int
    their_paddle: int
    knob: int
    our_score: int
    their_score: int


def read_memory(ram) -> PongMemory:
    at = lambda address: int(ram[address - 0x80])
    return PongMemory(at(BALL_X), at(BALL_Y), at(OUR_PADDLE), at(THEIR_PADDLE), at(KNOB), at(OUR_SCORE),
                      at(THEIR_SCORE))


class RealPong:
    """One game of Pong from its start in the emulator."""

    def __init__(self, state: str = "Start"):
        import retro

        self.env = retro.make(GAME, state=state)
        self.buttons = self.env.buttons
        self._press = {STAY: (), UP: ("UP",), DOWN: ("DOWN",)}

    def _buttons(self, action: int) -> np.ndarray:
        pressed = np.zeros(len(self.buttons), np.int8)
        for name in self._press[action]:
            pressed[self.buttons.index(name)] = 1
        return pressed

    def reset(self) -> np.ndarray:
        screen, _ = self.env.reset()
        return screen

    def step(self, action: int):
        """(screen, point: +1 we scored, -1 they did, 0 neither; game over)."""
        screen, reward, terminated, truncated, _ = self.env.step(self._buttons(action))
        return screen, int(reward), bool(terminated or truncated)

    def memory(self) -> PongMemory:
        return read_memory(self.env.get_ram())

    def save(self) -> bytes:
        return self.env.em.get_state()

    def load(self, state: bytes) -> None:
        self.env.em.set_state(state)

    def close(self) -> None:
        self.env.close()
