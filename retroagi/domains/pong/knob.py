"""Our paddle and the buttons, in plain arithmetic: exactly as the real
game and its emulator have them (measured 2026-11-15, docs/pong.md).

The cartridge is played with a paddle: a knob turned by hand. The emulator
stands the buttons in for the knob: holding a button turns a hidden knob
(its "charge", 1 to 3855, starting at 2048) by a step that grows while a
button is held, 0 on the very first frame and then 1, 2, 3, 4, 5 and 60 a
frame. The step stays at 60 from then on, released or not, until the
game is reset or a saved state is put back (both start the knob afresh, at
2048: the knob is not part of the saved state): in play every press turns
the knob 60 a frame from its first frame. Up turns the charge down, down turns it up; a turn
that would pass an end is not made.

The game times how long the knob's capacitor takes to charge, two
scanlines at a time, and reads a number from 38 (top) to 220 (bottom); the
reading is shown in the game's memory every second frame. On those frames
our paddle's top moves halfway to the reading, rounded down: it stays
between 38 and 219. Past the reading's ends the charge goes on (to 1 or
3855): a knob turned past an end needs as many frames to come back before
the paddle moves again.
"""

from dataclasses import dataclass

import numpy as np

from .game import DOWN, STAY, UP

START, TOP = 2048, 3856  # the charge at the start, and the bound it stays under
FULL = 60  # the step a frame once it has grown
GROWING = 5  # the step grows 1 a frame up to this, then jumps to FULL
# The emulator's capacitor timing: its resistance for a charge, and the cycles
# the capacitor needs at the emulator's 262 scanlines and 59.92 frames a second.
RESISTANCE, RANGE = 1400000, 4096
CYCLES = 1.216e-6 * 262 * float(np.float32(59.92))
# The game's reading: two scanlines (152 cycles) a count, from where its loop starts.
LOOP, LOOP_START, READ_FROM = 152, 135, 6
HIGHEST, LOWEST = 38, 220  # the reading's ends (the paddle's top: 38 to 219)


def reading(charge: int) -> int:
    """The number the game reads from a knob at ``charge``."""
    resistance = int(np.float32(RESISTANCE) * (np.float32(charge) / np.float32(RANGE)))
    needed = int(CYCLES * resistance)
    return min(LOWEST, max(HIGHEST, READ_FROM + 2 * ((needed + LOOP_START) // LOOP)))


@dataclass
class Knob:
    """The hidden knob: its charge, its step, whether a button was held last frame."""

    charge: int = START
    step: int = FULL
    held: bool = False

    def press(self, action: int) -> None:
        """One frame of ``action`` (STAY, UP or DOWN)."""
        if self.held:
            self.step = self.step + 1 if self.step < GROWING else FULL
        self.held = action != STAY
        if action == UP and self.charge > self.step:
            self.charge -= self.step
        elif action == DOWN and self.charge + self.step < TOP:
            self.charge += self.step

    def reading(self) -> int:
        return reading(self.charge)


def follow(paddle: int, knob_reading: int) -> int:
    """Our paddle's top after a frame on which the game reads the knob."""
    return (paddle + knob_reading) // 2
