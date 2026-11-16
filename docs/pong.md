# Pong: the second domain

Pong (the Atari 2600 cartridge) is the second game the framework is to
play, after Super Mario Bros. The same rules hold as for Mario
([requirements.md](requirements.md)): the real game is used to train
seeing, to measure how the game behaves so that a fast practice world
behaves the same, and to score the finished system; the policy learns in
the practice world, Block Pong, alone.

## The real game

`retroagi/domains/pong/game.py` plays the cartridge in stable-retro's
emulator (game `Pong-Atari2600`). The agent plays the right-hand paddle
with three actions: nothing, knob up, knob down. The game's memory gives
the ball's left and top, each paddle's top, the knob reading and both
scores; it is read for measuring and scoring only.

The cartridge image comes from the Arcade Learning Environment's package
(`ale-py` on PyPI, which carries the Atari 2600 cartridges for research
use): its `pong.bin` matches stable-retro's checksum and is imported once
with `python -m retro.import FOLDER`.

## How the game behaves (measured 2026-11-15)

`scripts/dev/pong_measure.py` plays 80,000 frames and reads the game's
memory every frame. In the game's units (pixels; heights grow downward):

| Rule | Measured |
|---|---|
| The ball across | 1 pixel a frame, always |
| The ball up or down | 0, 1 or 2 pixels a frame (3 for a frame at a wall turn) |
| The walls | it turns at heights 44 (top) and 207 (bottom) |
| Our paddle | turns the ball at x 188; 16 pixels tall from its top |
| Their paddle | turns the ball at x 68 |
| The bounce off a paddle | by the ball's height minus the paddle's top: −4 to 0 sends it 2 a frame up, 1 to 4 one up, 5 to 8 level, 9 to 12 one down, 13 to 16 two down |
| The opponent | moves 2 pixels a frame toward lining its paddle's top up with the ball, resting one frame in eight (1.75 a frame): a ball moving 2 a frame up or down outruns it |
| A point | the ball leaves past x 205 (theirs) or 52 (ours) |
| The serve | 63 frames after a point, from the middle (x 127 to 129), toward the side that lost the point, 1 pixel a frame up from height 127 or down from 129; the direction changes seldom |
| Our paddle and the buttons | the buttons turn a hidden knob (memory 0x84, read every other frame); the paddle moves every second frame halfway to the knob's reading, rounded down, between 38 and 219 |
| The knob | held, it settles at about 5 pixels a frame after a start that taps of up to 3 frames never get past (4 frames down, 7 up from rest at 182); held past an end it runs on beyond what is read and needs as long to come back |

The knob's start is the emulator's smoothing of a joystick standing in
for a paddle knob; it is the next thing to measure exactly, since the
controller will plan the paddle's moves with it.

## Next

1. The knob's response exactly (its start and its run past the ends).
2. Block Pong: these rules in plain arithmetic, drawn as boxes, its state
   by lookup; random serves and opponents as the practice worlds; checked
   against the game by setting up the same situation in both.
3. The domain's parts behind the framework's interfaces: the choices
   (where to meet the ball, with which part of the paddle), the
   controller that moves the paddle there, the outcome network's tokens.
4. Seeing: a vision network trained on the real game's frames, labelled
   from its memory.
