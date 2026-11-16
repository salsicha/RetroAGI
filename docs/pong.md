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

`scripts/dev/pong_measure.py` plays the game and reads its memory every
frame; the game's memory also holds the ball's direction across (0xBA), its
vertical speed (0xB8), our paddle where it is drawn (0xB3, a frame behind),
a frame count (0x88) and the frames left before a serve (0x8B). In the
game's units (pixels; heights grow downward; one step a frame), in the
order the game applies them each frame:

| Rule | Measured |
|---|---|
| The ball across | 1 pixel a frame, always |
| The ball up or down | its speed: 0, 1, 2 or 3 pixels a frame, up or down |
| The walls | going up at height 45 or less, or down at 206 or more, it turns, with a pixel more that frame |
| Our paddle | turns the ball while it is at x 188 to 192 and its height, rounded down to even (the game draws in two-scanline rows), is from 2 above the paddle's drawn top to 15 below (from 4 above for a ball falling 2 a frame from an odd height) |
| Their paddle | turns it at x 64 to 68, from 4 above its top to 14 below |
| The bounce | by the ball's height minus the paddle's top: 4 or more above sends it 3 a frame up, −3 to 0 two up, 1 to 4 one up, 5 to 8 level, 9 to 12 one down, 13 and below two down. A paddle that moves onto a ball still in its reach turns it there |
| The opponent | moves 2 pixels a frame toward lining its top up with the ball's height after the ball's move (toward 0 while the ball is out of play), except when the frame count is a multiple of 8: a ball moving 2 a frame up or down outruns it. At a serve it is first put at 128 |
| A point | the ball reaching x 205 (past us) or 52 (past them) scores; it stays there a frame, then is out of play 63 frames |
| The serve | toward the side that lost the point (toward them at the game's start), from x 127 (toward them) or 129 (toward us), one pixel a frame up from height 127 or down from 129, as the ball was going up or down (down if level) |
| Our paddle | on every frame whose count is odd, its top moves halfway to the knob's reading, rounded down: it stays between 38 and 219 |
| The knob | a hidden knob the buttons turn; the game reads it as 38 (top) to 220 (bottom), shown in memory (0x84) on alternate frames |

**The knob, exactly.** The emulator's own source (stable-retro's Stella
core, `Paddles.cxx` and `TIA.cxx`) says how it stands the buttons in for a
paddle, and the game agrees with it on every frame. Holding a button turns
a hidden charge (1 to 3855, 2048 at the start) by a step that grows while a
button is held: 0 on the very first frame, then 1, 2, 3, 4, 5, then 60 a
frame. The step stays at 60 from then on, released or not, so in play every
press moves the knob 60 a frame from its first frame. Up turns the charge
down; a turn that would pass an end is not made. The game times the knob's
capacitor two scanlines at a time: its reading is
`6 + 2 × ((cycles + 135) // 152)`, kept within 38 to 220, where `cycles` is
the emulator's charging time for the charge (its 262 scanlines and 59.92
frames a second). The charge goes on past where the reading stops (38 below
charge 376, 220 from 2472): a knob held past an end takes as many frames to
come back before the paddle moves. The knob is not part of the emulator's
saved state: every reset, and every saved state put back, starts it afresh
at 2048 with the step not yet grown, so our paddle heads for 181 until a
button is pressed.

## Block Pong

`retroagi/domains/pong/block_pong.py` is these rules in plain arithmetic
(`knob.py` the knob's), with nothing of the game loaded: one frame is a
few comparisons and additions, about 0.002 ms on one processor core (a
test keeps it under 0.1 ms).
Its state is a lookup: the ball's place, direction and speed, both
paddles, the knob, the frame count, the serve's wait and the score.

`scripts/dev/pong_compare.py` checks it against the game: before every
frame Block Pong is put in the game's state, both take the same action,
and every part of the state after is compared. Over 200,000 frames, with
the ball turned by our paddle 678 times:

| Part | Frames it disagrees on |
|---|---|
| Our paddle, as it is and as drawn | 0 |
| Their paddle | 0 |
| The serve's wait, the score | 0 |
| The ball's place, direction and speed | 17 |

The 17 are all a ball grazing our paddle's top pixel or two (or meeting
it at a wall), where what the game decides depends on how the hardware
draws the frame: about one paddle meeting in 40 at the rim. A test keeps
the agreement (`scripts/tests/test_pong_game.py`).

## The domain

`retroagi/domains/pong/domain.py` gives the framework what it asks of a
domain ([framework.md](framework.md)); nothing in the framework changed
for it but one thing a domain may now say: how long a choice may last
(`longest`).

**The choices.** While the ball comes toward us: return it with one of
five parts of our paddle (its top edge, upper part, middle, lower part,
bottom edge), which send it back steeply up, up, level, down or steeply
down. A return lasts until the ball comes back (or the point ends), so
what it leads to includes whether the opponent reaches it. Otherwise:
rest at one of three heights until the ball comes toward us.

**The controller** (`controller.py`), in plain arithmetic: it moves our
paddle by choosing the charge to turn the knob to. For a return it tries a
few plans in Block Pong's own rules from the present state (the ball's
flight, the knob, the paddle; the opponent plays no part in the ball's way
to us) and keeps the first that meets the ball with the chosen part: a
charge to turn to, or a charge and then its neighbour from a timed frame,
since a paddle at rest sits only every 5 or 6 pixels and a part is 4 tall.
Starting a return takes under half a millisecond as a rule (3 at the 90th
percentile), pressing on a few microseconds. Of 1,000 returns asked for in
random situations, 993 met the ball with the part chosen and 7 were
refused, 5 of them a part the paddle cannot bring to the ball's height.

**The practice worlds** (`world.py`): one point of Block Pong from a
situation drawn at random from the game's own ranges (a serve to come, or
a ball in flight anywhere, either way, at any of its speeds; the paddles
anywhere; our knob at rest or, as after the real game is reset, fresh),
and the "return" skill scene, passed once our paddle has turned the ball.
Losing the point is a death, winning it finishes the task; each return is
progress.

**The state** is Block Pong's own (a lookup); in the real game it is read
from the game's memory (`real.py`), the hidden knob followed from the
agent's own presses. Seeing for Pong is not built yet.

**What the hand-written chooser does** (`scripted_chooser.py`: every
return with the middle of the paddle). In 200 practice points it lost
none, won 24 and kept the other 176 rallies going until the 4,000-frame
limit: a level return comes straight back. In the real game, from its
memory, it returned the ball 250 times in 60,000 frames without a miss and
without a point either way. Sampling every candidate's outcome in 20
practice points (955 returns and rests, 3.5 seconds on one core) shows
what there is to learn: a return off the paddle's top or bottom edge won
the point 43% and 35% of the time, one off its other parts never. A rule
written by hand from that (always the bottom edge, the top edge where the
bottom cannot be done) wins the real game 21 to 0; the outcome network is
to find it in Block Pong alone.

## Next

1. The outcome network learned in Block Pong (the framework's sampling
   and training, `--domain pong`), on the project owner's go, then scored
   on the real game.
2. Seeing: a vision network trained on the real game's frames, labelled
   from its memory, so the real game is played from the screen.
