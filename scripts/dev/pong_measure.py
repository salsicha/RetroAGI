"""How the real Pong behaves, read from its memory: what Block Pong must
copy (docs/pong.md). Nothing learns.

Plays the right-hand paddle with a player that aims where the ball will
arrive (an offset drawn at random each rally so every part of the paddle
is met), records the game's memory every frame, and prints:

- the ball's moves a frame, and where it turns at the walls;
- where it turns at each paddle, and its vertical speed after, by where on
  the paddle it was met;
- how the opponent's paddle follows the ball;
- the serve: where, which way, how long after a point;
- how our paddle answers the buttons: taps of each length from rest, and
  the knob it follows.

    python scripts/dev/pong_measure.py [--frames 80000] [--output OUT.json]
"""

import argparse
import json
from collections import Counter, defaultdict

import numpy as np

from retroagi.domains.pong.game import DOWN, STAY, UP, RealPong

# The walls the ball turns at (its top, game units), for the aiming player.
TOP, BOTTOM = 44, 207
OUR_X = 188  # where the ball turns at our paddle


def arrival(x, y, vx, vy):
    """Where (height) a ball at (x, y) moving (vx, vy) a frame meets our
    paddle's column, bouncing between the walls (None: going away)."""
    if vx <= 0:
        return None
    span = BOTTOM - TOP
    h = (y + vy * (OUR_X - x) - TOP) % (2 * span)
    return TOP + (h if h <= span else 2 * span - h)


def record(frames: int, seed: int = 2) -> np.ndarray:
    """[frames, 8]: ball x, ball y, our paddle, theirs, knob, point, action, frame."""
    game, rng = RealPong(), np.random.default_rng(seed)
    game.reset()
    rows, previous, offset = [], None, 0
    for frame in range(frames):
        m = game.memory()
        target = 125
        if previous is not None and m.ball_x and previous[0]:
            vx, vy = m.ball_x - previous[0], m.ball_y - previous[1]
            if vx < 0 and frame % 7 == 0:
                offset = int(rng.integers(-11, 12))
            y = arrival(m.ball_x, m.ball_y, vx, vy)
            if y is not None:
                target = y - 6 - offset
        previous = (m.ball_x, m.ball_y)
        action = DOWN if target > m.our_paddle + 2 else UP if target < m.our_paddle - 2 else STAY
        _, point, over = game.step(action)
        m = game.memory()
        rows.append((m.ball_x, m.ball_y, m.our_paddle, m.their_paddle, m.knob, point, action, frame))
        if over:
            game.reset()
    game.close()
    return np.array(rows)


def ball(rows) -> dict:
    bx, by = rows[:, 0], rows[:, 1]
    steps, walls = Counter(), Counter()
    for i in range(1, len(bx) - 1):
        if bx[i] and bx[i - 1] and abs(bx[i] - bx[i - 1]) < 3 and abs(by[i] - by[i - 1]) < 4:
            steps[(int(bx[i] - bx[i - 1]), int(by[i] - by[i - 1]))] += 1
            e0, e1 = by[i] - by[i - 1], by[i + 1] - by[i]
            if e0 and e1 and np.sign(e0) != np.sign(e1) and abs(e1) < 4:
                walls[int(by[i])] += 1
    return {"moves a frame (dx, dy)": {str(k): v for k, v in steps.most_common()},
            "heights it turns at": dict(walls.most_common(6))}


def paddles(rows) -> dict:
    """Vertical speed after a turn (pixels a frame, over the next 4 frames),
    by the ball's height minus the paddle's top, for each side."""
    bx, by, ours, theirs = rows[:, 0], rows[:, 1], rows[:, 2], rows[:, 3]
    sides = {"ours": defaultdict(Counter), "theirs": defaultdict(Counter)}
    turn_x = {"ours": Counter(), "theirs": Counter()}
    for i in range(2, len(bx) - 6):
        d0, d1 = bx[i] - bx[i - 1], bx[i + 1] - bx[i]
        if d0 and d1 and np.sign(d0) != np.sign(d1) and abs(d0) < 3 and abs(d1) < 3:
            side = "ours" if d0 > 0 else "theirs"
            top = ours[i] if side == "ours" else theirs[i]
            sides[side][int(by[i] - top)][(by[i + 5] - by[i + 1]) / 4] += 1
            turn_x[side][int(bx[i])] += 1
    out = {}
    for side, table in sides.items():
        out[side] = {"turns at x": dict(turn_x[side].most_common(3)),
                     "speed after, by ball minus paddle top": {
                         k: max(table[k], key=table[k].get) for k in sorted(table) if sum(table[k].values()) >= 3}}
    return out


def opponent(rows) -> dict:
    bx, by, theirs = rows[:, 0], rows[:, 1], rows[:, 3]
    moves, pattern = Counter(), []
    for i in range(1, len(bx) - 1):
        if bx[i] and abs(theirs[i + 1] - theirs[i]) <= 2:
            moves[(int(np.sign(by[i] - theirs[i])), int(theirs[i + 1] - theirs[i]))] += 1
    chase = [int(theirs[i + 1] - theirs[i]) for i in range(1, len(bx) - 1)
             if bx[i] and abs(by[i] - theirs[i]) > 20 and abs(theirs[i + 1] - theirs[i]) <= 2]
    still = sum(1 for m in chase if m == 0) / max(1, len(chase))
    return {"move a frame by side of the ball (side, move)": {str(k): v for k, v in moves.most_common(6)},
            "share of frames it rests while chasing": round(still, 3)}


def serves(rows) -> dict:
    bx, by, point = rows[:, 0], rows[:, 1], rows[:, 5]
    out, delays = Counter(), Counter()
    for p in np.nonzero(point)[0]:
        j = p + 1
        while j < len(bx) - 8 and not (by[j] and by[j + 1] and bx[j + 1] != bx[j]):
            j += 1
        if j < len(bx) - 8:
            out[(int(point[p]), int(bx[j]), int(by[j]), int(bx[j + 1] - bx[j]), int(np.sign(by[j + 5] - by[j + 1])))] += 1
            delays[int(j - p)] += 1
    return {"(point, x, y, x step, up or down)": {str(k): v for k, v in out.most_common(8)},
            "frames from a point to the serve": dict(delays.most_common(3))}


def knob() -> dict:
    """Our paddle's move after a tap of each length from rest, and the
    knob readings it followed."""
    game = RealPong()
    game.reset()
    for _ in range(121):
        game.step(STAY)
    start = game.save()
    out = {}
    for frames in (1, 2, 3, 4, 5, 6, 8, 10, 14, 20):
        for name, action in (("up", UP), ("down", DOWN)):
            game.load(start)
            for _ in range(20):
                game.step(STAY)
            before, reads = game.memory().our_paddle, []
            for _ in range(frames):
                game.step(action)
                reads.append(game.memory().knob)
            for _ in range(20):
                game.step(STAY)
                reads.append(game.memory().knob)
            out[f"{name} {frames}"] = {"moved": game.memory().our_paddle - before, "knob": reads[1::2][:8]}
    game.close()
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--frames", type=int, default=80000)
    parser.add_argument("--output")
    args = parser.parse_args(argv)
    rows = record(args.frames)
    result = {"frames": args.frames, "points": int((rows[:, 5] != 0).sum()), "won": int((rows[:, 5] > 0).sum()),
              "ball": ball(rows), "paddles": paddles(rows), "opponent": opponent(rows), "serves": serves(rows),
              "knob": knob()}
    text = json.dumps(result, indent=1, default=str)
    if args.output:
        open(args.output, "w").write(text)
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
