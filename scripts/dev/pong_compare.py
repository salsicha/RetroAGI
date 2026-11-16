"""Block Pong against the real game, frame by frame: the same situation set
up in both, the same action taken, what follows compared. A measuring
device for making Block Pong behave as the game does (docs/pong.md);
nothing learns.

The real game is played by a player that aims where the ball will arrive
(the part of the paddle drawn at random each rally) and now and then
presses at random. Before every frame Block Pong is put in the game's state
(read from its memory; the hidden knob is followed from the emulator's
start and from each reset, knob.py), both take the frame's action, and each part of the state
after is compared. It prints the share of frames each part disagrees on,
and the first disagreements of each.

    python scripts/dev/pong_compare.py [--frames 100000] [--seed 0]
"""

import argparse
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from pong_measure import arrival  # noqa: E402

from retroagi.domains.pong.block_pong import PongState, step  # noqa: E402
from retroagi.domains.pong.game import DOWN, STAY, UP, RealPong  # noqa: E402
from retroagi.domains.pong.knob import Knob  # noqa: E402
from retroagi.domains.pong.real import state_from_memory  # noqa: E402

PARTS = ("ball_x", "ball_y", "across", "speed", "ours", "drawn", "theirs", "wait", "points")


def parts(state: PongState, m=None) -> dict:
    if m is None:
        return {name: getattr(state, name) for name in PARTS}
    return {"ball_x": m.ball_x, "ball_y": m.ball_y, "across": m.across, "speed": m.speed, "ours": m.our_paddle,
            "drawn": m.our_drawn, "theirs": m.their_paddle, "wait": m.serve_wait,
            "points": (m.our_score, m.their_score)}


def player(rng):
    previous, offset = [None], [0]

    def act(m, frame):
        target = 125
        if previous[0] is not None and m.ball_x and previous[0][0] and m.ball_y:
            vx, vy = m.ball_x - previous[0][0], m.ball_y - previous[0][1]
            if vx < 0 and frame % 7 == 0:
                offset[0] = int(rng.integers(-11, 12))
            y = arrival(m.ball_x, m.ball_y, vx, vy)
            if y is not None:
                target = y - 6 - offset[0]
        previous[0] = (m.ball_x, m.ball_y)
        action = DOWN if target > m.our_paddle + 2 else UP if target < m.our_paddle - 2 else STAY
        return int(rng.integers(0, 3)) if rng.random() < 0.05 else action

    return act


def compare(frames: int, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    game, act = RealPong(), player(rng)
    game.reset()
    knob = Knob(step=0)  # a new emulator's knob
    wrong, examples, compared, turns = Counter(), defaultdict(list), 0, 0
    for frame in range(frames):
        m = game.memory()
        action = act(m, frame)
        block = state_from_memory(m, Knob(knob.charge, knob.step, knob.held))
        step(block, action)
        _, _, over = game.step(action)
        knob.press(action)
        after = game.memory()
        compared += 1
        turns += m.across == 1 and after.across == -1
        theirs, ours = parts(block), parts(None, after)
        for name in PARTS:
            if theirs[name] != ours[name]:
                wrong[name] += 1
                if len(examples[name]) < 5:
                    examples[name].append({"frame": frame, "before": parts(None, m), "action": action,
                                           "block": theirs, "game": ours})
        if over:
            game.reset()
            knob = Knob(step=0)  # every reset (and every saved state put back) starts the knob afresh
    game.close()
    return {"frames": compared, "our paddle's turns": turns,
            "share wrong": {name: wrong[name] / compared for name in PARTS},
            "examples": dict(examples)}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--frames", type=int, default=100000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--examples", action="store_true")
    args = parser.parse_args(argv)
    result = compare(args.frames, args.seed)
    turns = result["our paddle's turns"]
    print(f"{result['frames']} frames, the ball turned by our paddle {turns} times")
    for name, share in result["share wrong"].items():
        print(f"{name:8} {share:.5f}  ({round(share * result['frames'])} frames)")
    if args.examples:
        for name, cases in result["examples"].items():
            for case in cases:
                print(name, case)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
