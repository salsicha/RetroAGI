"""The real Pong in the emulator (retroagi/domains/pong/game.py), when its
cartridge has been imported."""

import pytest

retro = pytest.importorskip("retro")


def _game():
    from retroagi.domains.pong.game import GAME, RealPong

    try:
        retro.data.get_romfile_path(GAME)
    except FileNotFoundError:
        pytest.skip("the Pong cartridge is not imported")
    return RealPong()


def test_the_knob_moves_our_paddle():
    from retroagi.domains.pong.game import DOWN, UP

    game = _game()
    try:
        game.reset()
        for _ in range(130):
            game.step(UP)
        high = game.memory().our_paddle
        for _ in range(60):
            game.step(DOWN)
        low = game.memory().our_paddle
    finally:
        game.close()
    assert high == 38 and low == 219  # the paddle's two ends


def test_the_ball_is_served_and_crosses_a_pixel_a_frame():
    from retroagi.domains.pong.game import STAY

    game = _game()
    try:
        game.reset()
        xs = []
        for _ in range(200):
            game.step(STAY)
            xs.append(game.memory().ball_x)
    finally:
        game.close()
    moving = [b - a for a, b in zip(xs, xs[1:]) if a and b and a != b]
    assert moving and all(abs(step) == 1 for step in moving)


def test_the_knob_and_paddle_follow_the_buttons_exactly():
    """Block Pong's knob (retroagi/domains/pong/knob.py) against the game's
    memory, frame by frame, over random holds and taps that reach both ends.
    A new emulator starts with the knob's step not yet grown."""
    import numpy as np

    from retroagi.domains.pong.knob import Knob, follow

    rng = np.random.default_rng(3)
    actions = []
    while len(actions) < 3000:
        actions += [int(rng.integers(0, 3))] * int(rng.integers(1, 30))
    game = _game()
    try:
        game.reset()
        knob, paddle = Knob(step=0), game.memory().our_paddle
        for frame, action in enumerate(actions):
            game.step(action)
            knob.press(action)
            seen = game.memory()
            if frame % 2 == 0:
                assert seen.knob == knob.reading(), frame
                paddle = follow(paddle, knob.reading())
            assert seen.our_paddle == paddle, frame
    finally:
        game.close()


def test_block_pong_behaves_as_the_game_frame_by_frame():
    """Block Pong put in the game's state before every frame, both given the
    same action (scripts/dev/pong_compare.py): our paddle, theirs, the serve
    and the score agree on every frame; the ball on all but the rare frame
    where it grazes our paddle's top rim."""
    _game().close()
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).parents[1] / "dev"))
    from pong_compare import compare

    result = compare(30000, seed=4)
    wrong = result["share wrong"]
    assert all(wrong[name] == 0 for name in ("ours", "drawn", "theirs", "wait", "points")), wrong
    assert all(wrong[name] < 0.0005 for name in ("ball_x", "ball_y", "across", "speed")), wrong
