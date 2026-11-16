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
