"""Block Pong's rules (retroagi/domains/pong/block_pong.py), without the
emulator."""

import time

from retroagi.domains.pong.block_pong import PongState, band, step
from retroagi.domains.pong.game import DOWN, STAY, UP
from retroagi.domains.pong.knob import Knob


def test_the_ball_crosses_a_pixel_a_frame_and_turns_at_the_walls():
    state = PongState(ball_x=120, ball_y=50, across=1, speed=-2, ours=40, drawn=40)
    ys = []
    for _ in range(6):
        step(state, STAY)
        ys.append(state.ball_y)
    assert state.ball_x == 126
    assert ys == [48, 46, 44, 47, 49, 51]  # a pixel more on the frame it turns


def test_our_paddle_turns_the_ball_by_where_it_meets_it():
    for meet, speed in ((0, -2), (2, -1), (6, 0), (10, 1), (14, 2)):
        state = PongState(ball_x=188, ball_y=120 + meet, across=1, speed=0, ours=120, drawn=120)
        step(state, STAY)
        assert (state.across, state.speed) == (-1, speed), meet
        assert band(meet) == speed


def test_a_missed_ball_is_their_point_and_is_served_back_toward_us():
    state = PongState(ball_x=195, ball_y=100, across=1, speed=1, ours=200, drawn=200)
    points, frames = 0, 0
    while state.ball_y == 0 or state.ball_x != 129:
        frames += 1
        points += step(state, STAY).point
        assert frames < 200
    assert points == -1 and state.points == (0, 1)
    assert (state.across, state.speed, state.ball_y) == (1, 1, 129)  # down from 129, as it was going


def test_the_opponent_follows_the_ball_and_rests_every_eighth_frame():
    state = PongState(ball_x=120, ball_y=150, across=-1, speed=0, theirs=100, frame=0)
    moves = []
    for _ in range(16):
        before = state.theirs
        step(state, STAY)
        moves.append(state.theirs - before)
    assert moves == [0] + [2] * 7 + [0] + [2] * 7


def test_the_knob_moves_our_paddle_every_second_frame():
    state = PongState(ball_x=120, ball_y=100, ours=181, drawn=181, frame=1, knob=Knob())
    for _ in range(20):
        step(state, UP)
    assert state.ours < 100
    for _ in range(40):
        step(state, DOWN)
    assert state.ours == 219


def test_block_pong_is_fast():
    state = PongState(ball_x=120, ball_y=100, across=1, speed=1)
    start = time.perf_counter()
    for frame in range(20000):
        step(state, (STAY, UP, DOWN)[frame // 7 % 3])
    assert (time.perf_counter() - start) / 20000 < 0.0001  # under 0.1 ms a frame
