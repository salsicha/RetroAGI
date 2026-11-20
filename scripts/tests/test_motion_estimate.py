"""Vision's estimate of Mario's motion (perception/estimator.py) against the
game's rules."""

from types import SimpleNamespace

from retroagi.domains.smb.perception.estimator import MotionEstimator


def _scene(standing: bool, top: int = 176, left: int = 100, right: int = 112, pipes=()):
    mario = SimpleNamespace(box=(left, top, right, top + 16), on_something=standing,
                            support="ground" if standing else "air")
    return SimpleNamespace(mario=mario, enemies=[], moving_platforms=[], blocks=[], surfaces=[], uncut_surfaces=[], gaps=[],
                           coins=[], power_ups=[], pipes=list(pipes))


def _pressed_on(pipes) -> int:
    """Mario's modelled speed after he runs (40) into what is ahead of his
    right edge (x 72, left of where the screen scrolls, so the camera is
    still) and stays there, his running sprite changing width each frame
    (its right edge still)."""
    estimate = MotionEstimator()
    estimate.motion.x_speed = 40
    for frame in range(8):
        estimate.observe(_scene(standing=True, left=59 + frame % 2, right=72, pipes=pipes))
    return estimate.motion.x_speed


def test_standing_stops_the_fall():
    """Seen standing, Mario does not fall (the game's landing and Block SMB
    zero his vertical speed every frame he stands): a model left at the
    fastest fall would tell the outcome network a standing Mario falls."""
    estimate = MotionEstimator()
    estimate.motion.y_speed, estimate.motion.y_force = 4, 112
    estimate.observe(_scene(standing=True))
    assert (estimate.motion.y_speed, estimate.motion.y_force) == (0, 0)


def test_a_jump_seen_a_frame_late_is_kept():
    """A jump the model has just started (rising) is not undone by a screen
    that still shows Mario on the ground."""
    estimate = MotionEstimator()
    estimate.motion.y_speed = -4
    estimate.observe(_scene(standing=True))
    assert estimate.motion.y_speed == -4


def test_a_jump_seen_a_frame_late_keeps_the_air_top_speed():
    """Jumping at the walking top speed (24), Mario keeps it in the air
    while right and run are held, as the game does (World 1-1, the jump
    across the pit between the last staircases). The screen shows the
    takeoff a frame late; taken for standing that frame, the model ran him
    on the ground to 25, which unlocks the running top speed in the air, and
    it had him at 30 where the game kept 24 (a flight check then let go of
    right and he hit the far staircase's side)."""
    estimate = MotionEstimator()
    estimate.motion.x_speed, estimate.motion.moving, estimate.motion.facing = 24, 1, 1
    estimate.executed(2, _scene(standing=True))  # jump and right pressed, standing
    estimate.executed(2, _scene(standing=True, top=172))  # rising, still shown standing
    for _ in range(10):
        estimate.executed(2, _scene(standing=False, top=150))
    assert estimate.motion.x_speed == 24


def test_in_the_air_the_fall_goes_on():
    estimate = MotionEstimator()
    estimate.motion.y_speed, estimate.motion.y_force = 3, 40
    estimate.observe(_scene(standing=False, top=120))
    assert (estimate.motion.y_speed, estimate.motion.y_force) == (3, 40)


def test_a_wall_stops_the_run():
    """Pressed against a pipe, Mario has stopped (the game's wall contact),
    though his changing sprite never lets the speed samples fill."""
    assert _pressed_on([(73, 160, 105, 208)]) == 0


def test_no_wall_no_stop():
    """With nothing solid just ahead the model's run is not cut off by a
    box that seems still (a scroll measured wrong did that in open
    running)."""
    assert _pressed_on([]) == 40
