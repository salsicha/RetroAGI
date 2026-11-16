"""Vision's estimate of Mario's motion (perception/estimator.py) against the
game's rules."""

from types import SimpleNamespace

from retroagi.domains.smb.perception.estimator import MotionEstimator


def _scene(standing: bool, top: int = 176):
    mario = SimpleNamespace(box=(100, top, 112, top + 16), on_something=standing,
                            support="ground" if standing else "air")
    return SimpleNamespace(mario=mario, enemies=[], moving_platforms=[])


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


def test_in_the_air_the_fall_goes_on():
    estimate = MotionEstimator()
    estimate.motion.y_speed, estimate.motion.y_force = 3, 40
    estimate.observe(_scene(standing=False, top=120))
    assert (estimate.motion.y_speed, estimate.motion.y_force) == (3, 40)
