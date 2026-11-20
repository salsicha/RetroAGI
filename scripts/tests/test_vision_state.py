"""The planner's state made from what vision shows (perception/state.py)."""

from types import SimpleNamespace

from retroagi.domains.smb.perception.state import perceived
from retroagi.domains.smb.physics import NESPlayerMotion

HOPPER = 5  # (vision's type of a Paratroopa that hops: state.TYPE_KINDS)


def _seen(track):
    """The state's enemies when vision shows Mario standing at the screen's
    left and ``track``."""
    mario = SimpleNamespace(box=(40, 193, 52, 208), on_something=True, support="ground")
    scene = SimpleNamespace(mario=mario, enemies=[], moving_platforms=[], blocks=[], surfaces=[], uncut_surfaces=[],
                            gaps=[], coins=[], power_ups=[], pipes=[])
    feedback = SimpleNamespace(camera_position=0.0, tracks=SimpleNamespace(tracks=[track]), feet_velocity=0.0,
                               motion=NESPlayerMotion(), motion_ready=True)
    return perceived(feedback, scene).enemies


def test_a_hopper_just_bounced_is_seen_rising():
    """A Paratroopa that has fallen 12 frames and bounced 3 frames ago is
    rising, as the game's memory shows at once. The bounce reverses its
    course and starts its averaged samples again; read from them it was
    taken to stand still for 7 frames, and a flight over it at 8-1 was
    pulled back short of it, where it came down on Mario."""
    moves = [(-0.5, 2.5)] * 12 + [(-0.5, -2.5)] * 3
    track = SimpleNamespace(kind="other", type=HOPPER, box=(190, 190, 202, 202), missed=0, samples=moves[-3:],
                            velocity=(-0.5, -2.5), identity=3, moves=moves,
                            places=lambda x, y: ((x, y),))
    (hopper,) = _seen(track)
    assert hopper.kind == "hopper"
    assert hopper.vy < -0.5  # (-0.95: the game memory's reading 3 frames after a bounce)
    assert abs(hopper.vx + 0.5) < 0.01
