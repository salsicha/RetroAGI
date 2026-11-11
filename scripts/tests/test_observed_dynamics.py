"""The dynamics model identified from what is seen
(framework/observed_dynamics.py): things here are plain boxes with tracks;
the model is told nothing of what they are."""

from dataclasses import dataclass
from typing import Optional

import pytest

from retroagi.domains.smb.planner import RULES
from retroagi.framework import registry
from retroagi.framework.observed_dynamics import ObservedDynamics

FLOOR = 208


@dataclass
class Thing:
    box: tuple
    vx: float = 0.0
    vy: float = 0.0
    ref: object = 0
    track: Optional[tuple] = None
    type: object = "a"
    path: object = None


@dataclass
class Solid:
    x0: float
    top: float
    x1: float
    bottom: float
    kind: str = "ground"


GROUND = Solid(0, FLOOR, 1000, 240)


def shown(model, places, solids, width=10, height=6, kind="a", ref=0, given=(0.0, 0.0)) -> Thing:
    """``model`` shown a thing at each of ``places`` (its box's left and
    top, a frame each) in turn, as a state source would give it; the thing
    as last shown."""
    for t in range(len(places)):
        x, y = places[t]
        thing = Thing((x, y, x + width, y + height), *given, ref, tuple(places[max(0, t - 31):t + 1]), kind)
        model.observe([thing], solids)
    return thing


def walk(x, y, speed, frames):
    """Whole-pixel places of a thing moving sideways at ``speed``."""
    return [(float(round(x + speed * t)), float(y)) for t in range(frames)]


def test_a_walker_moves_at_the_speed_its_track_shows_and_turns_at_a_wall():
    model = ObservedDynamics()
    wall = Solid(60, 160, 92, 240)
    # (its box ends 4 above the floor it walks on; the source's own estimate is wrong)
    thing = shown(model, walk(140, FLOOR - 10, -0.5, 24), [GROUND, wall], given=(0.3, 0.0))
    model.paths([thing], [GROUND, wall], 120)
    x_now = thing.box[0]
    assert thing.path[16][0] == pytest.approx(x_now - 8, abs=1) and thing.path[16][1] == FLOOR - 10
    assert min(p[0] for p in thing.path) >= wall.x1 - 1  # it never enters the wall
    assert thing.path[120][0] > thing.path[100][0]  # and walks back from it


def test_a_short_track_leaves_the_sources_estimate_standing():
    model = ObservedDynamics(measured=8)
    thing = shown(model, walk(140, FLOOR - 10, 0.0, 3), [GROUND], given=(-0.5, 0.0))
    model.paths([thing], [GROUND], 16)
    assert thing.path[16][0] == pytest.approx(140 - 8)


def test_a_type_seen_to_fall_falls_off_an_end_as_it_was_seen_to():
    model = ObservedDynamics(gravity=0.5)
    ledge = Solid(100, 100, 200, 116)
    # One thing of the type seen falling in the open at 0.2 a frame, a frame.
    shown(model, [(300.0, float(round(20 + 0.1 * t * t))) for t in range(20)], [GROUND], ref=1)
    assert model.seen["a"].gravity == pytest.approx(0.2, abs=0.03)
    # Another of the type walks off the ledge's end: it falls like the first.
    walker = shown(model, walk(120, 90, -0.5, 24), [GROUND, ledge], ref=2)
    model.paths([walker], [GROUND, ledge], 110)
    off = next(k for k, p in enumerate(walker.path) if p[1] > 90)
    assert walker.path[off + 10][1] - 90 == pytest.approx(0.1 * 11 * 11, abs=3)
    # A type not yet seen falling is taken to fall as the types seen do.
    other = Thing(walker.box, -0.5, 0.0, 3, walker.track, "b")
    model.observe([other], [GROUND, ledge])
    model.paths([other], [GROUND, ledge], 110)
    assert other.path[off + 10][1] == walker.path[off + 10][1]
    # With nothing seen to fall yet, by the prior (0.5): faster.
    fresh = ObservedDynamics(gravity=0.5)
    fresh.observe([other], [GROUND, ledge])
    fresh.paths([other], [GROUND, ledge], 110)
    assert other.path[off + 10][1] > walker.path[off + 10][1] + 8
    # Both land on the ground under the ledge (their feet 4 below their boxes, as on it) and walk on.
    assert walker.path[110][3] == pytest.approx(FLOOR - 4, abs=1) and other.path[110][3] == walker.path[110][3]
    assert walker.path[110][0] < walker.path[100][0]


def test_a_bounce_is_learned_from_seeing_one():
    model = ObservedDynamics()
    # Hops as a Paratroopa's: up at 3 a frame from the floor, 0.1 a frame heavier each frame.
    places, x, y, speed = [], 400.0, FLOOR - 15.0, -3.0
    for _ in range(150):
        places.append((float(round(x)), float(round(y))))
        x -= 0.5
        speed += 0.1
        y += speed
        if y >= FLOOR - 15:
            y, speed = FLOOR - 15.0, -3.0
    hopper = shown(model, places, [GROUND], width=12, height=12)
    seen = model.seen["a"]
    assert seen.gravity == pytest.approx(0.1, abs=0.02) and seen.bounce == pytest.approx(3.0, abs=0.4)
    assert seen.feet == pytest.approx(3, abs=1.5) and not seen.through
    # Its next landing is followed by a rise, not a rest.
    model.paths([hopper], [GROUND], 90)
    lowest = max(range(91), key=lambda k: hopper.path[k][1])
    assert lowest < 80 and hopper.path[lowest + 8][1] < hopper.path[lowest][1] - 12


def test_a_thing_coming_out_of_a_solid_stops_where_its_type_was_seen_to():
    model = ObservedDynamics()
    pipe = Solid(300, 144, 332, 240)
    # A plant: up out of the pipe at half a pixel a frame until its box ends 4 above the pipe, then still.
    rise = [(311.0, float(round(142 - 0.5 * t))) for t in range(17)] + [(311.0, 134.0)] * 12
    shown(model, rise, [GROUND, pipe], ref=1)
    seen = model.seen["a"]
    assert seen.through and seen.out == 4 and seen.pace == pytest.approx(0.5, abs=0.05)
    # The next one of its type, only just seen in its pipe: predicted up and out to there.
    second = Thing((411.0, 142.0, 421.0, 148.0), 0.0, 0.0, 2, ((411.0, 142.0),), "a")
    other_pipe = Solid(400, 144, 432, 240)
    model.observe([second], [GROUND, other_pipe])
    model.paths([second], [GROUND, other_pipe], 48)
    assert second.path[8][1] == pytest.approx(138, abs=1) and second.path[48][3] == 140


def test_a_thing_in_the_air_that_does_not_fall_flies_through_everything():
    model = ObservedDynamics()
    pipe = Solid(300, 144, 332, 240)
    bullet = shown(model, walk(400, 170, -1.5, 16), [GROUND, pipe], width=14, height=10)
    model.paths([bullet], [GROUND, pipe], 96)
    assert bullet.path[96][0] == pytest.approx(bullet.box[0] - 144, abs=2) and bullet.path[96][1] == 170


def test_super_mario_bros_takes_either_dynamics_model_by_name():
    domain = registry.load("smb")
    assert domain.controller().dynamics is RULES
    observed = registry.load("smb", ("dynamics=observed",))
    assert isinstance(observed.controller().dynamics, ObservedDynamics)
    assert observed.controller().dynamics is not observed.controller().dynamics  # (a new one each play)
    assert registry.used(observed) == ("dynamics=observed",) and domain.chosen == {"dynamics": "rules"}
    with pytest.raises(ValueError):
        registry.load("smb", ("dynamics=guessed",))
    # A practice level plays through with it (the simulator gives no tracks: its own speeds stand).
    from retroagi.framework.agent import Agent, play

    world = observed.practice(100003)
    agent = Agent(observed.truth_source(), observed.behaviour, observed.controller, observed.idle)
    agent.reset(0, "speed_run")
    assert play(world, agent, 600)["frames"] > 100


def test_a_vision_track_gives_its_places_through_a_turn():
    from retroagi.domains.smb.perception.tracks import Track

    track = Track((10, 20, 26, 36), "walker", moves=[(-1, 0), (-1, 0), (1, 0), (1, 2)])
    assert track.places(100.0, 50.0) == ((100.0, 48.0), (99.0, 48.0), (98.0, 48.0), (99.0, 48.0), (100.0, 50.0))
