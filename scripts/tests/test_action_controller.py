"""Actions to the executor, and following a moving target."""

import numpy as np

from retroagi.core.action_controller import ActionTarget, apply, destination_of, retarget
from retroagi.core.action_tokens import ActionToken
from retroagi.core.smb_ground_control import GroundMove
from retroagi.core.smb_scene_labels import MarioView, SceneObservation
from retroagi.core.smb_spatial_feedback import SpatialFeedback
from retroagi.core.target_tracker import (
    OBJECT_ROW,
    OBJECT_SLOTS,
    ROW,
    ObjectTracker,
    TargetTracker,
)

MARIO = (100, 200, 110, 212)


def scene():
    return SceneObservation(
        mario=MarioView(box=MARIO, facing_right=True, support="ground", on_something=True)
    )


def walker(x):
    rows = np.zeros((OBJECT_SLOTS, len(OBJECT_ROW)), np.float32)
    rows[0, ROW["identity"]] = 1
    rows[0, ROW["x0"] : ROW["y1"] + 1] = (x, 196, x + 16, 212)
    return rows


def test_destinations_follow_the_verbs_mode():
    outcome = {"dx": 49.6, "dy": -10.2}
    assert destination_of(ActionToken("hold"), outcome).mode == "hold"
    stomp = destination_of(ActionToken("stomp", ("enemies", 0)), outcome)
    assert (stomp.mode, stomp.x, stomp.y) == ("jump", 50, -10)
    run = destination_of(ActionToken("run_to", ("surfaces", 0)), {"dx": 900, "dy": 0})
    assert (run.mode, run.x) == ("run", 256)


def test_a_turned_walker_moves_the_runs_destination():
    tracker = ObjectTracker(TargetTracker())  # untrained: steady motion
    spatial = SpatialFeedback()
    spatial.travel = GroundMove(spatial, 60.0, 0.0, MARIO[3])
    # Stomp where the walker will be in 30 frames: Mario's end on its top.
    target = ActionTarget(ActionToken("stomp", ("enemies", 0)), 1, (0.0, 0.0), 30)
    x, moved = 140.0, []
    for frame in range(24):
        x += 1 if frame < 12 else -1  # turns at frame 12
        tracker.observe(walker(x))
        found = retarget(target, scene(), spatial, tracker)
        if found is not None:
            moved.append((frame, found))
    assert moved and 12 <= moved[0][0] <= 18
    frame, (dx, dy) = moved[0]
    # Steady motion before the turn put the walker's middle at 141 + 30 + 8
    # at frame 30; since it turned, its average movement (an untrained
    # tracker continues that) has slowed: the destination moves back.
    before_turn = 141 + 30 + 8 - (MARIO[0] + MARIO[2]) / 2
    assert dx < before_turn - 2
    assert abs(dy - (196 - MARIO[3])) < 1e-3
    apply((dx, dy), scene(), spatial)
    assert spatial.travel.distance == dx and spatial.travel.target is None


def test_a_flight_plans_its_rest_again():
    class Flight:
        goal, done, replanned = (0, 0), False, False

        def replan(self, scene, motion):
            self.replanned = True

    spatial = SpatialFeedback()
    spatial.flight = Flight()
    apply((30.0, -12.0), scene(), spatial)
    assert spatial.flight.goal == (135.0, 200.0) and spatial.flight.replanned
