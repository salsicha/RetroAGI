"""The choose-and-control controller (docs/choose-and-control-plan.md phase
1): rollouts with the NES motion model checked against the scene, the jump
planner's goals and its "infeasible" answer, and the controller's runs."""

from retroagi.core.smb_controller import Choice, Controller, State, stopping_distance
from retroagi.core.smb_physics import NESPlayerMotion
from retroagi.core.smb_planner import INSET, Enemy, Goal, Solid, plan_jump, rollout

FLOOR = Solid(0, 208, 4000, 240, "ground", ("surfaces", 0))
STANDING = (100.0, 192.0, 112.0, 208.0)  # Mario's body on the floor


def pipe(x0, top, slot=0):
    return Solid(x0, top, x0 + 32, 240, "pipe", ("pipes", slot))


def test_a_rollout_lands_hits_walls_and_bumps_blocks():
    motion = NESPlayerMotion()
    assert rollout(motion, STANDING, 0, 16, 1, [FLOOR], []).end == "landed"
    # A pipe in the way of a low hop: its side stops Mario, who lands back
    # on the floor before it; a high jump pressed against it rises over it.
    hop = rollout(motion, STANDING, 0, 1, 1, [FLOOR, pipe(116, 176)], [])
    assert hop.end == "landed" and hop.solid is FLOOR and hop.body[2] <= 116
    tall = rollout(motion, (104.0, 192.0, 116.0, 208.0), 0, 20, 1, [FLOOR, pipe(116, 176)], [])
    assert tall.end == "landed" and tall.solid.kind == "pipe"
    # A block just above Mario's head: jumping into it bumps it.
    block = Solid(98, 160, 114, 176, "block", ("blocks", 0))
    assert rollout(motion, STANDING, 0, 16, 0, [FLOOR, block], []).end == "bonk"
    # Running on a floor that ends: Mario falls.
    short = Solid(0, 208, 130, 240, "ground", ("surfaces", 0))
    assert rollout(motion, STANDING, 64, 0, 1, [short], []).end == "fell"


def test_the_planner_lands_on_a_pipe_inside_its_ends():
    solids = [FLOOR, pipe(150, 176)]
    plan = plan_jump(Goal("land", ("pipes", 0), 150, 182), NESPlayerMotion(), STANDING, solids, [], 1)
    assert plan is not None and plan.outcome.end == "landed"
    middle = (plan.outcome.body[0] + plan.outcome.body[2]) / 2
    assert 150 + INSET <= middle <= 182 - INSET
    assert plan.outcome.body[3] == 176
    assert len(plan.buttons) == plan.outcome.frame + 1


def test_too_far_is_infeasible():
    solids = [FLOOR, pipe(900, 144)]
    assert plan_jump(Goal("land", ("pipes", 0), 900, 932), NESPlayerMotion(), STANDING, solids, [], 1) is None


def test_stomping_and_passing_a_walker():
    walker = Enemy((170.0, 194.0, 186.0, 208.0), -0.5, 0.0, ("slots", 0))
    stomp = plan_jump(Goal("stomp", ("slots", 0)), NESPlayerMotion(), STANDING, [FLOOR], [walker], 1)
    # The stomp bounces Mario, who then lands.
    assert stomp is not None and stomp.outcome.stomped is walker and stomp.outcome.end == "landed"
    passing = plan_jump(Goal("pass", ("slots", 0)), NESPlayerMotion(), STANDING, [FLOOR], [walker], 1)
    assert passing is not None and passing.outcome.end == "landed"
    # Beyond where the walker is when Mario lands.
    assert passing.outcome.body[0] >= walker.at(passing.outcome.frame + 1)[2]


def test_a_run_stops_at_its_x():
    motion = NESPlayerMotion()
    controller = Controller()
    state = State(STANDING, motion, True, [FLOOR], [])
    controller.start(Choice("run", x=180), state)
    x, frames = STANDING[0], 0
    body = STANDING
    while controller.event is None and frames < 300:
        button = controller.press(State(body, motion, True, [FLOOR], []))
        dx, _, _ = motion.advance(direction={1: 1, 3: -1}.get(button, 0), jump=False, grounded=True, y=192, run=True)
        x += dx
        body = (x, 192.0, x + 12, 208.0)
        frames += 1
    while motion.x_speed:
        dx, _, _ = motion.advance(direction=0, jump=False, grounded=True, y=192, run=False)
        x += dx
    assert controller.event == "arrived"
    assert abs(x + 6 - 180) <= 6
    assert stopping_distance(NESPlayerMotion(), 1) == 0


def test_a_walker_turns_back_at_a_wall():
    from retroagi.core.smb_planner import enemy_paths

    # Walking right at half a pixel a frame into a pipe 20 pixels away: it
    # comes back, as the game turns walkers.
    walker = Enemy((200.0, 194.0, 216.0, 208.0), 0.5, 0.0, ("slots", 0))
    enemy_paths([walker], [FLOOR, pipe(236, 176)], 120)
    xs = [b[0] for b in walker.path]
    assert max(xs) <= 236 - 16 + 0.5 and xs[-1] < xs[60]


def test_the_compiled_motion_matches_the_motion_model():
    import random

    from retroagi.core.smb_planner import _MOTION_FIELDS, _advance, _motion_array

    rng = random.Random(0)
    for trial in range(300):
        motion = NESPlayerMotion(x_speed=rng.randint(-40, 40), moving=rng.choice((-1, 1)), facing=rng.choice((-1, 1)))
        array = _motion_array(motion)
        grounded, y = rng.random() < 0.7, 192.0
        for _ in range(60):
            direction, jump, run = rng.choice((-1, 0, 1)), rng.random() < 0.4, rng.random() < 0.8
            expected = motion.advance(direction=direction, jump=jump, grounded=grounded, y=y, run=run)
            got = _advance(array, direction, jump, grounded, y, run)
            assert tuple(int(v) for v in got) == tuple(int(v) for v in expected)
            assert [int(getattr(motion, f)) for f in _MOTION_FIELDS] == list(array)
            y += expected[1]
            grounded = rng.random() < 0.5
