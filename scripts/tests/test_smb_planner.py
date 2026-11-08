"""The choose-and-control controller (docs/choose-and-control-plan.md phase
1): rollouts with the NES motion model checked against the scene, the jump
planner's goals and its "infeasible" answer, and the controller's runs."""

from copy import copy

from retroagi.core.smb_controller import Choice, Controller, State, meets_soon, stopping_distance
from retroagi.core.smb_physics import NESPlayerMotion
from retroagi.core.smb_planner import (
    INSET,
    Enemy,
    Goal,
    Solid,
    enemy_paths,
    floor_behind,
    plan_flight,
    plan_jump,
    rollout,
    screen_edge,
)

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
    plan = plan_jump(Goal("land", ("pipes", 0), 150, 182, top=176), NESPlayerMotion(), STANDING, solids, [], 1)
    assert plan is not None and plan.outcome.end == "landed"
    middle = (plan.outcome.body[0] + plan.outcome.body[2]) / 2
    assert 150 + INSET <= middle <= 182 - INSET
    assert plan.outcome.body[3] == 176
    assert len(plan.buttons) == plan.outcome.frame + 1
    # The same pipe in another slot (relabelled after a scroll) is the same goal.
    renumbered = [FLOOR, pipe(150, 176, slot=3)]
    assert plan_jump(Goal("land", ("pipes", 0), 150, 182, top=176), NESPlayerMotion(), STANDING, renumbered, [], 1)


def test_too_far_is_infeasible():
    solids = [FLOOR, pipe(900, 144)]
    assert plan_jump(Goal("land", ("pipes", 0), 900, 932, top=144), NESPlayerMotion(), STANDING, solids, [], 1) is None


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


def test_a_flight_is_corrected_when_an_enemy_wakes_where_it_lands():
    walker = Enemy((170.0, 194.0, 186.0, 208.0), -0.5, 0.0, ("slots", 0))
    goal = Goal("pass", ("slots", 0), side=1)
    plan = plan_jump(goal, NESPlayerMotion(), STANDING, [FLOOR], [walker], 1)
    # Played to the middle of the flight.
    motion, (x0, y0, x1, y1), airborne = NESPlayerMotion(), STANDING, False
    for button in plan.buttons[: plan.pre_run + 12]:
        dx, dy, jumped = motion.advance(direction=1, jump=button in (2, 4, 5), grounded=not airborne, y=y0, run=True)
        airborne = airborne or jumped
        x0, x1, y0, y1 = x0 + dx, x1 + dx, y0 + (dy if airborne else 0), y1 + (dy if airborne else 0)
    body, frame = (x0, y0, x1, y1), plan.pre_run + 12
    walker_now = Enemy(walker.at(frame), -0.5, 0.0, ("slots", 0))
    rest = plan.buttons[frame:]
    hold_left = next((k for k, b in enumerate(rest) if b not in (2, 4, 5)), len(rest))
    # Nothing new: the jump goes on as planned.
    aim = (plan.outcome.body[0] + plan.outcome.body[2]) / 2
    same = plan_flight(goal, copy(motion), body, [FLOOR], [walker_now], 1, hold_left, aim)
    assert same.buttons == rest[: len(same.buttons)]
    # Something that cannot be stomped woken where Mario comes down: the
    # correction lands clear of it.
    landing = (plan.outcome.body[0] + plan.outcome.body[2]) / 2
    woken = Enemy((landing - 8, 194.0, landing + 8, 208.0), 0.0, 0.0, ("slots", 1), stompable=False)
    as_planned = rollout(copy(motion), body, 0, hold_left, 1, [FLOOR], [walker_now, woken], airborne=True)
    assert as_planned.end == "hit"
    corrected = plan_flight(goal, copy(motion), body, [FLOOR], [walker_now, woken], 1, hold_left, aim)
    assert corrected is not None and corrected.outcome.end == "landed"
    assert corrected.buttons != rest[: len(corrected.buttons)]


def test_a_floor_gap_an_enemy_stands_in_is_floor():
    left, right = Solid(500, 208, 697, 240, "ground", ("surfaces", 0)), Solid(713, 208, 900, 240, "ground", ("surfaces", 1))
    goomba = Enemy((700.0, 198.0, 710.0, 204.0), -0.5, 0.0, ("slots", 0))
    floor = [s for s in floor_behind([left, right], [goomba]) if s.kind == "ground"]
    assert [(s.x0, s.x1, s.ref) for s in floor] == [(500, 900, ("surfaces", 0))]
    # As vision boxes it: drawn, its last row one below the floor's top.
    seen = Enemy((697.0, 193.0, 713.0, 209.0), -0.5, 0.0, ("enemies", 4))
    assert len(floor_behind([left, right], [seen])) == 1
    # Two side by side hide a gap two of them wide.
    wide = Solid(737, 208, 900, 240, "ground", ("surfaces", 1))
    pair = [goomba, Enemy((723.0, 198.0, 733.0, 204.0), -0.5, 0.0, ("slots", 1))]
    assert len(floor_behind([left, wide], pair)) == 1
    assert len(floor_behind([left, wide], [goomba])) == 2
    # A real pit, or an enemy elsewhere, keeps the gap.
    assert len(floor_behind([left, right], [])) == 2
    falling = Enemy((700.0, 220.0, 710.0, 226.0), -0.5, 1.0, ("slots", 0))
    assert len(floor_behind([left, right], [falling])) == 2


def test_a_bounce_can_stomp_the_next_walker_too():
    # Two Goombas close together: the first stomp's bounce comes down on the
    # second, a second stomp, and Mario lands beyond them.
    first = Enemy((170.0, 194.0, 186.0, 208.0), -0.5, 0.0, ("slots", 0))
    second = Enemy((194.0, 194.0, 210.0, 208.0), -0.5, 0.0, ("slots", 1))
    plan = plan_jump(Goal("stomp", ("slots", 0)), NESPlayerMotion(), STANDING, [FLOOR], [first, second], 1)
    assert plan is not None and plan.outcome.end == "landed" and plan.outcome.stomped is first
    assert plan.outcome.body[0] > second.at(plan.outcome.frame)[2]


def test_a_run_ends_when_an_enemy_will_meet_mario_soon():
    running = NESPlayerMotion(x_speed=32, moving=1)  # two pixels a frame
    near = Enemy((172.0, 194.0, 188.0, 208.0), -0.5, 0.0, ("slots", 0))
    far = Enemy((400.0, 194.0, 416.0, 208.0), -0.5, 0.0, ("slots", 1))
    behind = Enemy((40.0, 194.0, 56.0, 208.0), -0.5, 0.0, ("slots", 2))
    above = Enemy((172.0, 130.0, 188.0, 144.0), -0.5, 0.0, ("slots", 3))
    state = State(STANDING, running, True, [FLOOR], [near])
    assert meets_soon(state, near)
    assert not any(meets_soon(state, e) for e in (far, behind, above))
    controller = Controller()
    controller.start(Choice("run", x=600, keep_speed=True), state)
    assert controller.press(state) == 1 and controller.event == "threat"  # still running on
    clear = State(STANDING, running, True, [FLOOR], [far, behind, above])
    controller.start(Choice("run", x=600, keep_speed=True), clear)
    controller.press(clear)
    assert controller.event is None


def test_a_walker_falls_off_a_ledge_and_walks_off_the_screen():
    # A Goomba on a brick row (its box 4 above its feet) walking left off
    # its end falls to the floor and walks on, past the screen's edge.
    row = Solid(100, 80, 148, 96, "block", ("blocks", 0))
    goomba = Enemy((104.0, 70.0, 114.0, 76.0), -0.5, 0.0, ("slots", 0))
    enemy_paths([goomba], [FLOOR, row, screen_edge(40.0)], 240)
    assert goomba.at(4)[3] == 76.0  # still on the row
    assert goomba.at(120)[3] == 204.0  # on the floor
    assert goomba.at(240)[0] < 40.0  # walked on past the edge
    # One already in the air keeps its own motion.
    flying = Enemy((200.0, 100.0, 210.0, 106.0), -0.5, 0.0, ("slots", 1))
    enemy_paths([flying], [FLOOR], 20)
    assert flying.at(20)[1] == 100.0
