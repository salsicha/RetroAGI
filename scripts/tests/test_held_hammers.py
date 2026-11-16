"""A hammer still held over its Hammer Bro's head is foreseen as Block SMB
(and the game) plays it: moving with the Bro until its hold is up, then
thrown toward Mario (planner.held, THROWN_HOLD); told apart from a thrown one
by what is seen: it has never moved up or down."""

from retroagi.domains.smb import planner as P
from retroagi.domains.smb.sim.levels import skill_scene
from retroagi.domains.smb.sim.world import SimSource, SimWorld


def test_held_hammers_are_foreseen_to_the_pixel():
    exact = checked = 0
    for seed in range(20):
        world = SimWorld(skill_scene(seed, "hammer_bros"))
        world.reset()
        source, env = SimSource(), world.env
        for _ in range(400):
            state = source.read(world)
            hammers = [e for e in state.enemies if e.kind == "thrown" and P.held(e)]
            if hammers:
                e = hammers[0]
                P.aim_throws(state.enemies, state.body)
                P.enemy_paths([e], state.solids, 60)
                worst = 0.0
                for k in range(60):
                    h = env.enemies[e.ref[1]]
                    worst = max(worst, abs(h["x"] - e.path[k][0]) + abs(h["y"] - e.path[k][1]))
                    world.step(0)
                checked += 1
                exact += worst < 1.0
                break
            world.step(0)
    assert checked >= 15 and exact / checked >= 0.8, (exact, checked)


def test_a_flying_hammer_is_not_held():
    flying = P.Enemy((0, 0, 8, 8), 1.0, -1.5, ("enemies", 0), kind="thrown", track=((0, 10), (1, 8.5)))
    assert not P.held(flying)
