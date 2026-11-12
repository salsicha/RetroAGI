"""How carefully a choice is carried out is part of the choice (user,
2026-11-12: "make them part of each choice"): every jump and run comes in
each of the controller's ways (controller.CARE), the controller carries
each out its own way, and outcomes sampled for other kinds of choice are
refused in training."""

import numpy as np
import pytest

from retroagi.domains.smb.choices import CARES, KINDS, SPACE
from retroagi.domains.smb.controller import CARE, USUAL, State
from retroagi.domains.smb.domain import DOMAIN
from retroagi.domains.smb.physics import NESPlayerMotion
from retroagi.domains.smb.planner import ENEMY_MARGIN, SAFE_AFTER, Enemy, Goal, Solid, plan_jump

FLOOR = Solid(0, 208, 4000, 240, "ground", ("surfaces", 0))
STANDING = (100.0, 192.0, 112.0, 208.0)


def test_every_jump_and_run_comes_in_each_care():
    goomba = Enemy((180.0, 198.0, 190.0, 204.0), -0.5, 0.0, ("slots", 0))
    step = Solid(160, 176, 176, 240, "ground", ("surfaces", 1))
    state = State(body=STANDING, motion=NESPlayerMotion(), grounded=True, solids=[FLOOR, step], enemies=[goomba])
    options = SPACE.candidates(state)
    for c in options:
        if c.choice.verb == "wait":
            assert c.choice.care == USUAL
        else:
            assert c.modifier % CARES == c.choice.care
    by = {}
    for c in options:
        by.setdefault((KINDS[c.kind], c.ref, c.modifier // CARES), []).append(c.choice.care)
    assert all(sorted(cares) == list(range(CARES)) for (kind, _, _), cares in by.items() if kind != "wait")
    # The usual way is the controller's way before the choice was made of it.
    assert (CARE[USUAL].berth, CARE[USUAL].clear) == (ENEMY_MARGIN, SAFE_AFTER)
    # Every candidate's (kind, modifier) has a head of its own in the network.
    heads = {(c.kind, c.modifier) for c in options}
    for kind, modifier in heads:
        spec = DOMAIN.space.spec.kinds[kind]
        assert modifier < spec.modifiers


def test_a_bold_landing_is_planned_where_a_careful_one_is_not():
    # A Goomba walks toward where Mario lands on the floor ahead: a landing
    # clear of it for a few frames is there, one clear for longer is not.
    found = {}
    for x in range(200, 320, 4):
        goomba = Enemy((float(x), 198.0, x + 10.0, 204.0), -0.5, 0.0, ("slots", 0))
        goal = Goal("land", FLOOR.ref, 150, 170, prefer="near", top=208)
        plans = [plan_jump(goal, NESPlayerMotion(), STANDING, [FLOOR], [goomba], 1, berth=c.berth,
                           safe_after=c.clear) for c in CARE]
        found[x] = [p is not None for p in plans]
    assert any(bold and not careful for bold, _, careful in found.values())
    assert all(bold or not careful for bold, _, careful in found.values())  # careful never where bold is not


def test_outcomes_sampled_for_other_kinds_of_choice_are_refused(tmp_path):
    from retroagi.framework.sampling import sample
    from retroagi.framework.training import Data

    shard = sample(DOMAIN, 5, "speed_run", kind="pit", explore=0.0)
    np.savez(tmp_path / "now.npz", **shard)
    Data(str(tmp_path), DOMAIN.space.spec, DOMAIN.ends, "cpu")  # (its own: taken)
    older = {k: v for k, v in shard.items() if k != "kinds"}  # (from before the kinds were recorded)
    np.savez(tmp_path / "older.npz", **older)
    with pytest.raises(ValueError, match="other kinds of choice"):
        Data(str(tmp_path), DOMAIN.space.spec, DOMAIN.ends, "cpu")
