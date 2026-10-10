"""No family may teach, and no policy may learn, one single value.

The families that teach jumping over or past a pit or an enemy start Mario at
varied distances from it (right next to it included) and follow it with a
second pit or enemy at a varied distance. Teacher qualification fails a
family whose labels collapse on one destination, and a training round whose
policy collapses on one cannot pass its gate.
"""

import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import random
from types import SimpleNamespace

import numpy as np
import pytest

from retroagi.core.tokens import SKILL_MODES, SKILL_X, SKILL_Y
from retroagi.stages.block_smb.layered_train import label_collapse, output_collapse, top_share
from retroagi.stages.block_smb.monte_carlo import sample_block_smb_monte_carlo_scenario
from retroagi.stages.block_smb.threat_variety import fit_patrols, follow_up, start_distance


def test_a_pit_cut_after_the_threat_bounds_the_landing():
    rng = random.Random(0)
    for _ in range(40):
        scenario = {"platforms": [[0, 220, 100, 20], [140, 220, 216, 20]], "enemies": []}
        kind, room, detail = follow_up(rng, scenario, 1, 140, 1, kinds=("pit",), window=(24, 72))
        first, rest = scenario["platforms"][1], scenario["platforms"][2]
        assert kind == "pit" and first == [140, 220, room, 20] and 12 <= room <= 72
        assert rest[0] == 140 + room + detail and rest[0] + rest[2] == 356 and rest[2] >= 32


def test_an_enemy_after_the_threat_patrols_only_its_own_floor():
    rng = random.Random(1)
    scenario = {"platforms": [[0, 220, 100, 20], [140, 220, 216, 20]], "enemies": []}
    kind, room, _ = follow_up(rng, scenario, 1, 140, 1, kinds=("enemy",))
    x, y, low, high, speed, direction = scenario["enemies"][0]
    assert kind == "enemy" and x == 140 + room and y + 14 == 220
    assert (low, high) == (140, 346) and direction in (-1, 1)
    scenario["enemies"].append([20, 206, 0, 999, 0.5, 1])
    fit_patrols(scenario)
    assert scenario["enemies"][1][2:4] == [0, 90]


def test_mario_often_starts_right_next_to_the_threat():
    rng = random.Random(2)
    distances = [start_distance(rng, 40) for _ in range(400)]
    assert min(distances) == 0 and max(distances) == 40
    assert 0.3 < sum(d <= 8 for d in distances) / len(distances) < 0.75


@pytest.mark.parametrize(
    "family",
    ["action_jump_gap", "skill_enemy_bypass", "single_gap", "enemy_hop", "action_platform_up"],
)
def test_threat_families_vary_where_mario_starts_and_what_follows(family):
    starts, kinds = set(), set()
    for index in range(12):
        sample = sample_block_smb_monte_carlo_scenario(
            split="train",
            seed=0,
            sample_index=index,
            family=family,
            difficulty="medium",
            calibrate_deadline=False,
        )
        starts.add(round(sample.scenario["mario"][0]))
        kinds.add(sample.parameters.get("then"))
    assert len(starts) >= 6, starts
    if family != "action_platform_up":
        assert {"pit", "enemy"} <= kinds, kinds


def episode(family, modes, xs, *, labelled=True):
    count = len(modes)
    indices = {
        "mode": np.array([SKILL_MODES.index(m) for m in modes]),
        "x": np.array([SKILL_X.index(x) for x in xs]),
        "y": np.full(count, SKILL_Y.index(0)),
    }
    labels = {**indices, "valid": np.ones(count, bool) if labelled else np.zeros(count, bool)}
    return SimpleNamespace(family=family, labels=labels, picks=dict(indices))


def test_a_family_whose_labels_repeat_one_destination_collapses():
    assert top_share([(72, 0), (73, 0), (70, -2), (40, 0)]) == ((72, 0), 0.75)
    assert top_share([72, 73, 70, 40]) == (72, 0.75)
    varied = [episode("a", ["jump"] * 4, [40, 52, 61, 75]) for _ in range(1)]
    varied += [episode("a", ["jump"] * 4, [44, 57, 66, 80])]
    single = [episode("b", ["jump", "run"], [72, 30 + i * 9]) for i in range(8)]
    found = label_collapse(varied + single, "skill")
    assert "a" not in found and found["b"]["jump"][1] == 1.0 and "run" not in found["b"]


def test_a_policy_that_repeats_one_value_its_teacher_did_not_collapses():
    taught = [episode("a", ["jump"] * 4, [40 + 9 * i + j for i in range(4)]) for j in range(3)]
    own = [episode("a", ["jump"] * 4, [72, 72, 73, 72], labelled=False) for _ in range(3)]
    assert output_collapse(own, taught, "skill")["a"]["jump"][1] > 0.9
    # Repeating what the teacher repeats is not a collapse of the policy.
    same = [episode("a", ["jump"] * 4, [72, 72, 73, 72]) for _ in range(3)]
    assert output_collapse(own, same, "skill") == {}
