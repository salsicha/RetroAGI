"""No kind of action may end at one value whatever the scene.

The families that teach jumping over or past a pit or an enemy start Mario at
varied distances from it (right next to it included) and follow it with a
second pit or enemy at a varied distance. Teacher qualification fails when
one end of a verb covers most of that verb's labels over every family
together, and a training round whose skill picks one verb far more often
than its teacher would at the same moments cannot pass its gate.
"""

import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import random
from types import SimpleNamespace

import numpy as np
import pytest

from retroagi.core.action_tokens import ACTION_WIDTH, VERBS
from retroagi.stages.block_smb.layered_train import (
    choices_against_teacher,
    label_collapse,
    output_collapse,
    top_share,
)
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


# Layouts of one-move lessons with an enemy are kept only once the teacher
# wins them (monte_carlo._teacher_wins): seconds each.
@pytest.mark.timeout(1200)
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


def episode(family, verbs, xs, *, labelled=True):
    """Decisions with these verbs, ending xs pixels ahead (the teacher's
    labels where ``labelled``; the learner's own actions otherwise)."""
    count = len(verbs)
    verb = np.array([VERBS.index(v) for v in verbs])
    labels = {
        "verb": verb,
        "pointer": np.zeros(count, int),
        "action_valid": np.ones(count, bool),
        "end_dx": np.array(xs),
        "end_dy": np.zeros(count, int),
        "valid": np.ones(count, bool) if labelled else np.zeros(count, bool),
    }
    used = np.zeros((count, ACTION_WIDTH), np.float32)
    used[np.arange(count), verb] = 1.0
    used[:, len(VERBS)] = np.array(xs) / 256
    return SimpleNamespace(
        family=family,
        labels=labels,
        picks={"verb": verb, "pointer": np.zeros(count, int)},
        used=used,
        played_teacher=np.full(count, labelled),
    )


def test_one_end_for_a_verb_over_every_family_collapses():
    assert top_share([(72, 0), (73, 0), (70, -2), (40, 0)]) == ((72, 0), 0.75)
    assert top_share([72, 73, 70, 40]) == (72, 0.75)
    varied = [episode("a", ["jump"] * 4, [40, 52, 61, 75]) for _ in range(1)]
    varied += [episode("a", ["jump"] * 4, [44, 57, 66, 80])]
    # A family may teach one end where its scenes call for it: judged over
    # every family's labels together, one end of a verb among varied ones is
    # not a collapse...
    single = [episode("b", ["jump", "run"], [72, 30 + i * 9]) for i in range(2)]
    assert label_collapse(varied + single, "skill") == {}
    # ...one end for most of a verb's labels over every family is.
    single = [episode("b", ["jump", "run"], [72, 30 + i * 9]) for i in range(12)]
    found = label_collapse(varied + single, "skill")["all families"]
    assert found["jump"][0] == (72, 0) and "run" not in found
    # Waits have no end: their length is what varies.
    waits = [episode("c", ["wait"] * 4, [0] * 4) for _ in range(12)]
    assert "wait" not in label_collapse(varied + waits, "skill").get("all families", {})


def own_choices(family, verbs, teacher_verbs, same=None):
    """Held-out decisions the skill played itself (``verbs``) next to the
    teacher's verb at the same moments; ``same``: whether each was the
    teacher's choice (verb and object; default: the verbs are equal)."""
    e = episode(family, verbs, [0] * len(verbs), labelled=False)
    e.teacher_verb = np.array([VERBS.index(v) for v in teacher_verbs])
    e.same_as_teacher = np.array(
        same if same is not None else [a == b for a, b in zip(verbs, teacher_verbs)]
    )
    return e


def test_a_skill_that_picks_one_verb_where_the_teacher_would_not_collapses():
    # Judged at the same moments as the teacher's choices (its run lengths
    # are the teacher's: carry_out), not by where its actions ended.
    teacher = ["run", "jump", "run", "run"]
    own = [own_choices("a", ["jump"] * 4, teacher) for _ in range(3)]
    found = output_collapse(own, [], "skill")
    assert found == {"a": {"jump": (None, 1.0, 0.25)}}
    # Picking what the teacher picks there is no collapse, however often.
    same = [own_choices("a", ["jump"] * 4, ["jump"] * 4) for _ in range(3)]
    assert output_collapse(same, [], "skill") == {}


def test_the_skills_choices_are_compared_with_the_teachers_at_the_same_moments():
    episodes = [
        own_choices("enemy_hop", ["jump", "run"], ["jump", "run"]),
        # The same verb aimed at another object is not the teacher's choice.
        own_choices("enemy_hop", ["jump", "run"], ["jump", "run"], [True, False]),
        own_choices("enemy_hop", ["jump", "jump"], ["jump", "run"]),
    ]
    # A decision the teacher played, or where it had no named choice, is not
    # the skill's own: left out.
    episodes[0].played_teacher = np.array([True, False])
    episodes[2].teacher_verb[0] = -1
    report = choices_against_teacher(episodes)
    assert report == {
        "enemy_hop": {
            "decisions": 4,
            "agreed": 0.5,
            "differ": {"run at another object": 1, "jump for run": 1},
        }
    }
