"""The action layer's training layouts: every combination of each family's parameters."""

import itertools

import numpy as np
import pytest

from retroagi.stages.block_smb.monte_carlo import (
    CombinationDraws,
    block_smb_parameter_combinations,
)


def every_combination(make, levels=3):
    """Run ``make(draws)`` for every combination; returns what each run made."""
    made, path = [], []
    while path is not None:
        draws = CombinationDraws(path, levels)
        made.append(make(draws))
        path = draws.next_path()
    return made


def test_every_combination_of_the_draws_is_made_once():
    made = every_combination(lambda d: (d.randint(0, 10), d.choice("xy"), d.uniform(0.0, 1.0)))
    expected = itertools.product((0, 5, 10), ("x", "y"), (0.0, 0.5, 1.0))
    assert sorted(made) == sorted(expected)  # both ends of every range, and the middle


def test_small_ranges_take_every_value():
    made = every_combination(lambda d: (d.randint(3, 4), d.randrange(0, 6, 3)))
    assert sorted(made) == [(3, 0), (3, 3), (4, 0), (4, 3)]


def test_draws_that_depend_on_earlier_ones_are_followed_on_every_branch():
    def make(d):
        if d.choice(("one", "two")) == "one":
            return ("one", d.randint(1, 3))
        return ("two", d.randint(7, 8), d.choice("ab"))

    assert sorted(every_combination(make)) == sorted(
        [("one", 1), ("one", 2), ("one", 3)] + [("two", a, b) for a in (7, 8) for b in "ab"]
    )


def test_a_family_sweep_covers_the_ends_of_its_ranges_with_verified_routes():
    layouts, dropped = block_smb_parameter_combinations("single_gap", "easy")
    assert len(layouts) == 27 and dropped == 0
    assert len({repr((layout["platforms"], layout["coins"])) for layout in layouts}) == 27
    gaps = [layout["platforms"][1][0] - sum(layout["platforms"][0][::2]) for layout in layouts]
    assert len(set(gaps)) > 1
    for layout in layouts:
        meta = layout["metadata"]["block_smb_monte_carlo"]
        assert meta["family"] == "single_gap" and meta["reachability"]["reachable"]
        assert meta["oracle"]["actions"]


class _MapPool:
    """Stands in for the worker pool: its pool's map runs in this process."""

    pool = type("Inline", (), {"map": staticmethod(map)})


def test_each_family_weighs_the_same_in_the_action_layers_training():
    from retroagi.stages.block_smb.layered_train import LayeredTrainConfig, combination_tasks

    config = LayeredTrainConfig(learner="action", families=("flat_run", "single_gap"))
    tasks, made = combination_tasks(config, _MapPool())
    sizes = {f: sum(t.family == f for t in tasks) for f in config.families}
    assert sizes == {"flat_run": 27, "single_gap": 81}  # 9 and 27 per difficulty
    totals = {f: sum(t.weight for t in tasks if t.family == f) for f in config.families}
    assert totals["flat_run"] == pytest.approx(totals["single_gap"])
    assert np.mean([t.weight for t in tasks]) == pytest.approx(1.0, rel=0.5)
    assert all(t.scenario is not None and t.label for t in tasks)
    assert made["single_gap:hard"] == {"layouts": 27, "no_route": 0}


def test_a_family_split_by_its_first_draws_makes_the_same_layouts():
    from retroagi.stages.block_smb.monte_carlo import combination_prefixes

    whole, _ = block_smb_parameter_combinations("single_gap", "medium")
    prefixes = combination_prefixes("single_gap", "medium", at_least=4)
    assert len(prefixes) >= 4
    parts = [
        layout
        for prefix in prefixes
        for layout in block_smb_parameter_combinations("single_gap", "medium", prefix=prefix)[0]
    ]

    def key(layout):
        return repr({k: v for k, v in layout.items() if k != "metadata"})

    assert sorted(map(key, parts)) == sorted(map(key, whole))
