"""The action layer's training layouts: every combination of every value of each
family's parameters."""

import itertools
import random

import pytest

from retroagi.stages.block_smb.monte_carlo import (
    CombinationDraws,
    ParameterDraws,
    block_smb_parameter_combinations,
    combination_prefixes,
    sample_block_smb_monte_carlo_scenario,
    uniform_values,
)


def every_combination(make):
    """Run ``make(draws)`` for every combination; returns what each run made."""
    made, path = [], []
    while path is not None:
        draws = CombinationDraws(path)
        made.append(make(draws))
        path = draws.next_path()
    return made


def test_every_value_of_every_draw_is_combined_once():
    made = every_combination(lambda d: (d.randint(0, 3), d.choice("xy"), d.uniform(0.0, 0.02)))
    expected = itertools.product(range(4), "xy", (0.0, 0.01, 0.02))
    assert sorted(made) == sorted(expected)


def test_ranges_with_steps_take_every_value():
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


def test_random_draws_take_the_sweeps_values():
    draws = ParameterDraws(random.Random(0))
    for _ in range(200):
        assert draws.uniform(0.4, 0.53) in uniform_values(0.4, 0.53)


def test_a_family_sweep_takes_every_value_with_verified_routes():
    layouts, dropped = block_smb_parameter_combinations("action_jump_gap", "easy")
    assert dropped == 0
    seen = {
        (meta["gap_width"], meta["edge_distance"])
        for meta in (
            layout["metadata"]["block_smb_monte_carlo"]["parameters"] for layout in layouts
        )
    }
    assert seen == set(itertools.product(range(8, 21), range(0, 5)))  # 13 widths x 5 distances
    for layout in layouts:
        meta = layout["metadata"]["block_smb_monte_carlo"]
        assert meta["family"] == "action_jump_gap" and meta["reachability"]["reachable"]


@pytest.mark.timeout(300)
def test_every_random_test_layout_is_one_the_sweep_covers():
    swept = {
        (p["enemy_distance"], p["enemy_speed"])
        for p in (
            layout["metadata"]["block_smb_monte_carlo"]["parameters"]
            for layout in block_smb_parameter_combinations("action_stomp", "hard")[0]
        )
    }
    for index in range(12):
        sample = sample_block_smb_monte_carlo_scenario(
            split="validation", seed=3, sample_index=index, family="action_stomp", difficulty="hard"
        )
        assert (sample.parameters["enemy_distance"], sample.parameters["enemy_speed"]) in swept


def test_a_family_split_by_its_first_draws_makes_the_same_layouts():
    whole, _ = block_smb_parameter_combinations("action_climb", "medium")
    prefixes = combination_prefixes("action_climb", "medium", at_least=4)
    assert len(prefixes) >= 4
    parts = [
        layout
        for prefix in prefixes
        for layout in block_smb_parameter_combinations("action_climb", "medium", prefix=prefix)[0]
    ]

    def key(layout):
        return repr({k: v for k, v in layout.items() if k != "metadata"})

    assert sorted(map(key, parts)) == sorted(map(key, whole))


class _MapPool:
    """Stands in for the worker pool: its pool's map runs in this process."""

    pool = type("Inline", (), {"map": staticmethod(map)})


def test_each_family_weighs_the_same_in_the_action_layers_training(tmp_path, monkeypatch):
    from retroagi.stages.block_smb.layered_train import LayeredTrainConfig, combination_tasks

    monkeypatch.setenv("RETROAGI_COMBINATION_CACHE", str(tmp_path))
    config = LayeredTrainConfig(learner="skill", families=("action_walk", "action_jump_gap"))
    tasks, made = combination_tasks(config, _MapPool())
    sizes = {f: sum(t.family == f for t in tasks) for f in config.families}
    assert sizes == {"action_walk": 177, "action_jump_gap": 165}
    totals = {f: sum(t.weight for t in tasks if t.family == f) for f in config.families}
    assert totals["action_walk"] == pytest.approx(totals["action_jump_gap"])
    assert all(t.scenario is not None and t.label for t in tasks)
    assert made["action_jump_gap:hard"] == {"layouts": 50, "no_route": 0}
    # Made again, the layouts come from the disk.
    again, _ = combination_tasks(config, _MapPool())
    assert [t.scenario for t in again] == [t.scenario for t in tasks]


def test_a_sweep_too_large_to_make_stops_the_run_and_names_the_family():
    from retroagi.stages.block_smb.layered_train import LayeredTrainConfig, combination_tasks

    config = LayeredTrainConfig(learner="skill", families=("action_stomp",), sweep_limit=10)
    with pytest.raises(ValueError, match="action_stomp"):
        combination_tasks(config, _MapPool())


def test_learning_keeps_every_episode_of_the_latest_round():
    from retroagi.stages.block_smb.layered_train import kept_episodes

    older, latest = list(range(100)), list(range(100, 180))
    assert kept_episodes(older, latest, capacity=50) == latest  # never cut the round
    assert kept_episodes(older, latest, capacity=120) == older[-40:] + latest
