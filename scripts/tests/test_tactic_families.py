"""Explicit tactics per layout, the route and retreat families, and the clones."""

import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import pytest

from retroagi.core.smb_executor import HOLD_GROUND
from retroagi.core.tokens import BACKWARD_TACTICS
from retroagi.stages.block_smb import tactic_schedule
from retroagi.stages.block_smb.env import MarioScenarioEnv
from retroagi.stages.block_smb.monte_carlo import (
    BLOCK_SMB_MC_FAMILIES,
    sample_block_smb_monte_carlo_scenario,
)
from retroagi.stages.block_smb.tactic_families import (
    CHOICE_FAMILIES,
    CLONE_FAMILIES,
    LOW_CHOICE_FAMILIES,
    TACTIC_FAMILIES,
    TACTIC_TRAINING_FAMILIES,
)
from retroagi.stages.block_smb.tactic_schedule import SCHEDULE_STANCES, segment
from retroagi.stages.block_smb.teacher_tokens import (
    _first_plan,
    episode_teacher,
    teacher_plan,
    teacher_tactic,
)


def _flat(tactics, *, platforms=None, enemies=(), mario=(40, 208)):
    return {
        "world_width": 400,
        "mario": list(mario),
        "platforms": platforms or [[0, 220, 400, 20]],
        "enemies": list(enemies),
        "goal": [300, 200, 16, 20],
        "tactics": tactics,
    }


def _run(env, action, frames):
    info = {}
    for _ in range(frames):
        _, _, done, _, info = env.step(action)
        if done:
            break
    return info


def _sample(family, index=0, difficulty="easy"):
    return sample_block_smb_monte_carlo_scenario(
        split="validation", seed=7, sample_index=index, family=family, difficulty=difficulty
    ).scenario


def test_a_forbidden_platform_ends_the_episode_as_a_loss():
    env = MarioScenarioEnv()
    env.reset(
        scenario=_flat(
            [segment("advance", 1, forbidden=[1])],
            platforms=[[0, 220, 100, 20], [100, 220, 300, 20]],
        )
    )
    info = _run(env, 1, 200)
    assert info["off_route"] and info["terminated"] and not env._goal_credited


def test_leaving_a_hold_area_early_loses_and_holding_it_then_advancing_wins():
    schedule = [segment("hold_area", 1, area=16, frames=40), segment("advance", 1)]
    env = MarioScenarioEnv()
    env.reset(scenario=_flat(schedule))
    assert _run(env, 1, 120)["off_route"] and not env._goal_credited
    env.reset(scenario=_flat(schedule))
    _run(env, 0, 40)
    assert tactic_schedule.current(env)["stance"] == "advance"
    _run(env, 1, 400)
    assert env._goal_credited


def test_the_goal_counts_only_after_the_retreat():
    schedule = [segment("retreat", -1, reach_x=60), segment("advance", 1)]
    env = MarioScenarioEnv()
    env.reset(scenario=_flat(schedule, mario=(120, 208)))
    info = _run(env, 1, 400)
    # Ignoring the retreat eventually scrolls its required line behind the
    # physical camera boundary. That route is now terminal, never a goal.
    assert not env._goal_credited and info["terminated"] and info["off_route"]
    assert env.camera_x > 60
    env.reset(scenario=_flat(schedule, mario=(120, 208)))
    _run(env, 3, 80)
    assert tactic_schedule.current(env)["stance"] == "advance"
    _run(env, 1, 400)
    assert env._goal_credited


@pytest.mark.parametrize("direction", [-1, 1])
def test_holding_ground_tracks_the_platform_and_rejects_walking_or_jumping(direction):
    from retroagi.stages.block_smb.env_state import restore_env_state, snapshot_env_state

    platform = {
        "x": 100,
        "y": 220,
        "w": 64,
        "h": 10,
        "moving": [60, 160, 0.7],
        "direction": direction,
    }
    scenario = _flat([segment("hold_area", area=1)], platforms=[platform], mario=(120, 204))
    env = MarioScenarioEnv()
    try:
        env.reset(scenario=scenario)
        saved = snapshot_env_state(env)
        offset = env.mario["x"] - env.platforms[0]["rect"].x
        teacher = episode_teacher(scenario)
        assert teacher_tactic(env, teacher).stance == "hold_ground"
        assert not _run(env, 0, 80)["off_route"]
        assert env.mario["x"] - env.platforms[0]["rect"].x == pytest.approx(offset, abs=1)
        assert abs(env.mario["x"] - saved["mario"]["x"]) > 10
        for action in (1, 3, 5):
            restore_env_state(env, saved)
            assert _run(env, action, 10)["off_route"]
    finally:
        env.close()


def test_segments_end_in_order_and_routes_follow_the_platform_underfoot():
    platforms = [[0, 220, 400, 20], [80, 196, 40, 24], [120, 172, 40, 48]]
    schedule = [segment("alternate_route", 1, route=[1, 2], on=2), segment("advance", 1)]
    env = MarioScenarioEnv()
    env.reset(scenario=_flat(schedule, platforms=platforms))
    assert tactic_schedule.next_route_platform(env) == 1
    with pytest.raises(ValueError):
        tactic_schedule.check_schedule([segment("advance", 1, on=0)], 1, 0)
    with pytest.raises(ValueError):
        segment("charge", 1)


def test_completed_route_destinations_do_not_return_after_backtracking():
    env = MarioScenarioEnv()
    try:
        env.reset(
            scenario=_flat(
                [segment("advance", route=[1, 2, 3])],
                platforms=[
                    [0, 220, 80, 20],
                    [80, 220, 60, 20],
                    [140, 220, 110, 20],
                    [250, 220, 150, 20],
                ],
            )
        )
        while env.mario["x"] < 160:
            assert not env.step(1)[2]
        assert tactic_schedule.next_route_platform(env) == 3
        while env.mario["x"] > 110:
            assert not env.step(3)[2]
        assert env.mario["_platform"] is env.platforms[1]
        assert tactic_schedule.next_route_platform(env) == 3
    finally:
        env.close()


def test_destination_under_spawn_is_consumed_before_the_first_decision():
    env = MarioScenarioEnv()
    try:
        env.reset(
            scenario=_flat(
                [segment("advance", route=[0, 1])],
                platforms=[[0, 220, 80, 20], [80, 220, 320, 20]],
            )
        )
        assert env.steps == 0
        assert tactic_schedule.next_route_platform(env) == 1
    finally:
        env.close()


def test_segment_completion_exposes_the_next_objective_on_the_same_frame():
    env = MarioScenarioEnv()
    try:
        env.reset(
            scenario=_flat(
                [
                    segment("advance", route=[1], on=1),
                    segment("advance", route=[0, 1, 2]),
                ],
                platforms=[[0, 220, 80, 20], [80, 220, 60, 20], [140, 220, 260, 20]],
            )
        )
        while env._tactic_index == 0:
            assert not env.step(1)[2]
        assert env.mario["_platform"] is env.platforms[1]
        assert tactic_schedule.next_route_platform(env) == 2
    finally:
        env.close()


def test_final_destination_ends_the_scenario_on_its_first_contact_frame():
    env = MarioScenarioEnv()
    try:
        scenario = _flat([segment("advance")])
        scenario["goal"] = [60, 200, 16, 20]
        env.reset(scenario=scenario)
        for _ in range(100):
            _, _, done, _, _ = env.step(1)
            contact = env.mario["x"] + env.mario["w"] > env.goal.left
            assert done == contact
            if done:
                assert env._goal_credited
                break
        else:
            pytest.fail("Mario never reached the destination")
    finally:
        env.close()


def test_the_monster_cannot_be_stomped_is_the_other_kind_and_wakes_on_screen():
    monster = {
        "kind": "monster",
        "x": 120,
        "y": 200,
        "patrol_min": 0,
        "patrol_max": 400,
        "speed": 0.6,
        "direction": -1,
    }
    env = MarioScenarioEnv()
    env.reset(scenario=_flat([segment("advance", 1)], enemies=[monster], mario=(116, 160)))
    info = _run(env, 0, 60)  # falls onto its back
    assert info["death"]
    assert set(env.instance_meta()[1].values()) == {"other"}
    far = dict(monster, x=380)
    env.reset(scenario=_flat([segment("advance", 1)], enemies=[far]))
    _run(env, 0, 30)
    assert env.enemies[0]["x"] == 380  # off screen: it has not woken
    env.close()


def test_every_family_states_its_schedule_and_every_schedule_stance_has_families():
    from retroagi.stages.block_smb.monte_carlo import _family_tactics

    stances = set()
    kinds = set()
    for family in BLOCK_SMB_MC_FAMILIES:
        if family in TACTIC_FAMILIES:
            continue  # made by tactic_families, with their own tactics
        scenario = {"mario": [20, 200], "goal": [200, 200, 16, 20]}
        for seg in _family_tactics(family, scenario):
            stances.add(seg["stance"])
            kinds.add(seg["kind"])
    for family in (
        "dead_end_retreat",
        "monster_retreat",
        "choice_alternate_route",
        "choice_hold_area",
    ):
        for seg in _sample(family)["tactics"]:
            stances.add(seg["stance"])
            kinds.add(seg["kind"])
    # Hold area also comes from the moving-platform, plant and monster segments.
    assert stances == set(SCHEDULE_STANCES) and {"bridge", "plant", "monster"} <= kinds


def test_clones_play_the_same_layouts_and_only_their_tactics_differ():
    for group in (("choice_advance", "choice_retreat"), LOW_CHOICE_FAMILIES):
        layouts = {family: _sample(family, index=3) for family in group}
        first = layouts[group[0]]
        for family, scenario in layouts.items():
            assert scenario["platforms"] == first["platforms"]
            assert scenario["mario"] == first["mario"]
        schedules = [str(scenario["tactics"]) for scenario in layouts.values()]
        assert len(set(schedules)) == len(group)


def test_in_the_clones_the_schedule_decides_the_first_tactic():
    first = {}
    for family in CHOICE_FAMILIES + LOW_CHOICE_FAMILIES + ("low_choice_alternate_route",):
        scenario = _sample(family, index=3)
        env = MarioScenarioEnv()
        env.reset(scenario=scenario, seed=0)
        teacher = episode_teacher(scenario)
        teacher.observe_frame(env)
        tactic = teacher_tactic(env, teacher)
        plan, _ = teacher_plan(env, teacher)
        first[family] = (tactic.stance, plan.action if plan else None)
        env.close()
    assert first["choice_advance"][0] == "advance"
    assert first["choice_hold_area"] == ("hold_ground", HOLD_GROUND)
    assert first["choice_retreat"][0] == "retreat"
    assert first["low_choice_advance"][0] == "advance"
    assert first["low_choice_alternate_route"][0] == "descend_forward"


@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
def test_alternate_route_has_observable_strategy_and_collects_upper_path_points(difficulty):
    from retroagi.stages.block_smb.monte_carlo import block_smb_monte_carlo_metadata
    from retroagi.stages.block_smb.teacher_tokens import teacher_strategy

    scenario = _sample("choice_alternate_route", difficulty=difficulty)
    direct = _sample("choice_advance", difficulty=difficulty)
    assert "choice_alternate_route" in TACTIC_TRAINING_FAMILIES
    assert "choice_alternate_route" not in CLONE_FAMILIES
    assert scenario["coins"]
    assert teacher_strategy(episode_teacher(scenario)).kind == "max_points"
    assert teacher_strategy(episode_teacher(direct)).kind == "speed_run"
    assert scenario["strategy_objective"]["points"] > 0
    env = MarioScenarioEnv()
    try:
        env.reset(scenario=scenario)
        state = episode_teacher(scenario)
        tactics = set()
        for button in block_smb_monte_carlo_metadata(scenario)["oracle"]["actions"]:
            state.observe_frame(env)
            tactics.add(teacher_tactic(env, state).stance)
            if env.step(button)[2]:
                break
        assert env._goal_credited
        assert env.points() >= scenario["strategy_objective"]["points"]
        assert {"climb_forward", "descend_forward", "advance"} <= tactics
    finally:
        env.close()


def test_the_monster_family_needs_backing_off_jumping_in_the_tunnel_fails():
    from retroagi.stages.block_smb.local_traversal import LocalObjective, safe_jump_holds

    scenario = _sample("monster_retreat", index=1, difficulty="medium")
    env = MarioScenarioEnv()
    env.reset(scenario=scenario, seed=0)
    env.render = lambda: None
    index = next(i for i, e in enumerate(env.enemies) if e.get("kind") == "monster")
    certified = False
    for _ in range(80):  # walk at it inside the tunnel, trying every jump
        monster = env.enemies[index]
        objective = LocalObjective(
            "enemy",
            monster["x"],
            monster["x"] + monster["w"] + 24,
            monster["y"],
            enemy_index=index,
        )
        certified |= bool(env.mario["on_ground"] and safe_jump_holds(env, objective, 1))
        _, _, done, _, _ = env.step(1)
        if done:
            break
    assert not certified
    assert tactic_schedule.current(env)["kind"] == "monster"
    env.close()


@pytest.mark.parametrize("start,enemy,line", [(378, 551, 412), (363, 546, 399)])
def test_monster_approach_wakes_offscreen_enemy_and_preserves_escape_room(start, enemy, line):
    from retroagi.stages.block_smb.monster import (
        APPROACH_MARGIN,
        RETREAT_ROOM,
        monster_choice,
    )

    env = MarioScenarioEnv()
    try:
        env.reset(
            scenario={
                "world_width": 800,
                "mario": [start, 208],
                "platforms": [[0, 220, 800, 20], [line + 4, 40, 151, 150]],
                "enemies": [
                    {
                        "kind": "monster",
                        "x": enemy,
                        "y": 200,
                        "speed": 0.6,
                        "direction": -1,
                        "patrol_min": start - 50,
                        "patrol_max": 700,
                    }
                ],
                "goal": [740, 200, 16, 20],
                "tactics": [
                    segment("advance", 1, kind="monster", keep_behind=line, past_enemy=0),
                    segment("advance", 1),
                ],
            }
        )
        assert enemy >= env.camera_x + env.width
        for _ in range(80):
            stance, action, holds = monster_choice(env)
            assert not holds  # approach must not jump into the tunnel
            _, _, done, _, info = env.step(action)
            assert not done and not info["death"]
            if env.enemies[0].get("awake"):
                break
        assert env.enemies[0].get("awake")
        # Even releasing the direction at this instant can stop before the lip.
        for _ in range(40):
            env.step(0)
            assert env.mario["x"] + env.mario["w"] + APPROACH_MARGIN <= line
            assert env.mario["x"] - env.camera_x - 1 >= RETREAT_ROOM
        assert abs(env.mario["vx"]) < 0.1
    finally:
        env.close()


def test_monster_approach_refuses_to_spend_missing_retreat_space():
    from retroagi.stages.block_smb.monster import monster_choice

    env = MarioScenarioEnv()
    try:
        env.reset(
            scenario={
                "world_width": 800,
                "mario": [378, 208],
                "platforms": [[365, 220, 435, 20]],
                "enemies": [
                    {"kind": "monster", "x": 551, "y": 200, "patrol_min": 365, "patrol_max": 780}
                ],
                "goal": [740, 200, 16, 20],
                "tactics": [
                    segment("advance", 1, kind="monster", keep_behind=412, past_enemy=0),
                    segment("advance", 1),
                ],
            }
        )
        assert monster_choice(env) == ("hold_area", 0, [])
    finally:
        env.close()


def test_composed_scenes_change_tactics_along_the_way():
    for family in ("chained_obstacles", "tactics_bridge_then_gap"):
        tactics = _sample(family)["tactics"]
        assert len(tactics) >= 3
        assert len({(seg["stance"], seg["kind"]) for seg in tactics}) >= 2


def test_the_tactic_learner_leaves_out_the_clones():
    from retroagi.stages.block_smb.layered_train import learner_families

    families = learner_families("tactic", BLOCK_SMB_MC_FAMILIES)
    assert not set(CLONE_FAMILIES) & set(families)


@pytest.mark.parametrize("split", ["train", "validation"])
@pytest.mark.parametrize("learner", ["skill", "tactic"])
@pytest.mark.parametrize(
    "family",
    TACTIC_TRAINING_FAMILIES,
)
def test_tactic_family_tasks_belong_only_to_tactic_training(learner, split, family):
    from retroagi.stages.block_smb.layered_train import LayeredTrainConfig, _tasks

    config = LayeredTrainConfig(learner=learner, families=(family,))
    tasks = _tasks(config, split, 1, 0, 0.0, False)
    expected = (1 if split == "train" else 3) if learner == "tactic" else 0
    assert len(tasks) == expected
    assert all(task.family == family for task in tasks)


def test_primitive_maneuvers_are_skill_families_without_an_action_learner():
    from retroagi.stages.block_smb.action_families import ACTION_FAMILIES
    from retroagi.stages.block_smb.layered_train import learner_families

    families = learner_families("skill", BLOCK_SMB_MC_FAMILIES)
    assert set(ACTION_FAMILIES) <= set(families)
    with pytest.raises(ValueError, match="unknown learned layer"):
        learner_families("action", BLOCK_SMB_MC_FAMILIES)
    assert not set(ACTION_FAMILIES) & set(learner_families("tactic", BLOCK_SMB_MC_FAMILIES))


@pytest.mark.timeout(300)
@pytest.mark.parametrize(
    "family,tactic",
    [
        ("action_walk", "advance"),
        ("action_walk_back", "retreat"),
        ("action_jump_gap", "advance"),
        ("action_climb", "climb_forward"),
        ("action_climb_back", "climb_backward"),
        ("action_descend", "descend_forward"),
        ("action_descend_back", "descend_backward"),
        ("action_stomp", "advance"),
        ("action_wait", "hold_ground"),
    ],
)
def test_a_single_action_family_asks_for_one_tactic_from_start_to_finish(family, tactic):
    for difficulty in ("easy", "hard"):
        made = _labels_along_the_route(family, difficulty)
        assert made and {t.stance for _, t in made} == {tactic}


def _labels_along_the_route(family, difficulty="easy"):
    """The teacher's tactic at the start of each stretch of its route."""
    sample = sample_block_smb_monte_carlo_scenario(
        split="validation", seed=0, sample_index=0, family=family, difficulty=difficulty
    )
    env = MarioScenarioEnv()
    env.reset(scenario=sample.scenario, seed=0)
    teacher = episode_teacher(sample.scenario)
    route, made, t = list(sample.oracle["actions"]), [], 0
    while t < len(route) and not env._goal_credited:
        teacher.observe_frame(env)
        plan = _first_plan(route[t:])
        made.append((plan.action, teacher_tactic(env, teacher)))
        for action in route[t : t + plan.frames]:
            env.step(action)
        t += plan.frames
    env.close()
    return made


@pytest.mark.parametrize("family", ["retreat_recovery"])
def test_going_back_to_the_left_is_never_labelled_advance(family):
    made = _labels_along_the_route(family)
    left = [tactic for action, tactic in made if action in (3, 4)]
    assert left
    for tactic in left:
        assert tactic.stance in BACKWARD_TACTICS
