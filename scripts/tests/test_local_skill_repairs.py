"""Local goals, observed contact and feedback survive curriculum boundaries."""

from dataclasses import replace

import pytest
import torch

from retroagi.core.smb_scene_labels import decode_scene, scene_from_labels, scene_targets
from retroagi.core.smb_spatial_feedback import SpatialFeedback
from retroagi.core.smb_trajectory import VisualTracks, plan_flight
from retroagi.core.tokens import SkillToken
from retroagi.stages.block_smb.env import MarioScenarioEnv
from retroagi.stages.block_smb.layered_train import learner_families
from retroagi.stages.block_smb.monte_carlo import (
    BLOCK_SMB_MC_FAMILIES,
    block_smb_monte_carlo_metadata,
    sample_block_smb_monte_carlo_scenario,
)
from retroagi.stages.block_smb.skill_families import LOW_ROUTE_SKILL_FAMILIES
from retroagi.stages.block_smb.teacher_tokens import episode_teacher, teacher_plan, teacher_skill


def sample(family, difficulty="easy"):
    return sample_block_smb_monte_carlo_scenario(
        split="validation", seed=0, sample_index=100, family=family, difficulty=difficulty
    ).scenario


@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
@pytest.mark.parametrize("family", ["low_choice_alternate_route", *LOW_ROUTE_SKILL_FAMILIES])
def test_lower_route_and_local_maneuvers_have_reachable_goals(family, difficulty):
    scenario = sample(family, difficulty)
    skills = learner_families("skill", BLOCK_SMB_MC_FAMILIES)
    assert (family in skills) == (family in LOW_ROUTE_SKILL_FAMILIES)
    env = MarioScenarioEnv()
    try:
        env.reset(scenario=scenario)
        for button in block_smb_monte_carlo_metadata(scenario)["oracle"]["actions"]:
            if env.step(button)[2]:
                break
        assert env._goal_credited
        if family == "low_choice_alternate_route":
            assert scenario["strategy"] == "max_points"
            assert env.points() >= scenario["strategy_objective"]["points"] > 0
        else:
            assert len(scenario["tactics"]) == 1
            assert len(scenario["tactics"][0]["route"]) == 1
    finally:
        env.close()


def test_hold_skill_finishes_at_timer_without_a_departure_or_gap_crossing():
    scenario = sample("choice_hold_area")
    env = MarioScenarioEnv()
    try:
        env.reset(scenario=scenario)
        duration = scenario["tactics"][0]["end"]["frames"]
        start = env.mario["x"]
        for frame in range(1, duration + 1):
            done = env.step(0)[2]
            assert done == (frame == duration)
        assert env._goal_credited and env.mario["x"] == start
        assert len(scenario["tactics"]) == 1
    finally:
        env.close()


def test_stomp_label_anchors_the_current_enemy_and_restores_probe_contact():
    scenario = sample("action_stomp_up_back", "hard")
    env = MarioScenarioEnv()
    try:
        env.reset(scenario=scenario)
        teacher = episode_teacher(scenario)
        teacher.observe_frame(env)
        plan, _ = teacher_plan(env, teacher, certify_holds=False)
        enemy = dict(env.enemies[0])
        before = (env.steps, env.mario["x"], env.mario["y"], env.stomped)
        goal = teacher_skill(env, teacher, plan)
        assert goal.mode == "jump" and enemy["speed"] > 0
        assert abs(env.mario["x"] + env.mario["w"] / 2 + goal.x - enemy["x"] - enemy["w"] / 2) <= 1
        assert abs(env.mario["y"] + env.mario["h"] + goal.y - enemy["y"]) <= 1
        assert before == (env.steps, env.mario["x"], env.mario["y"], env.stomped)
        assert env.enemies[0]["x"] == enemy["x"] and not env.enemies[0]["dead"]
    finally:
        env.close()


def test_zero_distance_repeated_commands_report_stall_but_holding_does_not():
    env = MarioScenarioEnv()
    try:
        env.reset(scenario={"mario": [40, 208], "platforms": [[0, 220, 256, 20]]})
        scene = scene_from_labels(env.scene_labels())
        feedback = SpatialFeedback()
        for _ in range(30):
            feedback.observe(scene)
            feedback.begin(SkillToken("run", 0, 0), scene)
        assert feedback.report(scene)[1] == 1
        feedback.begin(SkillToken("hold", 0, 0), scene)
        assert feedback.report(scene)[1] == 0
        feedback.begin(SkillToken("run", 20, 0), scene)
        assert feedback.report(scene)[1] == 0
    finally:
        env.close()


def test_airborne_replanning_can_extend_hold_but_never_repress_after_release():
    env = MarioScenarioEnv()
    try:
        env.reset(scenario={"mario": [120, 208], "platforms": [[0, 220, 256, 20]]})
        flight = plan_flight(
            scene_from_labels(env.scene_labels()), VisualTracks(), 0, SkillToken("jump", 30, -32)
        )
        # Exercise an underestimated initial hold against real game motion.
        flight.hold = 1
        buttons = []
        for _ in range(40):
            button = flight.press(scene_from_labels(env.scene_labels()))
            buttons.append(button)
            env.step(button)
        pressed = [button in (2, 4, 5) for button in buttons]
        assert pressed[:2] == [True, True]
        first_release = pressed.index(False)
        assert 1 < first_release <= 32
        assert not any(pressed[first_release:])
    finally:
        env.close()


def test_changing_mario_silhouette_does_not_reverse_observed_momentum():
    env = MarioScenarioEnv()
    try:
        env.reset(scenario={"mario": [136, 208], "platforms": [[0, 220, 256, 20]]})
        truth = scene_from_labels(env.scene_labels())
        feedback = SpatialFeedback()
        for box in ((131, 208, 146, 220), (135, 204, 146, 216), (136, 200, 146, 212)):
            scene = replace(truth, mario=replace(truth.mario, box=box))
            feedback.observe(scene)
            feedback.executed(4, scene)
        assert feedback.motion.x_speed < 0
    finally:
        env.close()


@pytest.mark.parametrize("standing", [False, True])
@pytest.mark.parametrize("contact_prediction", [False, True])
@pytest.mark.parametrize("support_prediction", ["air", "ground"])
def test_vision_contact_resolves_surface_and_side_of_ledge(
    standing, contact_prediction, support_prediction
):
    env = MarioScenarioEnv()
    try:
        env.reset(
            scenario={"mario": [40, 208], "platforms": [[0, 220, 256, 20], [70, 150, 30, 70]]}
        )
        if not standing:
            env.mario.update(x=60, y=160, on_ground=False)
        targets = scene_targets(env.scene_labels())
        one = torch.nn.functional.one_hot
        heads = {
            "pixel_logits": one(torch.as_tensor(targets["types"]).long(), 10).permute(2, 0, 1)[None]
            * 20.0,
            "kind_logits": one(torch.as_tensor(targets["kind"]).clamp_min(0), 4).permute(2, 0, 1)[
                None
            ]
            * 20.0,
            "facing_logits": one(torch.as_tensor(targets["facing"]), 2)[None].float() * 20,
            "support_logits": torch.tensor(
                [[20.0, 0.0, 0.0]] if support_prediction == "air" else [[0.0, 20.0, 0.0]]
            ),
            "on_something_logits": torch.tensor(
                [[0.0, 20.0]] if contact_prediction else [[20.0, 0.0]]
            ),
        }
        scene = decode_scene(heads)[0]
        assert scene.mario.on_something == standing
        assert (scene.mario.support == "air") == (not standing)
    finally:
        env.close()
