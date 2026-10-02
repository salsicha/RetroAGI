"""The shared description of a frame: objects as drawn, surfaces, gaps, blocks, pipes.

Both games' ground truth goes through scene_from_labels, and the vision
transformer's structure through the same structure_from_types, so these rules
are the definition every model and policy shares.
"""

import numpy as np
import pytest
import torch

from retroagi.core.smb_pixel_types import TYPE_ID
from retroagi.core.smb_scene_labels import (
    Gap,
    SceneLabels,
    Surface,
    decode_scene,
    scene_from_labels,
    scene_targets,
    structure_from_types,
)
from retroagi.stages.block_smb.env import MarioScenarioEnv

G, B, Q, P, E, M, L = (
    TYPE_ID[name]
    for name in ("ground", "brick", "question_block", "pipe", "enemy", "mario", "moving_platform")
)


def floor_with_pit(pit=(100, 140)):
    types = np.zeros((240, 256), np.uint8)
    types[208:] = G
    types[208:, pit[0] : pit[1]] = 0
    return types


def labels(types, instances=None, categories=None, kinds=None, standing=True, facing=True):
    return SceneLabels(
        types=types,
        instances=instances if instances is not None else np.full(types.shape, -1, np.int32),
        categories=categories or {},
        kinds=kinds or {},
        standing=standing,
        facing_right=facing,
    )


def test_floor_surfaces_and_the_gap_between_them():
    found = structure_from_types(floor_with_pit())
    assert found["surfaces"] == (Surface(8, 100, 208, False), Surface(140, 248, 208, False))
    assert found["gaps"] == (Gap(100, 140),)


def test_an_enemy_standing_on_the_floor_does_not_split_it():
    types = floor_with_pit()
    types[194:209, 30:46] = E  # feet sink one row into the floor's top row
    types[208, 36:40] = G  # the floor shows between the feet
    surfaces = structure_from_types(types)["surfaces"]
    assert Surface(8, 100, 208, False) in surfaces
    assert all(s.top == 208 for s in surfaces)


def test_brick_rows_question_corners_and_pipes():
    types = floor_with_pit()
    types[144:160, 48:96] = B
    types[144:160, 64:80] = Q
    types[144, 64] = types[144, 79] = 0  # rounded question block corners
    types[160:208, 176:208] = P
    found = structure_from_types(types)
    assert Surface(48, 96, 144, False) in found["surfaces"]
    assert Surface(176, 208, 160, False) in found["surfaces"]
    kinds = sorted((b.kind, b.box) for b in found["blocks"])
    assert kinds == [
        ("brick", (48, 144, 64, 160)),
        ("brick", (80, 144, 96, 160)),
        ("question_block", (64, 144, 80, 160)),
    ]
    assert found["pipes"] == ((176, 160, 208, 208),)


def test_mortar_lines_in_the_backdrop_colour_do_not_make_surfaces():
    types = np.zeros((240, 256), np.uint8)
    types[160:] = G
    types[168::8] = 0  # one-pixel mortar rows, as in castle bricks
    types[160:, 100::16] = 0  # one-pixel mortar columns
    surfaces = structure_from_types(types)["surfaces"]
    assert surfaces == (Surface(8, 248, 160, False),)


def test_nothing_outside_the_window_the_nes_shows_is_measured():
    types = floor_with_pit()
    types[100:110, 0:8] = B  # only in the hidden border
    found = structure_from_types(types)
    assert found["blocks"] == ()
    assert all(s.x0 >= 8 and s.x1 <= 248 for s in found["surfaces"])


def test_objects_are_their_drawn_boxes_and_mario_support_reads_the_pixels_below():
    types = floor_with_pit()
    instances = np.full(types.shape, -1, np.int32)
    types[192:208, 20:36] = M
    instances[192:208, 20:36] = 0
    types[190:200, 150:200] = L
    instances[190:200, 150:200] = 1
    types[176:190, 160:170] = M  # not Mario: a second drawn object on the lift
    instances[176:190, 160:170] = 2
    scene = scene_from_labels(
        labels(types, instances, {0: "mario", 1: "moving_platform", 2: "enemy"}, {2: "walker"})
    )
    assert scene.mario.box == (20, 192, 36, 208)
    assert scene.mario.support == "ground"
    assert scene.moving_platforms == ((150, 190, 200, 200),)
    assert [(e.box, e.kind) for e in scene.enemies] == [((160, 176, 170, 190), "walker")]
    on_lift = scene_from_labels(
        labels(types, np.where(instances == 0, -1, instances), {1: "moving_platform", 2: "mario"})
    )
    assert on_lift.mario.support == "moving_platform"
    assert (
        scene_from_labels(labels(types, instances, {0: "mario"}, standing=False)).mario.support
        == "air"
    )


def test_targets_decode_back_to_the_exact_scene():
    types = floor_with_pit()
    instances = np.full(types.shape, -1, np.int32)
    types[192:208, 20:36] = M
    instances[192:208, 20:36] = 0
    types[198:208, 60:70] = E
    instances[198:208, 60:70] = 1
    truth_labels = labels(types, instances, {0: "mario", 1: "enemy"}, {1: "defeated"}, facing=False)
    targets = scene_targets(truth_labels)
    one = torch.nn.functional.one_hot
    heads = {
        "pixel_logits": one(torch.as_tensor(targets["types"]).long(), 10).permute(2, 0, 1)[None]
        * 20.0,
        "kind_logits": one(torch.as_tensor(targets["kind"]).clamp_min(0), 4).permute(2, 0, 1)[None]
        * 20.0,
        "facing_logits": one(torch.as_tensor(targets["facing"]), 2)[None].float() * 20,
        "support_logits": one(torch.as_tensor(targets["support"]), 3)[None].float() * 20,
    }
    assert decode_scene(heads)[0] == scene_from_labels(truth_labels)


# ── Block SMB draws exactly what is reported ─────────────────────────────────


def block_scene(scenario, steps=0, action=0):
    env = MarioScenarioEnv()
    try:
        env.reset(scenario=scenario, seed=0)
        for _ in range(steps):
            env.step(action)
        return scene_from_labels(env.scene_labels()), env.scene_labels()
    finally:
        env.close()


def test_block_goomba_is_reported_by_its_drawn_body_and_feet():
    scene, _ = block_scene(
        {
            "world_width": 256,
            "mario": [20, 200],
            "platforms": [[0, 220, 256, 20]],
            "enemies": [[120, 206, 120, 120, 0.0]],
            "goal": [240, 200, 16, 20],
        }
    )
    (goomba,) = scene.enemies
    x0, y0, x1, y1 = goomba.box
    assert (x1 - x0, y1 - y0) == (10, 10)  # 6-pixel body over 4-pixel feet, not the 10x6 body
    assert goomba.kind == "walker"


def test_block_power_ups_are_drawn_and_reported_and_only_reward():
    scenario = {
        "world_width": 256,
        "mario": [20, 200],
        "platforms": [[0, 220, 256, 20]],
        "power_ups": [[60, 204]],
        "goal": [240, 200, 16, 20],
    }
    scene, truth = block_scene(scenario)
    assert scene.power_ups == ((60, 204, 76, 220),)
    assert (truth.types == TYPE_ID["power_up"]).any()
    env = MarioScenarioEnv()
    try:
        env.reset(scenario=scenario, seed=0)
        rewards = []
        for _ in range(40):
            _obs, _reward, _done, _truncated, info = env.step(1)
            rewards.append(info["reward_terms"]["power_up"])
        assert sum(rewards) > 0
        assert env.mario["h"] == 12  # Mario does not grow
    finally:
        env.close()


@pytest.mark.parametrize("phase, shown", [(90, False), (30, True), (8, True)])
def test_block_plant_inside_its_pipe_is_not_reported(phase, shown):
    # Phases (16 rising, 48 out, 16 sinking, 32 hidden): 90 hidden, 30 out, 8 half-risen.
    scenario = {
        "world_width": 256,
        "mario": [20, 200],
        "platforms": [[0, 220, 256, 20], [120, 180, 32, 40]],
        "platform_kinds": ["ground", "pipe"],
        "enemies": [{"kind": "piranha_plant", "x": 130, "pipe_top": 180, "phase": phase}],
        "goal": [240, 200, 16, 20],
    }
    scene, _ = block_scene(scenario)
    if not shown:
        assert scene.enemies == ()
    else:
        (seen,) = scene.enemies
        assert seen.kind == "plant"
        assert seen.box[3] <= 180  # only the part above the pipe's top is drawn
