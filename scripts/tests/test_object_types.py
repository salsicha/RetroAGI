"""Each enemy's object type (user, 2026-10-21): a number the game's labels tell
(Full SMB: the game's enemy codes), carried by vision to the policy and the
collision trackers, whose networks learn what each number means. Labels and
models without types (Block SMB's) tell none, and everything recorded or
trained before reads as it did."""

import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import numpy as np
import pytest
import torch

from retroagi.core.layered_policy import LayeredSMBPolicy
from retroagi.core.smb_kinematics import CONTEXT, SPRITE_SLOTS, SPRITE_STATE, TRACKED
from retroagi.core.smb_observer import C_SPANS, SEQ_LEN_C, before_enemy_types, observation_layout, pack_c
from retroagi.core.smb_scene_labels import (
    OBJECT_TYPE_SLOTS,
    EnemyView,
    MarioView,
    SceneLabels,
    SceneObservation,
    decode_scene,
    scene_from_labels,
    scene_targets,
)
from retroagi.core.target_tracker import FEATURES, OBSERVATION_WIDTH, CollisionTracker, record_features


def _labels(enemy_types):
    types = np.zeros((240, 256), np.uint8)
    instances = np.full((240, 256), -1, np.int32)
    from retroagi.core.smb_pixel_types import TYPE_ID

    types[100:116, 60:76] = TYPE_ID["mario"]
    instances[100:116, 60:76] = 0
    types[100:116, 120:136] = TYPE_ID["enemy"]
    instances[100:116, 120:136] = 1
    return SceneLabels(
        types=types,
        instances=instances,
        categories={0: "mario", 1: "enemy"},
        kinds={1: "walker"},
        standing=False,
        facing_right=True,
        stomping=False,
        enemy_types=enemy_types,
    )


def test_labels_that_tell_a_type_give_it_to_the_scene_and_the_vision_targets():
    told = _labels({1: 3})
    (enemy,) = scene_from_labels(told).enemies
    assert enemy.type == 3
    targets = scene_targets(told)["enemy_type"]
    assert (targets[100 // 8 : 116 // 8, 120 // 8 : 136 // 8] == 3).all()
    assert set(np.unique(targets)) == {-1, 3}
    # Labels without types (Block SMB's) tell none.
    assert scene_from_labels(_labels({})).enemies[0].type == -1
    assert (scene_targets(_labels({}))["enemy_type"] == -1).all()


def test_a_vision_model_with_a_type_head_reports_the_type_it_sees():
    from retroagi.core.vision import SceneVisionTransformer

    model = SceneVisionTransformer(dim=16, depth=1, heads=4, object_types=OBJECT_TYPE_SLOTS)
    heads = model.heads(np.zeros((1, 240, 256, 3), np.uint8))
    targets = scene_targets(_labels({1: 5}))
    one = lambda index, size: torch.nn.functional.one_hot(torch.as_tensor(index), size).float() * 20
    heads = {name: value.detach() for name, value in heads.items()}
    heads["pixel_logits"] = one(torch.as_tensor(targets["types"]).long(), 10).permute(2, 0, 1)[None]
    heads["kind_logits"] = one(torch.as_tensor(targets["kind"]).clamp_min(0), 4).permute(2, 0, 1)[None]
    heads["type_logits"] = one(torch.as_tensor(targets["enemy_type"]).clamp_min(0), OBJECT_TYPE_SLOTS)
    heads["type_logits"] = heads["type_logits"].permute(2, 0, 1)[None]
    (scene,) = decode_scene(heads)
    assert [e.type for e in scene.enemies] == [5]
    heads["type_logits"] = None  # a model without the head tells none
    assert [e.type for e in decode_scene(heads)[0].enemies] == [-1]
    # Its loss is on the cells whose labels tell a type.
    from retroagi.core.scene_vision import scene_losses, stack_targets

    out = model.heads(np.zeros((1, 240, 256, 3), np.uint8))
    losses = scene_losses(out, stack_targets([targets]))
    assert losses["enemy_type"] > 0
    assert "enemy_type" not in scene_losses({**out, "type_logits": None}, stack_targets([targets]))


def _scene(enemy_type):
    return SceneObservation(
        mario=MarioView(box=(60, 100, 76, 116), facing_right=True, support="ground", on_something=True),
        enemies=(EnemyView((120, 100, 136, 116), "walker", enemy_type),),
    )


def test_each_enemy_slots_type_is_in_what_was_spare_room():
    begin, end = C_SPANS["c_enemy_types"]
    assert end - begin == 6 and C_SPANS["c_reserved"][0] == end
    told, untold = pack_c(_scene(4)), pack_c(_scene(-1))
    assert told[begin] == 5.0 and not told[begin + 1 : end].any()
    assert not untold[begin:].any()
    # Everything else is laid out as before.
    assert np.array_equal(np.delete(told, begin), np.delete(untold, begin))
    earlier = observation_layout()
    spans = dict(earlier["c_spans"])
    spans["c_reserved"] = [spans.pop("c_enemy_types")[0], SEQ_LEN_C]
    earlier = {**earlier, "version": 2, "c_spans": spans}
    earlier.pop("object_type_slots")
    assert before_enemy_types(earlier) and not before_enemy_types(observation_layout())


def test_the_policy_reads_types_from_no_influence_and_learns_them():
    torch.manual_seed(0)
    policy = LayeredSMBPolicy().eval()
    rows = torch.as_tensor(np.stack([pack_c(_scene(4)), pack_c(_scene(-1))]))
    tokens, present = policy.scene._objects(rows)
    enemies = 2  # summary, Mario, then the enemy slots
    assert torch.equal(tokens[enemies][0], tokens[enemies][1])
    with torch.no_grad():
        policy.scene.enemy_type.weight[5] = 1.0
    tokens, _ = policy.scene._objects(rows)
    assert not torch.equal(tokens[enemies][0, 0], tokens[enemies][1, 0])


def test_a_checkpoint_from_before_types_loads_and_acts_as_it_did(tmp_path):
    from retroagi.core.world_model import scene_digest
    from retroagi.stages.block_smb import layered_train as lt

    torch.manual_seed(0)
    policy = LayeredSMBPolicy().eval()
    path = tmp_path / "now.pt"
    lt.save_layered_checkpoint(path, policy, lt.LayeredTrainConfig(output=str(tmp_path)), [], [],
                               {"reads": scene_digest(policy), "history": []})
    saved = torch.load(path, weights_only=False)
    layout = saved["observation_layout"]
    spans = dict(layout["c_spans"])
    spans["c_reserved"] = [spans.pop("c_enemy_types")[0], SEQ_LEN_C]
    layout = {**layout, "version": 2, "c_spans": spans}
    layout.pop("object_type_slots")
    saved["observation_layout"] = layout
    saved["state_dict"] = {
        k: v for k, v in saved["state_dict"].items() if not k.startswith("scene.enemy_type.")
    }
    old = tmp_path / "old.pt"
    torch.save(saved, old)
    loaded, checkpoint = lt.load_layered_checkpoint(old)
    assert "scene_reads_enemy_types" in checkpoint["load_migrations"]
    assert not loaded.scene.enemy_type.weight.any()
    assert checkpoint["world_model"]["reads"] == scene_digest(loaded)
    rows = torch.as_tensor(pack_c(_scene(-1)))[None]
    with torch.no_grad():
        assert torch.allclose(loaded.scene._objects(rows)[0][2], policy.scene._objects(rows)[0][2])


def test_the_trackers_read_the_type_and_an_earlier_tracker_loads(tmp_path):
    from retroagi.stages.block_smb.tracker_train import load_tracker

    assert TRACKED[-OBJECT_TYPE_SLOTS:] == tuple(f"type_{n}" for n in range(OBJECT_TYPE_SLOTS))
    torch.manual_seed(0)
    new = CollisionTracker(16)
    state = new.state_dict()
    # A collision_v2 tracker: no type inputs.
    first, count = TRACKED.index("type_0"), OBJECT_TYPE_SLOTS
    for name, at in (("gru.weight_ih_l0", first), ("head.0.weight", 16 + first)):
        state[name] = torch.cat((state[name][:, :at], state[name][:, at + count :]), 1)
    torch.save({"tracker": state, "width": 16, "layout": "collision_v2", "config": {}, "history": []},
               tmp_path / "v2.pt")
    loaded = load_tracker(tmp_path / "v2.pt")
    assert not loaded.gru.weight_ih_l0[:, first : first + count].any()
    assert loaded.gru.weight_ih_l0.shape[1] == OBSERVATION_WIDTH
    assert loaded.head[0].weight.shape[1] == 16 + FEATURES


def test_an_episode_recorded_before_types_has_none_told():
    frames = 2
    record = type("Old", (), {})()
    record.sprite_kinematics = np.zeros((frames, SPRITE_SLOTS, len(SPRITE_STATE)), np.float16)
    record.sprite_kinematics[:, 0, 0] = 1.0
    record.mario_kinematics = np.zeros((frames, 3), np.float16)
    record.enemy_context = np.ones((frames, 6, 8), np.float16)  # eight numbers an enemy then
    record.slot_identity = np.zeros((frames, SPRITE_SLOTS), np.int32)
    record.slot_identity[:, 0] = 2
    record.mario_position = np.zeros((frames, 2), np.float32)
    record.decision_frames = np.array([0], np.int32)
    record.action_verb = None
    features = record_features(record)
    types = slice(TRACKED.index("type_0"), len(TRACKED))
    assert not features[:, 0, types].any()
    assert (features[:, 0, TRACKED.index("walker") : TRACKED.index("type_0")] == 1.0).all()
    assert len(CONTEXT) == 8 + OBJECT_TYPE_SLOTS


def test_full_smb_labels_number_every_enemy_code():
    from retroagi.stages.full_smb.pixel_labels import (
        ENEMY_KIND_CLASSES,
        ENEMY_TYPE_NUMBERS,
        HAMMER_TYPE_NUMBER,
    )

    assert set(ENEMY_TYPE_NUMBERS) == set(ENEMY_KIND_CLASSES)
    numbers = set(ENEMY_TYPE_NUMBERS.values()) | {HAMMER_TYPE_NUMBER}
    assert all(0 <= n < OBJECT_TYPE_SLOTS for n in numbers)
