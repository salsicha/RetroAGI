"""The list of tracked things (core/smb_tracked_objects; user, 2026-10-21):
every enemy and moving platform vision has reported, up to 20, those out of
sight kept where they vanished, the one seen longest ago dropped when the
list is full; each keeps its row."""

import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

from dataclasses import replace

import numpy as np

from retroagi.core.smb_kinematics import CONTEXT, enemy_context, kinematics, slot_identities
from retroagi.core.smb_observer import pack_c
from retroagi.core.smb_scene_labels import scene_from_labels
from retroagi.core.smb_spatial_feedback import SpatialFeedback
from retroagi.core.smb_tracked_objects import (
    CAPACITY,
    COLUMN,
    Sighting,
    TrackedObjects,
    sightings,
    unseen_value,
)
from retroagi.stages.block_smb.env import MarioScenarioEnv


def play(scenario, frames, hide=()):
    """Watch ``scenario`` (Mario standing) through the executor and the list
    (the frames in ``hide`` with no enemy seen); the list as read each frame."""
    env = MarioScenarioEnv()
    try:
        env.reset(scenario=scenario)
        spatial, objects, read = SpatialFeedback(), TrackedObjects(), []
        for frame in range(frames):
            scene = scene_from_labels(env.scene_labels())
            if frame in hide:
                scene = replace(scene, enemies=())
            spatial.observe(scene)
            sprites, _ = kinematics(scene, spatial)
            identities = slot_identities(scene, spatial)
            position = (spatial.displacement, scene.mario.box[3])
            objects.update(*sightings(sprites, identities, enemy_context(scene), pack_c(scene), position))
            read.append(objects.read(position))
            spatial.executed(0, scene)
            env.step(0)
        return read
    finally:
        env.close()


def floor(enemies, extra=()):
    return {
        "world_width": 400,
        "mario": [40, 204],
        "platforms": [[0, 220, 400, 20], *extra],
        "enemies": enemies,
        "goal": [380, 200, 16, 20],
    }


def test_a_walker_out_of_sight_stays_where_it_vanished_and_is_the_same_when_seen_again():
    read = play(floor([[150, 206, 60, 300, 0.1, -1]]), 120, hide=range(20, 90))
    rows, keys, _ = read[19]
    (row,) = np.flatnonzero(keys)
    key, vanished = keys[row], rows[row, COLUMN["x"]]
    for t in range(20, 90):
        rows, keys, _ = read[t]
        assert keys[row] == key and rows[row, COLUMN["x"]] == vanished  # Mario stands still
        assert rows[row, COLUMN["unseen"]] == np.float32(unseen_value(t - 19))
    rows, keys, _ = read[119]
    assert keys[row] == key and rows[row, COLUMN["unseen"]] == 0.0  # seen again: the same
    assert np.count_nonzero(keys) == 1


def test_a_plant_down_in_its_pipe_stays_on_the_list_and_rises_as_itself():
    from retroagi.stages.block_smb.collision_families import _plant

    plant = _plant(150, 32, 188, 0, 32, 64, 16, 40)  # seen at full height, going down
    read = play({**floor([plant], [[150, 188, 32, 32]]), "platform_kinds": ["ground", "pipe"]}, 80)
    first = [k for k in read[0][1] if k]
    assert len(first) == 1
    hidden = [t for t, (rows, keys, _) in enumerate(read) if (rows[:, COLUMN["unseen"]] > 0).any()]
    assert len(hidden) > 30
    assert all([k for k in read[t][1] if k] == first for t in range(80))
    rows, keys, _ = read[-1]
    assert rows[keys == first[0], COLUMN["unseen"]][0] == 0.0 and rows[keys == first[0], COLUMN["plant"]][0] == 1.0


def _thing(identity, x, kind="walker"):
    context = np.zeros(len(CONTEXT), np.float32)
    context[CONTEXT.index(kind) if kind != "platform" else CONTEXT.index("width")] = 1.0
    context[CONTEXT.index("height")] = 1.0
    return Sighting(identity, kind, (float(x), 200.0), (0.0, 0.0), 1.0, context, 0)


def test_a_full_list_drops_the_one_seen_longest_ago_and_entries_keep_their_rows():
    objects = TrackedObjects()
    stays = _thing(1000, 0.0)
    for n in range(CAPACITY + 5):
        # One new thing a frame, seen once, far apart; one seen every frame.
        objects.update([_thing(n + 1, 40.0 * (n + 1)), stays])
    rows, keys, _ = objects.read((0.0, 0.0))
    assert np.count_nonzero(keys) == CAPACITY
    places = sorted(rows[keys > 0, COLUMN["x"]].tolist())
    # The one always seen, and the newest nineteen: the first five dropped.
    assert places == [0.0] + [40.0 * n for n in range(7, CAPACITY + 6)]
    row = int(np.flatnonzero(rows[:, COLUMN["x"]] == 0.0)[0])
    objects.update([stays])
    assert objects.read((0.0, 0.0))[1][row] == keys[row]


def test_an_enemy_seen_defeated_where_it_vanished_leaves_the_list():
    objects = TrackedObjects()
    objects.update([_thing(1, 100.0)])
    objects.update([], defeated=[(100.0, 216.0)])  # its middle and bottom
    assert not objects.read((0.0, 0.0))[1].any()
    objects.update([_thing(2, 100.0, "platform")])
    objects.update([], defeated=[(100.0, 216.0)])  # a platform is never an enemy
    assert objects.read((0.0, 0.0))[1].any()
