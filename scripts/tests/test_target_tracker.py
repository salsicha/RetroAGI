"""The target tracker: inputs from recorded object rows, learning where walkers
turn, and sensing when an object departs from its forecast."""

import random

import numpy as np
import torch

from retroagi.core.target_tracker import (
    OBJECT_ROW,
    OBJECT_SLOTS,
    ROW,
    TRACKER_HORIZONS,
    ObjectTracker,
    TargetTracker,
    steady_error,
    track_examples,
)
from retroagi.stages.block_smb.tracker_train import batch_loss, evaluate


def walker_rows(rng, frames=240):
    """One walker patrolling a floor between its ends, as recorded rows
    [frames, OBJECT_SLOTS, len(OBJECT_ROW)] (identity 1, slot 0)."""
    left = rng.randint(0, 100)
    right = left + rng.randint(48, 160)
    speed = rng.choice((0.5, 0.75, 1.0))
    x = rng.uniform(left, right - 16)
    direction = rng.choice((-1, 1))
    rows = np.zeros((frames, OBJECT_SLOTS, len(OBJECT_ROW)), np.float32)
    for t in range(frames):
        row = rows[t, 0]
        row[ROW["identity"]] = 1
        row[ROW["kind"]] = 0
        row[ROW["x0"] : ROW["y1"] + 1] = (x, 200, x + 16, 216)
        row[ROW["support_left"]], row[ROW["support_right"]] = left, right
        row[ROW["supported"]] = 1
        row[ROW["mario_x"]], row[ROW["mario_feet"]] = 40, 216
        x += speed * direction
        if x <= left or x + 16 >= right:
            direction = -direction
            x = min(max(x, left), right - 16)
    return rows


def test_examples_follow_one_identity():
    rows = walker_rows(random.Random(0), frames=100)
    rows[50:53, 0, ROW["identity"]] = 0  # not seen for three frames
    [(inputs, velocity, targets, seen, known, valid)] = track_examples(rows)
    assert len(inputs) == 100 and valid.sum() == 97
    assert not inputs[50:53, 0].any() and inputs[53, 0] == 1
    h = TRACKER_HORIZONS.index(4)
    assert seen[10, h] and not seen[47, h]  # 47 + 4 = 51 was not seen
    assert known[95, 0] and not known[97, h]
    x = (rows[:, 0, ROW["x0"]] + rows[:, 0, ROW["x1"]]) / 2
    assert np.isclose(targets[10, h, 0], x[14] - x[10])


def test_tracker_learns_where_walkers_turn():
    rng = random.Random(1)
    train = [e for _ in range(64) for e in track_examples(walker_rows(rng))]
    held = [e for _ in range(16) for e in track_examples(walker_rows(rng))]
    torch.manual_seed(0)
    model = TargetTracker(32)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    for _ in range(300):
        loss, _ = batch_loss(model, random.Random(_).sample(train, 32), "cpu")
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    stats = evaluate(model, held, "cpu", 64)
    steady = steady_error(held)
    # Far ahead, a walker has often turned: steady motion is far off.
    assert stats["error_64"] < 0.7 * steady[64]
    assert stats["error_1"] < 1.0


def test_play_tracker_senses_a_turn():
    rows = walker_rows(random.Random(3), frames=200)
    x = rows[:, 0, ROW["x0"]]
    turns = [t for t in range(2, 200) if (x[t] - x[t - 1]) * (x[t - 1] - x[t - 2]) < 0]
    assert turns
    tracker = ObjectTracker(TargetTracker(16))  # untrained: steady motion
    fired = []
    for t in range(200):
        tracker.observe(rows[t])
        if tracker.changed(1):
            fired.append(t)
    first = turns[0]
    assert any(first <= t <= first + 8 for t in fired)
    assert not [t for t in fired if t < first]
    where = tracker.where(1, 8)
    assert where is not None and abs(where[1] - 200) < 1e-3
