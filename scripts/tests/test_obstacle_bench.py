"""The obstacle test bench (docs/choose-and-control-plan.md phase 0): a
level's obstacles from its map, the saved state each starts from, and the
rules for passing, dying and stalling."""

from types import SimpleNamespace

import numpy as np

from retroagi.domains.smb.nes import obstacle_bench as ob
from retroagi.domains.smb.nes.obstacles import (
    APPROACH,
    BUCKET,
    WAKE_LEAD,
    LevelMap,
    _merged,
    obstacles_of,
    state_for,
)


def test_a_level_map_becomes_its_obstacles_in_order():
    level = LevelMap(
        pipes=[(450, 482, 176), (608, 640, 160)],
        pits=[(1104, 1136)],
        steps=[(2134, 2150, 192), (2150, 2166, 176)],
        # Two Goombas woken together, one later on its own.
        enemies=[(343, 203, 6), (360, 205, 6), (657, 530, 6)],
        finish_x=3160,
        furthest_x=3160,
    )
    found = obstacles_of(level)
    kinds = [o.kind for o in found]
    assert kinds == ["enemies", "pipe", "pipe", "enemies", "pit", "steps", "flag"]
    first = found[0]
    assert (first.x0, first.x1, first.approach_x) == (343, 360, 203 - WAKE_LEAD)
    assert found[1].approach_x == 450 - APPROACH
    assert (found[5].x0, found[5].x1) == (2134, 2166)  # the steps joined
    assert found[-1].name.endswith("flag_3160")
    assert [o.name[:2] for o in found] == [f"{k:02d}" for k in range(len(found))]


def test_pieces_seen_in_many_frames_count_once():
    assert _merged([(450, 482, 176), (451, 483, 176), (608, 640, 160)]) == [(450, 482, 176), (608, 640, 160)]


def test_an_obstacle_starts_from_the_state_kept_nearest_before_it():
    states = {10: b"a", 20: b"b", 30: b"c"}
    assert state_for(states, 25 * BUCKET) == (20, b"b")
    assert state_for(states, 5 * BUCKET) is None
    # A start that fails its check (Mario falling, an enemy on him) gives way
    # to the nearest earlier one that passes.
    assert state_for(states, 35 * BUCKET, check=lambda state: state != b"c") == (20, b"b")


class FakeGame:
    """Level x advancing ``speed`` a frame; dying at ``dies_at`` frames."""

    def __init__(self, speed, dies_at=None):
        self.speed, self.dies_at, self.frame = speed, dies_at, 0
        self.start = (0, 0)
        self.env = SimpleNamespace(get_ram=lambda: np.zeros(2048, np.uint8))

    def reset(self):
        self.frame = 0
        return None

    def step(self, button):
        self.frame += 1
        dying = self.dies_at is not None and self.frame >= self.dies_at
        return None, SimpleNamespace(dying=dying, finished=False)


class StillPlayer:
    def reset(self, seed, mode):
        pass

    def act(self, screen, game):
        return 0


def test_passing_dying_and_stalling(monkeypatch):
    obstacle = {"x0": 450, "x1": 482}
    games = []

    def truth(ram, start):
        game = games[-1]
        return {"level_x": 400 + game.speed * game.frame}

    monkeypatch.setattr(ob, "true_frame", truth)
    games.append(FakeGame(speed=2))
    passed = ob.attempt(games[-1], StillPlayer(), 0, "speed_run", obstacle)
    assert passed["end"] == "passed" and passed["frames"] == (482 + ob.PASSED - 400) // 2
    games.append(FakeGame(speed=2, dies_at=10))
    assert ob.attempt(games[-1], StillPlayer(), 0, "speed_run", obstacle)["end"] == "died"
    games.append(FakeGame(speed=0))
    stalled = ob.attempt(games[-1], StillPlayer(), 0, "speed_run", obstacle)
    assert stalled == {"end": "stalled", "frames": ob.ATTEMPT_FRAMES, "furthest": 400}
