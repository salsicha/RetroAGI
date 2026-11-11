"""The obstacle test bench (docs/choose-and-control-plan.md phase 0): a
level's obstacles from its map, the saved state each starts from, and the
rules for passing, dying and stalling."""

from retroagi.domains.smb.nes.obstacles import (
    APPROACH,
    BUCKET,
    WAKE_LEAD,
    LevelMap,
    _merged,
    obstacles_of,
    state_for,
)
from retroagi.framework import bench
from retroagi.framework.interfaces import Step


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


class FakeWorld:
    """Progress advancing ``speed`` a frame; dying at ``dies_at`` frames."""

    settled = True

    def __init__(self, speed, dies_at=None):
        self.speed, self.dies_at, self.frame = speed, dies_at, 0

    @property
    def progress(self):
        return 400 + self.speed * self.frame

    def reset(self):
        self.frame = 0

    def step(self, action):
        self.frame += 1
        dying = self.dies_at is not None and self.frame >= self.dies_at
        return Step(over=dying, died=dying)


class StillPlayer:
    idle = 0

    def reset(self, seed, mode):
        pass

    def act(self, world):
        return 0


def test_passing_dying_and_stalling():
    obstacle = {"x0": 450, "x1": 482}
    (passed,) = bench.attempts(FakeWorld(speed=2), StillPlayer(), 0, "speed_run", [obstacle])
    assert passed["end"] == "passed" and passed["frames"] == (482 + bench.PASSED - 400) // 2
    (died,) = bench.attempts(FakeWorld(speed=2, dies_at=10), StillPlayer(), 0, "speed_run", [obstacle])
    assert died["end"] == "died"
    (stalled,) = bench.attempts(FakeWorld(speed=0), StillPlayer(), 0, "speed_run", [obstacle])
    assert stalled == {"end": "stalled", "frames": bench.ATTEMPT_FRAMES, "furthest": 400}


def test_one_play_scores_every_obstacle_of_its_start_and_seeds_wait():
    near, far = {"x0": 450, "x1": 482}, {"x0": 600, "x1": 640}
    # The near one is passed on the way to the far one, as in a play of its own.
    first, second = bench.attempts(FakeWorld(speed=2), StillPlayer(), 0, "speed_run", [near, far])
    assert first == bench.attempts(FakeWorld(speed=2), StillPlayer(), 0, "speed_run", [near])[0]
    assert second["end"] == "passed" and second["frames"] > first["frames"]
    # Dying on the way counts against every obstacle not yet passed.
    first, second = bench.attempts(FakeWorld(speed=2, dies_at=80), StillPlayer(), 0, "speed_run", [near, far])
    assert (first["end"], second["end"]) == ("passed", "died")
    # The frames allowed count again from each obstacle passed: a slow play stalls only where it stops.
    far_off = {"x0": 2216, "x1": 2256}
    first, second = bench.attempts(FakeWorld(speed=2), StillPlayer(), 0, "speed_run", [near, far_off])
    assert second["end"] == "passed" and second["frames"] > bench.ATTEMPT_FRAMES
    first, second = bench.attempts(FakeWorld(speed=2), StillPlayer(), 0, "speed_run", [near, {"x0": 9000, "x1": 9040}])
    assert (first["end"], second["end"]) == ("passed", "stalled")
    assert second["frames"] == first["frames"] + bench.ATTEMPT_FRAMES
    # A later seed waits first: the world has moved on before the agent acts.
    waited = bench.attempts(FakeWorld(speed=2), StillPlayer(), 1, "speed_run", [near])[0]
    assert waited["frames"] == first["frames"] - bench.START_WAITS[1]
