"""A pit one tile wide in Block SMB behaves as in the game (measured in its
memory, scripts/dev/gap_rule.py): walking or running Mario crosses it, his
feet set on the far ledge from up to 3 pixels below its top (sim/env.py
STEP_UP); stopped over it, he falls in; a pit two tiles wide he falls into.
The planner meets a ledge's side the same way (planner.WALL_DEPTH)."""

from retroagi.domains.smb import planner
from retroagi.domains.smb.sim.env import STEP_UP
from retroagi.domains.smb.sim.levels import FLOOR, TILE
from retroagi.domains.smb.sim.world import SimWorld

PIT = 6 * TILE  # the pit's near edge (Mario starts at 40)
RIGHT = 1  # (Block SMB's button: right)


def scene(tiles: int) -> dict:
    return {"world_width": 800, "scroll_x": 115, "mario": [40.0, FLOOR - 12], "frame_budget": 600,
            "platforms": [[0, FLOOR, PIT, 32], [PIT + tiles * TILE, FLOOR, 800, 32]],
            "platform_kinds": ["ground", "ground"], "enemies": [], "task": {"family": "game_level"}}


def walk(tiles: int, let_go_at: float = 10_000.0) -> str:
    """How Mario's walk at the pit ends, pressing right until ``let_go_at``:
    "crossed", "fell" or "short" (stopped before it)."""
    world = SimWorld(scene(tiles))
    world.reset()
    mario = world.env.mario
    for _ in range(400):
        step = world.step(RIGHT if mario["x"] < let_go_at else 0)
        if step.over:
            return "fell"
        if mario["x"] > PIT + tiles * TILE + 16 and mario["on_ground"]:
            return "crossed"
    return "crossed" if mario["x"] > PIT else "short"


def test_mario_walks_across_a_one_tile_pit_and_into_a_two_tile_one():
    assert walk(1) == "crossed"
    assert walk(2) == "fell"


def test_stopped_over_a_one_tile_pit_mario_falls_in():
    """Let go before the pit, he slides on: short of it, into it (coming to
    rest over it), or over it (still fast enough), as in the game."""
    ends = [walk(1, let_go_at=PIT - back) for back in range(40)]
    assert {"short", "fell", "crossed"} <= set(ends)
    first_fall = ends.index("fell")
    assert ends[first_fall - 1] == "crossed" and "short" in ends[first_fall:]


def test_the_planner_and_block_smb_meet_a_ledge_alike():
    assert planner.WALL_DEPTH == STEP_UP == 3
