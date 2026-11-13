"""The sky levels' ledges in Block SMB's random levels (sim/levels.py;
measured in the block maps of Worlds 1-3, 3-3, 4-3, 5-3 and 6-3 read from
game memory): a stretch with no floor crossed on ledges of the game's
widths, heights and gaps, each within a jump of the one before, and a
scene of them that can be passed."""

from retroagi.domains.smb.sim import levels as made
from retroagi.domains.smb.sim.levels import FLOOR, TILE, skill_scene
from retroagi.domains.smb.sim.passable import passable



def test_a_ledges_scene_is_a_pit_crossed_on_ledges_each_within_a_jump_of_the_one_before():
    for seed in range(30):
        scene = skill_scene(seed, "ledges")
        fixed = [p for p in scene["platforms"] if not isinstance(p, dict)]  # (lifts are dictionaries)
        floor = [p for p in fixed if p[1] == FLOOR]
        gaps = [(a[0] + a[2], b[0]) for a, b in zip(sorted(floor), sorted(floor)[1:]) if b[0] > a[0] + a[2]]
        assert len(gaps) == 1  # one stretch with no floor
        start, end = gaps[0]
        ledges = sorted(p for p in fixed if p[1] < FLOOR and start <= p[0] < end)
        assert made.LEDGE_COUNT[0] <= len(ledges) <= made.LEDGE_COUNT[1]
        before_x, before_height = start, 0
        for x, top, width, _ in ledges:
            height = (FLOOR - top) // TILE
            assert width // TILE in made.LEDGE_TILES and 1 <= height <= max(made.LEDGE_HEIGHTS)
            assert height - before_height <= made.REACH
            if height > before_height:
                assert (x - before_x) // TILE <= made.RISE_GAP
            before_x, before_height = x + width, height


def test_ledges_scenes_can_be_passed():
    for seed in range(3):
        assert passable(skill_scene(seed, "ledges"))["passable"]
