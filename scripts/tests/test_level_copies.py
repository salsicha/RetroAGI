"""A real level's layout written in Block SMB's format (nes/clone.py), to
check that Block SMB behaves as the game does: the copy made from what was
captured. (The capture itself reads the game's memory and is not run here.)
A copy is never a practice world: test_game_levels checks the domain's
practice kinds."""

from retroagi.domains.smb.nes.clone import scenario
from retroagi.domains.smb.sim.levels import BOTTOM, FLOOR, TILE

CAPTURED = {
    "level": "LevelX-1",
    "pipes": [(320, 352, 160)],
    "pits": [(480, 528)],
    "steps": [(640, 656, 192), (656, 672, 176)],
    "floating": [(800, 816, 144)],
    "blocks": [(256, 144, "question_block"), (272, 144, "brick")],
    # (first seen x, its box's bottom, the game's code): a Goomba, a Koopa on the
    # floating tile, a plant in the pipe, a Hammer Bro (not in Block SMB).
    "enemies": [(400, 204, 0x06), (802, 140, 0x00), (331, 150, 0x0D), (900, 200, 0x05)],
    "finish": 1200,
    "seen_to": 1190,
}


def test_a_captured_level_becomes_a_block_smb_level_of_the_same_pieces():
    copy = scenario(CAPTURED)
    solids = {(p[0], p[1], p[2], kind) for p, kind in zip(copy["platforms"], copy["platform_kinds"])}
    # The floor is cut at the pit and where the pipe and the steps stand.
    floor = sorted(p[:3] for p, kind in zip(copy["platforms"], copy["platform_kinds"])
                   if kind == "ground" and p[1] == FLOOR)
    assert [f[0] for f in floor] == [0, 352, 528, 672] and floor[0][2] == 320 and floor[1][2] == 480 - 352
    assert (320, 160, 32, "pipe") in solids and (640, 192, TILE, "ground") in solids
    assert (800, 144, TILE, "ground") in solids and (256, 144, TILE, "question_block") in solids
    assert copy["goal"] == [1200, 0, TILE, BOTTOM] and copy["world_width"] > 1200
    # Each question block holds something; the enemies Block SMB has stand where they were first seen.
    assert list(copy["block_contents"].values()) == ["coin"]
    kinds = sorted((e.get("kind", "goomba"), e["x"]) for e in copy["enemies"])
    assert kinds == [("goomba", 400.0), ("koopa", 802.0), ("piranha_plant", 328.0)]
    koopa = next(e for e in copy["enemies"] if e.get("kind") == "koopa")
    assert koopa["game_rules"] and koopa["y"] == 144 - 14  # on the floating tile, not the floor under it
    assert copy["metadata"]["game_level"]["left_out"]["Hammer Bro"] == 1

