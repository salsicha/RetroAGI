"""The family atlas's sheets: each family's pictures tiled a row a difficulty
and a column a quantile, a layout the generator does not draw left blank."""

from PIL import Image

from retroagi.stages.block_smb.family_atlas import family_sheets


def test_a_family_sheet_tiles_its_pictures_and_leaves_rejected_layouts_blank(tmp_path):
    (tmp_path / "walk").mkdir()
    entries = []
    for difficulty in ("easy", "hard"):
        for q in (0.0, 1.0):
            if (difficulty, q) == ("hard", 1.0):
                entries.append({"family": "walk", "difficulty": difficulty, "quantile": q, "rejected": "edge_landing"})
                continue
            name = f"walk/{difficulty}_q{round(q * 100):03d}.png"
            Image.new("RGB", (40, 30), (255, 0, 0)).save(tmp_path / name)
            entries.append({"family": "walk", "difficulty": difficulty, "quantile": q, "image": name, "won": True})
    (path,) = family_sheets(tmp_path, entries)
    sheet = Image.open(path)
    # Two columns, two rows of 40 x 30 pictures with 8-pixel gaps under a heading.
    assert sheet.size == (2 * 48 + 8, 32 + 2 * 38 + 8)
    assert sheet.getpixel((8 + 20, 32 + 8 + 15)) == (255, 0, 0)  # easy, 0%
    assert sheet.getpixel((8 + 48 + 20, 32 + 8 + 38 + 20)) != (255, 0, 0)  # hard, 100%: not drawn
