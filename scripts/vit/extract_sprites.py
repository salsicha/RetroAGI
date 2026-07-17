"""
extract_sprites.py
==================
Slice accurate Super Mario Bros sprites out of the ripped sprite sheets
(from the well-known `justinmeister/Mario-Level-1` asset set) into individual
transparent PNGs that the scene generator composes.

Sheet coordinates are taken verbatim from that project's own component code
(data/components/*.py), so the crops are pixel-accurate to the original NES art.

Run:
    python scripts/vit/extract_sprites.py
Outputs:
    assets/sprites/<name>.png   (one transparent sprite per semantic class)
"""
import os
import numpy as np
from PIL import Image

HERE        = os.path.dirname(os.path.abspath(__file__))
PROJECT     = os.path.dirname(os.path.dirname(HERE))
SHEET_DIR   = os.path.join(PROJECT, "assets", "spritesheets")
OUT_DIR     = os.path.join(PROJECT, "assets", "sprites")

SKY_RGB = (107, 140, 255)   # cornflower-blue SMB sky (sampled from level_1.png)


def load(name):
    return Image.open(os.path.join(SHEET_DIR, f"{name}.png")).convert("RGBA")


def crop(sheet, x, y, w, h):
    """Crop (x,y,w,h) and keep the existing alpha channel."""
    return sheet.crop((x, y, x + w, y + h))


def autocrop_sky(img, sky=SKY_RGB, tol=12):
    """Replace sky-blue with transparency and trim to the content bbox.
    Used for tiles cropped out of the baked level image (pipes, hills, etc.)."""
    arr = np.array(img.convert("RGBA"))
    r, g, b = arr[..., 0].astype(int), arr[..., 1].astype(int), arr[..., 2].astype(int)
    is_sky = (abs(r - sky[0]) <= tol) & (abs(g - sky[1]) <= tol) & (abs(b - sky[2]) <= tol)
    arr[is_sky, 3] = 0
    ys, xs = np.where(arr[..., 3] > 0)
    if len(xs) == 0:
        return Image.fromarray(arr)
    return Image.fromarray(arr[ys.min():ys.max() + 1, xs.min():xs.max() + 1])


def save(img, name):
    os.makedirs(OUT_DIR, exist_ok=True)
    img.save(os.path.join(OUT_DIR, f"{name}.png"))
    print(f"  saved {name:14s} {img.size}")


def main():
    print("Extracting sprites...")

    tile  = load("tile_set")          # bricks, question blocks
    item  = load("item_objects")      # coins, mushroom
    enem  = load("smb_enemies_sheet") # goomba, koopa
    level = load("level_1")           # baked level: ground / pipe / hill / cloud / bush

    # ── Blocks (coords from data/components/bricks.py & coin_box.py) ──────────
    save(crop(tile, 16,  0, 16, 16), "brick")          # overworld brick
    save(crop(tile, 384, 0, 16, 16), "question_block") # ? block frame 0

    # ── Coin (round gold collectible coin from item_objects sheet) ────────────
    save(crop(item, 0, 96, 16, 16), "coin")

    # ── Mushroom power-up (data/components/powerups.py) ───────────────────────
    save(crop(item, 0, 0, 16, 16), "mushroom")

    # ── Enemies (data/components/enemies.py, smb_enemies_sheet) ───────────────
    save(crop(enem, 0,   4, 16, 16), "goomba")
    save(crop(enem, 150, 0, 16, 24), "koopa")

    # ── Mario (data/components/mario.py, mario_bros sheet) ────────────────────
    mario = load("mario_bros")
    save(crop(mario, 178, 32, 12, 16), "mario")        # small Mario, standing right

    # ── Ground tile (clean overworld ground brick, away from hills/bushes) ────
    save(crop(level, 144, 200, 16, 16), "ground")

    # ── Pipe / hill / cloud / bush — auto-cropped from the baked level ────────
    # The ground band in level_1.png starts at y=200, so these crops must extend
    # to y=200 or the sprite bottoms are truncated.
    # A clean 2-tile pipe (lip + body) lives just left of x=640 in level_1.png.
    save(autocrop_sky(crop(level, 600, 144, 40, 56)), "pipe")
    # Big green hill at the very start of the level.
    save(autocrop_sky(crop(level, 0,   128, 80, 72)), "hill")
    # First cloud, upper area.
    save(autocrop_sky(crop(level, 145, 32, 56, 40)),  "cloud")
    # Bush cluster mid-screen.
    save(autocrop_sky(crop(level, 376, 160, 64, 40)), "bush")

    print(f"Done. Sprites in {OUT_DIR}")


if __name__ == "__main__":
    main()
