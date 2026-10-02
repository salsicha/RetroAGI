"""Count what appears on screen at once in both games, to size the policy's object lists.

The policy receives fixed-length lists (smb_scene_labels.SceneObservation
packed by the observer): so many enemies, coins, surfaces, and so on. This
survey runs the shared ground-truth labels (scene_from_labels) over Block SMB
training frames and Full SMB frames from every level start, and reports for
each list the 99th and 99.9th percentiles and the maximum number per frame.
The chosen size of each list is the larger of the two games' 99.9th
percentiles (at least 1).

It also counts frames where two objects of one kind have their centres in the
same 8x8 cell (the vision transformer finds one object per cell and kind),
and can draw the labels onto sample frames for inspection.

Example:
    python scripts/vision/survey_scene_slots.py --pictures /tmp/scene_survey
"""

import argparse
import json
import math
import random
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from itertools import islice
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from retroagi.core.smb_scene_labels import (
    SceneObservation,
    scene_counts,
    scene_from_labels,
)

LISTS = ("enemies", "coins", "power_ups", "moving_platforms", "pipes", "blocks", "surfaces", "gaps")
CELL = 8


def _same_cell(scene: SceneObservation) -> list[str]:
    """Kinds of object with two centres in one 8x8 cell."""
    found = []
    groups = {
        "enemy": [e.box for e in scene.enemies],
        "coin": scene.coins,
        "power_up": scene.power_ups,
        "moving_platform": scene.moving_platforms,
    }
    for name, boxes in groups.items():
        cells = [((x0 + x1) // 2 // CELL, (y0 + y1) // 2 // CELL) for x0, y0, x1, y1 in boxes]
        if len(cells) != len(set(cells)):
            found.append(name)
    return found


def _summary(counts: list[dict]) -> dict:
    table = {}
    for name in LISTS:
        values = np.array([c[name] for c in counts])
        table[name] = {
            "mean": float(values.mean()),
            "p99": float(np.percentile(values, 99)),
            "p99_9": float(np.percentile(values, 99.9)),
            "max": int(values.max()),
        }
    return table


def draw(image: np.ndarray, scene: SceneObservation, path: Path, scale: int = 3) -> None:
    """Save the frame with every reported object drawn on it."""
    from PIL import Image, ImageDraw

    picture = Image.fromarray(image).resize((256 * scale, 240 * scale), Image.NEAREST)
    pen = ImageDraw.Draw(picture)

    def box(b, colour, label=""):
        x0, y0, x1, y1 = (v * scale for v in b)
        pen.rectangle((x0, y0, x1 - 1, y1 - 1), outline=colour, width=2)
        if label:
            pen.text((x0 + 2, y0 + 1), label, fill=colour)

    for b in scene.pipes:
        box(b, (0, 255, 0), "pipe")
    for b in scene.blocks:
        box(b.box, (255, 160, 0), "?" if b.kind == "question_block" else "brick")
    for s in scene.surfaces:
        colour = (0, 255, 255) if s.moving else (255, 255, 255)
        pen.line(
            (s.x0 * scale, s.top * scale, s.x1 * scale - 1, s.top * scale), fill=colour, width=3
        )
    for g in scene.gaps:
        pen.rectangle(
            (g.x0 * scale, 232 * scale, g.x1 * scale - 1, 240 * scale - 1), fill=(255, 0, 0)
        )
    for b in scene.moving_platforms:
        box(b, (0, 255, 255), "lift")
    for b in scene.coins:
        box(b, (255, 255, 0))
    for b in scene.power_ups:
        box(b, (255, 128, 0), "power-up")
    for e in scene.enemies:
        box(e.box, (255, 0, 255), e.kind)
    if scene.mario.box is not None:
        facing = ">" if scene.mario.facing_right else "<"
        box(scene.mario.box, (255, 0, 0), f"Mario {facing} {scene.mario.support}")
    picture.save(path)


# ── Block SMB ─────────────────────────────────────────────────────────────────


def block_survey(frames: int, seed: int, workers: int, pictures: Path | None) -> dict:
    from retroagi.stages.block_smb.vision_frames import family_layouts, frame_stream

    with ProcessPoolExecutor(workers) as pool:
        layouts = family_layouts("train", seed, 2, executor=pool)
    counts, collisions = [], Counter()
    rng = random.Random(seed)
    shown = 0
    for frame in islice(frame_stream(layouts, seed), frames):
        scene = scene_from_labels(frame.scene)
        counts.append(scene_counts(scene))
        collisions.update(_same_cell(scene))
        if pictures is not None and shown < 12 and rng.random() < 0.02:
            draw(frame.image, scene, pictures / f"block_{shown:02d}_{frame.family}.png")
            shown += 1
    return {"frames": len(counts), "lists": _summary(counts), "same_cell": dict(collisions)}


# ── Full SMB ──────────────────────────────────────────────────────────────────


def _full_play(level: str, seed: int, every: int, picture_dir: str | None):
    from retroagi.stages.full_smb.vision_frames import labelled_frames

    refusals = Counter()
    counts, collisions = [], Counter()
    rng = random.Random(seed)
    shown = 0
    for frame in labelled_frames(level, frames=9000, seed=seed, every=every, refusals=refusals):
        scene = scene_from_labels(frame.scene)
        counts.append(scene_counts(scene))
        collisions.update(_same_cell(scene))
        if picture_dir is not None and shown < 2 and rng.random() < 0.01:
            draw(frame.image, scene, Path(picture_dir) / f"full_{level}_{seed}_{shown}.png")
            shown += 1
    return counts, collisions, refusals


def full_survey(plays: int, every: int, seed: int, workers: int, pictures: Path | None) -> dict:
    from retroagi.stages.full_smb.vision_frames import LEVELS

    rng = random.Random(seed)
    jobs = [(level, rng.randrange(2**31), every) for level in LEVELS for _ in range(plays)]
    counts, collisions, refusals = [], Counter(), Counter()
    with ProcessPoolExecutor(workers) as pool:
        for play_counts, play_collisions, play_refusals in pool.map(
            _full_play, *zip(*jobs), [str(pictures) if pictures else None] * len(jobs)
        ):
            counts += play_counts
            collisions += play_collisions
            refusals += play_refusals
    return {
        "frames": len(counts),
        "refused_frames": sum(refusals.values()),
        "lists": _summary(counts),
        "same_cell": dict(collisions),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--block-frames", type=int, default=20_000)
    parser.add_argument("--full-plays", type=int, default=2, help="plays of each level start")
    parser.add_argument("--every", type=int, default=6, help="Full SMB: keep every n-th frame")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--pictures", type=Path, help="directory for labelled sample frames")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "data/scene_slot_survey.json")
    args = parser.parse_args()
    if args.pictures is not None:
        args.pictures.mkdir(parents=True, exist_ok=True)

    games = {
        "block_smb": block_survey(args.block_frames, args.seed, args.workers, args.pictures),
        "full_smb": full_survey(
            args.full_plays, args.every, args.seed, args.workers, args.pictures
        ),
    }
    slots = {
        name: max(1, max(math.ceil(games[g]["lists"][name]["p99_9"]) for g in games))
        for name in LISTS
    }
    report = {"games": games, "slots": slots}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    for game, result in games.items():
        print(f"\n{game}: {result['frames']} frames")
        print(f"{'list':<18}{'mean':>7}{'p99':>7}{'p99.9':>7}{'max':>6}")
        for name in LISTS:
            row = result["lists"][name]
            print(
                f"{name:<18}{row['mean']:>7.2f}{row['p99']:>7.1f}{row['p99_9']:>7.1f}{row['max']:>6}"
            )
        print("two centres in one 8x8 cell:", result["same_cell"] or "never")
    print("\nchosen list sizes:", slots)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
