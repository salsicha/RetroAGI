"""Measure the Full SMB vision transformer on held-out real frames.

Frames: the test levels (vision_frames.TEST_LEVELS), which training never
sees, each played several times from its saved start by the random player.
Labels are read from game memory (pixel_labels.label_frame); frames memory
cannot fully explain are refused and counted, never measured. The
measurements and the printed table are the shared ones
(retroagi.core.pixel_vision.evaluate_pixel_vision and vision_report), so they
mean exactly what scripts/vision/evaluate_block_vision.py reports for the
Block SMB model:

- pixels correct, and each type's "found" (share of its true pixels given that
  type) and "correct" (share of pixels given that type that truly are it):
  compare() over all frames' pixels pooled together;
- frames where Mario is found: of frames whose true labels show Mario, the
  share whose predicted labels show Mario;
- Mario position error: pixels between the predicted and true label images'
  Mario centres (mario_position), over frames where both show him;
- standing/air agreement: mario_standing of the predicted labels against the
  game's own standing flag, over frames whose true labels show Mario;
- enemies seen: a true enemy object (an enemy with at least one true enemy
  pixel) is seen if any of its pixels is labelled enemy.

Example:
    python scripts/vision/evaluate_full_vision.py --checkpoint data/full_vit/full_vit_pixel.pth
"""

import argparse
import json
import random
import sys
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from retroagi.core import select_device
from retroagi.core.pixel_vision import evaluate_pixel_vision, vision_report
from retroagi.stages.full_smb.vision import DEFAULT_FULL_VIT_CHECKPOINT, load_full_vit_checkpoint
from retroagi.stages.full_smb.vision_frames import TEST_LEVELS, labelled_frames

PLAY_FRAMES = 9_000


def _play(level: str, seed: int, every: int) -> tuple[list, Counter]:
    refusals = Counter()
    frames = list(
        labelled_frames(level, frames=PLAY_FRAMES, seed=seed, every=every, refusals=refusals)
    )
    return frames, refusals


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_FULL_VIT_CHECKPOINT)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--plays", type=int, default=4, help="plays of each test level")
    parser.add_argument("--every", type=int, default=4, help="measure every n-th frame")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output", type=Path, help="JSON path (default: beside checkpoint)")
    args = parser.parse_args()

    started = time.time()
    device = select_device(args.device)
    loaded = load_full_vit_checkpoint(args.checkpoint, device=device, freeze=True)
    rng = random.Random(args.seed)
    plays = [
        (level, rng.randrange(2**31), args.every)
        for level in TEST_LEVELS
        for _ in range(args.plays)
    ]
    refusals = Counter()

    def frames():
        with ProcessPoolExecutor(args.workers) as pool:
            for play_frames, play_refusals in pool.map(_play, *zip(*plays)):
                refusals.update(play_refusals)
                yield from play_frames

    metrics = evaluate_pixel_vision(loaded.model, frames(), batch_size=args.batch_size)
    print(f"Full SMB vision transformer, test levels: {', '.join(TEST_LEVELS)}")
    print(vision_report(metrics))
    print(f"Frames refused (not fully explained by memory): {sum(refusals.values())}")
    output = args.output or args.checkpoint.with_name(f"{args.checkpoint.stem}_evaluation.json")
    report = {
        "checkpoint": str(args.checkpoint),
        "levels": list(TEST_LEVELS),
        "seed": args.seed,
        "plays": len(plays),
        "every": args.every,
        "refused_frames": dict(refusals),
        "learned_numbers": sum(p.numel() for p in loaded.model.parameters()),
        "seconds": round(time.time() - started, 1),
        **metrics,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"\nWrote {output}")


if __name__ == "__main__":
    main()
