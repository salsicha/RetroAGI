"""Measure the Block SMB vision transformer on held-out validation frames.

Frames: every Monte Carlo family at every difficulty on the VALIDATION split,
each layout played with its teacher route and with a perturbed teacher route.
The measurements and the printed table are the shared ones
(retroagi.core.pixel_vision.evaluate_pixel_vision and vision_report), so the
Full SMB model's evaluation reports the same numbers with the same meaning:

- pixels correct, and each type's "found" (share of its true pixels given that
  type) and "correct" (share of pixels given that type that truly are it):
  compare() over all frames' pixels pooled together;
- frames where Mario is found: of frames whose true labels show Mario, the
  share whose predicted labels show Mario;
- Mario position error: pixels between the predicted and true label images'
  Mario centres (mario_position), over frames where both show him;
- standing/air agreement: mario_standing of the predicted labels against the
  simulator's on_ground, over frames whose true labels show Mario;
- enemies seen: a true enemy object (a drawn enemy with at least one true
  enemy pixel) is seen if any of its pixels is labelled enemy.

Example:
    python scripts/vision/evaluate_block_vision.py --checkpoint data/block_vit/block_vit_pixel.pth
"""

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from retroagi.core import select_device
from retroagi.core.pixel_vision import evaluate_pixel_vision, vision_report
from retroagi.stages.block_smb.vision import (
    DEFAULT_BLOCK_VIT_CHECKPOINT,
    load_block_vit_checkpoint,
)
from retroagi.stages.block_smb.vision_frames import family_layouts, held_out_frames


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_BLOCK_VIT_CHECKPOINT)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--repeats", type=int, default=2, help="layouts per family and difficulty"
    )
    parser.add_argument("--keep", type=float, default=0.2, help="share of frames measured")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output", type=Path, help="JSON path (default: beside checkpoint)")
    args = parser.parse_args()

    started = time.time()
    device = select_device(args.device)
    loaded = load_block_vit_checkpoint(args.checkpoint, device=device, freeze=True)
    with ProcessPoolExecutor(args.workers) as pool:
        layouts = family_layouts("validation", args.seed, args.repeats, executor=pool)
    metrics = evaluate_pixel_vision(
        loaded.model,
        held_out_frames(layouts, args.seed, keep=args.keep),
        batch_size=args.batch_size,
    )
    print("Block SMB vision transformer, validation split")
    print(vision_report(metrics))
    output = args.output or args.checkpoint.with_name(f"{args.checkpoint.stem}_evaluation.json")
    report = {
        "checkpoint": str(args.checkpoint),
        "split": "validation",
        "seed": args.seed,
        "layouts": len(layouts),
        "routes": ["teacher", "perturbed"],
        "keep": args.keep,
        "learned_numbers": sum(p.numel() for p in loaded.model.parameters()),
        "seconds": round(time.time() - started, 1),
        **metrics,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"\nWrote {output}")


if __name__ == "__main__":
    main()
