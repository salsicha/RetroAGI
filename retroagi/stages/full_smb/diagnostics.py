"""Measure the Full SMB vision transformer on held-out real frames.

This is `retroagi diagnose-vision --game smb --stage full`. Frames come from
fresh plays of every level start (vision_frames.LEVELS), each played from its
saved start by the random player in vision_frames, with seeds training does
not use. Every
frame's true scene is read from game memory (pixel_labels.label_frame);
a frame memory cannot fully explain is counted by its reason and never
measured. The measurements are the shared ones
(retroagi.core.scene_vision.evaluate_scene_vision), so they mean exactly what
the Block SMB diagnostic reports. scripts/vision/evaluate_full_vision.py takes
the same measurement with several worker processes.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any, Iterator, Optional, Sequence

from retroagi.core import select_device, to_plain_data
from retroagi.core.scene_vision import evaluate_scene_vision
from retroagi.stages.full_smb.vision import DEFAULT_FULL_VIT_CHECKPOINT, load_full_vit_checkpoint
from retroagi.stages.full_smb.vision_frames import LEVELS, labelled_frames

# Frames per play before the level is restarted; most plays finish sooner.
PLAY_FRAMES = 9_000


def run_full_smb_vision_diagnostic(
    model: Any,
    *,
    seed: int = 0,
    plays: int = 1,
    every: int = 4,
    frames: int = PLAY_FRAMES,
    batch_size: int = 32,
) -> dict[str, Any]:
    """The shared vision measurements on test-level frames.

    Each test level is played ``plays`` times, every ``every``-th frame is
    labelled from game memory, and ``refused_frames`` counts the frames memory
    could not fully explain, by reason.
    """
    if plays <= 0:
        raise ValueError("plays must be positive")
    rng = random.Random(seed)
    play_seeds = [(level, rng.randrange(2**31)) for level in LEVELS for _ in range(plays)]
    refusals: Counter = Counter()

    def frame_stream() -> Iterator[Any]:
        for level, play_seed in play_seeds:
            yield from labelled_frames(
                level, frames=frames, seed=play_seed, every=every, refusals=refusals
            )

    metrics = evaluate_scene_vision(model, frame_stream(), batch_size=batch_size)
    return {
        "levels": list(LEVELS),
        "plays": len(play_seeds),
        **metrics,
        "refused_frames": dict(refusals),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="retroagi diagnose-vision --stage full")
    parser.add_argument("--vision-checkpoint", type=Path, default=DEFAULT_FULL_VIT_CHECKPOINT)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--plays", type=int, default=1, help="plays of each test level")
    parser.add_argument("--every", type=int, default=4, help="measure every n-th frame")
    parser.add_argument("--frames", type=int, default=PLAY_FRAMES, help="frames per play")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--output", type=Path)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    device = select_device(args.device)
    loaded = load_full_vit_checkpoint(args.vision_checkpoint, device=device, freeze=True)
    result = run_full_smb_vision_diagnostic(
        loaded.model,
        seed=args.seed,
        plays=args.plays,
        every=args.every,
        frames=args.frames,
        batch_size=args.batch_size,
    )
    payload = {
        "config": {
            "seed": args.seed,
            "plays": args.plays,
            "every": args.every,
            "frames": args.frames,
            "batch_size": args.batch_size,
            "device": str(device),
        },
        "vision": {
            "checkpoint_path": str(loaded.path),
            "frozen": loaded.frozen,
        },
        **result,
    }
    output = json.dumps(to_plain_data(payload), indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output + "\n", encoding="utf-8")
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
