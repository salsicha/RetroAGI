"""Play Full SMB levels with a layered agent trained in Block SMB.

The agent sees each level only through the Full SMB vision transformer and
plays exactly as deployed: every token is its own, under the strategy switch
given here (the goal is always to the right). Game memory is read only to
score each run: how far into the level Mario got, and whether he died or
finished.

Example:
    python -m retroagi.stages.full_smb.layered_eval \\
        --checkpoint artifacts/block_smb/layered/best.pt --levels Level1-1 Level5-1
"""

import argparse
import json
from pathlib import Path

from retroagi.core.smb_agent import SMBAgents
from retroagi.core.smb_observer import VisionObserver
from retroagi.core.tokens import DEFAULT_STRATEGY, STRATEGIES, StrategyToken

from .play import POLICY_TEST_LEVELS, play_full_levels
from .vision import DEFAULT_FULL_VIT_CHECKPOINT, load_full_vit_checkpoint


def evaluate_on_full(
    checkpoint: Path,
    levels=POLICY_TEST_LEVELS,
    *,
    frames: int = 4000,
    device: str = "cuda",
    vision_checkpoint: Path = DEFAULT_FULL_VIT_CHECKPOINT,
    strategy: str = DEFAULT_STRATEGY.kind,
) -> dict:
    from retroagi.stages.block_smb.layered_train import load_layered_checkpoint

    policy, saved = load_layered_checkpoint(checkpoint, device)
    policy.eval()
    observer = VisionObserver(load_full_vit_checkpoint(vision_checkpoint, device=device).model)
    switch = StrategyToken(strategy, 1)  # Full SMB's goal is always to the right
    agents = SMBAgents(observer, policy, device, len(levels), switch=switch)
    runs = play_full_levels(agents, levels, frames)
    return {
        "checkpoint": str(checkpoint),
        "trained_layers": saved["trained_layers"],
        "strategy": strategy,
        "levels": {
            run.level: {
                "furthest_x": run.furthest_x,
                "frames": run.frames,
                "died": run.died,
                "finished": run.finished,
            }
            for run in runs
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--levels", nargs="+", default=list(POLICY_TEST_LEVELS))
    parser.add_argument("--frames", type=int, default=4000)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--vision-checkpoint", type=Path, default=DEFAULT_FULL_VIT_CHECKPOINT)
    parser.add_argument("--strategy", choices=STRATEGIES, default=DEFAULT_STRATEGY.kind)
    args = parser.parse_args()
    result = evaluate_on_full(
        args.checkpoint,
        tuple(args.levels),
        frames=args.frames,
        device=args.device,
        vision_checkpoint=args.vision_checkpoint,
        strategy=args.strategy,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
