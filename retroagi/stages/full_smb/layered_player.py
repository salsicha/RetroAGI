"""The layered agent on the obstacle bench (obstacle_bench.run_bench).

    python -m retroagi.stages.full_smb.layered_player --bench DIR \\
        --checkpoint AGENT.pt --tracker TRACKER.pt [--runs 4] [--output OUT.json]
"""

import argparse
import json
from pathlib import Path

from .obstacle_bench import run_bench


class LayeredPlayer:
    """The layered agent as the 1-1 playthroughs play it: the Full SMB vision,
    the checkpoint's layers and world model, its collision trackers; the
    tactic and skill choices sampled from seeded randomness."""

    def __init__(self, checkpoint: str, tracker: str, device: str = "cuda",
                 vision: str = "data/full_vit/full_vit_scene.pth"):
        import torch

        from retroagi.core.smb_agent import SMBAgents
        from retroagi.core.smb_observer import VisionObserver
        from retroagi.stages.block_smb import layered_train as lt
        from retroagi.stages.block_smb.tracker_train import load_tracker

        from .vision import load_full_vit_checkpoint

        self.torch = torch
        policy, _ = lt.load_layered_checkpoint(checkpoint, device)
        policy.eval()
        model = load_full_vit_checkpoint(vision, device=device).model.eval()
        self.agents = SMBAgents(VisionObserver(model), policy, device, 1, tracker=load_tracker(tracker, device),
                                world_model=True, watch_threats=True)

    def reset(self, seed: int, mode: str) -> None:
        from retroagi.core.tokens import StrategyToken

        self.torch.manual_seed(1000 * (mode == "max_points") + seed)
        self.agents.reset(0, StrategyToken(mode, 1))

    def act(self, screen, game) -> int:
        with self.torch.no_grad():
            return self.agents.act([screen], [0], sample=("skill", "tactic"))[0].button


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--bench", type=Path, required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--tracker", required=True)
    parser.add_argument("--runs", type=int, default=4)
    parser.add_argument("--only", nargs="*", default=())
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    results = run_bench(args.bench, lambda: LayeredPlayer(args.checkpoint, args.tracker), args.runs, only=args.only)
    print(f"obstacles passed on average: {results['passed']:.2f}")
    if args.output:
        args.output.write_text(json.dumps(results, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
