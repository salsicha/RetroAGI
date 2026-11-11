"""The layered agent on the obstacle bench (framework/bench.run_bench).

    python -m retroagi.legacy.full_smb.layered_player --bench DIR \\
        --checkpoint AGENT.pt --tracker TRACKER.pt [--runs 4] [--output OUT.json]
"""

import argparse
from pathlib import Path

from retroagi.domains.smb.nes.world import NESWorld
from retroagi.framework.bench import report, run_bench


class LayeredPlayer:
    """The layered agent as the 1-1 playthroughs play it: the Full SMB vision,
    the checkpoint's layers and world model, its collision trackers; the
    tactic and skill choices sampled from seeded randomness."""

    idle = 0  # (the action that does nothing)

    def __init__(self, checkpoint: str, tracker: str, device: str = "cuda",
                 vision: str = "data/full_vit/full_vit_scene.pth"):
        import torch

        from retroagi.domains.smb.nes.vision import load_full_vit_checkpoint
        from retroagi.legacy.block_smb import layered_train as lt
        from retroagi.legacy.block_smb.tracker_train import load_tracker
        from retroagi.legacy.core.smb_agent import SMBAgents
        from retroagi.legacy.core.smb_observer import VisionObserver

        self.torch = torch
        policy, _ = lt.load_layered_checkpoint(checkpoint, device)
        policy.eval()
        model = load_full_vit_checkpoint(vision, device=device).model.eval()
        self.agents = SMBAgents(VisionObserver(model), policy, device, 1, tracker=load_tracker(tracker, device),
                                world_model=True, watch_threats=True)

    def reset(self, seed: int, mode: str) -> None:
        from retroagi.legacy.core.tokens import StrategyToken

        self.torch.manual_seed(1000 * (mode == "max_points") + seed)
        self.agents.reset(0, StrategyToken(mode, 1))

    def act(self, world) -> int:
        with self.torch.no_grad():
            return self.agents.act([world.screen], [0], sample=("skill", "tactic"))[0].button


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--bench", type=Path, required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--tracker", required=True)
    parser.add_argument("--runs", type=int, default=4)
    parser.add_argument("--only", nargs="*", default=())
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    results = run_bench(args.bench, NESWorld, lambda: LayeredPlayer(args.checkpoint, args.tracker),
                        ("speed_run", "max_points"), args.runs, only=args.only)
    report(results, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
