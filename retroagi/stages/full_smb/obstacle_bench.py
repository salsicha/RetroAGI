"""Score a player on each obstacle of a level (the obstacle test bench,
docs/choose-and-control-plan.md phase 0; user, 2026-11-07).

From each obstacle's saved state (obstacles.save_bench) the player plays
until Mario stands PASSED pixels beyond the obstacle's end (passed), dies
(died) or ATTEMPT_FRAMES pass (stalled). Nothing learns. A player has
``reset(seed, mode)`` and ``act(screen, game) -> button``; LayeredPlayer
wraps the layered agent (core.smb_agent.SMBAgents) as the 1-1 playthroughs
play it.

    python -m retroagi.stages.full_smb.obstacle_bench --bench DIR \\
        (--checkpoint AGENT.pt --tracker TRACKER.pt | --player scripted) \\
        [--runs 4] [--output OUT.json]

``--player scripted``: the hand-written chooser reading game memory with the
choose-and-control controller (scripted_chooser), testing the controller
alone.
"""

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Callable, Sequence

from .pixel_labels import PLAYER_FLOAT_STATE
from .play import FullSMBGame
from .truth import true_frame

# Pixels beyond an obstacle's end where Mario, standing, has passed it.
PASSED = 24
# Frames a player has for one obstacle.
ATTEMPT_FRAMES = 900
MODES = ("speed_run", "max_points")


def attempt(game: FullSMBGame, player, seed: int, mode: str, obstacle: dict) -> dict:
    """One try at ``obstacle`` from the game's start state: its end ("passed",
    "died", "stalled"), the frames it took and Mario's furthest level x."""
    screen = game.reset()
    player.reset(seed, mode)
    goal = obstacle["x1"] + PASSED
    furthest = 0
    for frame in range(ATTEMPT_FRAMES):
        screen, outcome = game.step(player.act(screen, game))
        ram = game.env.get_ram()
        truth = true_frame(ram, game.start)
        furthest = max(furthest, truth["level_x"])
        if outcome.dying:
            return {"end": "died", "frames": frame + 1, "furthest": furthest}
        standing = int(ram[PLAYER_FLOAT_STATE]) == 0
        if outcome.finished or (truth["level_x"] >= goal and standing):
            return {"end": "passed", "frames": frame + 1, "furthest": furthest}
    return {"end": "stalled", "frames": ATTEMPT_FRAMES, "furthest": furthest}


def run_bench(bench: Path, make_player: Callable[[], object], runs: int = 4,
              modes: Sequence[str] = MODES, only: Sequence[str] = ()) -> dict:
    """Every obstacle of ``bench`` tried ``runs`` times in each mode (seeds 0
    to runs - 1); ``only``: obstacle names to try (all when empty)."""
    index = json.loads((bench / "index.json").read_text())
    player = make_player()
    results = {"level": index["level"], "runs": runs, "obstacles": []}
    for obstacle in index["obstacles"]:
        if only and obstacle["name"] not in only:
            continue
        state = (bench / f"{obstacle['name']}.state").read_bytes()
        game = FullSMBGame(index["level"], state)
        tries = {}
        try:
            for mode in modes:
                tries[mode] = [attempt(game, player, seed, mode, obstacle) for seed in range(runs)]
        finally:
            game.close()
        ends = Counter(t["end"] for mode_tries in tries.values() for t in mode_tries)
        total = sum(ends.values())
        results["obstacles"].append({**obstacle, "passed": ends["passed"] / max(1, total),
                                     "ends": dict(ends), "tries": tries})
        print(f"{obstacle['name']:24} passed {ends['passed']:2}/{total}  died {ends['died']:2}  "
              f"stalled {ends['stalled']:2}", flush=True)
    passed = [o["passed"] for o in results["obstacles"]]
    results["passed"] = sum(passed) / max(1, len(passed))
    return results


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
    parser.add_argument("--player", choices=("layered", "scripted"), default="layered")
    parser.add_argument("--checkpoint")
    parser.add_argument("--tracker")
    parser.add_argument("--runs", type=int, default=4)
    parser.add_argument("--only", nargs="*", default=())
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if args.player == "scripted":
        from .scripted_chooser import ScriptedPlayer

        level = json.loads((args.bench / "index.json").read_text())["level"]
        make_player = lambda: ScriptedPlayer(level)  # noqa: E731
    else:
        if not (args.checkpoint and args.tracker):
            parser.error("the layered player needs --checkpoint and --tracker")
        make_player = lambda: LayeredPlayer(args.checkpoint, args.tracker)  # noqa: E731
    results = run_bench(args.bench, make_player, args.runs, only=args.only)
    print(f"obstacles passed on average: {results['passed']:.2f}")
    if args.output:
        args.output.write_text(json.dumps(results, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
