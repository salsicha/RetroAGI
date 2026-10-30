"""Train the world model and the collision trackers from the saved datasets
(user, 2026-10-18: "After the tactics run is done then the models need to be
trained"; the datasets are made once and not made again).

- The world model (the policy's LSTM, ``policy.memory``) learns from the
  skill and tactic datasets (each round's episodes: layered_train.save_round),
  their executed actions and the teacher's tried ones
  (world_model_train.episode_data); the scene encoder it reads stays as it is.
- The collision trackers learn from those and from the collision lessons
  (collision_train: the GRU's own data; nothing else learns from it).

Each episode is turned into the networks' inputs and labels once and the
episodes are let go; both networks then make ``epochs`` passes. The policy
(with its world model) is saved as policy.pt and the trackers as tracker.pt;
they are measured afterwards on fresh layouts, by the tactic layer's run
that starts from them (layered_train.train_layer's evaluation).

Example:
    python -m retroagi.stages.block_smb.dataset_train \\
        --source artifacts/block_smb/skill_data_20261018/best.pt \\
        --tracker artifacts/block_smb/collision_20261018/tracker.pt \\
        --rounds artifacts/block_smb/skill_data_20261018/episodes \\
                 artifacts/block_smb/tactic_data_20261018/episodes \\
        --collision artifacts/block_smb/collision_20261018/dataset \\
        --output artifacts/block_smb/predictions_20261018
"""

import argparse
import gzip
import json
import pickle
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import torch


@dataclass
class DatasetTrainConfig:
    source: str = ""  # the policy whose world model learns (its scene encoder is read, not changed)
    tracker: str = ""  # the trackers to go on training ("" for new ones)
    rounds: tuple = ()  # folders of saved rounds (episodes/: round_NN/), for both networks
    collision: tuple = ()  # collision lessons' dataset folders, for the trackers only
    output: str = "artifacts/block_smb/predictions"
    epochs: int = 20
    device: str = "cuda"
    seed: int = 0


def _round_folders(folder: Path) -> list:
    from .layered_train import ROUND_COMPLETE

    return sorted(p for p in Path(folder).glob("round_*") if (p / ROUND_COMPLETE).exists())


def _collision_records(folder: Path) -> list:
    summary = json.loads((Path(folder) / "complete.json").read_text())
    out = []
    for family in summary["families"]:
        with gzip.open(Path(folder) / f"{family}.pkl.gz", "rb") as handle:
            out += pickle.load(handle)
    return out


def train(config: DatasetTrainConfig) -> dict:
    from . import layered_train as lt
    from . import world_model_train as wm
    from retroagi.core.target_tracker import CollisionTracker

    from .tracker_train import TrackerConfig, batch_loss, load_tracker, save_tracker, track_sequences

    random.seed(config.seed)
    torch.manual_seed(config.seed)
    device = torch.device(config.device)
    output = Path(config.output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "config.json").write_text(json.dumps(asdict(config), indent=2) + "\n")
    policy, saved = lt.load_layered_checkpoint(config.source, device)
    policy.eval()
    tracker = (
        load_tracker(config.tracker, device)
        if config.tracker
        else CollisionTracker(TrackerConfig().width).to(device)
    )
    world, tracks, counts = [], [], {}
    started = time.time()
    for folder in config.rounds:
        episodes = 0
        for round_folder in _round_folders(folder):
            records, _ = lt.load_round(round_folder)
            for record in records:
                if len(record.decision_frames) and record.truth is not None:
                    world.append(wm.episode_data(wm.kept(record)))
                if record.truth is not None:
                    tracks += track_sequences(record)
            episodes += len(records)
        counts[str(folder)] = episodes
    for folder in config.collision:
        records = _collision_records(folder)
        for record in records:
            tracks += track_sequences(record)
        counts[str(folder)] = len(records)
    print(
        f"[datasets] {len(world)} episodes for the world model, {len(tracks)} enemy tracks for "
        f"the trackers, from {counts}; {time.time() - started:.0f} s",
        flush=True,
    )
    for parameter in policy.parameters():
        parameter.requires_grad_(False)
    for parameter in policy.memory.parameters():
        parameter.requires_grad_(True)
    settings = wm.WorldModelConfig()
    world_optimizer = torch.optim.AdamW(
        policy.memory.parameters(), lr=settings.learning_rate, weight_decay=settings.weight_decay
    )
    tracker_settings = TrackerConfig()
    tracker_optimizer = torch.optim.AdamW(
        tracker.parameters(), lr=tracker_settings.learning_rate, weight_decay=tracker_settings.weight_decay
    )
    history = []
    for epoch in range(config.epochs):
        started = time.time()
        policy.memory.train()
        random.shuffle(world)
        world_losses = []
        for batch in wm.batches(world, settings):
            loss = wm._forward(policy, batch, device, settings.tried_weight)
            world_optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.memory.parameters(), 1.0)
            world_optimizer.step()
            world_losses.append(float(loss))
        policy.memory.eval()
        tracker.train()
        random.shuffle(tracks)
        tracker_losses = []
        for start in range(0, len(tracks), tracker_settings.batch_tracks):
            loss, _ = batch_loss(tracker, tracks[start : start + tracker_settings.batch_tracks], device)
            tracker_optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(tracker.parameters(), 1.0)
            tracker_optimizer.step()
            tracker_losses.append(float(loss))
        tracker.eval()
        history.append(
            {
                "epoch": epoch,
                "world_model_loss": float(np.mean(world_losses)) if world_losses else None,
                "tracker_loss": float(np.mean(tracker_losses)) if tracker_losses else None,
                "seconds": round(time.time() - started, 1),
            }
        )
        print(json.dumps(history[-1]), flush=True)
        save_tracker(output / "tracker.pt", tracker, TrackerConfig(width=tracker.width), history)
        from retroagi.core.world_model import scene_digest

        world_model = {
            "reads": scene_digest(policy),
            "history": history,
            "trained_with": "datasets_v1",
            "datasets": counts,
        }
        lt.save_layered_checkpoint(
            output / "policy.pt",
            policy,
            lt.LayeredTrainConfig(**{**lt.saved_settings(saved["config"]), "output": str(output)}),
            list(saved.get("trained_layers", ())),
            saved.get("history", []),
            world_model,
        )
    (output / "history.json").write_text(json.dumps(history, indent=2) + "\n")
    return {"history": history, "datasets": counts}


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--source", required=True)
    parser.add_argument("--tracker", default="")
    parser.add_argument("--rounds", nargs="*", default=[])
    parser.add_argument("--collision", nargs="*", default=[])
    parser.add_argument("--output", default=DatasetTrainConfig.output)
    parser.add_argument("--epochs", type=int, default=DatasetTrainConfig.epochs)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    train(
        DatasetTrainConfig(
            source=args.source,
            tracker=args.tracker,
            rounds=tuple(args.rounds),
            collision=tuple(args.collision),
            output=args.output,
            epochs=args.epochs,
            device=args.device,
            seed=args.seed,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
