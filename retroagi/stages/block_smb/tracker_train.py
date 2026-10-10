"""Train the target tracker (core/target_tracker) on the objects vision
tracked in played episodes, and measure it against steady motion.

No teacher is needed: every recorded frame says where each tracked enemy and
moving platform was (EpisodeTask.record_objects), so from any frame the
tracker learns where each will be at each horizon. Episodes come from
predictor_train's collection (it records objects too) or any recorded set.

Step 3 of docs/action-predictor-controller.md.
"""

import json
import pickle
import random
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import torch

from retroagi.core.target_tracker import (
    TRACKER_HORIZONS,
    TargetTracker,
    steady_error,
    track_examples,
    tracker_loss,
)


@dataclass(frozen=True)
class TrackerConfig:
    episodes: str = "artifacts/block_smb/action_predictor/episodes.pkl"
    width: int = 48
    epochs: int = 60
    batch_tracks: int = 256
    learning_rate: float = 2e-3
    weight_decay: float = 0.01
    seed: int = 2026101001
    device: str = "cuda"
    output: str = "artifacts/block_smb/target_tracker"


def examples_of(episodes) -> list:
    out = []
    for episode in episodes:
        if getattr(episode, "objects", None) is not None and len(episode.objects):
            out.extend(track_examples(episode.objects))
    return out


def _padded(batch, device):
    length = max(len(e[0]) for e in batch)
    parts = []
    for k, dtype in enumerate((np.float32, np.float32, np.float32, bool, bool, bool)):
        first = batch[0][k]
        array = np.zeros((len(batch), length, *first.shape[1:]), dtype)
        for i, e in enumerate(batch):
            array[i, : len(e[k])] = e[k]
        parts.append(torch.as_tensor(array, device=device))
    return parts


def batch_loss(model, batch, device):
    inputs, velocity, targets, seen, known, valid = _padded(batch, device)
    states, _ = model.run(inputs)
    predicted = model.forecast(states, velocity)
    mask = valid[..., None] & known
    return tracker_loss(
        {k: v[valid] for k, v in predicted.items()},
        targets[valid],
        (seen & mask)[valid],
        known[valid],
    )


@torch.no_grad()
def evaluate(model, examples, device, batch_tracks: int) -> dict:
    model.eval()
    sums, counts = defaultdict(float), defaultdict(int)
    for start in range(0, len(examples), batch_tracks):
        batch = examples[start : start + batch_tracks]
        _, stats = batch_loss(model, batch, device)
        weight = sum(int(e[5].sum()) for e in batch)
        for key, value in stats.items():
            sums[key] += value * weight
            counts[key] += weight
    model.train()
    return {key: round(sums[key] / counts[key], 2) for key in sums}


def train_tracker(config: TrackerConfig, episodes=None) -> dict:
    """Train on the training episodes, evaluate on the validation ones
    (``episodes``: (train, validation), else read from config.episodes)."""
    output = Path(config.output)
    output.mkdir(parents=True, exist_ok=True)
    if episodes is None:
        episodes = pickle.loads(Path(config.episodes).read_bytes())
    train, validation = (examples_of(part) for part in episodes)
    if not train:
        raise ValueError("no recorded objects: collect episodes with record_objects")
    baseline = steady_error(validation)
    print(f"[tracker] steady motion on held-out tracks: {baseline}", flush=True)
    torch.manual_seed(config.seed)
    rng = random.Random(config.seed)
    model = TargetTracker(config.width).to(config.device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    history, best = [], None
    for epoch in range(config.epochs):
        started = time.time()
        order = list(range(len(train)))
        rng.shuffle(order)
        for start in range(0, len(order), config.batch_tracks):
            batch = [train[i] for i in order[start : start + config.batch_tracks]]
            loss, _ = batch_loss(model, batch, config.device)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        held_out = evaluate(model, validation, config.device, config.batch_tracks)
        history.append({"epoch": epoch, "validation": held_out, "seconds": time.time() - started})
        print(f"[tracker] epoch {epoch}: {held_out}", flush=True)
        score = held_out.get(f"error_{TRACKER_HORIZONS[-3]}")
        if score is not None and (best is None or score < best):
            best = score
            save_tracker(output / "best.pt", model, config, history)
    save_tracker(output / "last.pt", model, config, history)
    (output / "history.json").write_text(
        json.dumps({"steady_motion": baseline, "history": history}, indent=2) + "\n"
    )
    return {"steady_motion": baseline, "best": best, "history": history}


def save_tracker(path, model, config: TrackerConfig, history) -> None:
    torch.save(
        {
            "tracker": model.state_dict(),
            "width": model.width,
            "config": asdict(config),
            "history": history,
        },
        path,
    )


def load_tracker(path, device="cpu") -> TargetTracker:
    saved = torch.load(path, map_location=device, weights_only=False)
    model = TargetTracker(saved["width"]).to(device)
    model.load_state_dict(saved["tracker"])
    return model


def main(argv: Optional[Sequence[str]] = None) -> int:
    import argparse
    import dataclasses

    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    for field in dataclasses.fields(TrackerConfig):
        parser.add_argument(
            f"--{field.name.replace('_', '-')}", type=type(field.default), default=field.default
        )
    config = TrackerConfig(**vars(parser.parse_args(argv)))
    print(json.dumps(train_tracker(config)["best"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
