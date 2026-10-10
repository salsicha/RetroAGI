"""Train the action predictor (core/action_predictor) on the teacher's trials,
and measure its errors on held-out layouts.

Data: skill episodes the teacher plays and labels through the production
pipeline (vision, executor, episode budget), recording at each decision the
teacher's choice and every jump it could make there, with their outcomes
(EpisodeTask.record_actions). The scene encoder and the scene memory are a
trained policy's, kept fixed: the predictor reads them as the skill does.

Step 2 of docs/action-predictor-controller.md.
"""

import dataclasses
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

from retroagi.core.action_predictor import COLUMN, ActionPredictor, outcome_loss
from retroagi.core.action_tokens import VERBS
from retroagi.core.layered_policy import PolicySettings, forecast_features

from . import layered_train as lt
from .monte_carlo import BLOCK_SMB_MC_FAMILIES


@dataclass(frozen=True)
class PredictorConfig:
    # The policy whose scene encoder and memory the predictor reads.
    source: str = "artifacts/block_smb/skill_margin_teacher_20261009/best.pt"
    families: tuple[str, ...] = ()  # empty: every skill family
    train_layouts_per_family: int = 24
    validation_layouts_per_difficulty: int = 3
    epochs: int = 40
    batch_frames: int = 16384
    learning_rate: float = 1e-3
    weight_decay: float = 0.05
    workers: int = 15
    lanes: int = 8
    seed: int = 2026101000
    device: str = "cuda"
    output: str = "artifacts/block_smb/action_predictor"


def _layer_config(config: PredictorConfig, saved: dict) -> lt.LayeredTrainConfig:
    settings = dict(saved["config"])
    families = config.families or lt.learner_families("skill", BLOCK_SMB_MC_FAMILIES)
    settings.update(
        learner="skill",
        families=tuple(families),
        output=config.output,
        device=config.device,
        workers=config.workers,
        lanes=config.lanes,
        seed=config.seed,
        train_layouts_per_family=config.train_layouts_per_family,
        validation_layouts_per_difficulty=config.validation_layouts_per_difficulty,
    )
    return lt.LayeredTrainConfig(**settings)


def collect(pool, tasks: Sequence[lt.EpisodeTask]) -> list:
    """The teacher's episodes on ``tasks``, with its actions recorded."""
    demonstrations = [
        dataclasses.replace(t, teacher_share=1.0, label=True, explore=False, record_actions=True)
        for t in pool.with_scenarios(tasks)
    ]
    return pool.play(demonstrations, teacher_only=True)


def _decisions(episodes):
    """Every decision of the batch: (episode, frame) tensors, and the action
    records of the decisions that have any, with each record's decision."""
    episode, frame, rows, records = [], [], [], []
    for i, e in enumerate(episodes):
        for k, f in enumerate(e.decision_frames):
            if e.actions is not None and k < len(e.actions):
                kept = e.actions[k][e.actions[k][:, COLUMN["verb"]] >= 0]
                rows.extend([len(frame)] * len(kept))
                records.extend(kept)
            episode.append(i)
            frame.append(int(f))
    return episode, frame, rows, records


def predictor_inputs(policy, episodes, device):
    """What the predictor reads at every decision with actions, from the
    fixed policy (no gradients), and the action records:
    (scene tokens, present, memory, enemy forecast, rows, records) or None."""
    episode, frame, rows, records = _decisions(episodes)
    if not records:
        return None
    a, b, c, _, _, _ = lt._padded(episodes, device)
    d = {
        "episode": torch.tensor(episode, device=device),
        "frame": torch.tensor(frame, device=device),
    }
    e, f = d["episode"], d["frame"]
    used = sorted(set(rows))
    with torch.no_grad():
        memory = lt.action_memory(policy, a, b, c, d, episodes)[used]
        tokens, present = policy.encode_scene(
            (a[e[used], f[used]], b[e[used], f[used]], c[e[used], f[used]])
        )
        forecast = forecast_features(policy.memory.enemies(memory))
    position = {row: k for k, row in enumerate(used)}
    rows = torch.tensor([position[r] for r in rows], device=device)
    records = torch.tensor(np.asarray(records), device=device)
    return tokens, present, memory, forecast, rows, records


def predict(predictor, inputs):
    tokens, present, memory, forecast, rows, records = inputs
    return predictor(
        tokens,
        present,
        memory,
        forecast,
        rows,
        records[:, COLUMN["verb"]].long(),
        records[:, COLUMN["pointer"]].long(),
    )


@torch.no_grad()
def evaluate(policy, predictor, episodes, device, batch_frames: int) -> dict:
    """The predictor's errors on ``episodes``: overall, and per verb."""
    predictor.eval()
    totals = defaultdict(lambda: defaultdict(float))
    for batch in lt._batches(episodes, batch_frames, random.Random(0)):
        inputs = predictor_inputs(policy, batch, device)
        if inputs is None:
            continue
        predicted, records = predict(predictor, inputs), inputs[-1]
        verbs = records[:, COLUMN["verb"]].long()
        for name in ("all", *VERBS):
            mask = (
                torch.ones_like(verbs, dtype=torch.bool)
                if name == "all"
                else verbs == VERBS.index(name)
            )
            if not mask.any():
                continue
            _, stats = outcome_loss({k: v[mask] for k, v in predicted.items()}, records[mask])
            count = stats.pop("actions")
            totals[name]["actions"] += count
            for key, value in stats.items():
                totals[name][key] += value * count
    predictor.train()
    return {
        name: {
            key: (value if key == "actions" else round(value / entry["actions"], 3))
            for key, value in entry.items()
        }
        for name, entry in totals.items()
    }


def train_predictor(config: PredictorConfig, episodes=None) -> dict:
    """Collect the teacher's episodes (or use ``episodes``: (train,
    validation) lists), train the predictor, and save it with its history."""
    output = Path(config.output)
    output.mkdir(parents=True, exist_ok=True)
    policy, saved = lt.load_layered_checkpoint(Path(config.source), config.device)
    policy.eval()
    layer = _layer_config(config, saved)
    data = output / "episodes.pkl"
    if episodes is None and data.exists():
        episodes = pickle.loads(data.read_bytes())
    if episodes is None:
        pool = lt.EpisodePool(layer, policy.settings)
        try:
            pool.publish(policy)
            train = collect(
                pool, lt._tasks(layer, "train", layer.train_layouts_per_family, 0, 1.0, True)
            )
            validation = collect(
                pool,
                lt._tasks(
                    layer, "validation", layer.validation_layouts_per_difficulty, 0, 0.0, False
                ),
            )
        finally:
            pool.close()
        episodes = (train, validation)
        data.write_bytes(pickle.dumps(episodes))
    train, validation = episodes
    torch.manual_seed(config.seed)
    predictor = ActionPredictor(policy.settings).to(config.device)
    optimizer = torch.optim.AdamW(
        predictor.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    rng = random.Random(config.seed)
    history, best = [], None
    for epoch in range(config.epochs):
        started, seen = time.time(), defaultdict(float)
        for batch in lt._batches(train, config.batch_frames, rng):
            inputs = predictor_inputs(policy, batch, config.device)
            if inputs is None:
                continue
            loss, stats = outcome_loss(predict(predictor, inputs), inputs[-1])
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(predictor.parameters(), 1.0)
            optimizer.step()
            seen["loss"] += float(loss) * stats["actions"]
            seen["actions"] += stats["actions"]
        held_out = evaluate(policy, predictor, validation, config.device, config.batch_frames)
        entry = {
            "epoch": epoch,
            "train_loss": round(seen["loss"] / max(1, seen["actions"]), 4),
            "validation": held_out,
            "seconds": round(time.time() - started, 1),
        }
        history.append(entry)
        overall = held_out.get("all", {})
        print(
            f"[predictor] epoch {epoch}: loss {entry['train_loss']}, held-out end "
            f"{overall.get('end_error_pixels')} px, frames {overall.get('frames_error')}, "
            f"object {overall.get('object_error_pixels')} px, "
            f"success {overall.get('success_accuracy')}",
            flush=True,
        )
        score = overall.get("end_error_pixels")
        if score is not None and (best is None or score < best):
            best = score
            save_predictor(output / "best.pt", predictor, config, history)
    save_predictor(output / "last.pt", predictor, config, history)
    (output / "history.json").write_text(json.dumps(history, indent=2) + "\n")
    return {"best_end_error_pixels": best, "history": history}


def save_predictor(path, predictor, config: PredictorConfig, history) -> None:
    torch.save(
        {
            "predictor": predictor.state_dict(),
            "settings": asdict(PolicySettings(**_settings_of(predictor))),
            "config": asdict(config),
            "history": history,
        },
        path,
    )


def _settings_of(predictor) -> dict:
    return {
        "width": predictor.verb.embedding_dim,
        "memory_width": predictor.memory.in_features,
        "heads": predictor.attention.num_heads,
    }


def load_predictor(path, device="cpu") -> tuple[ActionPredictor, dict]:
    saved = torch.load(path, map_location=device, weights_only=False)
    predictor = ActionPredictor(PolicySettings(**saved["settings"])).to(device)
    predictor.load_state_dict(saved["predictor"])
    return predictor, saved


def main(argv: Optional[Sequence[str]] = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    for field in dataclasses.fields(PredictorConfig):
        if field.name == "families":
            parser.add_argument("--families", nargs="*", default=())
        else:
            parser.add_argument(
                f"--{field.name.replace('_', '-')}", type=type(field.default), default=field.default
            )
    args = parser.parse_args(argv)
    config = PredictorConfig(**{**vars(args), "families": tuple(args.families)})
    print(json.dumps(train_predictor(config)["best_end_error_pixels"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
