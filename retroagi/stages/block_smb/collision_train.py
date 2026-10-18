"""The collision lessons' dataset and the collision trackers' training on it
(user, 2026-10-18: the trackers fail for want of contacts to learn from;
collision lessons, used to train them and nothing else).

1. Every layout of every collision family (collision_families: the full
   sweep of their parameters, each difficulty) is played through the
   training pipeline, Mario carrying out the lesson's one scripted action,
   the simulator's truth recorded.
2. A lesson is kept when Mario and an enemy touched during the action (Mario
   hit, or hitting it: a stomp), as the recorded truth shows (the contact
   labels collision_labels makes), or when the action ran to its end with no
   contact (a miss: the trackers must know when none comes; user,
   2026-10-18). Any other is a failed layout (a contact before the action, a
   death without one, a timeout): recorded in failures.json with its
   parameters and layout, not kept.
3. The kept lessons are saved, one gzip-compressed file per family
   (dataset/<family>.pkl.gz, with dataset/complete.json written last), and
   read back instead of played again.
4. The collision trackers (the GRU and its head), from the given trackers,
   learn from every kept lesson; then they are measured on freshly drawn
   layouts of each family, played the same way, per family, contact kind
   and Mario's action (tracker_train.evaluate), against the constant-speed
   baseline.

Example:
    python -m retroagi.stages.block_smb.collision_train \\
        --source artifacts/block_smb/skill_continue_20261017/best.pt \\
        --tracker artifacts/block_smb/skill_continue_20261017/tracker.pt \\
        --output artifacts/block_smb/collision_20261018
"""

import argparse
import dataclasses
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

from .collision_families import COLLISION_FAMILIES, DIFFICULTIES, every_layout, random_layouts


@dataclass
class CollisionConfig:
    source: str = ""  # a policy checkpoint: its settings and vision run the lessons
    tracker: str = ""  # the trackers to go on training ("" for new ones)
    output: str = "artifacts/block_smb/collision"
    families: tuple = COLLISION_FAMILIES
    epochs: int = 40
    batch_tracks: int = 128
    learning_rate: float = 1e-3
    measure_per_difficulty: int = 10  # fresh layouts per family and difficulty, measured after
    workers: int = 8
    lanes: int = 8
    device: str = "cuda"
    seed: int = 0


def _tasks(layouts, sample_offset: int = 0) -> list:
    from . import layered_train as lt

    return [
        lt.EpisodeTask(
            index=i,
            family=scenario["metadata"]["block_smb_monte_carlo"]["family"],
            split="train",
            seed=0,
            sample_index=sample_offset + i,
            teacher_share=1.0,
            label=True,
            frames=int(scenario["frame_budget"]),
            scenario=scenario,
            difficulty=params["difficulty"],
            record_truth=True,
        )
        for i, (scenario, params) in enumerate(layouts)
    ]


def touched(record) -> Optional[str]:
    """How the lesson's action went: "stomp" or "hit" (Mario and an enemy
    touched during it: the lesson ends by the contact, a stomp ending it as
    won, at the frame the truth labels show it), "miss" (the action ran to
    its end with no contact), or None (a failed layout)."""
    from retroagi.core.collision_labels import KINDS, action_ends, collision_labels

    if record.end == "action_done":
        return "miss"
    if record.end not in ("death", "goal"):
        return None
    labels = collision_labels(record.truth, action_ends(record.decision_frames, record.frames))
    t, enemy = np.nonzero(labels.frames > 0)
    if not len(t):
        return None
    meets = t + labels.frames[t, enemy]  # the frame each label says the contact comes
    first = int(np.argmin(meets))
    if meets[first] > record.frames + 1:
        return None  # only after the recording's end
    return KINDS[int(labels.kind[t[first], enemy[first]])]


def _starts(scenario) -> bool:
    """Whether the lesson gets to its action: the pictures watched before it
    end in no contact (the environment refuses one that does)."""
    from .env import MarioScenarioEnv

    env = MarioScenarioEnv()
    try:
        env.reset(scenario=scenario, seed=0)
        return True
    except ValueError:
        return False
    finally:
        env.close()


def _play(pool, layouts, offset=0) -> list:
    return pool.play(_tasks(layouts, offset), teacher_only=True)


def make_dataset(config: CollisionConfig, pool, output: Path) -> dict:
    """Play every layout (or read the saved dataset back): {family: [kept
    records]}; failed layouts are recorded."""
    folder = output / "dataset"
    if (folder / "complete.json").exists():
        summary = json.loads((folder / "complete.json").read_text())
        kept = {}
        for family in summary["families"]:
            with gzip.open(folder / f"{family}.pkl.gz", "rb") as handle:
                kept[family] = pickle.load(handle)
        return kept
    folder.mkdir(parents=True, exist_ok=True)
    kept, failures, summary = {}, [], {"families": {}, "failed": {}, "contacts": {}}
    for family in config.families:
        every = [layout for d in DIFFICULTIES for layout in every_layout(family, d)]
        layouts = []
        for scenario, params in every:
            if _starts(scenario):
                layouts.append((scenario, params))
            else:
                failures.append({"family": family, "params": params, "end": "touched_before_the_action",
                                 "frames": 0, "scenario": scenario})
        started = time.time()
        records = _play(pool, layouts)
        kept[family] = []
        contacts = {"stomp": 0, "hit": 0, "miss": 0}
        for (scenario, params), record in zip(layouts, records):
            kind = touched(record)
            if kind is None:
                failures.append(
                    {"family": family, "params": params, "end": record.end, "frames": record.frames,
                     "scenario": scenario}
                )
                continue
            contacts[kind] += 1
            kept[family].append(record)
        with gzip.open(folder / f"{family}.tmp", "wb", compresslevel=6) as handle:
            pickle.dump(kept[family], handle, protocol=5)
        (folder / f"{family}.tmp").replace(folder / f"{family}.pkl.gz")
        summary["families"][family] = len(kept[family])
        summary["failed"][family] = len(every) - len(kept[family])
        summary["contacts"][family] = contacts
        print(
            f"[collision] {family}: {len(kept[family])}/{len(every)} kept "
            f"({contacts['stomp']} stomps, {contacts['hit']} hits, {contacts['miss']} misses), "
            f"{time.time() - started:.0f} s",
            flush=True,
        )
        (output / "failures.json").write_text(json.dumps(failures, indent=1) + "\n")
    (folder / "complete.json").write_text(json.dumps(summary, indent=2) + "\n")
    return kept


def train(config: CollisionConfig) -> dict:
    from . import layered_train as lt
    from retroagi.core.target_tracker import CollisionTracker

    from .tracker_train import TrackerConfig, batch_loss, evaluate, load_tracker, save_tracker, track_sequences

    random.seed(config.seed)
    torch.manual_seed(config.seed)
    output = Path(config.output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "config.json").write_text(json.dumps(asdict(config), indent=2) + "\n")
    device = torch.device(config.device)
    policy, saved = lt.load_layered_checkpoint(config.source, device)
    settings = lt.saved_settings(saved["config"])
    layer = lt.LayeredTrainConfig(
        **{**settings, "learner": "skill", "output": str(output), "workers": config.workers,
           "lanes": config.lanes, "device": config.device, "tracker": None}
    )
    pool = lt.EpisodePool(layer, policy.settings)
    try:
        pool.publish(policy)
        kept = make_dataset(config, pool, output)
        sequences = [s for records in kept.values() for r in records for s in track_sequences(r)]
        print(f"[collision] {len(sequences)} enemy tracks from "
              f"{sum(len(v) for v in kept.values())} lessons", flush=True)
        tracker = (
            load_tracker(config.tracker, device) if config.tracker else CollisionTracker(TrackerConfig().width).to(device)
        )
        optimizer = torch.optim.AdamW(tracker.parameters(), lr=config.learning_rate, weight_decay=0.01)
        history = []
        for epoch in range(config.epochs):
            started = time.time()
            tracker.train()
            random.shuffle(sequences)
            losses = []
            for start in range(0, len(sequences), config.batch_tracks):
                loss, _ = batch_loss(tracker, sequences[start : start + config.batch_tracks], device)
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(tracker.parameters(), 1.0)
                optimizer.step()
                losses.append(float(loss))
            tracker.eval()
            history.append({"epoch": epoch, "loss": float(np.mean(losses)), "seconds": round(time.time() - started, 1)})
            print(json.dumps(history[-1]), flush=True)
            save_tracker(output / "tracker.pt", tracker, TrackerConfig(width=tracker.width), history)
        # Measured after training, on freshly drawn layouts (never trained on).
        measured = {}
        for family in config.families:
            layouts = [
                layout
                for d in DIFFICULTIES
                for layout in random_layouts(family, d, config.measure_per_difficulty, seed=1)
            ]
            layouts = [layout for layout in layouts if _starts(layout[0])]
            records = [r for r in _play(pool, layouts, offset=100_000) if touched(r) is not None]
            tracks = [s for r in records for s in track_sequences(r)]
            measured[family] = {
                "lessons": len(records),
                "of": len(layouts),
                "trackers": evaluate(tracker, tracks, device) if tracks else None,
            }
            hit = (measured[family]["trackers"] or {}).get("all/all", {})
            print(
                f"[collision] measured {family}: {len(records)}/{len(layouts)} kept; "
                + ", ".join(f"{k} {hit[k]:.2f}" for k in ("hit16_recall", "hit32_recall", "hit48_recall") if k in hit),
                flush=True,
            )
        report = {"history": history, "measured": measured}
        (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        return report
    finally:
        pool.close()


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    for setting in dataclasses.fields(CollisionConfig):
        if setting.name == "families":
            parser.add_argument("--families", nargs="+", default=list(COLLISION_FAMILIES))
            continue
        parser.add_argument(f"--{setting.name.replace('_', '-')}", type=type(setting.default), default=setting.default)
    args = parser.parse_args(argv)
    values = {f.name: getattr(args, f.name) for f in dataclasses.fields(CollisionConfig)}
    values["families"] = tuple(values["families"])
    train(CollisionConfig(**values))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
