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

from retroagi.core.action_predictor import COLUMN, ActionPredictor, outcome_loss, reading_digest
from retroagi.core.action_tokens import VERBS
from retroagi.core.layered_policy import PolicySettings, forecast_features

from . import layered_train as lt
from .monte_carlo import BLOCK_SMB_MC_FAMILIES


@dataclass(frozen=True)
class PredictorConfig:
    # The policy whose scene encoder and memory the predictor reads.
    source: str = "artifacts/block_smb/skill_margin_teacher_20261009/best.pt"
    families: tuple[str, ...] = ()  # empty: every skill family
    # The training layouts of the first ``train_rounds`` rounds of a skill
    # run with this seed (``train_layouts_per_family`` each), and its
    # validation layouts: with the seed of a teacher validation, every
    # layout was won by the teacher there.
    train_layouts_per_family: int = 8
    train_rounds: int = 3
    validation_layouts_per_difficulty: int = 3
    epochs: int = 40
    batch_frames: int = 16384
    learning_rate: float = 1e-3
    weight_decay: float = 0.05
    workers: int = 15
    lanes: int = 8
    seed: int = 2026100909  # teacher_validation_dense_20261010's
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
    """The teacher's episodes on ``tasks``, with its actions recorded (and
    every frame's tracked objects, the target tracker's data)."""
    demonstrations = [
        dataclasses.replace(
            t,
            teacher_share=1.0,
            label=True,
            explore=False,
            record_actions=True,
            record_objects=True,
        )
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


# A family's concentration on one end is explained by its scenes when the
# predictor, which learns ends from scenes of every family, reproduces that
# family's held-out ends within EXPLAINED_PIXELS, and its predicted end moves
# with the action's object (at least EXPLAINED_FOLLOW of a shift of the object
# in the picture): it follows the geometry, not a value learned for the
# family.
EXPLAINED_PIXELS = 8.0
EXPLAINED_FOLLOW = 0.5


def _shifted_rows(c, verbs, pointers, shift):
    """The decision rows ``c`` [N, SEQ_LEN_C] with each action's object moved
    ``shift`` pixels right (its x numbers; a hold's row unchanged)."""
    from retroagi.core.action_tokens import pointer_target
    from retroagi.core.smb_observer import _SLOT_WIDTH, C_SPANS

    c = c.clone()
    for k, (verb, pointer) in enumerate(zip(verbs.tolist(), pointers.tolist())):
        if VERBS[verb] == "hold":
            continue
        name, slot = pointer_target(int(pointer))
        start = C_SPANS[f"c_{name}"][0] + slot * _SLOT_WIDTH[name]
        for column in (1, 2) if name in ("surfaces", "gaps") else (1, 3):
            c[k, start + column] += shift / 256
    return c


@torch.no_grad()
def explained(policy, predictor, episodes, device, shift: float = 8.0) -> dict:
    """Whether each family's action ends are explained by its scenes: per
    family and verb, over the teacher's chosen actions, the most common end's
    share (within 2 pixels), the predictor's mean error on them (pixels), and
    how far its predicted end moves when the action's object is moved
    ``shift`` pixels right in the picture (as a fraction of the shift);
    "explained" where the share is at most layered_train.COLLAPSE_SHARE, or
    the error and the following meet EXPLAINED_PIXELS and EXPLAINED_FOLLOW
    (or there are too few actions to judge: fewer than COLLAPSE_MIN)."""
    predictor.eval()
    gathered = defaultdict(lambda: {"ends": [], "errors": [], "follow": []})
    for batch in lt._batches(episodes, 16384, random.Random(0)):
        episode, frame, rows, records = _decisions(batch)
        if not records:
            continue
        records = np.asarray(records)
        chosen = records[:, COLUMN["chosen"]] > 0.5
        if not chosen.any():
            continue
        families = [batch[episode[r]].family for r in rows]
        inputs = predictor_inputs(policy, batch, device)
        tokens, present, memory, forecast, index, recorded = inputs
        keep = torch.tensor(chosen, device=device)
        index, recorded = index[keep], recorded[keep]
        verbs = recorded[:, COLUMN["verb"]].long()
        pointers = recorded[:, COLUMN["pointer"]].long()
        base = predictor(tokens, present, memory, forecast, index, verbs, pointers)["mean"]
        # The same decisions with each action's object moved.
        a, b, c, _, _, _ = lt._padded(batch, device)
        used = sorted(set(rows))
        e = torch.tensor([episode[r] for r in used], device=device)
        f = torch.tensor([frame[r] for r in used], device=device)
        moved_c = _shifted_rows(c[e, f][index], verbs, pointers, shift)
        moved_tokens, moved_present = policy.encode_scene((a[e, f][index], b[e, f][index], moved_c))
        moved = predictor(
            moved_tokens,
            moved_present,
            memory[index],
            forecast[index],
            torch.arange(len(index), device=device),
            verbs,
            pointers,
        )["mean"]
        target = recorded[:, [COLUMN["dx"], COLUMN["dy"]]]
        error = (base[:, :2] - target).norm(dim=-1)
        follow = (moved[:, 0] - base[:, 0]) / shift
        names = [f for f, kept in zip(families, chosen) if kept]
        for k, family in enumerate(names):
            verb = VERBS[int(verbs[k])]
            if verb == "hold" or recorded[k, COLUMN["safe"]] < 0.5:
                continue
            entry = gathered[(family, verb)]
            entry["ends"].append(tuple(int(v) for v in target[k].round().tolist()))
            entry["errors"].append(float(error[k]))
            entry["follow"].append(float(follow[k]))
    out = {}
    for (family, verb), entry in sorted(gathered.items()):
        value, share = lt.top_share(entry["ends"])
        error = float(np.mean(entry["errors"]))
        follow = float(np.mean(entry["follow"]))
        out.setdefault(family, {})[verb] = {
            "actions": len(entry["ends"]),
            "most_common_end": list(value),
            "share": round(share, 3),
            "error_pixels": round(error, 2),
            "follows_object": round(follow, 2),
            # Fewer than COLLAPSE_MIN actions are too few to judge.
            "explained": len(entry["ends"]) < lt.COLLAPSE_MIN
            or share <= lt.COLLAPSE_SHARE
            or (error <= EXPLAINED_PIXELS and follow >= EXPLAINED_FOLLOW),
        }
    return out


def train_predictor(config: PredictorConfig, episodes=None) -> dict:
    """Collect the teacher's episodes (or use ``episodes``: (train,
    validation) lists), train the predictor, and save it with its history."""
    output = Path(config.output)
    output.mkdir(parents=True, exist_ok=True)
    policy, saved = lt.load_layered_checkpoint(Path(config.source), config.device)
    policy.eval()
    reads = reading_digest(policy)
    layer = _layer_config(config, saved)
    data = output / "episodes.pkl"
    if episodes is None and data.exists():
        episodes = pickle.loads(data.read_bytes())
    if episodes is None:
        pool = lt.EpisodePool(layer, policy.settings)
        try:
            pool.publish(policy)
            train = collect(
                pool,
                [
                    task
                    for r in range(config.train_rounds)
                    for task in lt._tasks(
                        layer, "train", layer.train_layouts_per_family, r, 1.0, True
                    )
                ],
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
            save_predictor(output / "best.pt", predictor, config, history, reads)
    save_predictor(output / "last.pt", predictor, config, history, reads)
    (output / "history.json").write_text(json.dumps(history, indent=2) + "\n")
    # Whether each family's concentrated ends follow from its scenes.
    chosen, _ = load_predictor(output / "best.pt", config.device)
    report = explained(policy, chosen, validation, config.device)
    (output / "explained.json").write_text(json.dumps(report, indent=2) + "\n")
    unexplained = {
        family: [verb for verb, entry in verbs.items() if not entry["explained"]]
        for family, verbs in report.items()
        if any(not entry["explained"] for entry in verbs.values())
    }
    print(f"[predictor] ends not explained by the scene: {unexplained or 'none'}", flush=True)
    return {"best_end_error_pixels": best, "history": history, "unexplained": unexplained}


def save_predictor(path, predictor, config: PredictorConfig, history, reads: str) -> None:
    """``reads``: the fingerprint of the scene encoder and memory it was
    trained on (action_predictor.reading_digest)."""
    torch.save(
        {
            "predictor": predictor.state_dict(),
            "settings": asdict(PolicySettings(**_settings_of(predictor))),
            "config": asdict(config),
            "history": history,
            "reads": reads,
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
