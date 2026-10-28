"""Train the collision trackers (core/target_tracker) on recorded episodes:
for every enemy vision tracked, every frame, the inputs the trackers had
then (target_tracker.record_features: the executor's kinematics, what
vision showed, the action in progress) and labels made from what really
happened (core/collision_labels: Mario's real path until his action ended,
then standing, against the enemy's real path).

Episodes are collected with the simulator's truth recorded
(EpisodeTask.record_truth): the teacher playing every layout, and the skill
playing the skill families' (the teacher carrying its actions out), so both
the teacher's and the skill's actions are in the data. Held-out layouts
(the validation split) measure the trackers, per verb in progress and enemy
kind, beside a constant-velocity baseline (the enemy keeps its estimated
velocity, Mario stands).
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

from retroagi.core.action_tokens import VERBS
from retroagi.core.collision_labels import KINDS, action_ends, collision_labels, slot_labels
from retroagi.core.smb_kinematics import CONTEXT, TRACKED
from retroagi.core.smb_tracked_objects import COLUMN as LIST_COLUMN
from retroagi.core.target_tracker import (
    BINS,
    CLASSES,
    NONE,
    OBSERVATION_WIDTH,
    POSITION_SCALE,
    ROWS,
    CollisionTracker,
    allowed_classes,
    collision_loss,
    record_features,
)

# The horizons (frames) a hit within is measured at.
HIT_WITHIN = (16, 32, 48, 96)
# The fields kept of each collected episode (the rest is not needed).
KEPT = (
    "family",
    "split",
    "sample_index",
    "difficulty",
    "won",
    "end",
    "decision_frames",
    "truth",
    "slot_match",
    "sprite_kinematics",
    "mario_kinematics",
    "slot_identity",
    "enemy_context",
    "src_c",
    "mario_position",
    "action_verb",
    "action_pointer",
    "commands",
    "ended",
    "played_teacher",
)


@dataclass
class TrackerConfig:
    episodes: str = "artifacts/block_smb/collision_tracker/episodes.pkl"
    output: str = "artifacts/block_smb/collision_tracker"
    width: int = 64
    epochs: int = 40
    batch_tracks: int = 128
    learning_rate: float = 2e-3
    weight_decay: float = 0.01
    device: str = "cuda"
    seed: int = 0


class Kept:
    """A collected episode's KEPT fields as attributes."""

    def __init__(self, values: dict):
        self.__dict__.update(values)


def kept(record) -> Kept:
    return Kept({name: getattr(record, name) for name in KEPT})


def track_sequences(record) -> list:
    """One sequence per entry of the list of tracked things in a recorded
    episode (smb_tracked_objects.record_objects; its enemies): for each frame
    it was on the list (seen, or out of sight where it vanished), its inputs
    and labels: {"observations" [L, OBSERVATION_WIDTH], "features" [L,
    FEATURES], "allowed" [L, CLASSES], "place" [L, 2], "seen" [L] (a contact
    was seen), "labelled" [L] (its true enemy is known), "frames", "kind",
    "censor" [L], "verb" [L] (the action in progress; -1 none), "enemy_kind"
    [L] (CONTEXT index), "unseen" [L] (out of sight)}."""
    features, objects = record_features(record)
    frames = len(features)
    # A moving platform's entry shows a true platform, not an enemy.
    platform = objects["rows"][:, :, LIST_COLUMN["platform"]] > 0.5
    labels = slot_labels(
        collision_labels(record.truth, action_ends(record.decision_frames, frames)),
        np.where(platform, -1, objects["truths"]),
        ROWS,
    )
    verb_now = np.full(frames, -1, np.int64)
    starts = [int(f) for f in record.decision_frames] + [frames]
    for k, (f, e) in enumerate(zip(starts, starts[1:])):
        verb_now[f:e] = int(record.action_verb[k]) if record.action_verb is not None else -1
    keys = objects["keys"]
    kinds = features[:, :, TRACKED.index("walker") : TRACKED.index("other") + 1]
    unseen = objects["rows"][:, :, TRACKED.index("unseen")] > 0
    out = []
    for key in np.unique(keys[keys > 0]):
        t, slot = np.nonzero((keys == key) & (features[:, :, 0] > 0))
        if not len(t):
            continue
        frames_, kind, censor = (labels[n][t, slot] for n in ("frames", "kind", "censor"))
        out.append(
            {
                "observations": features[t, slot, :OBSERVATION_WIDTH],
                "features": features[t, slot],
                "allowed": allowed_classes(frames_, kind, censor),
                "place": labels["place"][t, slot].astype(np.float32),
                "seen": frames_ > 0,
                "labelled": labels["labelled"][t, slot],
                "frames": frames_,
                "kind": kind,
                "censor": censor,
                "verb": verb_now[t],
                "enemy_kind": np.argmax(kinds[t, slot], -1),
                "unseen": unseen[t, slot],
            }
        )
    return out


def _padded(sequences, key, device, dtype=torch.float32):
    longest = max(len(s[key]) for s in sequences)
    first = np.asarray(sequences[0][key])
    out = np.zeros((len(sequences), longest) + first.shape[1:], first.dtype)
    for i, s in enumerate(sequences):
        out[i, : len(s[key])] = s[key]
    return torch.as_tensor(out, device=device, dtype=dtype)


def _forward(model, sequences, device):
    """The head's outputs over a batch of sequences, and the mask of real,
    labelled frames."""
    observations = _padded(sequences, "observations", device)
    features = _padded(sequences, "features", device)
    out = model.answer(model.observe(observations), features)
    lengths = torch.as_tensor([len(s["features"]) for s in sequences], device=device)
    real = torch.arange(observations.shape[1], device=device)[None] < lengths[:, None]
    labelled = _padded(sequences, "labelled", device, torch.bool) & real
    return out, labelled


def batch_loss(model, sequences, device):
    out, mask = _forward(model, sequences, device)
    allowed = _padded(sequences, "allowed", device, torch.bool)[mask]
    place = _padded(sequences, "place", device)[mask]
    seen = _padded(sequences, "seen", device, torch.bool)[mask]
    flat = {name: value[mask] for name, value in out.items()}
    return collision_loss(flat, allowed, place, seen)


def _baseline_hit_frames(features) -> np.ndarray:
    """The constant-velocity baseline's frames until a hit (inf: none): the
    enemy keeps its estimated velocity, Mario stands; a hit when it reaches
    him at his height."""
    x = features[:, 1] * POSITION_SCALE
    y = features[:, 2] * POSITION_SCALE
    vx = features[:, 3]
    width = features[:, TRACKED.index("width")] * 16
    gap = np.abs(x) - (width / 2 + 5)
    closing = np.where(x > 0, -vx, vx)
    level = np.abs(y + 6) <= 12
    frames = np.where((closing > 0.05) & level, np.maximum(gap, 0) / np.maximum(closing, 1e-6), np.inf)
    return np.where(gap <= 0, 1.0, frames)


@torch.no_grad()
def evaluate(model, sequences, device) -> dict:
    """Per verb in progress × enemy kind ("all" × "all" included; and
    "remembered" × "all", the rows of enemies out of sight): for hits
    within HIT_WITHIN frames, precision and recall (the trackers' and the
    baseline's); for seen contacts, the time error (median; shares within 2
    and 4 frames), the place error (median pixels) and stomp-or-hit
    accuracy."""
    model.eval()
    rows = defaultdict(lambda: defaultdict(list))
    for start in range(0, len(sequences), 256):
        batch = sequences[start : start + 256]
        out, mask = _forward(model, batch, device)
        probability = torch.softmax(out["logits"], -1)[mask].cpu().numpy()
        place = out["place"][mask].cpu().numpy()
        keys = {
            name: np.concatenate([s[name][s["labelled"]] for s in batch])
            for name in ("frames", "kind", "censor", "verb", "enemy_kind", "features", "unseen")
        }
        keys["place_label"] = np.concatenate([s["place"][s["labelled"]] for s in batch])
        hit_by = np.cumsum(probability[:, BINS:NONE], -1)
        stomp_mass = probability[:, :BINS].sum(-1)
        contact_by = np.cumsum(probability[:, :BINS] + probability[:, BINS:NONE], -1)
        baseline = _baseline_hit_frames(keys["features"])
        for i in range(len(probability)):
            verb = int(keys["verb"][i])
            groups = [
                ("all", "all"),
                (VERBS[verb] if verb >= 0 else "none", "all"),
                ("all", CONTEXT[int(keys["enemy_kind"][i])]),
            ]
            if keys["unseen"][i]:
                groups.append(("remembered", "all"))
            frames, kind, censor = int(keys["frames"][i]), int(keys["kind"][i]), int(keys["censor"][i])
            hit = frames > 0 and KINDS[kind] == "hit"
            for f in HIT_WITHIN:
                if frames == 0 and 0 < censor <= f:
                    continue  # not known whether it would have come by then
                truth = hit and frames <= f
                predicted = hit_by[i, (f - 1) // 2] >= 0.5
                base = baseline[i] <= f
                for g in groups:
                    rows[g][f"hit{f}"].append((truth, predicted, base))
            if frames > 0:
                contact = contact_by[i, -1]
                median = int(np.searchsorted(contact_by[i], contact / 2)) * 2 + 1.5
                stomp_predicted = stomp_mass[i] >= contact / 2
                for g in groups:
                    rows[g]["time"].append(abs(median - frames))
                    rows[g]["place"].append(float(np.abs(place[i] - keys["place_label"][i]).sum()))
                    rows[g]["kind"].append(stomp_predicted == (KINDS[kind] == "stomp"))
    report = {}
    for (verb, kind), values in rows.items():
        entry = {}
        for f in HIT_WITHIN:
            cases = values.get(f"hit{f}", [])
            if not cases:
                continue
            truth, predicted, base = (np.asarray(c, bool) for c in zip(*cases))
            for name, guess in (("", predicted), ("baseline_", base)):
                hits = (truth & guess).sum()
                entry[f"{name}hit{f}_precision"] = float(hits / max(1, guess.sum()))
                entry[f"{name}hit{f}_recall"] = float(hits / max(1, truth.sum()))
            entry[f"hit{f}_cases"] = int(truth.sum())
        if values.get("time"):
            time_error = np.asarray(values["time"])
            entry["time_error_median"] = float(np.median(time_error))
            entry["time_within_2"] = float(np.mean(time_error <= 2))
            entry["time_within_4"] = float(np.mean(time_error <= 4))
            entry["place_error_median"] = float(np.median(values["place"]))
            entry["kind_accuracy"] = float(np.mean(values["kind"]))
            entry["contacts"] = len(time_error)
        report[f"{verb}/{kind}"] = entry
    return report


def _sequences(episodes) -> list:
    return [s for e in episodes for s in track_sequences(e)]


def train_tracker(config: TrackerConfig) -> dict:
    """Train on the collected episodes; keep the best held-out loss."""
    random.seed(config.seed)
    torch.manual_seed(config.seed)
    train, validation = [], []
    for path in config.episodes.split(","):
        a, b = pickle.loads(Path(path).read_bytes())
        train += a
        validation += b
    train_sequences, held = _sequences(train), _sequences(validation)
    if not train_sequences:
        raise ValueError("no tracked enemies: collect episodes with record_truth")
    output = Path(config.output)
    output.mkdir(parents=True, exist_ok=True)
    model = CollisionTracker(config.width).to(config.device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    history, best = [], None
    for epoch in range(config.epochs):
        started = time.time()
        model.train()
        random.shuffle(train_sequences)
        losses = []
        for start in range(0, len(train_sequences), config.batch_tracks):
            loss, _ = batch_loss(
                model, train_sequences[start : start + config.batch_tracks], config.device
            )
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss))
        model.eval()
        with torch.no_grad():
            held_losses = [
                float(batch_loss(model, held[s : s + 256], config.device)[0])
                for s in range(0, len(held), 256)
            ]
        entry = {
            "epoch": epoch,
            "train_loss": float(np.mean(losses)),
            "held_loss": float(np.mean(held_losses)) if held_losses else None,
            "seconds": round(time.time() - started, 1),
        }
        history.append(entry)
        print(json.dumps(entry), flush=True)
        if entry["held_loss"] is not None and (best is None or entry["held_loss"] < best["held_loss"]):
            best = entry
            save_tracker(output / "best.pt", model, config, history)
    save_tracker(output / "last.pt", model, config, history)
    model = load_tracker(output / "best.pt", config.device)
    report = {"best": best, "held_out": evaluate(model, held, config.device)}
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    (output / "history.json").write_text(json.dumps(history, indent=2) + "\n")
    return report


def collect(
    source: str,
    output: str,
    *,
    families: Sequence[str] = (),
    rounds: int = 2,
    layouts: int = 8,
    validation_layouts: int = 2,
    seed: int = 2026101601,
    workers: int = 15,
    lanes: int = 8,
) -> Path:
    """Episodes with the simulator's truth recorded (EpisodeTask.record_truth):
    the teacher playing every family's layouts, and the skill (``source``'s,
    the teacher carrying its actions out) playing the skill families'.
    Pickled (train, validation) lists of their KEPT fields in
    ``output``/episodes.pkl; the validation split's layouts are held out.

    By default the strategy families are left out: their enemies move as the
    others' do, and each of their layouts first needs a speed-run reference
    replayed through vision."""
    import dataclasses

    from . import layered_train as lt
    from .monte_carlo import BLOCK_SMB_MC_FAMILIES
    from .strategy_families import STRATEGY_FAMILIES

    every = families or [f for f in BLOCK_SMB_MC_FAMILIES if f not in STRATEGY_FAMILIES]
    policy, saved = lt.load_layered_checkpoint(Path(source), "cpu")
    settings = lt.saved_settings(saved["config"])
    settings.update(
        learner="skill",
        families=tuple(every),
        output=output,
        workers=workers,
        lanes=lanes,
        seed=seed,
        predictor=None,
        tracker=None,
        extra_layouts="",
    )
    skill = lt.LayeredTrainConfig(**settings)
    tactic = lt.LayeredTrainConfig(**{**settings, "learner": "tactic"})
    Path(output).mkdir(parents=True, exist_ok=True)
    pool = lt.EpisodePool(skill, policy.settings)
    try:
        pool.publish(policy)

        def play(config, split, count, share, teacher_only):
            tasks = (
                [t for r in range(rounds) for t in lt._tasks(config, "train", count, r, share, True)]
                if split == "train"
                else lt._tasks(config, "validation", count, 0, share, True)
            )
            tasks = [
                dataclasses.replace(t, teacher_share=share, label=True, record_truth=True)
                for t in pool.with_scenarios(tasks)
            ]
            return [kept(r) for r in pool.play(tasks, teacher_only=teacher_only)]

        sets = {}
        for split, count in (("train", layouts), ("validation", validation_layouts)):
            sets[split] = (
                # Every family, the teacher playing.
                play(skill, split, count, 1.0, True)
                + play(tactic, split, count, 1.0, True)
                # The skill families, the skill playing.
                + play(skill, split, count, 0.0, False)
            )
    finally:
        pool.close()
    path = Path(output) / "episodes.pkl"
    path.write_bytes(pickle.dumps((sets["train"], sets["validation"])))
    return path


# collision_v2 (2026-10-21): the observation gained "unseen" (how long the
# enemy has been out of sight; smb_kinematics.TRACKED), after the sprite's
# kinematics. collision_v3 (2026-10-21): its context gained the enemy's object
# type (type_0 ...), before Mario's kinematics. An earlier tracker loads with
# those inputs reading nothing (zero weights).
LAYOUT = "collision_v3"
# Each layout's step to the next: the inputs it added (their first column in
# TRACKED, and how many).
_ADDED = {
    "collision_v1": ("collision_v2", "unseen", 1),
    "collision_v2": ("collision_v3", "type_0", len(TRACKED) - TRACKED.index("type_0")),
}


def save_tracker(path, model, config: TrackerConfig, history) -> None:
    torch.save(
        {
            "tracker": model.state_dict(),
            "width": model.width,
            "layout": LAYOUT,
            "config": asdict(config),
            "history": history,
        },
        path,
    )


def _with_inputs(state: dict, width: int, column: int, count: int) -> dict:
    """A tracker's weights with ``count`` inputs added at ``column`` of the
    observation (zero weights: they read nothing until trained), for the GRU's
    input and the head's (which reads the GRU state, then the frame's inputs)."""
    state = dict(state)
    for name, at in (("gru.weight_ih_l0", column), ("head.0.weight", width + column)):
        w = state[name]
        added = torch.zeros(w.shape[0], count, dtype=w.dtype, device=w.device)
        state[name] = torch.cat((w[:, :at], added, w[:, at:]), 1)
    return state


def load_tracker(path, device="cpu") -> CollisionTracker:
    saved = torch.load(path, map_location=device, weights_only=False)
    layout = saved.get("layout")
    if layout not in (*_ADDED, LAYOUT):
        raise ValueError(f"{path} is not a collision tracker (the old target tracker?)")
    model = CollisionTracker(saved["width"]).to(device)
    state = saved["tracker"]
    while layout != LAYOUT:
        layout, first, count = _ADDED[layout]
        state = _with_inputs(state, saved["width"], TRACKED.index(first), count)
    model.load_state_dict(state)
    return model


def main(argv: Optional[Sequence[str]] = None) -> int:
    import argparse
    import dataclasses

    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    for field in dataclasses.fields(TrackerConfig):
        parser.add_argument(
            f"--{field.name.replace('_', '-')}", type=type(field.default), default=field.default
        )
    parser.add_argument(
        "--collect-from", help="first collect episodes with this policy checkpoint's skill"
    )
    parser.add_argument(
        "--collect-families",
        nargs="*",
        default=(),
        help="the families to collect (default: every family but the strategy ones)",
    )
    args = vars(parser.parse_args(argv))
    source = args.pop("collect_from")
    families = tuple(args.pop("collect_families"))
    config = TrackerConfig(**args)
    if source:
        collected = collect(source, config.output, families=families)
        config = dataclasses.replace(config, episodes=str(collected))
    print(json.dumps(train_tracker(config)["best"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
