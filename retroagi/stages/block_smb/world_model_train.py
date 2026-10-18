"""Train the world model (core/world_model: the policy's LSTM, run once per
action) on recorded episodes: each executed action's measured outcome
(core/action_outcomes), taken in order (its state carried from action to
action), and at each decision the other actions the teacher tried
(teacher_actions.action_rows), evaluated from the state before it without
advancing it.

The scene encoder is the policy's, frozen (trained with the skill): the
world model records its digest and refuses a changed one. Held-out layouts
(the validation split) measure it per verb and per source (executed or
tried) against the bars an outcome must meet before anything relies on it
(BARS).
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

from retroagi.core.action_outcomes import COLUMN as EXECUTED, executed_outcomes
from retroagi.core.action_tokens import MODE_VERBS, VERBS
from retroagi.core.layered_policy import POINTER_START
from retroagi.core.tokens import SKILL_MODES
from retroagi.core.world_model import (
    ACTION_RECORD,
    COLUMN,
    CONTEXT_WIDTH,
    MARIO_TOKEN,
    OUTCOME_FIELDS,
    outcome_loss,
    scene_digest,
)

# The proposed bars (plan, step 6): the median end error of jumps and runs
# (pixels), and the share of waits whose length is within 4 frames.
BARS = {"end_error_median": 6.0, "wait_within_4": 0.8}
# The fields kept of each collected episode: the scene rows only at its
# decisions.
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
    "action_verb",
    "action_pointer",
    "commands",
    "ended",
    "execution_feedback",
    "actions",
    "played_teacher",
    "action_context",
    "labels",
)


@dataclass
class WorldModelConfig:
    source: str = ""  # the policy checkpoint whose scene encoder it reads
    episodes: str = "artifacts/block_smb/world_model/episodes.pkl"
    output: str = "artifacts/block_smb/world_model"
    epochs: int = 40
    batch_episodes: int = 32
    learning_rate: float = 1e-3
    weight_decay: float = 0.01
    tried_weight: float = 1.0  # the tried actions' share of the loss
    device: str = "cuda"
    seed: int = 0


class Kept:
    def __init__(self, values: dict):
        self.__dict__.update(values)


def kept(record) -> Kept:
    """A collected episode's KEPT fields, and its scene rows at its decisions."""
    values = {name: getattr(record, name) for name in KEPT}
    frames = np.asarray(record.decision_frames, np.int64)
    values["rows"] = tuple(getattr(record, name)[frames] for name in ("src_a", "src_b", "src_c"))
    return Kept(values)


def executed_rows(record) -> np.ndarray:
    """[D, len(ACTION_RECORD)] the executed actions as world-model records
    (verb -1 where the action is unknown: no loss, but the LSTM still steps
    with its command's mode as the verb)."""
    from . import layered_train as lt

    measured = executed_outcomes(record, lt.END_REASONS.index("no_progress"))
    rows = np.zeros((len(measured), len(ACTION_RECORD)), np.float32)
    for name in ("verb", *OUTCOME_FIELDS, "has_object", "success", "won", "room"):
        rows[:, COLUMN[name]] = measured[:, EXECUTED[name]]
    pointer = measured[:, EXECUTED["pointer"]]
    rows[:, COLUMN["pointer"]] = np.where(pointer >= 0, pointer + POINTER_START, MARIO_TOKEN)
    rows[:, COLUMN["executed"]] = 1.0
    # The teacher's choice: where it played a skill decision (a tactic
    # learner's episodes carry the skill's own actions).
    named = record.labels.get("action_valid") if record.labels else None
    chosen = np.asarray(record.played_teacher, bool) & (
        np.asarray(named, bool) if named is not None else False
    )
    rows[:, COLUMN["chosen"]] = chosen.astype(np.float32)
    return rows


def episode_data(record) -> dict:
    """What training reads of one episode, per decision: scene rows,
    kinematics, feedback, the executed record and the tried ones (those
    duplicating the executed action dropped: its measured outcome is kept)."""
    executed = executed_rows(record)
    steps = executed[:, COLUMN["verb"]].astype(np.int64)
    unnamed = steps < 0
    # An unnamed command steps the LSTM with its mode's verb.
    steps[unnamed] = [
        VERBS.index(MODE_VERBS[SKILL_MODES[int(m)]] if 0 <= int(m) < len(SKILL_MODES) else "wait")
        for m in record.commands[unnamed, 0]
    ]
    tried = record.actions if record.actions is not None else np.zeros((len(executed), 0, len(ACTION_RECORD)), np.float32)
    if tried.shape[-1] != len(ACTION_RECORD):
        raise ValueError("episodes recorded with another action record (an older vocabulary): collect again")
    tried = tried.copy()
    for k in range(len(executed)):
        same = (tried[k, :, COLUMN["verb"]] == executed[k, COLUMN["verb"]]) & (
            tried[k, :, COLUMN["pointer"]] == executed[k, COLUMN["pointer"]]
        )
        tried[k, same, COLUMN["verb"]] = -1
    frames = np.asarray(record.decision_frames, np.int64)
    context = record.action_context
    if context is None or len(context) != len(executed):
        context = np.zeros((len(executed), CONTEXT_WIDTH), np.float32)
    return {
        "rows": record.rows,
        "context": np.asarray(context, np.float32),
        "sprites": record.sprite_kinematics[frames].astype(np.float32),
        "mario": record.mario_kinematics[frames].astype(np.float32),
        "feedback": np.asarray(record.execution_feedback, np.float32),
        "executed": executed,
        "steps": steps,
        "step_pointers": executed[:, COLUMN["pointer"]].astype(np.int64),
        "tried": tried,
        "family": record.family,
    }


def _forward(policy, episodes, device, tried_weight=1.0, collect=None):
    """The loss over a batch of episodes' decisions, in order. ``collect``:
    a list to append (verb, source, family, predicted mean, record) to, for
    evaluation."""
    model = policy.memory
    counts = [len(e["executed"]) for e in episodes]
    offsets = np.cumsum([0] + counts)
    a, b, c = (np.concatenate([e["rows"][i] for e in episodes]) for i in range(3))
    with torch.no_grad():
        tokens, present = policy.encode_scene(
            (
                torch.as_tensor(a, dtype=torch.long, device=device),
                torch.as_tensor(b, dtype=torch.long, device=device),
                torch.as_tensor(c, dtype=torch.float32, device=device),
            )
        )
    sprites = torch.as_tensor(np.concatenate([e["sprites"] for e in episodes]), device=device)
    mario = torch.as_tensor(np.concatenate([e["mario"] for e in episodes]), device=device)
    feedback = torch.as_tensor(np.concatenate([e["feedback"] for e in episodes]), device=device)
    context = torch.as_tensor(np.concatenate([e["context"] for e in episodes]), device=device)
    tokens = model.with_motion(tokens, sprites, mario)
    hidden, cell = model.initial(len(episodes), device)
    losses, weights = [], []
    for k in range(max(counts)):
        active = [i for i, n in enumerate(counts) if n > k]
        at = torch.as_tensor([offsets[i] + k for i in active], device=device)
        state = (hidden[active], cell[active])
        verbs = torch.as_tensor([episodes[i]["steps"][k] for i in active], device=device)
        pointers = torch.as_tensor([episodes[i]["step_pointers"][k] for i in active], device=device)
        stepped = model.advance(tokens, feedback, verbs, pointers, state, rows=at, context=context[at])
        records = torch.as_tensor(np.stack([episodes[i]["executed"][k] for i in active]), device=device)
        named = records[:, COLUMN["verb"]] >= 0
        if named.any():
            predicted = model.readout(
                tokens, present, stepped[0], verbs, pointers, rows=at, context=context[at]
            )
            predicted = {name: value[named] for name, value in predicted.items()}
            loss, _ = outcome_loss(predicted, records[named])
            losses.append(loss)
            weights.append(float(named.sum()))
            if collect is not None:
                for j, i in enumerate(np.asarray(active)[named.cpu().numpy()]):
                    collect.append(("executed", episodes[i]["family"], predicted["mean"][j], records[named][j], torch.sigmoid(predicted["success"][j])))
        # The teacher's other tries, from the state before this action.
        tried_rows, tried_at, tried_from = [], [], []
        for j, i in enumerate(active):
            for row in episodes[i]["tried"][k]:
                if row[COLUMN["verb"]] >= 0:
                    tried_rows.append(row)
                    tried_at.append(offsets[i] + k)
                    tried_from.append(j)
        if tried_rows:
            records = torch.as_tensor(np.stack(tried_rows), device=device)
            source = torch.as_tensor(tried_from, device=device)
            rows_at = torch.as_tensor(tried_at, device=device)
            verbs = records[:, COLUMN["verb"]].long()
            pointers = records[:, COLUMN["pointer"]].long()
            branch = model.advance(
                tokens,
                feedback,
                verbs,
                pointers,
                (state[0][source], state[1][source]),
                rows=rows_at,
                context=context[rows_at],
            )
            predicted = model.readout(
                tokens, present, branch[0], verbs, pointers, rows=rows_at, context=context[rows_at]
            )
            loss, _ = outcome_loss(predicted, records)
            losses.append(loss * tried_weight)
            weights.append(float(len(records)) * tried_weight)
            if collect is not None:
                for j in range(len(records)):
                    i = active[tried_from[j]]
                    collect.append(("tried", episodes[i]["family"], predicted["mean"][j], records[j], torch.sigmoid(predicted["success"][j])))
        hidden, cell = hidden.clone(), cell.clone()
        hidden[active], cell[active] = stepped
    total = sum(l * w for l, w in zip(losses, weights)) / max(1.0, sum(weights))
    return total


@torch.no_grad()
def evaluate(policy, episodes, device) -> dict:
    """Per verb × source ("all" included): end error (median, 90th
    percentile), moving-object error (median), frames error (median, exact
    lengths only), waits within 4 frames, success (Brier score), count; and
    whether each verb meets BARS."""
    policy.memory.eval()
    collected = []
    for start in range(0, len(episodes), 16):
        _forward(policy, episodes[start : start + 16], device, collect=collected)
    groups = defaultdict(list)
    for source, family, mean, record, success in collected:
        for verb in (VERBS[int(record[COLUMN["verb"]])], "all"):
            for key in ((verb, source), (verb, "all")):
                groups[key].append((mean.cpu().numpy(), record.cpu().numpy(), float(success)))
    report = {}
    for (verb, source), items in groups.items():
        means = np.stack([m for m, _, _ in items])
        records = np.stack([r for _, r, _ in items])
        success = np.asarray([s for _, _, s in items])
        succeeded = records[:, COLUMN["success"]] > 0.5
        entry = {"count": len(items), "success_brier": float(np.mean((success - succeeded) ** 2))}
        if succeeded.any():
            target = records[succeeded][:, [COLUMN[n] for n in OUTCOME_FIELDS]]
            error = np.abs(means[succeeded] - target)
            end = np.hypot(error[:, 0], error[:, 1])
            entry["end_error_median"] = float(np.median(end))
            entry["end_error_p90"] = float(np.percentile(end, 90))
            has = records[succeeded][:, COLUMN["has_object"]] > 0.5
            if has.any():
                entry["object_error_median"] = float(np.median(np.hypot(error[has, 2], error[has, 3])))
            exact = records[succeeded][:, COLUMN["frames_censored"]] < 0.5
            if exact.any():
                entry["frames_error_median"] = float(np.median(error[exact, 4]))
                if verb == "wait":
                    entry["wait_within_4"] = float(np.mean(error[exact, 4] <= 4))
        report[f"{verb}/{source}"] = entry
    passed = {
        verb: report.get(f"{verb}/all", {}).get("end_error_median", 1e9) <= BARS["end_error_median"]
        for verb in ("run", "jump")
    }
    passed["wait"] = report.get("wait/all", {}).get("wait_within_4", 0.0) >= BARS["wait_within_4"]
    return {"by_verb": report, "passed": passed}


def train_world_model(config: WorldModelConfig) -> dict:
    """Train ``config.source``'s world model on the collected episodes (its
    other weights frozen); save the policy with the best held-out loss, its
    world model's report and the encoder digest it read."""
    from . import layered_train as lt

    random.seed(config.seed)
    torch.manual_seed(config.seed)
    policy, saved = lt.load_layered_checkpoint(config.source, config.device)
    train, validation = [], []
    for path in config.episodes.split(","):
        a, b = pickle.loads(Path(path).read_bytes())
        train += a
        validation += b
    train_data = [episode_data(e) for e in train if len(e.decision_frames)]
    held = [episode_data(e) for e in validation if len(e.decision_frames)]
    for parameter in policy.parameters():
        parameter.requires_grad_(False)
    for parameter in policy.memory.parameters():
        parameter.requires_grad_(True)
    optimizer = torch.optim.AdamW(
        policy.memory.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    output = Path(config.output)
    output.mkdir(parents=True, exist_ok=True)
    reads = scene_digest(policy)
    layer_config = lt.LayeredTrainConfig(**{**lt.saved_settings(saved["config"]), "output": str(output)})
    history, best = [], None
    for epoch in range(config.epochs):
        started = time.time()
        policy.eval()
        policy.memory.train()
        random.shuffle(train_data)
        losses = []
        for start in range(0, len(train_data), config.batch_episodes):
            loss = _forward(
                policy, train_data[start : start + config.batch_episodes], config.device, config.tried_weight
            )
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.memory.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss))
        policy.memory.eval()
        with torch.no_grad():
            held_losses = [
                float(_forward(policy, held[s : s + 16], config.device)) for s in range(0, len(held), 16)
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
            torch.save(policy.memory.state_dict(), output / "best_memory.pt")
    policy.memory.load_state_dict(torch.load(output / "best_memory.pt", map_location=config.device))
    report = evaluate(policy, held, config.device)
    world_model = {
        "reads": reads,
        "report": report["by_verb"],
        "passed": report["passed"],
        "history": history,
        "episodes": config.episodes,
    }
    lt.save_layered_checkpoint(
        output / "policy.pt", policy, layer_config, saved["trained_layers"], saved.get("history", []), world_model
    )
    (output / "report.json").write_text(json.dumps({"best": best, **report}, indent=2) + "\n")
    return {"best": best, **report}


def collect(
    source: str,
    output: str,
    *,
    families: Sequence[str] = (),
    rounds: int = 2,
    layouts: int = 8,
    validation_layouts: int = 2,
    seed: int = 2026101602,
    workers: int = 15,
    lanes: int = 8,
) -> Path:
    """Episodes with the simulator's truth and the teacher's tried actions
    recorded (record_truth, record_actions): the teacher playing the skill
    families' layouts and the skill (``source``'s, the teacher carrying its
    actions out) playing them. Pickled (train, validation) lists of their
    KEPT fields in ``output``/episodes.pkl."""
    import dataclasses

    from . import layered_train as lt
    from .monte_carlo import BLOCK_SMB_MC_FAMILIES

    policy, saved = lt.load_layered_checkpoint(Path(source), "cpu")
    settings = lt.saved_settings(saved["config"])
    settings.update(
        learner="skill",
        families=tuple(families or lt.learner_families("skill", BLOCK_SMB_MC_FAMILIES)),
        output=output,
        workers=workers,
        lanes=lanes,
        seed=seed,
        predictor=None,
        tracker=None,
        extra_layouts="",
    )
    config = lt.LayeredTrainConfig(**settings)
    Path(output).mkdir(parents=True, exist_ok=True)
    pool = lt.EpisodePool(config, policy.settings)
    try:
        pool.publish(policy)

        def play(split, count, share):
            tasks = (
                [t for r in range(rounds) for t in lt._tasks(config, "train", count, r, share, True)]
                if split == "train"
                else lt._tasks(config, "validation", count, 0, share, True)
            )
            tasks = [
                dataclasses.replace(
                    t, teacher_share=share, label=True, record_truth=True, record_actions=True
                )
                for t in pool.with_scenarios(tasks)
            ]
            return [kept(r) for r in pool.play(tasks)]

        sets = {
            split: play(split, count, 1.0) + play(split, count, 0.0)
            for split, count in (("train", layouts), ("validation", validation_layouts))
        }
    finally:
        pool.close()
    path = Path(output) / "episodes.pkl"
    path.write_bytes(pickle.dumps((sets["train"], sets["validation"])))
    return path


def main(argv: Optional[Sequence[str]] = None) -> int:
    import argparse
    import dataclasses

    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    for field in dataclasses.fields(WorldModelConfig):
        parser.add_argument(
            f"--{field.name.replace('_', '-')}", type=type(field.default), default=field.default
        )
    parser.add_argument("--collect", action="store_true", help="first collect episodes")
    args = vars(parser.parse_args(argv))
    collecting = args.pop("collect")
    config = WorldModelConfig(**args)
    if collecting:
        collected = collect(config.source, config.output)
        config = dataclasses.replace(config, episodes=str(collected))
    print(json.dumps(train_world_model(config)["passed"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
