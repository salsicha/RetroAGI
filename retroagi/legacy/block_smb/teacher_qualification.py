"""Qualify demonstrations on the exact layouts and limits consumed by training."""

import json
from dataclasses import replace
from pathlib import Path


def qualify_teachers(pool, tasks, report_path):
    """The teacher plays each prepared layout through the training pipeline;
    returns (tasks, records) of those it finished with every decision
    labelled. A layout it fails (its objective not reached, or a decision
    with no teacher command) is recorded, with its layout, in the report and
    left out: its data is not trained on (user, 2026-10-17).

    Prepared scenarios are shared with the learner: no redraw, enlarged frame
    budget, or separate replay loop is allowed. Even for tactic training the
    skill destinations come from the teacher, rather than the learned skill.
    """
    from .layered_train import STALL_FRAMES, label_collapse

    if any(task.scenario is None for task in tasks):
        raise ValueError("teacher qualification requires the actual prepared layouts")
    demonstrations = [replace(task, teacher_share=1.0, label=True, explore=False) for task in tasks]
    records = pool.play(demonstrations, teacher_only=True)
    if len(records) != len(tasks):
        raise RuntimeError("teacher qualification returned an incomplete set of episodes")
    rows = []
    for task, record in zip(tasks, records):
        valid = record.labels["valid"]
        complete_labels = len(valid) > 0 and bool(valid.all())
        # Skill decisions where the teacher's command names no action (its
        # object is not one the scene reports): nothing there to imitate.
        named = record.labels.get("action_valid")
        unnamed = int((valid & ~named.astype(bool)).sum()) if named is not None else 0
        row = {
            "family": task.family,
            "split": task.split,
            "seed": task.seed,
            "sample_index": task.sample_index,
            "difficulty": task.difficulty,
            "frame_limit": max(task.frames, int(task.scenario.get("frame_budget", 0))),
            "frames": record.frames,
            "end": record.end,
            "won": bool(record.won),
            "complete_labels": complete_labels,
            "unnamed_choices": unnamed,
            "passed": bool(record.won and complete_labels),
        }
        if not row["passed"]:
            row["scenario"] = task.scenario
        rows.append(row)
    failures = [row for row in rows if not row["passed"]]
    kept = [(task, record) for task, record, row in zip(tasks, records, rows) if row["passed"]]
    # No family may teach the skill one value: a destination (within two
    # pixels) covering most of a mode's labels (layered_train.label_collapse).
    learner = getattr(getattr(pool, "config", None), "learner", None)
    collapsed = label_collapse([r for _, r in kept], "skill") if learner == "skill" else {}
    unnamed = {}
    for row in rows:
        if row["unnamed_choices"]:
            unnamed[row["family"]] = unnamed.get(row["family"], 0) + row["unnamed_choices"]
    report = {
        "execution": "training_episode_pipeline",
        "stall_frames": STALL_FRAMES,
        "total": len(rows),
        "passed": len(rows) - len(failures),
        # Left out of training, with their layouts in "episodes".
        "failed": [
            f"{row['family']}[{row['sample_index']}] {row['difficulty']}: {row['end']} at {row['frames']} frames"
            + ("" if row["complete_labels"] else " (missing labels)")
            for row in failures
        ],
        "unnamed_choices": unnamed,
        "labels_collapsed": {
            family: {kind: list(v) for kind, v in kinds.items()}
            for family, kinds in collapsed.items()
        },
        "all_passed": not failures and not collapsed,
        "episodes": rows,
    }
    path = Path(report_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n")
    print(
        f"[teacher qualification] {report['passed']}/{report['total']} passed"
        f"{' (failed, left out: ' + '; '.join(report['failed'][:8]) + ')' if failures else ''}"
        f"{', labels collapse in ' + ', '.join(sorted(collapsed)) if collapsed else ''}"
        f", {sum(unnamed.values())} unnamed choices: {path}",
        flush=True,
    )
    if collapsed:
        raise RuntimeError(
            "Teacher labels collapse on one value in: "
            + "; ".join(
                f"{family} {kind} {v[0]} {v[1]:.0%} of {v[2]}"
                for family, kinds in collapsed.items()
                for kind, v in kinds.items()
            )
            + f". Details: {path}"
        )
    return [task for task, _ in kept], [record for _, record in kept]
