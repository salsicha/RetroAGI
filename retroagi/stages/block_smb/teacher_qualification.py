"""Qualify demonstrations on the exact layouts and limits consumed by training."""

import json
from dataclasses import replace
from pathlib import Path


def qualify_teachers(pool, tasks, report_path):
    """Fail before learning if a full spatial demonstration cannot finish.

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
            "passed": bool(record.won and complete_labels),
        }
        if not row["passed"]:
            row["scenario"] = task.scenario
        rows.append(row)
    failures = [row for row in rows if not row["passed"]]
    # No family may teach the skill one value: a destination (within two
    # pixels) covering most of a mode's labels (layered_train.label_collapse).
    learner = getattr(getattr(pool, "config", None), "learner", None)
    collapsed = label_collapse(records, "skill") if learner == "skill" else {}
    report = {
        "execution": "training_episode_pipeline",
        "stall_frames": STALL_FRAMES,
        "total": len(rows),
        "passed": len(rows) - len(failures),
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
        f"{', labels collapse in ' + ', '.join(sorted(collapsed)) if collapsed else ''}: {path}",
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
    if failures:
        examples = ", ".join(
            f"{row['family']}[{row['sample_index']}] {row['end']} at {row['frames']} frames"
            + (" (missing destination labels)" if not row["complete_labels"] else "")
            for row in failures[:8]
        )
        raise RuntimeError(f"Teacher qualification failed: {examples}. Details: {path}")
    return records
