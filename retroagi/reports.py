"""Architecture sweep comparison reports for experiment and promotion manifests."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from retroagi.core import BASELINE_ARCHITECTURE_NAME, to_plain_data
from retroagi.experiments import _architecture_config_item


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="retroagi report",
        description="Build a comparison report from experiment or promotion manifests.",
    )
    parser.add_argument(
        "--input",
        action="append",
        required=True,
        type=Path,
        help="manifest JSON to include; repeat for architecture sweeps",
    )
    parser.add_argument("--output", required=True, type=Path, help="comparison report JSON path")
    parser.add_argument(
        "--baseline-architecture",
        default=BASELINE_ARCHITECTURE_NAME,
        help="architecture name to use as the delta baseline",
    )
    parser.add_argument(
        "--baseline-config",
        action="append",
        default=None,
        type=_architecture_config_item,
        metavar="KEY=VALUE",
        help="optional baseline architecture config matcher; may be repeated",
    )
    return parser


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    baseline_config = dict(args.baseline_config or ())
    runs = []
    rows = []
    for input_path in args.input:
        manifest = _load_manifest(input_path)
        run = _run_summary(input_path, manifest)
        run_rows = _manifest_rows(input_path, manifest, run)
        runs.append({**run, "row_count": len(run_rows)})
        rows.extend(run_rows)

    baseline = _select_baseline(rows, args.baseline_architecture, baseline_config)
    _attach_regression_deltas(rows, baseline)
    return {
        "inputs": [str(path) for path in args.input],
        "baseline": _baseline_summary(baseline),
        "runs": runs,
        "rows": rows,
        "summary": {
            "run_count": len(runs),
            "row_count": len(rows),
            "game_count": len({row["game_key"] for row in rows}),
            "game_row_counts": _game_row_counts(rows),
            "passed_count": sum(1 for row in rows if row.get("passed") is True),
            "failed_count": sum(1 for row in rows if row.get("passed") is False),
            "stopped_count": sum(1 for row in rows if row.get("status") == "stopped"),
            "skipped_count": sum(1 for row in rows if row.get("status") == "skipped"),
        },
    }


def _load_manifest(path: Path) -> dict[str, Any]:
    loaded = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError(f"manifest {path} must contain a JSON object")
    return loaded


def _run_summary(path: Path, manifest: Mapping[str, Any]) -> dict[str, Any]:
    architecture = _architecture(manifest)
    game = _game(manifest)
    return {
        "input": str(path),
        "manifest_type": _manifest_type(manifest),
        "architecture": architecture,
        "architecture_key": _architecture_key(architecture),
        "game": game,
        "game_key": _game_key(game),
        "seed": manifest.get("seed"),
        "device": manifest.get("device"),
        "passed": manifest.get("passed"),
        "artifacts_dir": manifest.get("artifacts_dir"),
    }


def _manifest_rows(
    path: Path,
    manifest: Mapping[str, Any],
    run: Mapping[str, Any],
) -> list[dict[str, Any]]:
    if "rungs" in manifest:
        return _promotion_rows(path, manifest, run)
    return _experiment_rows(path, manifest, run)


def _experiment_rows(
    path: Path,
    manifest: Mapping[str, Any],
    run: Mapping[str, Any],
    *,
    rung: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    rows = []
    stages = manifest.get("stages", [])
    if not isinstance(stages, Sequence):
        return rows
    for stage in stages:
        if not isinstance(stage, Mapping):
            continue
        stage_name = str(stage.get("stage", "unknown-stage"))
        row = _base_row(path, run, kind="stage", name=stage_name)
        if rung is not None:
            row["rung"] = rung.get("name")
            row["comparison_key"] = f"{rung.get('name')}:{stage_name}"
            row["runtime_seconds"] = rung.get("runtime_seconds")
            row["automatic_gates"] = list(rung.get("automatic_gates", []))
        row.update(
            {
                "stage": stage_name,
                "status": "passed" if stage.get("passed") else "failed",
                "passed": stage.get("passed"),
                "metrics": _numeric_metrics(stage.get("metrics", {})),
                "gates": list(stage.get("gates", [])),
                "artifacts": _artifact_paths(stage),
                "command": stage.get("command"),
            }
        )
        rows.append(row)
    return rows


def _promotion_rows(
    path: Path,
    manifest: Mapping[str, Any],
    run: Mapping[str, Any],
) -> list[dict[str, Any]]:
    rows = []
    rungs = manifest.get("rungs", [])
    if not isinstance(rungs, Sequence):
        return rows
    for rung in rungs:
        if not isinstance(rung, Mapping):
            continue
        rung_name = str(rung.get("name", "unknown-rung"))
        row = _base_row(path, run, kind="rung", name=rung_name)
        rung_metrics = rung.get("metrics", {})
        row.update(
            {
                "rung": rung_name,
                "status": rung.get("status"),
                "passed": rung.get("passed"),
                "runtime_seconds": rung.get("runtime_seconds"),
                # Rungs without a nested experiment (e.g. the Full SMB rungs)
                # carry their metrics directly on the rung record.
                "metrics": dict(rung_metrics) if isinstance(rung_metrics, Mapping) else {},
                "gates": list(rung.get("automatic_gates", [])),
                "artifacts": _rung_artifacts(rung),
            }
        )
        rows.append(row)
        experiment = rung.get("experiment")
        if isinstance(experiment, Mapping):
            rows.extend(_experiment_rows(path, experiment, run, rung=rung))
    return rows


def _base_row(path: Path, run: Mapping[str, Any], *, kind: str, name: str) -> dict[str, Any]:
    return {
        "input": str(path),
        "kind": kind,
        "name": name,
        "comparison_key": name,
        "architecture": run["architecture"],
        "architecture_key": run["architecture_key"],
        "game": run["game"],
        "game_key": run["game_key"],
        "seed": run.get("seed"),
        "device": run.get("device"),
    }


def _architecture(manifest: Mapping[str, Any]) -> dict[str, Any]:
    architecture = manifest.get("architecture", {})
    if not isinstance(architecture, Mapping):
        architecture = {}
    return {
        "name": architecture.get("name"),
        "config": (
            dict(architecture.get("config", {}))
            if isinstance(architecture.get("config", {}), Mapping)
            else {}
        ),
    }


def _architecture_key(architecture: Mapping[str, Any]) -> str:
    config = architecture.get("config", {})
    return (
        f"{architecture.get('name')} "
        f"{json.dumps(config, sort_keys=True, separators=(',', ':'))}"
    )


def _game(manifest: Mapping[str, Any]) -> dict[str, Any]:
    game = manifest.get("game", {})
    if not isinstance(game, Mapping):
        game = {}
    backend = game.get("backend", {})
    if not isinstance(backend, Mapping):
        backend = {}
    stage_ladder = game.get("stage_ladder", [])
    if not isinstance(stage_ladder, Sequence) or isinstance(stage_ladder, (str, bytes)):
        stage_ladder = []
    return {
        "name": game.get("name"),
        "family": game.get("family"),
        "backend": backend.get("name"),
        "stage_ladder": [
            stage.get("name")
            for stage in stage_ladder
            if isinstance(stage, Mapping) and stage.get("name") is not None
        ],
    }


def _game_key(game: Mapping[str, Any]) -> str:
    return str(game.get("name") or "unknown-game")


def _manifest_type(manifest: Mapping[str, Any]) -> str:
    if "rungs" in manifest:
        return "promotion"
    if "stages" in manifest:
        return "experiment"
    return "unknown"


def _numeric_metrics(metrics: Any) -> dict[str, float]:
    if not isinstance(metrics, Mapping):
        return {}
    return {
        str(key): float(value)
        for key, value in metrics.items()
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    }


def _artifact_paths(stage: Mapping[str, Any]) -> dict[str, str]:
    artifacts = {}
    for field in ("summary_path", "checkpoint_path", "log_path"):
        value = stage.get(field)
        if value is not None:
            artifacts[field] = str(value)
    return artifacts


def _rung_artifacts(rung: Mapping[str, Any]) -> dict[str, str]:
    artifacts = {}
    rung_artifacts = rung.get("artifacts")
    if isinstance(rung_artifacts, Mapping):
        for key, value in rung_artifacts.items():
            if value is not None:
                artifacts[str(key)] = str(value)
    value = rung.get("experiment_manifest_path")
    if value is not None:
        artifacts["experiment_manifest_path"] = str(value)
    return artifacts


def _select_baseline(
    rows: Sequence[Mapping[str, Any]],
    baseline_architecture: str,
    baseline_config: Mapping[str, Any],
) -> dict[str, Any] | None:
    candidates = []
    for row in rows:
        architecture = row.get("architecture", {})
        if not isinstance(architecture, Mapping):
            continue
        if architecture.get("name") != baseline_architecture:
            continue
        config = architecture.get("config", {})
        if baseline_config and config != baseline_config:
            continue
        candidates.append(dict(row))
    return candidates[0] if candidates else None


def _baseline_summary(baseline: Mapping[str, Any] | None) -> dict[str, Any] | None:
    if baseline is None:
        return None
    return {
        "architecture": baseline.get("architecture"),
        "architecture_key": baseline.get("architecture_key"),
        "input": baseline.get("input"),
        "comparison_key": baseline.get("comparison_key"),
    }


def _attach_regression_deltas(
    rows: list[dict[str, Any]], baseline: Mapping[str, Any] | None
) -> None:
    baseline_metrics = _baseline_metric_index(rows, baseline)
    for row in rows:
        deltas = {}
        metrics = row.get("metrics", {})
        if not isinstance(metrics, Mapping):
            metrics = {}
        for metric, value in metrics.items():
            baseline_value = baseline_metrics.get((row["game_key"], row["comparison_key"], metric))
            if baseline_value is None:
                continue
            delta = float(value) - baseline_value
            deltas[metric] = {
                "baseline": baseline_value,
                "actual": float(value),
                "delta": delta,
                "percent_delta": (delta / baseline_value) if baseline_value != 0 else None,
            }
        row["regression_deltas"] = deltas


def _baseline_metric_index(
    rows: Sequence[Mapping[str, Any]],
    baseline: Mapping[str, Any] | None,
) -> dict[tuple[str, str, str], float]:
    if baseline is None:
        return {}
    baseline_key = baseline.get("architecture_key")
    index = {}
    for row in rows:
        if row.get("architecture_key") != baseline_key:
            continue
        metrics = row.get("metrics", {})
        if not isinstance(metrics, Mapping):
            continue
        for metric, value in metrics.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                index.setdefault(
                    (str(row["game_key"]), str(row["comparison_key"]), str(metric)),
                    float(value),
                )
    return index


def _game_row_counts(rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        key = str(row.get("game_key", "unknown-game"))
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    report = build_report(args)
    output = json.dumps(to_plain_data(report), indent=2, sort_keys=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(output + "\n", encoding="utf-8")
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
