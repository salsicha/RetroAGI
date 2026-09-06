"""Summarize completed learnability runs without treating missing evidence as a pass."""

import argparse
import hashlib
import json
from pathlib import Path

from retroagi.stages.block_smb.monte_carlo import BLOCK_SMB_MC_FAMILIES


def qualified_result(result):
    test = result.get("test_evaluation")
    if not result.get("passed") or not isinstance(test, dict):
        return False
    for difficulty in ("easy", "medium", "hard"):
        counts = test.get("counts", {}).get(difficulty)
        if not counts or counts[1] < 10 or counts[0] / counts[1] < 0.9:
            return False
    return True


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("runs", nargs="+", type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--seeds", nargs="+", type=int, default=[101, 202, 303])
    p.add_argument("--shared-dir", type=Path)
    p.add_argument("--shared-holdout", type=Path)
    p.add_argument("--recheck-dirs", nargs="*", type=Path, default=[])
    args = p.parse_args()
    latest = {}
    run_metadata = {}
    for root in args.runs:
        arguments = json.loads((root / "arguments.json").read_text())
        if not arguments.get("autonomous"):
            continue
        manifest_path = root / "manifest.json"
        manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
        run_metadata[str(root)] = dict(
            arguments=arguments,
            config=manifest.get("config"),
            source_manifest_sha256=hashlib.sha256(
                json.dumps(manifest.get("sources", {}), sort_keys=True).encode()
            ).hexdigest(),
        )
        for path in root.glob("*/result.json"):
            result = json.loads(path.read_text())
            key = (result["family"], result["seed"])
            if key not in latest or path.stat().st_mtime > latest[key][0]:
                latest[key] = (path.stat().st_mtime, path, result)
    rechecks = {}
    for root in args.recheck_dirs:
        for path in root.glob("*_seed*.json"):
            record = json.loads(path.read_text())
            key = (record["family"], record["seed"], record["checkpoint"])
            if key not in rechecks or path.stat().st_mtime > rechecks[key][0]:
                rechecks[key] = (path.stat().st_mtime, path, record)
    rows = []
    for family in BLOCK_SMB_MC_FAMILIES:
        seeds = {}
        for seed in args.seeds:
            item = latest.get((family, seed))
            seeds[str(seed)] = (
                None
                if item is None
                else dict(
                    passed=qualified_result(item[2]),
                    test=item[2].get("test_evaluation"),
                    validation=item[2]["evaluation"],
                    result_path=str(item[1]),
                )
            )
            if item is not None:
                checkpoint = str(item[1].parent / "policy.pth")
                recheck = rechecks.get((family, seed, checkpoint))
                if recheck is not None:
                    cell = seeds[str(seed)]
                    cell["original_test"] = cell["test"]
                    cell["test"] = recheck[2]["evaluation"]
                    cell["recheck_path"] = str(recheck[1])
                    cell["passed"] &= qualified_result(
                        dict(passed=cell["test"]["passed"], test_evaluation=cell["test"])
                    )
        rows.append(
            dict(
                family=family,
                seeds=seeds,
                passed=all(r is not None and r["passed"] for r in seeds.values()),
            )
        )
    shared = None
    if args.shared_dir and (args.shared_dir / "result.json").exists():
        shared = json.loads((args.shared_dir / "result.json").read_text())
    shared_passed = bool(
        shared
        and shared.get("passed")
        and all(
            qualified_result(
                dict(passed=True, test_evaluation=shared.get("test_families", {}).get(f))
            )
            for f in BLOCK_SMB_MC_FAMILIES
        )
    )
    extra_holdout = None
    if args.shared_holdout:
        if args.shared_holdout.exists():
            extra_holdout = json.loads(args.shared_holdout.read_text())
        checkpoint = args.shared_dir / "policy.pth" if args.shared_dir else None
        shared_passed = bool(
            shared_passed
            and extra_holdout
            and extra_holdout.get("passed")
            and checkpoint
            and checkpoint.exists()
            and extra_holdout.get("checkpoint_sha256")
            == hashlib.sha256(checkpoint.read_bytes()).hexdigest()
            and all(
                qualified_result(
                    dict(passed=True, test_evaluation=extra_holdout.get("families", {}).get(f))
                )
                for f in BLOCK_SMB_MC_FAMILIES
            )
        )
    report = dict(
        qualified=all(r["passed"] for r in rows) and shared_passed,
        shared_holdout=extra_holdout,
        shared_result=shared,
        shared_run=(
            {
                name: json.loads((args.shared_dir / filename).read_text())
                for name, filename in (("arguments", "arguments.json"), ("config", "config.json"))
                if (args.shared_dir / filename).exists()
            }
            if args.shared_dir
            else None
        ),
        families=rows,
        shared_passed=shared_passed,
        shared_dir=str(args.shared_dir),
        recheck_dirs=[str(path) for path in args.recheck_dirs],
        runs={
            root: metadata
            for root, metadata in run_metadata.items()
            if any(str(item[1].parent.parent) == root for item in latest.values())
        },
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    lines = [
        "# Block SMB learning audit",
        "",
        "Pass requires two successive autonomous validations, then at least 9/10 successes at each difficulty on a separate test split. Each family needs all three training seeds. One shared policy must also pass; missing evidence is incomplete.",
        "",
        "Demonstration runs use supervised learning. Evaluation supplies no demonstration actions or family-provided A actions. Percentages are observed level-completion rates, not confidence bounds.",
        "",
        "| Family | " + " | ".join(f"Seed {seed}" for seed in args.seeds) + " | Qualified |",
        "| --- | " + " | ".join("---" for _ in args.seeds) + " | --- |",
    ]
    for row in rows:
        cells = []
        for result in row["seeds"].values():
            if result is None:
                cells.append("Pending")
            elif result["test"] is None:
                cells.append("Failed validation")
            else:
                cells.append(
                    "/".join(
                        f'{100*result["test"]["rates"][d]:.0f}%' for d in ("easy", "medium", "hard")
                    )
                )
        lines.append(
            "| "
            + row["family"]
            + " | "
            + " | ".join(cells)
            + " | "
            + ("Yes" if row["passed"] else "No")
            + " |"
        )
    lines += [
        "",
        "Rates are easy / medium / hard. Matching current-code rechecks replace historical test rates when available. The JSON companion records both results and run configurations. Isolated families may use different teaching and control recipes; they do not establish that one common recipe is reliable across seeds.",
        "",
        f'Shared policy qualified: **{shared_passed}**. Full audit qualified: **{report["qualified"]}**.',
        "",
    ]
    if extra_holdout:
        lines += [
            f"Additional frozen-policy holdout seed: **{extra_holdout['arguments']['seed']}**.",
            "",
            "| Family | Shared policy: easy / medium / hard |",
            "| --- | --- |",
        ]
        for family in BLOCK_SMB_MC_FAMILIES:
            result = extra_holdout.get("families", {}).get(family)
            rates = (
                "/".join(f"{100*result['rates'][d]:.0f}%" for d in ("easy", "medium", "hard"))
                if result
                else "Pending"
            )
            lines.append(f"| {family} | {rates} |")
        lines.append("")
    args.output.with_suffix(".md").write_text("\n".join(lines))
    print(
        json.dumps(
            dict(
                qualified=report["qualified"],
                qualified_families=sum(r["passed"] for r in rows),
                families=len(rows),
            )
        )
    )


if __name__ == "__main__":
    main()
