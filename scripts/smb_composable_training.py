"""Fresh, staged 30-epoch composable SMB training with explicit promotion gates."""

import argparse
import hashlib
import json
import os
from dataclasses import asdict
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import torch

from retroagi.core.smb_components import SMBComponentContract, export_bundle
from retroagi.core.smb_learning import (
    block_stage,
    collect_case,
    collect_reactive_case,
    make_model,
    playback,
    rows_to_data,
    save_dataset,
)
from retroagi.stages.block_smb.demonstrations import fit_demonstrations
from retroagi.stages.block_smb.monte_carlo import BLOCK_SMB_MC_FAMILIES
from retroagi.stages.block_smb.nes_curriculum import sample_nes_case
from scripts.smb_perception_training import block_clips, train_perception


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, default=str) + "\n")
    temporary.replace(path)


def samples(config, split, count, *, offset=0, families=None):
    result = []
    for family in families or config["families"]:
        for i in range(count):
            result.append(
                sample_nes_case(
                    family=family,
                    split=split,
                    seed=config["seed"],
                    index=offset + i,
                    difficulty=("easy", "medium", "hard")[i % 3],
                )
            )
    return result


def collect(model, cases, vision, *, log):
    rows = []
    episodes = []
    for sample in cases:
        stage = block_stage(sample, vision=vision, device=next(model.parameters()).device)
        try:
            data, result = collect_case(
                model,
                stage,
                sample.oracle["actions"],
                family=BLOCK_SMB_MC_FAMILIES.index(sample.family),
            )
            if not data:
                data, result = collect_reactive_case(
                    model, stage, family=BLOCK_SMB_MC_FAMILIES.index(sample.family)
                )
            episodes.append(
                dict(
                    id=sample.scenario_id,
                    family=sample.family,
                    split=sample.split,
                    start=len(rows),
                    length=len(data),
                    **result,
                )
            )
            rows.extend(data)
        finally:
            stage.env.close()
    log(
        dict(
            phase="demonstrations",
            accepted=sum(e["success"] for e in episodes),
            attempted=len(episodes),
            frames=len(rows),
        )
    )
    missing = set(s.family for s in cases) - {e["family"] for e in episodes if e["success"]}
    if missing:
        raise RuntimeError(f"No executor-verified demonstrations for {sorted(missing)}")
    return rows_to_data(rows), episodes


def evaluate(model, cases, vision, *, max_steps=320):
    results = []
    for sample in cases:
        stage = block_stage(sample, vision=vision, device=next(model.parameters()).device)
        try:
            result = playback(model, stage, max_steps=max_steps)
        finally:
            stage.env.close()
        results.append(
            dict(
                id=sample.scenario_id,
                family=sample.family,
                difficulty=sample.difficulty_bin,
                **{k: v for k, v in result.items() if k != "actions"},
            )
        )
    rates = {
        family: {
            difficulty: sum(
                r["success"]
                for r in results
                if r["family"] == family and r["difficulty"] == difficulty
            )
            / max(1, sum(r["family"] == family and r["difficulty"] == difficulty for r in results))
            for difficulty in ("easy", "medium", "hard")
        }
        for family in sorted(set(s.family for s in cases))
    }
    return dict(
        rates=rates, results=results, minimum=min(v for d in rates.values() for v in d.values())
    )


def run(config, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)

    def log(event):
        print(json.dumps(event, default=str), flush=True)
        with (output / "events.jsonl").open("a") as stream:
            stream.write(json.dumps(event, default=str) + "\n")
        write_json(output / "status.json", event)

    torch.set_num_threads(1)
    torch.manual_seed(config["seed"])
    torch.use_deterministic_algorithms(True)
    sources = sorted(Path("retroagi").rglob("*.py")) + sorted(Path("scripts").glob("smb_*.py"))
    write_json(
        output / "manifest.json",
        dict(
            config=config,
            sources={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
            fresh_initialization=True,
            init_checkpoint=None,
            fixed_scenes=[],
            runtime="smb_scene_v2",
            full_level_qualified=False,
        ),
    )
    try:
        log(dict(phase="physics_audit"))
        from scripts.smb_physics_audit import audit

        physics = audit()
        write_json(output / "physics.json", physics)
        if not physics["exact_motion_gate"]:
            raise RuntimeError("NES motion gate failed")
        log(dict(phase="source_perception_dataset"))
        train = samples(config, "train", config["perception_layouts_per_family"])
        validation = samples(config, "validation", config["perception_validation_per_family"])
        clips = block_clips(train, output / "block_train_clips")
        validation_clips = block_clips(validation, output / "block_validation_clips")
        vision, perception = train_perception(
            clips,
            validation_clips,
            output / "block_perception",
            steps=config["perception_updates"],
            device=config["device"],
            seed=config["seed"],
            log=log,
        )
        if not perception["qualified"]:
            raise RuntimeError(
                "Block collision-perception gate failed; see per-class and edge metrics"
            )
        # Each family starts with independent fresh core weights. Validation
        # controls promotion; the test split is evaluated once after promotion.
        family_reports = {}
        for family in config["families"]:
            log(dict(phase="family_learning", family=family))
            torch.manual_seed(config["seed"])
            model = make_model(
                hidden_dim=config["hidden_dim"], device=config["device"], provider="perceived"
            )
            data, episodes = collect(
                model,
                samples(config, "train", config["family_train_layouts"], families=[family]),
                vision,
                log=log,
            )
            optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"])
            checks = samples(config, "validation", config["family_eval_layouts"], families=[family])
            streak = 0
            for round_index in range(config["family_rounds"]):
                loss = fit_demonstrations(
                    model,
                    optimizer,
                    data,
                    steps=config["family_updates"],
                    seed=config["seed"] + round_index,
                    decision_durations_only=True,
                    walk_durations=False,
                    prioritized=True,
                )
                result = evaluate(model, checks, vision)
                streak = streak + 1 if result["minimum"] >= config["family_gate"] else 0
                log(
                    dict(
                        phase="family_learning",
                        family=family,
                        round=round_index + 1,
                        loss=loss,
                        minimum=result["minimum"],
                        rates=result["rates"],
                        successive_passes=streak,
                    )
                )
                if streak >= 2:
                    break
            family_reports[family] = dict(validation=result, passed=streak >= 2)
            if streak >= 2:
                test = evaluate(
                    model,
                    samples(config, "test", config["family_eval_layouts"], families=[family]),
                    vision,
                )
                family_reports[family].update(
                    test=test, passed=test["minimum"] >= config["family_gate"]
                )
            write_json(output / "family_learning.json", family_reports)
        if not all(r["passed"] for r in family_reports.values()):
            raise RuntimeError("Family learnability gate failed; shared full-volume phase withheld")
        log(dict(phase="full_volume_initialization", epochs=config["epochs"]))
        # Fresh core again: independent family qualifiers are not a hidden warm start.
        torch.manual_seed(config["seed"])
        model = make_model(
            hidden_dim=config["hidden_dim"], device=config["device"], provider="perceived"
        )
        optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"])
        data, episodes = collect(
            model,
            samples(config, "train", config["demonstration_layouts_per_family"], offset=10000),
            vision,
            log=log,
        )
        save_dataset(data, output / "demonstrations.pth", episodes=episodes, provider="perceived")
        fit_demonstrations(
            model,
            optimizer,
            data,
            steps=config["bootstrap_updates"],
            seed=config["seed"],
            decision_durations_only=True,
            walk_durations=False,
            prioritized=True,
        )
        validation = samples(
            config, "validation", config["validation_layouts_per_family"], offset=10000
        )
        for epoch in range(1, config["epochs"] + 1):
            new = samples(
                config,
                "train",
                config["train_layouts_per_family_per_epoch"],
                offset=20000 + epoch * 1000,
            )
            fresh, _ = collect(model, new, vision, log=log)
            combined = type(data)(
                **{k: torch.cat((getattr(data, k), getattr(fresh, k))) for k in asdict(data)}
            )
            loss = fit_demonstrations(
                model,
                optimizer,
                combined,
                steps=config["rehearsal_updates"],
                seed=config["seed"] + epoch,
                decision_durations_only=True,
                walk_durations=False,
                prioritized=True,
            )
            result = evaluate(model, validation, vision)
            log(
                dict(
                    phase="full_volume",
                    epoch=epoch,
                    epochs=config["epochs"],
                    loss=loss,
                    minimum=result["minimum"],
                    rates=result["rates"],
                )
            )
            export_bundle(
                model,
                output / f"epoch_{epoch:02d}",
                contract=SMBComponentContract(),
                architecture={
                    "hidden_dim": config["hidden_dim"],
                    "controller_schedule": "constant",
                },
                perception=output / "block_perception" / "perception.pth",
            )
            write_json(output / f"validation_{epoch:02d}.json", result)
        test = evaluate(
            model,
            samples(config, "test", config["validation_layouts_per_family"], offset=10000),
            vision,
        )
        write_json(output / "block_test.json", test)
        if result["minimum"] < config["family_gate"] or test["minimum"] < config["family_gate"]:
            raise RuntimeError("Shared Block held-out family gate failed")
        log(dict(phase="recurrent_context_qualification"))
        from copy import deepcopy
        from dataclasses import replace

        from retroagi.core.smb_learning import fit_ordered_sequences, recurrent_causal_probe
        from retroagi.core.smb_runtime import attach_runtime

        baseline = deepcopy(model.state_dict())
        baseline_runtime = model.smb_runtime_contract
        attach_runtime(model, replace(baseline_runtime, recurrent_state=True).manifest())
        sequence_data, sequence_episodes = collect(
            model,
            samples(config, "train", config["sequence_layouts_per_family"], offset=95000),
            vision,
            log=log,
        )
        sequence_training = fit_ordered_sequences(model, sequence_data, sequence_episodes)
        causal = recurrent_causal_probe(model, sequence_data)
        recurrent_result = evaluate(model, validation, vision)
        recurrent_qualified = (
            causal["causal_path_present"]
            and recurrent_result["minimum"] >= config["family_gate"]
            and recurrent_result["minimum"] >= result["minimum"]
        )
        write_json(
            output / "recurrent_context.json",
            dict(
                training=sequence_training,
                causal=causal,
                validation=recurrent_result,
                qualified=recurrent_qualified,
            ),
        )
        if not recurrent_qualified:
            model.load_state_dict(baseline)
            attach_runtime(model, baseline_runtime.manifest())
        log(dict(phase="emulator_approaches", recurrent_context=recurrent_qualified))
        from scripts.smb_contact_audit import compare_contacts
        from scripts.smb_emulator_curriculum import (
            capture,
            evaluate_matrix,
            full_level_tests,
            qualify_individual_approaches,
        )

        approaches = capture(
            output / "emulator",
            approaches=config["emulator_approaches"],
            device=config["device"],
            variations={
                split: [(3 * i + offset, i % 5) for i in range(count)]
                for split, offset, count in (
                    ("train", 0, config["emulator_train_variations"]),
                    ("validation", 1, config["emulator_eval_variations"]),
                    ("test", 2, config["emulator_eval_variations"]),
                )
            },
        )
        contact_result = compare_contacts(output / "emulator")
        write_json(output / "contact_audit.json", contact_result)
        # Split perception by approach identity, not adjacent frames from the
        # same trajectory. Control tests separately vary each known approach.
        full_clips = []
        for case in approaches["cases"]:
            split = "validation" if case["approach"] % 3 == 2 else "train"
            full_clips.append(
                dict(
                    file=str((output / "emulator" / case["pixels"]).resolve()),
                    scenario_id=case["id"],
                    split=split,
                    frames=case["frames"],
                    label_source=case["label_source"],
                )
            )
        from scripts.smb_teacher_audit import audit_teacher

        audit_teacher(
            [c for c in full_clips if c["split"] == "validation"],
            output / "cnn_teacher_audit.json",
            device=config["device"],
        )
        full_vision, metrics = train_perception(
            [c for c in full_clips if c["split"] == "train"],
            [c for c in full_clips if c["split"] == "validation"],
            output / "full_perception",
            steps=config["perception_updates"],
            device=config["device"],
            seed=config["seed"] + 1,
            log=log,
        )
        if not metrics["qualified"]:
            raise RuntimeError("Full real-frame collision-perception gate failed")
        individual = qualify_individual_approaches(
            output / "emulator",
            full_vision,
            updates=config["emulator_learning_updates"],
            device=config["device"],
            hidden_dim=config["hidden_dim"],
        )
        matrix = evaluate_matrix(
            model,
            output / "emulator",
            full_vision,
            device=config["device"],
            block_replay=(sequence_data, sequence_episodes),
            retention_evaluator=lambda candidate: evaluate(candidate, validation, vision),
        )
        if (
            all(r["passed"] for r in individual)
            and not approaches["unresolved"]
            and all(r["success"] for r in matrix["perceived_frozen"]["evaluations"])
        ):
            write_json(output / "full_level_tests.json", full_level_tests(model, full_vision))
        export_bundle(
            model,
            output / "full_bundle",
            contract=SMBComponentContract(
                recurrent_state=model.smb_runtime_contract.recurrent_state
            ),
            architecture={"hidden_dim": config["hidden_dim"], "controller_schedule": "constant"},
            perception=output / "full_perception" / "perception.pth",
        )
        log(
            dict(
                phase="local_transfer_complete",
                full_level_qualified=False,
                unresolved_approaches=len(approaches["unresolved"]),
                next_gate=matrix["next_gate"],
            )
        )
    except Exception as error:
        log(dict(phase="gate_failed", error=str(error), full_level_qualified=False))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=Path("scripts/configs/smb_composable_full_volume.json")
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    if (
        config["epochs"] != 30
        or config.get("init_checkpoint")
        or config.get("resume_path")
        or config.get("fixed_scenes")
    ):
        raise ValueError("This pipeline requires a fresh 30-epoch generated curriculum")
    if set(config["families"]) != set(BLOCK_SMB_MC_FAMILIES):
        raise ValueError("All 21 families must be present")
    run(config, args.output_dir)


if __name__ == "__main__":
    main()
