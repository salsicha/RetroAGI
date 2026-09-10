"""Fresh, staged 30-epoch composable SMB training with explicit promotion gates."""

import argparse
import hashlib
import json
import os
import shutil
from dataclasses import fields
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import torch

from retroagi.core.smb_coaching import COACHING_CONTRACT
from retroagi.core.smb_components import SMBComponentContract, export_bundle
from retroagi.core.smb_learning import (
    block_stage,
    collect_case,
    collect_reactive_case,
    collect_stomp_recovery_case,
    collect_stomp_scroll_case,
    make_model,
    playback,
    rows_to_data,
    save_dataset,
)
from retroagi.core.smb_scene import SCENE_ENCODER
from retroagi.stages.block_smb.demonstrations import fit_demonstrations
from retroagi.stages.block_smb.monte_carlo import BLOCK_SMB_MC_FAMILIES
from retroagi.stages.block_smb.nes_curriculum import sample_nes_case
from scripts.smb_perception_training import block_clips, train_perception

POLICY_STATE_FAMILIES = frozenset(
    "tall_pipe_jump pipe_mount stair_climb platform_chain mixed_section "
    "chained_obstacles full_smb_opening_proxy enemy_stomp stomp_mount".split()
)


class QualificationFailure(RuntimeError):
    """A measured qualification requirement failed, rather than a runtime error."""


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, default=str) + "\n")
    temporary.replace(path)


def samples(config, split, count, *, offset=0, families=None, log=None):
    result = []
    from retroagi.stages.block_smb.nes_curriculum import canonical_families

    for family in canonical_families(families or config["families"]):
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
        if log:
            log(
                dict(
                    phase="source_perception_dataset",
                    split=split,
                    family=family,
                    layouts=len(result),
                )
            )
    return result


def collect(model, cases, vision, *, log):
    rows = []
    episodes = []
    for number, sample in enumerate(cases, 1):
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
            if data:
                alternative, alt_result = collect_reactive_case(
                    model,
                    stage,
                    family=BLOCK_SMB_MC_FAMILIES.index(sample.family),
                    seed=sample.sample_seed % (2**31),
                    takeoff_distance=24 + sample.sample_index % 45,
                    variant=1 + sample.sample_index % 13,
                )
                episodes.append(
                    dict(
                        id=sample.scenario_id,
                        family=sample.family,
                        split=sample.split,
                        route_variant=True,
                        start=len(rows),
                        length=len(alternative),
                        **alt_result,
                    )
                )
                rows.extend(alternative)
                if sample.family in POLICY_STATE_FAMILIES or sample.family in (
                    "bridge_wait",
                    "moving_bridge",
                ):
                    modes = (
                        ("decision", "brake")
                        if sample.family in ("bridge_wait", "moving_bridge")
                        else ("takeoff", "miss")
                    )
                    for mode in modes:
                        variation, variation_result = collect_reactive_case(
                            model,
                            stage,
                            family=BLOCK_SMB_MC_FAMILIES.index(sample.family),
                            seed=sample.sample_seed % (2**31),
                            policy_rollout=mode,
                            policy_takeoff_number=(
                                1 + sample.sample_index % 3
                                if sample.family
                                in (
                                    "chained_obstacles",
                                    "full_smb_opening_proxy",
                                    "mixed_section",
                                    "platform_chain",
                                    "stair_climb",
                                )
                                else 1
                            ),
                        )
                        episodes.append(
                            dict(
                                id=sample.scenario_id,
                                family=sample.family,
                                split=sample.split,
                                route_variant=True,
                                takeoff_variant="actual_" + mode,
                                start=len(rows),
                                length=len(variation),
                                **variation_result,
                            )
                        )
                        rows.extend(variation)
                if sample.family == "enemy_stomp":
                    for takeoff in ("nearby", "policy"):
                        variation, variation_result = collect_reactive_case(
                            model,
                            stage,
                            family=BLOCK_SMB_MC_FAMILIES.index(sample.family),
                            seed=sample.sample_seed % (2**31),
                            takeoff_delay=3 + sample.sample_index % 7 if takeoff == "nearby" else 0,
                            policy_takeoff=takeoff == "policy",
                        )
                        episodes.append(
                            dict(
                                id=sample.scenario_id,
                                family=sample.family,
                                split=sample.split,
                                route_variant=True,
                                takeoff_variant=takeoff,
                                start=len(rows),
                                length=len(variation),
                                **variation_result,
                            )
                        )
                        rows.extend(variation)
                    scrolling, scrolling_result = collect_stomp_scroll_case(
                        model,
                        stage,
                        family=BLOCK_SMB_MC_FAMILIES.index(sample.family),
                        seed=sample.sample_seed % (2**31),
                        offset=80 + 16 * (sample.sample_index % 5),
                    )
                    episodes.append(
                        dict(
                            id=sample.scenario_id,
                            family=sample.family,
                            split=sample.split,
                            route_variant=True,
                            takeoff_variant="policy_scrolling",
                            start=len(rows),
                            length=len(scrolling),
                            **scrolling_result,
                        )
                    )
                    rows.extend(scrolling)
                    recovery, recovery_result = collect_stomp_recovery_case(
                        model,
                        stage,
                        family=BLOCK_SMB_MC_FAMILIES.index(sample.family),
                        seed=sample.sample_seed % (2**31),
                        offset=12 + sample.sample_index % 9,
                        # Moving away behind a one-way camera can make a fast
                        # overshoot physically unrecoverable. Retain momentum
                        # without authoring those impossible reset states.
                        velocity=2.5 if sample.parameters["enemy_speed"] == 0 else 1.25,
                    )
                    episodes.append(
                        dict(
                            id=sample.scenario_id,
                            family=sample.family,
                            split=sample.split,
                            route_variant=True,
                            recovery_variant="stomp_overshoot",
                            start=len(rows),
                            length=len(recovery),
                            **recovery_result,
                        )
                    )
                    rows.extend(recovery)
        finally:
            stage.env.close()
        if number == 1 or number % 10 == 0:
            log(
                dict(
                    phase="demonstrations",
                    completed=number,
                    attempted=len(cases),
                    family=sample.family,
                    frames=len(rows),
                )
            )
    log(
        dict(
            phase="demonstrations",
            accepted=sum(e["success"] for e in episodes if not e.get("route_variant")),
            attempted=len(cases),
            accepted_variants=sum(e["success"] for e in episodes if e.get("route_variant")),
            safe_duration_sets=sum(
                e.get("safe_duration_sets", 0) for e in episodes if e["success"]
            ),
            frames=len(rows),
            policy_diagnostics={
                key: sum(e.get("policy_diagnostics", {}).get(key, 0) for e in episodes)
                for key in (
                    "decisions",
                    "jump_proposals",
                    "unsafe_takeoffs",
                    "unsafe_holds",
                    "missed_brakes",
                    "missed_waits",
                )
            },
        )
    )
    missing = set(s.family for s in cases) - {e["family"] for e in episodes if e["success"]}
    if missing:
        raise QualificationFailure(f"No executor-verified demonstrations for {sorted(missing)}")
    return rows_to_data(rows), episodes


def concatenate_demonstrations(left, right):
    return type(left)(
        **{f.name: torch.cat((getattr(left, f.name), getattr(right, f.name))) for f in fields(left)}
    )


def train_epoch(model, optimizer, config, vision, *, epoch, data=None, episodes=None, log):
    """Collect and learn inside this numbered epoch, retaining earlier replay."""
    total = config["train_layouts_per_family_per_epoch"]
    chunk = config.get("epoch_chunk_layouts_per_family", 3)
    budget = config["rehearsal_updates"]
    if total < 1 or chunk < 1 or budget < 1:
        raise ValueError("Epoch layout counts and update budget must be positive")
    chunks = (total + chunk - 1) // chunk
    episodes = list(episodes or [])
    completed = 0
    weighted_loss = 0.0

    def epoch_log(event):
        log(dict(epoch=epoch, epochs=config["epochs"], **event))

    for number, offset in enumerate(range(0, total, chunk), 1):
        count = min(chunk, total - offset)
        epoch_log(
            dict(
                phase="full_volume_epoch",
                stage="collecting_demonstrations",
                batch=number,
                batches=chunks,
                updates=completed,
                total_updates=budget,
            )
        )
        fresh, fresh_episodes = collect(
            model,
            samples(config, "train", count, offset=20000 + epoch * 1000 + offset),
            vision,
            log=lambda event: epoch_log(
                dict(
                    stage="collecting_demonstrations",
                    batch=number,
                    batches=chunks,
                    updates=completed,
                    total_updates=budget,
                    **event,
                )
            ),
        )
        start = len(data.action) if data is not None else 0
        episodes.extend({**e, "start": e["start"] + start} for e in fresh_episodes)
        data = fresh if data is None else concatenate_demonstrations(data, fresh)
        # Interleave updates without adding any budget outside the epoch.
        updates = number * budget // chunks - completed
        if updates:
            epoch_log(
                dict(
                    phase="full_volume_epoch",
                    stage="training",
                    batch=number,
                    batches=chunks,
                    updates=completed,
                    total_updates=budget,
                )
            )
            loss = fit_demonstrations(
                model,
                optimizer,
                data,
                steps=updates,
                seed=config["seed"] + epoch * 1000 + number,
                decision_durations_only=True,
                walk_durations=False,
                prioritized=True,
                adaptive_groups=True,
            )
            completed += updates
            weighted_loss += loss * updates
            epoch_log(
                dict(
                    phase="full_volume_epoch",
                    stage="training",
                    batch=number,
                    batches=chunks,
                    updates=completed,
                    total_updates=budget,
                    loss=loss,
                    replay_frames=len(data.action),
                    layouts_per_family=offset + count,
                    decision_groups=getattr(model, "last_demonstration_groups", []),
                )
            )
    return data, episodes, weighted_loss / completed


def evaluate(model, cases, vision, *, max_steps=320, log=None):
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
        if log and (len(results) % 10 == 0 or len(results) == len(cases)):
            family_rows = [r for r in results if r["family"] == sample.family]
            log(
                dict(
                    phase="family_validation_progress",
                    stage="validation",
                    completed=len(results),
                    attempted=len(cases),
                    family=sample.family,
                    family_completed=len(family_rows),
                    family_successes=sum(r["success"] for r in family_rows),
                    family_deaths=sum(r["death"] for r in family_rows),
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


def reuse_perception(checkpoint, output, *, device, log):
    """Reuse only qualified vision weights; policy initialization stays fresh."""
    from retroagi.core.smb_perception import DenseSMBPerception

    source = Path(checkpoint).resolve(strict=True)
    payload = torch.load(source, map_location="cpu", weights_only=True)
    metrics = payload.get("metrics", {})
    if not metrics.get("qualified") or not metrics.get("collision_labels"):
        raise QualificationFailure("Reused perception lacks qualified collision-label metrics")
    vision = DenseSMBPerception.load(source, device=device)
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    destination = Path(output) / "block_perception"
    destination.mkdir()
    shutil.copyfile(source, destination / "perception.pth")
    if hashlib.sha256((destination / "perception.pth").read_bytes()).hexdigest() != digest:
        raise ValueError("Reused perception checksum mismatch")
    write_json(destination / "metrics.json", metrics)
    original = source.parent / "provenance.json"
    if original.exists():
        shutil.copyfile(original, destination / "source_provenance.json")
    provenance = dict(
        source=str(source),
        sha256=digest,
        reused=True,
        interface="dense_smb_perception_v1",
        policy_weights_reused=False,
        source_provenance_sha256=(
            hashlib.sha256(original.read_bytes()).hexdigest() if original.exists() else None
        ),
    )
    write_json(destination / "provenance.json", provenance)
    log(dict(phase="source_perception_reused", **provenance))
    return vision, metrics


def run(config, output):
    from retroagi.stages.block_smb.nes_curriculum import canonical_families

    retired = {
        "bootstrap_updates",
        "demonstration_layouts_per_family",
        "demonstration_chunk_layouts_per_family",
    } & config.keys()
    if retired:
        raise ValueError(f"Removed bootstrap settings: {sorted(retired)}")
    config = {**config, "families": canonical_families(config["families"])}
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)

    def log(event):
        event = dict(event, timestamp=datetime.now(timezone.utc).isoformat())
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
            fresh_policy_initialization=True,
            fresh_perception_initialization=not bool(config.get("perception_checkpoint")),
            policy_schedule="numbered_epochs_only",
            family_aliases={"wait_timing": "bridge_wait"},
            independent_families=len(config["families"]),
            init_checkpoint=None,
            fixed_scenes=[],
            runtime="smb_scene_v2",
            scene_encoder=SCENE_ENCODER,
            coaching=COACHING_CONTRACT,
            replay_sampling="adaptive_decision_groups_v1",
            policy_state_families=sorted(POLICY_STATE_FAMILIES),
            objective_contract="observable_traversal_v4",
            full_level_qualified=False,
        ),
    )
    try:
        log(dict(phase="physics_audit"))
        from scripts.smb_physics_audit import audit

        physics = audit()
        write_json(output / "physics.json", physics)
        if not physics["exact_motion_gate"]:
            raise QualificationFailure("NES motion gate failed")
        if config.get("perception_checkpoint"):
            vision, perception = reuse_perception(
                config["perception_checkpoint"], output, device=config["device"], log=log
            )
        else:
            log(dict(phase="source_perception_dataset"))
            train = samples(config, "train", config["perception_layouts_per_family"], log=log)
            validation = samples(
                config, "validation", config["perception_validation_per_family"], log=log
            )
            clips = block_clips(train, output / "block_train_clips")
            validation_clips = block_clips(validation, output / "block_validation_clips")
            log(
                dict(
                    phase="source_perception_training",
                    device=config["device"],
                    updates=config["perception_updates"],
                )
            )
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
            raise QualificationFailure(
                "Block collision-perception gate failed; see per-class and edge metrics"
            )
        log(
            dict(
                phase="full_volume_initialization",
                epochs=config["epochs"],
                families=config["families"],
            )
        )
        # One fresh shared core learns every family; validation does not gate
        # the start of training or reset this model between families.
        torch.manual_seed(config["seed"])
        model = make_model(
            hidden_dim=config["hidden_dim"], device=config["device"], provider="perceived"
        )
        optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"])
        data, episodes, validation = None, [], None
        for epoch in range(1, config["epochs"] + 1):
            data, episodes, loss = train_epoch(
                model,
                optimizer,
                config,
                vision,
                epoch=epoch,
                data=data,
                episodes=episodes,
                log=log,
            )
            save_dataset(
                data, output / "demonstrations.pth", episodes=episodes, provider="perceived"
            )
            log(
                dict(
                    phase="full_volume_epoch",
                    epoch=epoch,
                    epochs=config["epochs"],
                    stage="validation",
                    updates=config["rehearsal_updates"],
                    total_updates=config["rehearsal_updates"],
                )
            )
            if validation is None:
                validation = samples(
                    config, "validation", config["validation_layouts_per_family"], offset=10000
                )
            result = evaluate(
                model,
                validation,
                vision,
                log=lambda event: log(dict(epoch=epoch, epochs=config["epochs"], **event)),
            )
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
            raise QualificationFailure("Shared Block held-out family gate failed")
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
            raise QualificationFailure("Full real-frame collision-perception gate failed")
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
        log(
            dict(
                phase=(
                    "gate_failed" if isinstance(error, QualificationFailure) else "runtime_failed"
                ),
                error=str(error),
                error_type=type(error).__name__,
                full_level_qualified=False,
            )
        )
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=Path("scripts/configs/smb_composable_full_volume.json")
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--perception-checkpoint",
        type=Path,
        help="Reuse qualified vision only; initialize a fresh policy",
    )
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    if args.perception_checkpoint:
        config["perception_checkpoint"] = str(args.perception_checkpoint.resolve())
    if (
        config["epochs"] != 30
        or config.get("init_checkpoint")
        or config.get("resume_path")
        or config.get("fixed_scenes")
    ):
        raise ValueError("This pipeline requires a fresh 30-epoch generated curriculum")
    from retroagi.stages.block_smb.nes_curriculum import canonical_families

    if set(canonical_families(config["families"])) != set(
        canonical_families(BLOCK_SMB_MC_FAMILIES)
    ):
        raise ValueError(
            "All 20 independent families must be present (wait_timing aliases bridge_wait)"
        )
    run(config, args.output_dir)


if __name__ == "__main__":
    main()
