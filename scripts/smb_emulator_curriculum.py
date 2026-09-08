"""Individual NES approaches, nearby physical variations and frozen-component tests.

Teacher searches run only during dataset construction. Evaluation restores an
explicit start and executes only the learned policy. Seeds alone are not treated
as distinct emulator layouts. No actor is adapted by the frozen-core experiment.
"""

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from retroagi.core.smb_learning import adapt_world_model, playback, primitive, rows_to_data, runtime
from retroagi.core.smb_physics import NES_JUMP_FRAMES
from retroagi.core.smb_runtime import attach_runtime, make_smb_executor
from retroagi.core.smb_scene import canonical_rgb
from retroagi.core.smb_supervision import OracleSceneVision, collision_labels
from retroagi.stages.block_smb.local_traversal import local_target_distance, support_edge_distance
from retroagi.stages.full_smb.adapter import FullSMBEnvConfig, FullSMBStage
from scripts.full_smb_adapt_geometry import candidate_jumps, geometry


def make_stage(*, state="Level1-1", vision=None, provider="oracle", device="cpu"):
    stage = FullSMBStage(env_config=FullSMBEnvConfig(state=state), vision=vision or object())
    stage.configure_policy_runtime(runtime(provider))
    if vision is None:
        stage.vision = OracleSceneVision(lambda: geometry(stage)["scene"], device=device)
    stage.reset(seed=0)
    return stage


def capture(
    directory, *, state="Level1-1", approaches=8, max_steps=2400, device="cpu", variations=None
):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    cases = []
    unresolved = []
    seen_states = set()
    duplicates = []
    # Physical waiting changes enemy phase; walk offsets change takeoff speed
    # and distance. The tuple sets are disjoint and persisted in the manifest.
    variations = variations or {
        "train": [(0, 0), (4, 2), (8, 4)],
        "validation": [(2, 1), (6, 3)],
        "test": [(1, 2), (7, 1)],
    }
    stage = make_stage(state=state, device=device)
    try:
        for split, settings in variations.items():
            for variation, (wait, walk) in enumerate(settings):
                stage.reset(seed=0)
                for _ in range(wait):
                    stage.step(0)
                count = 0
                steps = 0
                last_target = -1
                while count < approaches and steps < max_steps:
                    g = geometry(stage)
                    m = g["scene"].mario
                    t = g["objective"]
                    near = (
                        t.kind in ("mount", "enemy") and local_target_distance(g["scene"], t) < 48
                    ) or (t.kind == "gap" and support_edge_distance(g["scene"], 1) < 40)
                    target = t.right + g["scroll"]
                    if m["on_ground"] and near and target > last_target + 8:
                        for _ in range(walk):
                            stage.step(1)
                            steps += 1
                        g = geometry(stage)
                        t = g["objective"]
                        target = t.right + g["scroll"]
                        start = stage.save_emulator_state()
                        ram = stage.env.get_ram()
                        parts = [
                            ram[a:b].tobytes()
                            for a, b in (
                                (0x1D, 0x23),
                                (0x57, 0x5E),
                                (0x6D, 0x74),
                                (0x86, 0x8D),
                                (0x9F, 0xA6),
                                (0xB5, 0xBC),
                                (0xCE, 0xD5),
                                (0x400, 0x439),
                                (0x500, 0x6A0),
                                (0x703, 0x70B),
                                (0x71C, 0x71E),
                            )
                        ]
                        fingerprint = hashlib.sha256(b"".join(parts)).hexdigest()
                        candidates = verify_safe_exits(
                            stage, local_candidates(stage, t, NES_JUMP_FRAMES), target
                        )
                        if not candidates:
                            failed_name = f"{state}_{split}_v{variation}_a{count}_unresolved.pth"
                            torch.save(start, directory / failed_name)
                            unresolved.append(
                                dict(
                                    split=split,
                                    variation=variation,
                                    approach=count,
                                    world_x=g["world_x"],
                                    kind=t.kind,
                                    reason="no_verified_jump",
                                    snapshot=failed_name,
                                )
                            )
                            break
                        index, actions = candidates[len(candidates) // 2]
                        name = f"{state}_{split}_v{variation}_a{count}"
                        torch.save(start, directory / f"{name}.pth")
                        frames = []
                        labels = []
                        for action in actions:
                            frames.append(canonical_rgb(stage._last_observation).copy())
                            labels.append(collision_labels(geometry(stage)["scene"]))
                            stage.step(action)
                            steps += 1
                        np.savez_compressed(
                            directory / f"{name}.npz",
                            images=np.stack(frames),
                            labels=np.stack(labels),
                        )
                        duplicate = fingerprint in seen_states
                        seen_states.add(fingerprint)
                        retained = duplicates if duplicate else cases
                        retained.append(
                            dict(
                                id=name,
                                physical_fingerprint=fingerprint,
                                approach=count,
                                state=state,
                                split=split,
                                variation=dict(wait=wait, walk=walk),
                                snapshot=f"{name}.pth",
                                pixels=f"{name}.npz",
                                frames=len(actions),
                                actions=actions,
                                duration_index=index,
                                safe_durations=[i for i, _ in candidates],
                                target=target,
                                kind=t.kind,
                                start_x=g["world_x"],
                                observation_contract=runtime().manifest(),
                                label_source="nes_collision_instrumentation",
                                teacher="snapshot_search",
                                full_level_qualified=False,
                            )
                        )
                        last_target = target
                        count += 1
                    else:
                        _, _, done, truncated, info = stage.step(1)
                        steps += 1
                        if done or truncated or info["full_smb_signals"]["death"]:
                            break
    finally:
        stage.close()
    report = dict(
        cases=cases,
        unresolved=unresolved,
        duplicates=duplicates,
        variations=variations,
        full_level_qualified=False,
    )
    (directory / "approaches.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


@torch.no_grad()
def collect_local(model, stage, start, case, *, family=0):
    """One verified local arc, with actual one-frame dynamics targets."""
    obs = stage.load_emulator_state(start)
    stage._scene_start_frame = stage._geometry_frame
    executor = make_smb_executor(model)
    rows = []
    dead = False
    for frame, intent in enumerate(case["actions"]):
        batch = stage.encode_observation(obs)
        committed = executor.committed_action
        execution = executor.execute(
            intent, batch=batch, motor_primitives=primitive(case["duration_index"])
        )
        obs, _, done, truncated, info = stage.step(execution.action)
        following = stage.encode_observation(obs)
        valid = [i in case["safe_durations"] for i in range(16)]
        rows.append(
            (
                batch.src_a.cpu(),
                batch.src_b.cpu(),
                batch.src_c.cpu(),
                batch.metadata["smb_geometry"]["skill_goal"].cpu(),
                intent,
                int(committed) if committed is not None else intent,
                case["duration_index"],
                committed is None,
                following.src_c.cpu(),
                family,
                valid,
            )
        )
        dead = info["full_smb_signals"]["death"] or truncated
        if done or dead:
            break
    actual = geometry(stage)
    safe = not dead and actual["scene"].mario["on_ground"] and actual["world_x"] >= case["target"]
    return rows if safe else []


def evaluate_matrix(
    model,
    directory,
    full_vision,
    *,
    adaptation_steps=500,
    device="cuda",
    block_replay=None,
    retention_evaluator=None,
):
    directory = Path(directory)
    manifest = json.loads((directory / "approaches.json").read_text())
    cases = manifest["cases"]
    results = {}
    original = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    original_runtime = model.smb_runtime_contract.manifest()
    try:
        for lane in ("oracle_frozen", "perceived_frozen", "perceived_world_model"):
            model.load_state_dict(original)
            provider = "oracle" if lane == "oracle_frozen" else "perceived"
            attach_runtime(
                model,
                replace(
                    runtime(provider), recurrent_state=original_runtime["recurrent_state"]
                ).manifest(),
            )
            stage = make_stage(
                vision=full_vision if provider == "perceived" else None,
                provider=provider,
                device=device,
            )
            adaptation = None
            try:
                if lane == "perceived_world_model":
                    rows = []
                    episodes = []
                    for case in cases:
                        if case["split"] != "train":
                            continue
                        start = torch.load(directory / case["snapshot"], weights_only=False)
                        # Perception history in a snapshot belongs to the source
                        # provider. The target starts from reset history in both
                        # collection and evaluation.
                        start = replace(start, perceived_state=None, perceived_cache=None)
                        local = collect_local(model, stage, start, case)
                        episodes.append(dict(start=len(rows), length=len(local)))
                        rows.extend(local)
                    if rows:
                        data = rows_to_data(rows)
                        if block_replay is not None:
                            replay, replay_episodes = block_replay
                            from dataclasses import fields

                            offset = len(data.action)
                            data = type(data)(
                                **{
                                    f.name: torch.cat(
                                        (getattr(data, f.name), getattr(replay, f.name))
                                    )
                                    for f in fields(data)
                                }
                            )
                            episodes.extend(
                                {**e, "start": e["start"] + offset} for e in replay_episodes
                            )
                        if model.smb_runtime_contract.recurrent_state:
                            from retroagi.core.smb_learning import fit_ordered_sequences

                            adaptation = fit_ordered_sequences(
                                model, data, episodes, world_model_only=True
                            )
                        else:
                            adaptation = adapt_world_model(model, data, steps=adaptation_steps)
                evaluations = []
                for case in cases:
                    if case["split"] == "train":
                        continue
                    start = torch.load(directory / case["snapshot"], weights_only=False)
                    start = replace(start, perceived_state=None, perceived_cache=None)
                    result = playback(
                        model, stage, start=start, target=case["target"], max_steps=120
                    )
                    evaluations.append(
                        dict(
                            id=case["id"],
                            approach=case["approach"],
                            split=case["split"],
                            kind=case["kind"],
                            **result,
                        )
                    )
                results[lane] = dict(adaptation=adaptation, evaluations=evaluations)
                if lane == "perceived_world_model" and adaptation is not None:
                    from retroagi.core.smb_components import SMBComponentContract, export_bundle

                    results[lane]["block_retention"] = (
                        retention_evaluator(model) if retention_evaluator else None
                    )
                    export_bundle(
                        model,
                        directory / "adapted_components",
                        contract=SMBComponentContract(
                            recurrent_state=model.smb_runtime_contract.recurrent_state
                        ),
                        architecture={
                            "hidden_dim": model.agent.d_model,
                            "controller_schedule": "constant",
                        },
                    )
            finally:
                stage.close()
    finally:
        model.load_state_dict(original)
        model.requires_grad_(True)
        attach_runtime(model, original_runtime)
    results["full_level_qualified"] = False
    results["world_model_feedback_enabled"] = original_runtime["recurrent_state"]
    results["next_gate"] = (
        "Enable sequence-trained recurrent context or refinement in BOTH stages and demonstrate held-out improvement before claiming LSTM-only transfer."
    )
    (directory / "transfer_matrix.json").write_text(json.dumps(results, indent=2) + "\n")
    return results


def local_candidates(stage, target, durations):
    """Try direct jumps, then a verified short retreat from difficult starts."""
    candidates = candidate_jumps(stage, target, durations)
    if candidates:
        return candidates
    start = stage.save_emulator_state()
    try:
        for retreat in (4, 8, 12):
            stage.load_emulator_state(start)
            dead = False
            for _ in range(retreat):
                _, _, done, truncated, info = stage.step(3)
                if done or truncated or info["full_smb_signals"]["death"]:
                    dead = True
                    break
            if dead:
                continue
            current = geometry(stage)
            old = geometry_from_state_target(target, start, stage, current)
            candidates = candidate_jumps(stage, old, durations)
            if candidates:
                return [(index, [3] * retreat + actions) for index, actions in candidates]
        return []
    finally:
        stage.load_emulator_state(start)


def geometry_from_state_target(target, start, stage, current):
    """Preserve target world coordinates across a candidate camera movement."""
    from retroagi.stages.block_smb.local_traversal import LocalObjective

    saved = stage.save_emulator_state()
    try:
        stage.load_emulator_state(start)
        scroll = geometry(stage)["scroll"]
    finally:
        stage.load_emulator_state(saved)
    return LocalObjective(
        target.kind,
        target.left + scroll - current["scroll"],
        target.right + scroll - current["scroll"],
        target.top,
    )


def qualify_individual_approaches(
    directory,
    vision,
    *,
    updates=1000,
    device="cuda",
    hidden_dim=128,
    min_evaluations=100,
    success_gate=0.99,
):
    """Scratch controls establish learnability; they never overwrite transferred actors."""
    from retroagi.core.smb_learning import make_model
    from retroagi.stages.block_smb.demonstrations import fit_demonstrations

    directory = Path(directory)
    cases = json.loads((directory / "approaches.json").read_text())["cases"]
    reports = []
    for provider in ("oracle", "perceived"):
        for approach in sorted({c["approach"] for c in cases}):
            torch.manual_seed(19200 + approach)
            model = make_model(hidden_dim=hidden_dim, device=device, provider=provider)
            stage = make_stage(
                vision=vision if provider == "perceived" else None, provider=provider, device=device
            )
            rows = []
            accepted = []
            try:
                for case in cases:
                    if case["approach"] != approach or case["split"] != "train":
                        continue
                    start = replace(
                        torch.load(directory / case["snapshot"], weights_only=False),
                        perceived_state=None,
                        perceived_cache=None,
                    )
                    local = collect_local(model, stage, start, case)
                    if local:
                        rows.extend(local)
                        accepted.append(case["id"])
                result = dict(
                    provider=provider,
                    approach=approach,
                    train_starts=accepted,
                    evaluations=[],
                    passed=False,
                )
                if rows:
                    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
                    result["loss"] = fit_demonstrations(
                        model,
                        optimizer,
                        rows_to_data(rows),
                        steps=updates,
                        decision_durations_only=True,
                        walk_durations=False,
                        prioritized=True,
                    )
                    corrections = []
                    for case in cases:
                        if case["approach"] != approach or case["split"] != "train":
                            continue
                        start = replace(
                            torch.load(directory / case["snapshot"], weights_only=False),
                            perceived_state=None,
                            perceived_cache=None,
                        )
                        corrections.extend(policy_visited_corrections(model, stage, start, case))
                    result["policy_visited_corrections"] = len(corrections)
                    if corrections:
                        fit_demonstrations(
                            model,
                            optimizer,
                            rows_to_data(rows + corrections),
                            steps=max(1, updates // 2),
                            decision_durations_only=True,
                            walk_durations=False,
                            prioritized=True,
                        )
                    for case in cases:
                        if case["approach"] != approach or case["split"] == "train":
                            continue
                        start = replace(
                            torch.load(directory / case["snapshot"], weights_only=False),
                            perceived_state=None,
                            perceived_cache=None,
                        )
                        outcome = playback(
                            model, stage, start=start, target=case["target"], max_steps=120
                        )
                        result["evaluations"].append(
                            dict(id=case["id"], split=case["split"], **outcome)
                        )
                    result["success_rate"] = sum(r["success"] for r in result["evaluations"]) / max(
                        1, len(result["evaluations"])
                    )
                    result["passed"] = (
                        len(result["evaluations"]) >= min_evaluations
                        and result["success_rate"] >= success_gate
                    )
                reports.append(result)
                (directory / "individual_learning.json").write_text(
                    json.dumps(reports, indent=2) + "\n"
                )
            finally:
                stage.close()
    return reports


def full_level_tests(model, vision, *, states=("Level1-1", "Level1-2", "Level2-1"), max_steps=5000):
    """Independent whole-level completion evidence; local success is insufficient."""
    result = []
    for state in states:
        stage = make_stage(
            state=state, vision=vision, provider="perceived", device=next(model.parameters()).device
        )
        try:
            for wait in (0, 5, 11):
                stage.reset(seed=0)
                for _ in range(wait):
                    stage.step(0)
                outcome = playback(
                    model, stage, start=stage.save_emulator_state(), max_steps=max_steps
                )
                result.append(dict(state=state, initial_wait=wait, **outcome))
        finally:
            stage.close()
    return dict(trials=result, full_level_qualified=all(r["success"] for r in result))


@torch.no_grad()
def policy_visited_corrections(model, stage, start, case, *, max_steps=120):
    """Training-only DAgger queries on states reached by the current policy."""
    from retroagi.stages.full_smb.train import _policy_action_logits_and_state

    obs = stage.load_emulator_state(start)
    executor = make_smb_executor(model)
    memory = None
    rows = []
    last_query = -100
    for frame in range(max_steps):
        batch = stage.encode_observation(obs)
        g = geometry(stage)
        if g["scene"].mario["on_ground"] and frame - last_query >= 12:
            snapshot = stage.save_emulator_state()
            # Search and label with a separate executor, then restore all the
            # policy-owned state before continuing the unassisted rollout.
            actions = verify_safe_exits(
                stage, local_candidates(stage, g["objective"], NES_JUMP_FRAMES), case["target"]
            )
            if actions:
                index, sequence = actions[len(actions) // 2]
                query = {
                    **case,
                    "actions": sequence,
                    "duration_index": index,
                    "safe_durations": [i for i, _ in actions],
                    "start_x": g["world_x"],
                }
                rows.extend(collect_local(model, stage, snapshot, query))
            stage.load_emulator_state(snapshot)
            model.smb_executor = executor
            last_query = frame
        forward = _policy_action_logits_and_state(
            model, batch, device=next(model.parameters()).device, world_model_state=memory
        )
        memory = (
            forward.next_world_model_state if model.smb_runtime_contract.recurrent_state else None
        )
        action = executor.execute(
            int(forward.logits.argmax(-1)), batch=batch, motor_primitives=forward.motor_primitives
        ).action
        obs, _, done, truncated, info = stage.step(action)
        if done or truncated or info["full_smb_signals"]["death"]:
            break
    return rows


def verify_safe_exits(stage, candidates, target_world_x, *, continuation_budget=64):
    """A local teacher must retain safe continuation beyond the obstacle."""
    start = stage.save_emulator_state()
    verified = []
    try:
        for index, original in candidates:
            stage.load_emulator_state(start)
            actions = []
            stable = 0
            for frame in range(len(original) + continuation_budget):
                action = original[frame] if frame < len(original) else 1
                _, _, done, truncated, info = stage.step(action)
                actions.append(action)
                actual = geometry(stage)
                if info["full_smb_signals"]["death"] or truncated:
                    break
                if info["full_smb_signals"]["completion"]:
                    verified.append((index, actions))
                    break
                stable = (
                    stable + 1
                    if actual["scene"].mario["on_ground"] and actual["world_x"] >= target_world_x
                    else 0
                )
                if stable >= 3:
                    verified.append((index, actions))
                    break
                if done:
                    break
        return verified
    finally:
        stage.load_emulator_state(start)
