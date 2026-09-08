"""Frozen emulator transfer evaluation, with reproducible input/action traces.

A Block SMB source gate is an eligibility check, never a full-level success
claim. These diagnostics do not update weights or use future emulator states.
"""

import argparse
import hashlib
import json
from dataclasses import replace
from pathlib import Path

import torch

from retroagi.core.smb_runtime import make_smb_executor
from retroagi.stages.full_smb.adapter import FullSMBEnvConfig, FullSMBStage
from retroagi.stages.full_smb.train import _policy_action_logits_and_state
from retroagi.stages.full_smb.transfer import load_transferred_full_smb_policy


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--steps", type=int, default=2400)
    p.add_argument("--episodes", type=int, default=3)
    p.add_argument(
        "--initial-walk-offset",
        type=int,
        default=0,
        help="Held-out start perturbation in real frames; recorded explicitly",
    )
    p.add_argument("--state", default="Level1-1")
    p.add_argument("--save-state", type=Path)
    p.add_argument("--zero-visual-tokens", action="store_true")
    p.add_argument("--device", default="cuda")
    args = p.parse_args()
    torch.set_num_threads(1)
    if args.steps <= 0 or args.episodes <= 0 or args.initial_walk_offset < 0:
        raise ValueError("Positive evaluation budgets and nonnegative start offset required")
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / "summary.json").exists():
        raise FileExistsError("Use a fresh evaluation directory")
    policy = load_transferred_full_smb_policy(args.checkpoint, device=args.device)
    model = policy.model
    if not hasattr(model, "smb_runtime_contract"):
        raise ValueError("Probe requires a checkpoint with an explicit shared runtime contract")
    if args.zero_visual_tokens:
        model.smb_runtime_contract = replace(
            model.smb_runtime_contract, visual_tokens="zero_ablation"
        )
    start = None
    if args.save_state:
        from retroagi.stages.full_smb.save_states import load_full_smb_save_state_payload

        start = load_full_smb_save_state_payload(args.save_state)["state"]
    s = FullSMBStage(
        env_config=FullSMBEnvConfig(state=args.state),
        vision=policy.vision,
        start_emulator_state=start,
    )
    s.configure_policy_runtime(model.smb_runtime_contract)
    summaries = []
    try:
        for episode in range(args.episodes):
            obs = s.reset(seed=71000 + episode)
            # Explicit initial waits diversify enemy timing; seeds alone do not
            # create independent worlds in this deterministic NES integration.
            for _ in range(episode * 7):
                obs, _, _, _, _ = s.step(0)
            for _ in range(args.initial_walk_offset + episode * 5):
                obs, _, terminal, truncated, _ = s.step(1)
                if terminal or truncated:
                    raise RuntimeError("Evaluation start perturbation ended the episode")
            executor = make_smb_executor(model)
            memory = None
            total_return = 0.0
            max_x = 0
            last_x = None
            stalled = 0
            completed = False
            dead = False
            unavailable = set()
            unsupported = set()
            counts = {}
            initial_x = None
            with (args.output / f"episode{episode}.jsonl").open("w") as trace:
                with torch.no_grad():
                    for frame in range(args.steps):
                        batch = s.encode_observation(obs)
                        g = batch.metadata["smb_geometry"]
                        if initial_x is None:
                            initial_x = g["world_x"]
                        unavailable.update(g["unavailable_features"])
                        unsupported.update(g["unsupported_objects"])
                        f = _policy_action_logits_and_state(
                            model, batch, device=torch.device(args.device), world_model_state=memory
                        )
                        memory = (
                            f.next_world_model_state
                            if model.smb_runtime_contract.recurrent_state
                            else None
                        )
                        intent = int(f.logits.argmax(-1))
                        execution = executor.execute(
                            intent, batch=batch, motor_primitives=f.motor_primitives
                        )
                        obs, reward, terminal, truncated, info = s.step(execution.action)
                        total_return += float(reward)
                        x = g["world_x"]
                        max_x = max(max_x, x)
                        stalled = stalled + 1 if last_x == x else 0
                        last_x = x
                        signal = info.get("full_smb_signals", {})
                        dead = bool(signal.get("death"))
                        completed = bool(signal.get("completion"))
                        counts[str(execution.action)] = counts.get(str(execution.action), 0) + 1
                        trace.write(
                            json.dumps(
                                dict(
                                    frame=frame,
                                    x=x,
                                    box=g["player_box"],
                                    scroll=g["scroll"],
                                    objective=g["objective"].kind,
                                    target=[
                                        g["objective"].left,
                                        g["objective"].right,
                                        g["objective"].top,
                                    ],
                                    src_a=batch.src_a.tolist(),
                                    src_b=batch.src_b.tolist(),
                                    src_c=batch.src_c.tolist(),
                                    skill_goal=g["skill_goal"].tolist(),
                                    intent=intent,
                                    action=execution.action,
                                    hold_frames=execution.hold_frames,
                                    reward=float(reward),
                                    support=g["support"],
                                    unavailable=g["unavailable_features"],
                                    death=dead,
                                    score=signal.get("score") or 0,
                                    coins=signal.get("coins") or 0,
                                    episode_return=total_return,
                                    completion=completed,
                                )
                            )
                            + "\n"
                        )
                        if terminal or truncated or dead or completed or stalled >= 180:
                            break
            row = dict(
                episode=episode,
                frames=frame + 1,
                max_x=max_x,
                progress=max_x - initial_x,
                completed=completed,
                death=dead,
                score=signal.get("score") or 0,
                coins=signal.get("coins") or 0,
                episode_return=total_return,
                terminated=terminal,
                truncated=truncated,
                stalled=stalled >= 180,
                actions=counts,
                unavailable_features=sorted(unavailable),
                unsupported_objects=sorted(unsupported),
            )
            summaries.append(row)
            print(json.dumps(row), flush=True)
    finally:
        s.close()
    from retroagi.stages.full_smb.success import evaluate_full_smb_success_threshold

    tasks = {
        "Level1-1": "benchmark_1_1_start",
        "Level1-2": "benchmark_1_2_start",
        "Level2-1": "benchmark_2_1_start",
    }
    threshold = None
    if args.save_state is None and args.state in tasks:
        threshold = evaluate_full_smb_success_threshold(
            tasks[args.state],
            {
                "max_progress": max(e["max_x"] for e in summaries),
                "completion_rate": sum(e["completed"] for e in summaries) / len(summaries),
                "survival_rate": sum(not e["death"] for e in summaries) / len(summaries),
                "mean_score": sum(e["score"] for e in summaries) / len(summaries),
                "mean_coins": sum(e["coins"] for e in summaries) / len(summaries),
                "death_count": sum(e["death"] for e in summaries),
                "mean_return": sum(e["episode_return"] for e in summaries) / len(summaries),
            },
            evaluation_episodes=args.episodes,
            evaluation_max_steps=args.steps,
        )
    result = dict(
        checkpoint=str(args.checkpoint),
        checkpoint_sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        contract=model.smb_runtime_contract.manifest(),
        source_hashes={
            str(p): hashlib.sha256(p.read_bytes()).hexdigest()
            for root in (Path("retroagi"), Path("scripts"))
            for p in root.glob("**/*.py")
        },
        arguments={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        episodes=summaries,
        completion_rate=sum(e["completed"] for e in summaries) / len(summaries),
        threshold=threshold,
        full_level_qualified=bool(threshold and threshold["threshold_met"]),
    )
    (args.output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
