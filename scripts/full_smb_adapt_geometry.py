"""Bounded real-emulator demonstration adaptation of a transferred policy.

The teacher branches emulator snapshots ONLY while collecting training labels.
Evaluation/play never call this module. Each retained jump must land safely or
complete the level. Native Full SMB visual features and missing NES mechanics
are present in training exactly as they are in policy inference.
"""

import argparse
import json
from dataclasses import replace
from pathlib import Path

import torch

from retroagi.core import save_checkpoint
from retroagi.stages.block_smb.demonstrations import DemonstrationBatch, fit_demonstrations
from retroagi.stages.block_smb.local_traversal import local_target_distance, support_edge_distance
from retroagi.stages.full_smb.adapter import FullSMBEnvConfig, FullSMBStage
from retroagi.stages.full_smb.transfer import load_transferred_full_smb_policy


def geometry(stage):
    return stage.smb_geometry.observe(
        stage.env.get_ram(),
        frame=stage._geometry_frame,
        terminated=stage._last_terminal,
        truncated=stage._last_truncated,
    )


def candidate_jumps(stage, target, durations):
    snapshot = stage.save_emulator_state()
    initial = geometry(stage)
    scroll = initial["scroll"]
    left, right, top = target.left + scroll, target.right + scroll, target.top
    candidates = []
    try:
        for index, hold in enumerate(durations):
            stage.load_emulator_state(snapshot)
            air = False
            actions = []
            for frame in range(90):
                action = 2 if frame < hold else 1
                _, _, terminal, truncated, info = stage.step(action)
                actions.append(action)
                g = geometry(stage)
                m = g["scene"].mario
                air |= not m["on_ground"]
                signals = info["full_smb_signals"]
                if signals["death"] or truncated:
                    break
                if signals["completion"]:
                    candidates.append((index, actions))
                    break
                if terminal:
                    break
                if air and m["on_ground"]:
                    x = g["world_x"]
                    feet = m["y"] + m["h"]
                    if x >= right or (
                        target.kind in ("gap", "mount") and x + m["w"] > left and feet <= top + 2
                    ):
                        candidates.append((index, actions))
                    break
    finally:
        stage.load_emulator_state(snapshot)
    return candidates


def collect(stage, *, episodes, limit, durations):
    rows = []
    runs = []
    for episode in range(episodes):
        obs = stage.reset(seed=81000 + episode)
        for _ in range(episode * 3):
            obs, _, _, _, _ = stage.step(1)
        steps = 0
        failed = False
        complete = False
        last_x = -1
        stuck = 0
        jumps = 0
        while steps < limit:
            g = geometry(stage)
            m = g["scene"].mario
            t = g["objective"]
            x = g["world_x"]
            stuck = stuck + 1 if x == last_x else 0
            last_x = x
            if stuck > 120:
                failed = True
                break
            distance = local_target_distance(g["scene"], t)
            near = (t.kind in ("enemy", "mount") and distance < 50 - episode % 3 * 5) or (
                t.kind == "gap" and support_edge_distance(g["scene"], 1) < 42 - episode % 3 * 4
            )
            candidates = candidate_jumps(stage, t, durations) if m["on_ground"] and near else []
            if candidates:
                # Interior successful holds, with deterministic variation.
                selected = candidates[
                    len(candidates) // 2 if episode % 2 == 0 else len(candidates) // 3
                ]
                index, actions = selected
                valid = torch.zeros(16, dtype=torch.bool)
                valid[[i for i, _ in candidates]] = True
                jumps += 1
            else:
                index, actions = 0, [1]
                valid = torch.ones(16, dtype=torch.bool)
            chunk = []
            for j, action in enumerate(actions):
                # Branching restored the observation along with the emulator.
                obs = stage._last_observation
                batch = stage.encode_observation(obs)
                current = geometry(stage)
                obs, _, terminal, truncated, info = stage.step(action)
                next_batch = stage.encode_observation(obs)
                chunk.append(
                    (
                        batch,
                        current["skill_goal"],
                        action,
                        2 if candidates else action,
                        index,
                        j == 0,
                        next_batch.src_c.detach().cpu(),
                        valid.clone(),
                    )
                )
                steps += 1
                signals = info["full_smb_signals"]
                complete = signals["completion"]
                failed = signals["death"] or truncated
                if terminal or failed or complete:
                    break
            # Successful local transitions can be retained even if a later,
            # unrelated section fails. Never label a failed arc as successful.
            if not failed:
                rows.extend(chunk)
            if failed or complete:
                break
        row = dict(
            episode=episode,
            frames=steps,
            completed=bool(complete),
            failed=bool(failed),
            jumps=jumps,
            max_x=geometry(stage)["world_x"],
        )
        runs.append(row)
        print(json.dumps(row), flush=True)
    if not rows:
        raise RuntimeError("No successful emulator training transitions")
    data = DemonstrationBatch(
        a=torch.cat([r[0].src_a.cpu() for r in rows]),
        b=torch.cat([r[0].src_b.cpu() for r in rows]),
        c=torch.cat([r[0].src_c.detach().cpu() for r in rows]),
        goal=torch.cat([r[1].cpu() for r in rows]),
        action=torch.tensor([r[2] for r in rows]),
        motor_action=torch.tensor([r[3] for r in rows]),
        duration=torch.tensor([r[4] for r in rows]),
        actor_mask=torch.tensor([r[5] for r in rows]),
        next_c=torch.cat([r[6] for r in rows]),
        family=torch.zeros(len(rows), dtype=torch.long),
        valid_durations=torch.stack([r[7] for r in rows]),
    )
    return data, runs


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--episodes", type=int, default=4)
    p.add_argument("--steps", type=int, default=1800)
    p.add_argument("--updates", type=int, default=1000)
    p.add_argument("--dataset", type=Path)
    args = p.parse_args()
    if min(args.episodes, args.steps, args.updates) <= 0:
        raise ValueError("Positive collection and adaptation budgets required")
    if (args.output / "policy.pth").exists() or (args.output / "demonstrations.pth").exists():
        raise FileExistsError("Use a fresh adaptation directory")
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    torch.manual_seed(82001)
    policy = load_transferred_full_smb_policy(args.checkpoint, device="cuda")
    model = policy.model
    if args.dataset:
        saved = torch.load(args.dataset, weights_only=False)
        data, runs = saved["data"], saved["runs"]
        from retroagi.core.smb_runtime import SMBRuntimeContract

        recorded = SMBRuntimeContract(**saved["contract"])
        for name in ("schema", "motion_observations", "jump_hold_frames", "geometry_source"):
            if getattr(recorded, name) != getattr(model.smb_runtime_contract, name):
                raise ValueError(f"Cached demonstration contract mismatch: {name}")
    else:
        stage = FullSMBStage(env_config=FullSMBEnvConfig(state="Level1-1"), vision=policy.vision)
        stage.configure_policy_runtime(model.smb_runtime_contract)
        try:
            data, runs = collect(
                stage,
                episodes=args.episodes,
                limit=args.steps,
                durations=model.smb_runtime_contract.jump_hold_frames,
            )
        finally:
            stage.close()
        torch.save(
            {"data": data, "runs": runs, "contract": model.smb_runtime_contract.manifest()},
            args.output / "demonstrations.pth",
        )
    # Adapt the numeric action/duration paths to NES physics, real perception,
    # and unavailable patrol/coyote features; retain the transferred hierarchy.
    for name, param in model.named_parameters():
        param.requires_grad_(
            name.startswith(("agent.action_state_head.", "agent.duration_state_head."))
        )
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=0.0003)
    loss = fit_demonstrations(
        model,
        optimizer,
        data,
        steps=args.updates,
        seed=82001,
        decision_durations_only=True,
        walk_durations=False,
        prioritized=True,
    )
    checkpoint = policy.checkpoint
    checkpoint["config"]["smb_runtime_contract"] = replace(
        model.smb_runtime_contract, visual_tokens="native_adapted"
    ).manifest()
    checkpoint["states"]["model"] = model.state_dict()
    checkpoint.setdefault("metadata", {})["emulator_adaptation"] = {
        "training_level": "Level1-1",
        "training_seeds": [81000 + int(run["episode"]) for run in runs],
        "dataset_source": str(args.dataset) if args.dataset else None,
        "updates": args.updates,
        "frames": len(data.action),
        "loss": loss,
        "runs": runs,
        "full_level_qualified": False,
    }
    save_checkpoint(args.output / "policy.pth", checkpoint)
    (args.output / "training.json").write_text(
        json.dumps(checkpoint["metadata"]["emulator_adaptation"], indent=2) + "\n"
    )
    print(
        json.dumps({"updates": args.updates, "frames": len(data.action), "loss": loss}), flush=True
    )


if __name__ == "__main__":
    main()
