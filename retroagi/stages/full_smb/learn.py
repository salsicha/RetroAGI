"""Full SMB learning (user, 2026-10-17): the agent plays Full SMB as deployed,
every network engaged, and every network goes on learning from its
playthroughs.

- Played in both strategy modes, speed run and max points, each round the
  same number of playthroughs of each (practice.play_recorded: the vision
  network reads the screen, the tactic and skill layers choose, the world
  model carries out the skill's actions, the collision trackers end waits
  and aim stomps). The layers sample their choices.
- What each playthrough earns follows the mode it is played in (practice:
  progress, time, points, death, the goal): the skill and the tactic layers
  learn from those rewards (layered_train's reward learning, as Block SMB's
  reward rounds: a clipped policy-gradient objective, a critic, an entropy
  bonus; the tactic layer's choices change only once its critic predicts
  the returns well enough).
- The world model and the collision trackers learn from the game's truth
  recorded in every playthrough (layered_train.Predictions, continuing the
  trackers trained in Block SMB); the vision network from frames labelled
  from game memory (pixel_labels), as it was first trained.
- Levels: 1-1 first, played in both modes until each mode has finished it,
  then the next level, its start captured when a playthrough finished the
  level before (user, 2026-10-17). Nothing is held out.

Measured per round and mode: playthroughs, finishes, how far Mario got, the
frames to each 500-pixel checkpoint (speed run) and the points between them
(max points); per level and mode, the playthroughs until the first finish.

Example:
    python -m retroagi.stages.full_smb.learn \\
        --source artifacts/block_smb/tactic_20261017/passed.pt \\
        --tracker artifacts/block_smb/tactic_20261017/tracker.pt \\
        --output artifacts/full_smb/learn_20261018
"""

import argparse
import dataclasses
import json
import multiprocessing
import random
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import torch

from .practice import MODES, play_recorded
from .vision import DEFAULT_FULL_VIT_CHECKPOINT


@dataclass
class FullLearnConfig:
    source: str = ""  # the agent: the tactic run's checkpoint (passed.pt)
    tracker: str = ""  # its collision trackers (tracker.pt beside it)
    vision_checkpoint: str = str(DEFAULT_FULL_VIT_CHECKPOINT)
    output: str = "artifacts/full_smb/learn"
    level: str = "Level1-1"  # where play starts
    playthroughs_per_mode: int = 8  # each round
    rounds: int = 500
    frames: int = 6000  # a playthrough's limit
    workers: int = 8
    label_every: int = 8  # vision labels: every this many frames
    epochs: int = 4  # passes over a round's playthroughs by the reward learning
    vision_learning_rate: float = 3e-5
    vision_batch: int = 16
    device: str = "cuda"
    seed: int = 0


# ── Workers: one emulator a process ──────────────────────────────────────────

_WORKER: dict = {}


def _initialize(settings: dict, vision_checkpoint: str, tracker_width: int, device: str) -> None:
    from retroagi.core.layered_policy import LayeredSMBPolicy, PolicySettings
    from retroagi.core.smb_observer import VisionObserver
    from retroagi.core.target_tracker import CollisionTracker

    from .vision import load_full_vit_checkpoint

    torch.set_num_threads(1)
    vision = load_full_vit_checkpoint(vision_checkpoint, device=device).model.eval()
    _WORKER.update(
        device=device,
        vision=vision,
        observer=VisionObserver(vision),
        policy=LayeredSMBPolicy(PolicySettings(**settings)).to(device).eval(),
        tracker=CollisionTracker(tracker_width).to(device).eval(),
        version=-1,
    )


def _play(job) -> object:
    """One playthrough with the published weights (practice.play_recorded)."""
    from retroagi.core.smb_agent import SMBAgents

    version, weights, level, start_state, mode, index, frames, label_every, capture = job
    if _WORKER["version"] != version:
        published = torch.load(weights, map_location=_WORKER["device"], weights_only=False)
        _WORKER["policy"].load_state_dict(published["policy"])
        _WORKER["tracker"].load_state_dict(published["tracker"])
        _WORKER["vision"].load_state_dict(published["vision"])
        _WORKER["version"] = version
    agents = SMBAgents(
        _WORKER["observer"],
        _WORKER["policy"],
        _WORKER["device"],
        1,
        tracker=_WORKER["tracker"],
        world_model=True,
        watch_threats=True,
    )
    with torch.no_grad():
        return play_recorded(
            agents,
            level,
            mode=mode,
            frames=frames,
            index=index,
            start_state=start_state,
            explore=True,
            label_every=label_every,
            capture_next=capture,
        )


# ── Learning ─────────────────────────────────────────────────────────────────


class VisionLearner:
    """The vision network going on learning from frames labelled from game
    memory (core.scene_vision's losses, rare pixel types weighted up)."""

    def __init__(self, model, config: FullLearnConfig, device):
        self.model, self.config, self.device = model, config, device
        self.optimizer = torch.optim.AdamW(
            model.parameters(), lr=config.vision_learning_rate, weight_decay=0.05
        )

    def learn(self, labelled: Sequence) -> Optional[float]:
        from retroagi.core.scene_vision import (
            frame_targets,
            pixel_type_weights,
            scene_losses,
            screens_to_device,
            stack_targets,
        )

        if not labelled:
            return None
        frames = list(labelled)
        random.shuffle(frames)
        targets = [frame_targets(f.scene) for f in frames]
        weights = pixel_type_weights(np.stack([t["types"].numpy() for t in targets])).to(self.device)
        self.model.train()
        losses = []
        size = self.config.vision_batch
        for start in range(0, len(frames), size):
            images = screens_to_device(
                np.stack([f.image for f in frames[start : start + size]]), self.device
            )
            batch = {
                k: v.to(self.device) for k, v in stack_targets(targets[start : start + size]).items()
            }
            out = self.model.heads(images)
            loss = scene_losses(out, batch, weights)["total"]
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            losses.append(float(loss))
        self.model.eval()
        return float(np.mean(losses))


def _reward_learning(policy, records, learner: str, config, optimizer, device, epochs, critic_ready):
    """A layer's reward learning on a round's playthroughs (layered_train's
    learner_losses / tactic_losses, as Block SMB's reward rounds): its mean
    losses and statistics."""
    from retroagi.stages.block_smb import layered_train as lt

    rng = random.Random(len(records))
    totals: dict = {}
    parameters = [p for group in optimizer.param_groups for p in group["params"]]
    policy.train()
    for _ in range(epochs):
        for batch in lt._batches(records, config.batch_frames, rng):
            if learner == "tactic":
                losses, stats = lt.tactic_losses(
                    policy, batch, config, device, by_reward=True, critic_ready=critic_ready
                )
            else:
                losses, stats = lt.learner_losses(policy, "skill", batch, 1.0, device, rl=config)
            if not losses:
                continue
            loss = sum(losses.values())
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, 1.0)
            optimizer.step()
            for name, value in {**losses, **stats}.items():
                totals.setdefault(name, []).append(float(value))
    policy.eval()
    return {name: float(np.mean(values)) for name, values in totals.items()}


def _mode_summary(played: list) -> dict:
    finished = [p for p in played if p["finished"]]
    checkpoints = [p["checkpoint_frames"] for p in played]
    deepest = max((len(c) for c in checkpoints), default=0)
    return {
        "playthroughs": len(played),
        "finished": len(finished),
        "success": len(finished) / len(played) if played else 0.0,
        "furthest_x": float(np.mean([p["furthest_x"] for p in played])) if played else 0.0,
        "best_furthest_x": max((p["furthest_x"] for p in played), default=0),
        "fewest_frames_to_finish": min((p["frames"] for p in finished), default=None),
        "points": float(np.mean([p["points"] for p in played])) if played else 0.0,
        "most_points_finished": max((p["points"] for p in finished), default=None),
        # Per checkpoint, over the playthroughs that reached it: the mean
        # frames to reach it and how many reached it.
        "checkpoint_frames": [
            float(np.mean([c[k] for c in checkpoints if len(c) > k])) for k in range(deepest)
        ],
        "checkpoint_reached": [sum(len(c) > k for c in checkpoints) for k in range(deepest)],
        "ends": {e: sum(p["end"] == e for p in played) for e in sorted({p["end"] for p in played})},
    }


def learn(config: FullLearnConfig) -> dict:
    """Play and learn (module notes) for ``config.rounds`` rounds."""
    from retroagi.stages.block_smb import layered_train as lt
    from retroagi.stages.block_smb.tracker_train import load_tracker
    from retroagi.core.scene_vision import save_scene_vision_checkpoint

    from .vision import load_full_vit_checkpoint

    random.seed(config.seed)
    torch.manual_seed(config.seed)
    device = torch.device(config.device)
    output = Path(config.output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "config.json").write_text(json.dumps(asdict(config), indent=2) + "\n")
    policy, saved = lt.load_layered_checkpoint(config.source, device)
    policy.eval()
    tracker = load_tracker(config.tracker, device)
    vision = load_full_vit_checkpoint(config.vision_checkpoint, device=device, freeze=False).model
    vision.eval()
    settings = lt.saved_settings(saved["config"])
    # The reward learning's settings (layered_train's, for each layer).
    skill_rl = lt.LayeredTrainConfig(**{**settings, "learner": "skill", "device": config.device})
    tactic_rl = lt.LayeredTrainConfig(**{**settings, "learner": "tactic", "device": config.device})
    # Each layer's own weights learn from rewards (the scene encoder they
    # share stays as Block SMB left it); the world model learns from the
    # recorded truth.
    for parameter in policy.parameters():
        parameter.requires_grad_(False)
    for module in (policy.skill, policy.tactic):
        for parameter in module.parameters():
            parameter.requires_grad_(True)
    skill_optimizer = torch.optim.AdamW(
        policy.skill.parameters(), lr=skill_rl.reward_learning_rate, weight_decay=skill_rl.weight_decay
    )
    tactic_optimizer = torch.optim.AdamW(
        policy.tactic.parameters(), lr=tactic_rl.reward_learning_rate, weight_decay=tactic_rl.weight_decay
    )
    predictions = lt.Predictions(policy, skill_rl, device, output, tracker=tracker)
    seeing = VisionLearner(vision, config, device)
    world_model = saved.get("world_model")
    context = multiprocessing.get_context("spawn")
    pool = ProcessPoolExecutor(
        config.workers,
        mp_context=context,
        initializer=_initialize,
        initargs=(asdict(policy.settings), config.vision_checkpoint, tracker.width, config.device),
    )
    level, start_state = config.level, None
    progress: dict = {}  # level -> mode -> {"played", "first_finish"}
    next_level = None  # (name, emulator state) captured from a finish of this level
    history: list = []
    weights = output / "published.pt"
    critic_ready = False
    try:
        for round_index in range(config.rounds):
            started = time.time()
            torch.save(
                {
                    "policy": policy.state_dict(),
                    "tracker": predictions.tracker.state_dict(),
                    "vision": vision.state_dict(),
                },
                weights,
            )
            counts = progress.setdefault(level, {m: {"played": 0, "first_finish": None} for m in MODES})
            jobs = [
                (
                    round_index,
                    str(weights),
                    level,
                    start_state,
                    mode,
                    round_index * 1000 + MODES.index(mode) * 100 + i,
                    config.frames,
                    config.label_every,
                    next_level is None,
                )
                for mode in MODES
                for i in range(config.playthroughs_per_mode)
            ]
            played = list(pool.map(_play, jobs))
            play_time = time.time() - started
            for job, p in zip(jobs, played):
                mode = job[4]
                counts[mode]["played"] += 1
                if p.summary["finished"] and counts[mode]["first_finish"] is None:
                    counts[mode]["first_finish"] = counts[mode]["played"]
                if p.next_state is not None and next_level is None:
                    next_level = p.next_state
            started = time.time()
            skill_records = [p.skill for p in played if len(p.skill.decision_frames)]
            tactic_records = [p.tactic for p in played if p.tactic is not None]
            if tactic_records:
                explained = lt.critic_check(policy, tactic_records, tactic_rl, device)
                critic_ready = explained is not None and explained >= tactic_rl.critic_ready
            else:
                explained = None
            losses = {
                "skill": _reward_learning(
                    policy, skill_records, "skill", skill_rl, skill_optimizer, device, config.epochs, False
                ),
                "tactic": _reward_learning(
                    policy, tactic_records, "tactic", tactic_rl, tactic_optimizer, device, config.epochs, critic_ready
                ),
                "predictions": predictions.train(skill_records),
                "vision": seeing.learn([f for p in played for f in p.labelled]),
            }
            learn_time = time.time() - started
            entry = {
                "round": round_index,
                "level": level,
                "modes": {
                    mode: _mode_summary([p.summary for job, p in zip(jobs, played) if job[4] == mode])
                    for mode in MODES
                },
                "playthroughs_until_first_finish": {
                    name: {m: c["first_finish"] for m, c in modes.items()} for name, modes in progress.items()
                },
                "critic_explained_variance": explained,
                "critic_ready": critic_ready,
                "losses": losses,
                "seconds": {"play": round(play_time, 1), "learn": round(learn_time, 1)},
            }
            history.append(entry)
            print(_round_line(entry), flush=True)
            summary = {"rounds": history}
            lt.save_layered_checkpoint(
                output / "policy.pt",
                policy,
                lt.LayeredTrainConfig(**{**settings, "output": str(output)}),
                list(saved.get("trained_layers", ())),
                summary,
                world_model,
            )
            save_scene_vision_checkpoint(
                output / "vision.pth",
                vision,
                stage="full_smb_learning",
                metrics={"loss": losses["vision"]},
                trainer="retroagi.stages.full_smb.learn",
            )
            (output / "history.json").write_text(json.dumps(summary, indent=2) + "\n")
            with (output / "playthroughs.jsonl").open("a") as log:
                for p in played:
                    log.write(json.dumps({"round": round_index, **p.summary}) + "\n")
            if all(counts[m]["first_finish"] is not None for m in MODES) and next_level is not None:
                # Both modes have finished this level: on to the next (its
                # start captured when one of them finished this one).
                (level, start_state), next_level = next_level, None
                (output / "levels").mkdir(exist_ok=True)
                (output / "levels" / f"{level}.state").write_bytes(start_state)
                print(f"[full smb] both modes finished; next level: {level}", flush=True)
    finally:
        pool.shutdown(cancel_futures=True)
    return {"rounds": history, "progress": progress}


def _round_line(entry: dict) -> str:
    parts = []
    for mode, m in entry["modes"].items():
        parts.append(
            f"{mode}: finished {m['finished']}/{m['playthroughs']}, furthest {m['furthest_x']:.0f} "
            f"(best {m['best_furthest_x']}), points {m['points']:.0f}, ends {m['ends']}"
        )
    firsts = entry["playthroughs_until_first_finish"].get(entry["level"], {})
    return (
        f"[full smb] round {entry['round']:03d} {entry['level']}: " + "; ".join(parts)
        + f"; first finish after {firsts}; critic ready {entry['critic_ready']}"
        + f"; seconds {entry['seconds']}"
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    for setting in dataclasses.fields(FullLearnConfig):
        parser.add_argument(
            f"--{setting.name.replace('_', '-')}", type=type(setting.default), default=setting.default
        )
    args = parser.parse_args(argv)
    config = FullLearnConfig(**{f.name: getattr(args, f.name) for f in dataclasses.fields(FullLearnConfig)})
    learn(config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
