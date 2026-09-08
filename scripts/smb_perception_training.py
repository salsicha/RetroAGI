"""Train dense collision perception; keep clip splits and label provenance explicit."""

import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from retroagi.core.smb_perception import DenseSMBPerception
from retroagi.core.smb_scene import block_oracle_scene
from retroagi.core.smb_supervision import collision_labels
from retroagi.core.vision import image_tensor
from retroagi.stages.block_smb.env import MarioScenarioEnv


def block_clips(cases, directory, *, stride=8):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    clips = []
    for number, sample in enumerate(cases):
        env = MarioScenarioEnv()
        env.render_goal = False
        images = []
        labels = []
        try:
            obs, _ = env.reset(scenario=sample.scenario)
            for frame, action in enumerate(sample.oracle["actions"]):
                if frame % stride == 0:
                    images.append(obs.copy())
                    labels.append(collision_labels(block_oracle_scene(env)["scene"]))
                obs, _, done, truncated, _ = env.step(action)
                if done or truncated:
                    break
        finally:
            env.close()
        path = directory / f"clip_{number:05d}.npz"
        np.savez_compressed(path, images=np.stack(images), labels=np.stack(labels))
        clips.append(
            dict(
                file=str(path.resolve()),
                scenario_id=sample.scenario_id,
                split=sample.split,
                family=sample.family,
                label_source="block_collision_instrumentation",
                pixels="block_renderer",
                frames=len(images),
            )
        )
    (directory / "clips.json").write_text(json.dumps(clips, indent=2) + "\n")
    return clips


class ClipDataset:
    def __init__(self, clips):
        self.clips = clips
        self.index = [(i, j) for i, c in enumerate(clips) for j in range(c["frames"])]
        self.cache = {}
        if not self.index:
            raise ValueError("Empty perception split")

    def batch(self, indices):
        images = []
        labels = []
        for index in indices:
            i, j = self.index[index]
            if i not in self.cache:
                # Bounded cache; datasets are compressed on disk.
                if len(self.cache) >= 16:
                    self.cache.pop(next(iter(self.cache)))
                with np.load(self.clips[i]["file"]) as data:
                    self.cache[i] = (data["images"], data["labels"])
            image, label = self.cache[i]
            images.append(image[j])
            labels.append(label[j])
        return np.stack(images), np.stack(labels)


@torch.no_grad()
def evaluate_perception(model, dataset, *, batch_size=8):
    model.eval()
    confusion = torch.zeros(7, 7, dtype=torch.int64)
    box_errors = []
    missing = 0
    from retroagi.core.smb_tracking import component_boxes

    for start in range(0, len(dataset.index), batch_size):
        images, labels = dataset.batch(range(start, min(start + batch_size, len(dataset.index))))
        pred = model.encode(images).semantic_ids.cpu().numpy()
        valid = labels != 255
        counts = np.bincount(
            (labels[valid].astype(np.int64) * 7 + pred[valid]).ravel(), minlength=49
        )
        confusion += torch.from_numpy(counts.reshape(7, 7))
        for actual, estimate in zip(labels, pred):
            for cls in (1, 5, 6):
                a = component_boxes(actual == cls)
                b = component_boxes(estimate == cls)
                for rect in a:
                    if not b:
                        missing += 1
                        box_errors.append(256.0)
                        continue
                    match = min(
                        b,
                        key=lambda r: abs(r.centerx - rect.centerx) + abs(r.centery - rect.centery),
                    )
                    box_errors.append(
                        max(
                            abs(x - y)
                            for x, y in zip(
                                (rect.left, rect.top, rect.right, rect.bottom),
                                (match.left, match.top, match.right, match.bottom),
                            )
                        )
                    )
    intersection = confusion.diag()
    union = confusion.sum(0) + confusion.sum(1) - intersection
    iou = {str(i): float(intersection[i] / union[i]) if union[i] else None for i in range(7)}
    return dict(
        iou=iou,
        confusion=confusion.tolist(),
        body_edge_p95=float(np.percentile(box_errors, 95)) if box_errors else None,
        missed_bodies=missing,
        frames=len(dataset.index),
        collision_labels=True,
    )


def collision_cross_entropy(logits, target, *, weight):
    """Use pixel rows to preserve weighted CE with deterministic CUDA kernels.

    The spatial NLL reduction rejects strict determinism on CUDA. Flattening
    pixels selects the deterministic matrix NLL path with the same objective,
    class weighting, ignored pixels, and gradients.
    """
    return F.cross_entropy(
        logits.movedim(1, -1).reshape(-1, logits.shape[1]),
        target.reshape(-1),
        weight=weight,
        ignore_index=255,
    )


def translated_batch(images, labels, rng, *, max_x=128, max_y=64):
    """Expose screen positions, patch phases and clipped bodies without wrapping.

    New pixels are unsupervised (255), rather than invented collision labels.
    This is perception-only augmentation; physical policy trajectories are intact.
    """
    shifted_images = np.zeros_like(images)
    shifted_labels = np.full_like(labels, 255)
    height, width = labels.shape[-2:]
    for i in range(len(images)):
        dx = int(rng.integers(-max_x, max_x + 1))
        dy = int(rng.integers(-max_y, max_y + 1))
        sx, sy = max(0, -dx), max(0, -dy)
        tx, ty = max(0, dx), max(0, dy)
        w, h = width - abs(dx), height - abs(dy)
        if w > 0 and h > 0:
            shifted_images[i, ty : ty + h, tx : tx + w] = images[i, sy : sy + h, sx : sx + w]
            shifted_labels[i, ty : ty + h, tx : tx + w] = labels[i, sy : sy + h, sx : sx + w]
    return shifted_images, shifted_labels


def train_perception(
    train_clips,
    validation_clips,
    directory,
    *,
    steps=4000,
    batch_size=8,
    device="cuda",
    seed=0,
    dim=128,
    depth=3,
    log=None,
):
    if {c["scenario_id"] for c in train_clips} & {c["scenario_id"] for c in validation_clips}:
        raise ValueError("Perception clips leak across train/validation")
    if any(c["split"] != "train" for c in train_clips) or any(
        c["split"] != "validation" for c in validation_clips
    ):
        raise ValueError("Incorrect perception split")
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    train = ClipDataset(train_clips)
    validation = ClipDataset(validation_clips)
    torch.manual_seed(seed)
    class_weights = [0.1, 5.0, 1.0, 4.0, 4.0, 5.0, 3.0]
    model = DenseSMBPerception(
        dim=dim, depth=depth, class_weights=class_weights, refinement_channels=16
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    rng = np.random.default_rng(seed)
    weights = torch.tensor(class_weights, device=device)
    for step in range(steps):
        model.train()
        images, labels = train.batch(rng.integers(len(train.index), size=batch_size))
        images, labels = translated_batch(images, labels, rng)
        target = torch.as_tensor(labels, device=device).long()
        # Optimize weighted logits; inference removes the known class prior.
        vision = model(image_tensor(images, device=torch.device(device)), calibrated=False)
        loss = collision_cross_entropy(vision.semantic_logits, target, weight=weights)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step == 0 or (step + 1) % 100 == 0 or step + 1 == steps:
            event = dict(phase="perception", step=step + 1, loss=float(loss.detach()))
            if log:
                log(event)
            else:
                print(json.dumps(event), flush=True)
    metrics = evaluate_perception(model, validation)
    observed = [v for k, v in metrics["iou"].items() if k != "0" and v is not None]
    metrics["qualified"] = bool(
        observed
        and min(observed) >= 0.9
        and metrics["body_edge_p95"] is not None
        and metrics["body_edge_p95"] <= 2
        and not metrics["missed_bodies"]
    )
    metrics["absent_validation_classes"] = [int(k) for k, v in metrics["iou"].items() if v is None]
    metrics["full_level_qualified"] = False
    metrics["training_translation"] = dict(max_x=128, max_y=64, padding_label=255)
    metrics["class_weights"] = class_weights
    metrics["probability_correction"] = "subtract_log_training_class_weights"
    event = dict(
        phase="perception_validation",
        **{
            key: metrics[key]
            for key in ("qualified", "iou", "body_edge_p95", "missed_bodies", "frames")
        },
    )
    if log:
        log(event)
    else:
        print(json.dumps(event), flush=True)
    model.eval().requires_grad_(False)
    model.save(directory / "perception.pth", metrics=metrics)
    (directory / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    (directory / "provenance.json").write_text(
        json.dumps(dict(train=train_clips, validation=validation_clips), indent=2) + "\n"
    )
    return model, metrics
