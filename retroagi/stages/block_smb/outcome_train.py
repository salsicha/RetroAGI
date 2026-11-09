"""Train the outcome network (core/outcome_network.py) on sampled outcomes
(outcome_sampling.py) and measure it on outcomes sampled on fresh levels.

Each step takes BATCH sampled choices at random, runs the network on their
states and compares its predictions for those choices with what happened:
the chances (death during the choice or after it, infeasible, reaching the
flag) by cross-entropy, the amounts (progress, frames, points, at the
network's SCALES) by a Huber loss on choices the executor could carry out.
The measures, per kind of choice: how well the death chance ranks deaths
first (the area under the ROC curve), its mean error, how often
"infeasible" and "stuck" are called right, and the progress error in
pixels.
"""

import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from retroagi.core.devices import select_device
from retroagi.core.outcome_network import AMOUNTS, CHANCES, SCALES, OutcomeConfig, OutcomeNetwork, pick
from retroagi.core.smb_choices import KINDS

from .outcome_sampling import ENDS, OUTCOME_FIELDS, load

BATCH = 2048
STEPS = 20_000
LEARNING_RATE = 3e-4
INFEASIBLE = ENDS.index("infeasible")


def targets(outcomes: np.ndarray) -> np.ndarray:
    """[N, PREDICTIONS] from the sampled OUTCOME_FIELDS: death (during or
    after), infeasible (the executor ended it before its first frame: found
    infeasible, or a run blocked at once; a blocked run counted as feasible
    was picked again and again against a wall), finished, stuck, then
    progress, frames and points at the network's SCALES."""
    f = {name: outcomes[:, k] for k, name in enumerate(OUTCOME_FIELDS)}
    nothing = (f["frames"] == 0) & (f["died"] == 0) & (f["finished"] == 0)
    return np.stack([
        np.maximum(f["died"], f["died_after"]),
        ((f["end"] == INFEASIBLE) | nothing).astype(np.float32),
        f["finished"],
        f["stuck"],
        f["progress"] / SCALES[0],
        f["frames"] / SCALES[1],
        f["points"] / SCALES[2],
    ], axis=1).astype(np.float32)


class Data:
    """A folder of shards on ``device``: the states' tokens, the choices
    (state, kind, slot, modifier) and their targets."""

    def __init__(self, folder: str, device):
        d = load(folder)
        self.mario = torch.as_tensor(d["mario"], device=device)
        self.surfaces = torch.as_tensor(d["surfaces"], device=device)
        self.enemies = torch.as_tensor(d["enemies"], device=device)
        self.items = torch.as_tensor(d["items"], device=device)
        self.choices = torch.as_tensor(d["choices"].astype(np.int64), device=device)
        self.targets = torch.as_tensor(targets(d["outcomes"]), device=device)

    def __len__(self):
        return len(self.choices)

    def outputs(self, model, rows):
        c = self.choices[rows]
        s = c[:, 0]
        out = model(self.mario[s], self.surfaces[s], self.enemies[s], self.items[s])
        return pick(out, c[:, 1], c[:, 2], c[:, 3]), self.targets[rows]


def loss_of(predicted, target):
    chances = nn.functional.binary_cross_entropy_with_logits(predicted[:, list(CHANCES)], target[:, list(CHANCES)])
    feasible = target[:, 1] < 0.5
    amounts = nn.functional.smooth_l1_loss(predicted[feasible][:, list(AMOUNTS)], target[feasible][:, list(AMOUNTS)]) \
        if feasible.any() else predicted.sum() * 0.0
    return chances + amounts


def _auc(score: np.ndarray, label: np.ndarray) -> float:
    positives, negatives = label.sum(), len(label) - label.sum()
    if positives == 0 or negatives == 0:
        return float("nan")
    ranks = np.empty(len(score))
    ranks[np.argsort(score)] = np.arange(1, len(score) + 1)
    return float((ranks[label > 0.5].sum() - positives * (positives + 1) / 2) / (positives * negatives))


def measure(model, data: Data, batch: int = 8192) -> dict:
    """The measures on every choice of ``data``, overall and per kind."""
    model.eval()
    predicted, target = [], []
    with torch.no_grad():
        for start in range(0, len(data), batch):
            rows = torch.arange(start, min(start + batch, len(data)), device=data.choices.device)
            p, t = data.outputs(model, rows)
            predicted.append(p.cpu())
            target.append(t.cpu())
    model.train()
    p, t = torch.cat(predicted).numpy(), torch.cat(target).numpy()
    kinds = data.choices[:, 1].cpu().numpy()
    chance = 1 / (1 + np.exp(-p[:, list(CHANCES)]))
    out = {}
    for name, sel in [("all", np.ones(len(p), bool))] + [(k, kinds == i) for i, k in enumerate(KINDS)]:
        if not sel.any():
            continue
        feasible = sel & (t[:, 1] < 0.5)
        out[name] = {
            "choices": int(sel.sum()),
            "deaths": float(t[sel, 0].mean()),
            "death_auc": _auc(chance[sel, 0], t[sel, 0]),
            "death_error": float(np.abs(chance[sel, 0] - t[sel, 0]).mean()),
            "infeasible_right": float(((chance[sel, 1] > 0.5) == (t[sel, 1] > 0.5)).mean()),
            "stuck_right": float(((chance[sel, 3] > 0.5) == (t[sel, 3] > 0.5)).mean()),
            "progress_error_px": float(np.abs(p[feasible, 4] - t[feasible, 4]).mean() * SCALES[0]) if feasible.any() else None,
        }
    return out


def train(data_folder: str, eval_folder: str, output: str, steps: int = STEPS, batch: int = BATCH,
          seed: int = 0, config: OutcomeConfig = OutcomeConfig()) -> dict:
    torch.manual_seed(seed)
    device = select_device("auto")
    data, fresh = Data(data_folder, device), Data(eval_folder, device)
    model = OutcomeNetwork(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, steps)
    Path(output).mkdir(parents=True, exist_ok=True)
    log = []
    for step in range(1, steps + 1):
        rows = torch.randint(0, len(data), (batch,), device=device)
        predicted, target = data.outputs(model, rows)
        loss = loss_of(predicted, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        schedule.step()
        if step % (steps // 10) == 0 or step == steps:
            measures = measure(model, fresh)
            log.append({"step": step, "loss": float(loss), "fresh": measures})
            a = measures["all"]
            print(f"step {step} loss {float(loss):.4f} fresh: death auc {a['death_auc']:.3f} "
                  f"error {a['death_error']:.3f} infeasible right {a['infeasible_right']:.3f} "
                  f"stuck right {a['stuck_right']:.3f} "
                  f"progress {a['progress_error_px']:.1f} px", flush=True)
    torch.save({"config": config.__dict__, "model": model.state_dict(), "log": log}, f"{output}/outcome.pt")
    Path(f"{output}/measures.json").write_text(json.dumps(log[-1], indent=1))
    return log[-1]


def load_model(path: str, device="cpu") -> OutcomeNetwork:
    saved = torch.load(path, map_location=device)
    model = OutcomeNetwork(OutcomeConfig(**saved["config"])).to(device)
    model.load_state_dict(saved["model"])
    return model.eval()


def main(argv=None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Train the outcome network on sampled outcomes")
    parser.add_argument("--data", required=True)
    parser.add_argument("--fresh", required=True, help="outcomes sampled on levels not trained on")
    parser.add_argument("--output", required=True)
    parser.add_argument("--steps", type=int, default=STEPS)
    parser.add_argument("--batch", type=int, default=BATCH)
    args = parser.parse_args(argv)
    final = train(args.data, args.fresh, args.output, args.steps, args.batch)
    print(json.dumps(final["fresh"], indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
