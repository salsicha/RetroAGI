"""Train the outcome network (outcome_network.py) on sampled outcomes
(sampling.py) and measure it on outcomes sampled in fresh practice worlds.

Each step takes BATCH sampled choices at random, runs the network on their
states and compares its predictions for those choices with what happened:
the chances (death during the choice or after it, infeasible, finishing the
task, stuck) by cross-entropy, the amounts (progress, frames, points, at
the network's SCALES) by a Huber loss on choices the controller could carry
out.

How the task ends after each choice (the chance it is finished, "reach",
and the frames lost after it on the way, "lost": docs/requirements.md,
section 7, and chooser's module notes) is learned by bootstrapping. A
choice that killed ends the task unfinished; one that finished it,
finished, with nothing lost after it. Any other left the agent in a state
the sampling kept, with the candidates open there: from it the network
being trained picks the candidate the chooser's rule would take
(chooser.best_ends, in the speed run's way, on each candidate's own frames
lost, from its predicted frames and progress at the domain's pace, and
those after it), a copy of the network (refreshed every TARGET_EVERY
steps) says how that one ends, and the choice's end is that: its chance,
and its frames lost, its own and those after it. Each refresh carries the guess one choice further ahead,
until it reaches the task's end. (Picked and judged by the same network,
the best of many guesses came out too hopeful: nearly every choice was
given a 98 to 99% chance of finishing, 2026-11-23.) Outcomes sampled
before such states were kept teach no end.

The measures, per kind of choice: how well the death chance ranks deaths
first (the area under the ROC curve), its mean error, how often
"infeasible" and "stuck" are called right, the progress error, and on all
choices the end's errors against its bootstrapped targets (by the network
measured).

    python -m retroagi.framework.training --domain smb --data FOLDER --fresh FOLDER --output FOLDER
"""

import copy
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from . import registry
from .devices import select_device
from .outcome_network import (
    AMOUNTS,
    CHANCES,
    SCALES,
    ChoiceSpec,
    OutcomeConfig,
    PREDICTIONS,
    OutcomeNetwork,
    all_cells,
    cell_count,
    pick,
    save,
)
from .chooser import best_ends, frames_lost
from .sampling import OUTCOME_FIELDS, kinds_of, load

BATCH = 2048
# (the larger network went on learning to 40,000: outcome_network.OutcomeConfig)
STEPS = 40_000
LEARNING_RATE = 3e-4
# Steps between refreshes of the copy of the network the ends are
# bootstrapped from.
TARGET_EVERY = 500
DEATH_, INFEASIBLE_, FINISHED_, STUCK_, PROGRESS_, FRAMES_ = range(6)
REACH, LOST = PREDICTIONS.index("reach"), PREDICTIONS.index("lost")


def targets(outcomes: np.ndarray, ends: tuple) -> np.ndarray:
    """[N, PREDICTIONS] from the sampled OUTCOME_FIELDS: death (during or
    after), infeasible (the controller ended it before its first frame:
    found infeasible, or blocked at once; a blocked run counted as feasible
    was picked again and again against a wall), finished, stuck, then
    progress, frames and points at the network's SCALES; then the task's end
    where the choice itself settles it (reach: 0 if it killed, 1 if it
    finished; lost after it: 0 if it finished), NaN where it is
    bootstrapped (Data.aim)."""
    f = {name: outcomes[:, k] for k, name in enumerate(OUTCOME_FIELDS)}
    nothing = (f["frames"] == 0) & (f["died"] == 0) & (f["finished"] == 0)
    died = np.maximum(f["died"], f["died_after"])
    done = (died == 0) & (f["finished"] > 0)
    return np.stack([
        died,
        ((f["end"] == ends.index("infeasible")) | nothing).astype(np.float32),
        f["finished"],
        f["stuck"],
        f["progress"] / SCALES[0],
        f["frames"] / SCALES[1],
        f["points"] / SCALES[2],
        np.where(died > 0, 0.0, np.where(done, 1.0, np.nan)),
        np.where(done, 0.0, np.nan),
    ], axis=1).astype(np.float32)


class Data:
    """A folder of shards on ``device``: the states' tokens (one tensor per
    token group of ``spec``), the choices (state, kind, slot, modifier) and
    their targets."""

    def __init__(self, folder: str, spec: ChoiceSpec, ends: tuple, device, pace: float = 1.0):
        d = load(folder)  # (one folder, or several: their shards together)
        if d["kinds"] != [kinds_of(spec)]:
            raise ValueError(f"outcomes in {folder} were sampled for other kinds of choice than the domain's "
                             f"(a row's kind and modifier would mean another choice): {d['kinds']}")
        self.spec = spec
        # (sampled when a group had fewer numbers: zeros for those added since, sampling._joined)
        self.tokens = [torch.as_tensor(np.pad(d[g.name], [(0, 0)] * (d[g.name].ndim - 1)
                                              + [(0, g.width - d[g.name].shape[-1])]), device=device)
                       for g in spec.groups]
        self.choices = torch.as_tensor(d["choices"].astype(np.int64), device=device)
        self.targets = torch.as_tensor(targets(d["outcomes"], ends), device=device)
        self.pace = pace
        # The states choices left the agent in, stored small (sampling's
        # module notes): each group's tokens not all zero, their slots, and
        # each state's count of them, from next_start; each state's
        # candidates as one bit for each network output.
        self.left_in = torch.as_tensor(d["choice_next"].astype(np.int64), device=device)
        self.next_cells = torch.as_tensor(d.get("next_cells", np.zeros((0, (cell_count(spec) + 7) // 8), np.uint8)),
                                          device=device)
        self.next_tokens, self.next_slot, self.next_start, self.next_count = [], [], [], []
        for g in spec.groups:
            k = d.get(f"next_{g.name}", np.zeros((0, g.width), np.float16))
            self.next_tokens.append(torch.as_tensor(np.pad(k, [(0, 0), (0, g.width - k.shape[-1])]), device=device))
            if g.slots:
                count = torch.as_tensor(d.get(f"next_{g.name}_count", np.zeros(0, np.uint8)).astype(np.int64),
                                        device=device)
                self.next_count.append(count)
                self.next_start.append(torch.cumsum(count, 0) - count)
                self.next_slot.append(torch.as_tensor(d.get(f"next_{g.name}_slot", np.zeros(0, np.uint8)),
                                                      device=device))
            else:
                self.next_count.append(None)
                self.next_start.append(None)
                self.next_slot.append(None)
        self.bits = 2 ** torch.arange(7, -1, -1, device=device, dtype=torch.uint8)

    def __len__(self):
        return len(self.choices)

    def outputs(self, model, rows, ahead=None):
        """The predictions for the choices ``rows`` and their targets, the
        ends bootstrapped from the network ``ahead`` (none: those not
        settled by the choice itself are NaN), the choice after each picked
        by ``model``."""
        c = self.choices[rows]
        out = model(*(t[c[:, 0]] for t in self.tokens))
        target = self.targets[rows]
        if ahead is not None:
            target = self.aim(ahead, rows, target, model)
        return pick(self.spec, out, c[:, 1], c[:, 2], c[:, 3]), target

    def aim(self, ahead, rows, target, chooser=None):
        """``target`` with the ends of the choices ``rows`` that left the
        agent in a kept state bootstrapped from the network ``ahead``, the
        choice after picked by ``chooser`` (None: ``ahead``; module
        notes)."""
        left = self.left_in[rows]
        has = left >= 0
        if not has.any():
            return target
        reach, lost = self.ends_from(ahead, left[has], chooser)
        target = target.clone()
        aimed = target[has]
        aimed[:, REACH] = reach
        aimed[:, LOST] = lost / SCALES[3]
        target[has] = aimed
        return target

    def frames_lost(self, p):
        """The frames each choice of the network's outputs ``p`` [N, P]
        loses to the task's end, its own and after it."""
        return frames_lost(p[:, FRAMES_] * SCALES[1], p[:, PROGRESS_] * SCALES[0], p[:, LOST] * SCALES[3], self.pace)

    def end_tokens(self, states):
        """The kept states ``states`` [B]: their tokens, one tensor per
        group as forward reads them."""
        tokens = []
        for g, kept, slot, start, count in zip(self.spec.groups, self.next_tokens, self.next_slot,
                                              self.next_start, self.next_count):
            if not g.slots:
                tokens.append(kept[states].float())
                continue
            n = count[states]
            owner = torch.repeat_interleave(torch.arange(len(states), device=states.device), n)
            first = torch.cumsum(n, 0) - n
            rows = start[states][owner] + torch.arange(len(owner), device=states.device) - first[owner]
            dense = torch.zeros(len(states), g.slots, g.width, device=states.device)
            dense[owner, slot[rows].long()] = kept[rows].float()
            tokens.append(dense)
        return tokens

    def ends_from(self, ahead, states, chooser=None):
        """From each kept state ``states`` [B]: the choice the chooser's
        rule takes there on the network ``chooser``'s predictions (None:
        ``ahead``'s), and the chance the network ``ahead`` gives of
        finishing the task by it and the frames it loses to the end, its
        own and after it (0 and NaN where no candidate there can be carried
        out)."""
        with torch.no_grad():
            tokens = self.end_tokens(states)
            p = all_cells(ahead(*tokens))
            cells = p.shape[1]
            candidate = ((self.next_cells[states][:, :, None] & self.bits) > 0).reshape(len(states), -1)[:, :cells]
            p = p.reshape(-1, p.shape[-1])
            q = p if chooser is None or chooser is ahead else all_cells(chooser(*tokens)).reshape(-1, p.shape[-1])
            owner = torch.arange(len(states), device=states.device).repeat_interleave(cells)
            allowed = candidate.reshape(-1) & (torch.sigmoid(q[:, INFEASIBLE_]) < 0.5)
            best = best_ends(torch.sigmoid(q[:, REACH]), self.frames_lost(q), allowed, owner, len(states))
            found = best >= 0
            at = best.clamp(min=0)
            reach, lost = torch.sigmoid(p[at, REACH]), self.frames_lost(p[at])
            return (torch.where(found, reach, torch.zeros_like(reach)),
                    torch.where(found, lost, torch.full_like(lost, float("nan"))))


def loss_of(predicted, target):
    """Cross-entropy on the chances, a Huber loss on the amounts, both only
    where the choice could be carried out and the target is known (the
    first four chances: always)."""
    first, amounts_now = list(CHANCES[:4]), list(AMOUNTS[:3])
    loss = nn.functional.binary_cross_entropy_with_logits(predicted[:, first], target[:, first])
    feasible = target[:, INFEASIBLE_] < 0.5
    if feasible.any():
        loss = loss + nn.functional.smooth_l1_loss(predicted[feasible][:, amounts_now],
                                                   target[feasible][:, amounts_now])
    for k, chance in ((REACH, True), (LOST, False)):
        known = feasible & torch.isfinite(target[:, k])
        if known.any():
            loss = loss + (nn.functional.binary_cross_entropy_with_logits if chance else nn.functional.smooth_l1_loss)(
                predicted[known, k], target[known, k])
    return loss


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
            p, t = data.outputs(model, rows, ahead=model)
            predicted.append(p.cpu())
            target.append(t.cpu())
    model.train()
    p, t = torch.cat(predicted).numpy(), torch.cat(target).numpy()
    kinds = data.choices[:, 1].cpu().numpy()
    chance = 1 / (1 + np.exp(-p[:, list(CHANCES)]))
    out = {}
    names = data.spec.kind_names()
    for name, sel in [("all", np.ones(len(p), bool))] + [(k, kinds == i) for i, k in enumerate(names)]:
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
            "progress_error": float(np.abs(p[feasible, 4] - t[feasible, 4]).mean() * SCALES[0]) if feasible.any() else None,
        }
        known = feasible & np.isfinite(t[:, REACH])
        if known.any():
            out[name]["reach_error"] = float(np.abs(1 / (1 + np.exp(-p[known, REACH])) - t[known, REACH]).mean())
            out[name]["reach"] = float(t[known, REACH].mean())
        known = feasible & np.isfinite(t[:, LOST])
        if known.any():
            out[name]["lost_error"] = float(np.abs(p[known, LOST] - t[known, LOST]).mean() * SCALES[3])
    return out


def train(domain, data_folder: str, eval_folder: str, output: str, steps: int = STEPS, batch: int = BATCH,
          seed: int = 0, config: OutcomeConfig = OutcomeConfig()) -> dict:
    torch.manual_seed(seed)
    device = select_device("auto")
    spec = domain.space.spec
    data, fresh = (Data(folder, spec, domain.ends, device, domain.pace) for folder in (data_folder, eval_folder))
    model = OutcomeNetwork(spec, config).to(device)
    ahead = copy.deepcopy(model).eval()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, steps)
    Path(output).mkdir(parents=True, exist_ok=True)
    log = []
    for step in range(1, steps + 1):
        if step % TARGET_EVERY == 0:
            ahead.load_state_dict(model.state_dict())
        rows = torch.randint(0, len(data), (batch,), device=device)
        predicted, target = data.outputs(model, rows, ahead=ahead)
        loss = loss_of(predicted, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        schedule.step()
        if step % max(1, steps // 10) == 0 or step == steps:
            measures = measure(model, fresh)
            log.append({"step": step, "loss": float(loss), "fresh": measures})
            a = measures["all"]
            print(f"step {step} loss {float(loss):.4f} fresh: death auc {a['death_auc']:.3f} "
                  f"error {a['death_error']:.3f} infeasible right {a['infeasible_right']:.3f} "
                  f"stuck right {a['stuck_right']:.3f} "
                  f"progress {a['progress_error']:.1f} reach error {a.get('reach_error', float('nan')):.3f} "
                  f"lost {a.get('lost_error', float('nan')):.0f}", flush=True)
    save(model, f"{output}/outcome.pt", log)
    Path(f"{output}/measures.json").write_text(json.dumps(log[-1], indent=1))
    return log[-1]


def main(argv=None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Train the outcome network on sampled outcomes")
    parser.add_argument("--domain", default="smb")
    parser.add_argument("--data", required=True, nargs="+",
                        help="one or more folders of sampled outcomes (trained on together)")
    parser.add_argument("--fresh", required=True, help="outcomes sampled in practice worlds not trained on")
    parser.add_argument("--output", required=True)
    parser.add_argument("--steps", type=int, default=STEPS)
    parser.add_argument("--batch", type=int, default=BATCH)
    defaults = OutcomeConfig()
    parser.add_argument("--width", type=int, default=defaults.width, help="numbers per token inside the network")
    parser.add_argument("--layers", type=int, default=defaults.layers, help="transformer layers")
    parser.add_argument("--heads", type=int, default=defaults.heads, help="attention heads a layer")
    args = parser.parse_args(argv)
    config = OutcomeConfig(width=args.width, layers=args.layers, heads=args.heads)
    final = train(registry.load(args.domain), args.data, args.fresh, args.output, args.steps, args.batch,
                  config=config)
    print(json.dumps(final["fresh"], indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
