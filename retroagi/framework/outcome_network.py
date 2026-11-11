"""The outcome network: from the state, one pass predicts the outcome of
every candidate choice at once.

It reads the state as tokens in groups (a ChoiceSpec, given by the domain's
choice space): the agent's own token, then a fixed number of slots for each
kind of thing around it (in Super Mario Bros: the surfaces, the enemies and
the items on screen). A small transformer lets every token see the others;
then a head on each token gives the outcomes of the choices about that
thing, and a head on the agent's token those about no thing. Each choice's
outcome is PREDICTIONS: the chance the agent dies during it or soon after
with no way out, the chance the controller finds it infeasible, the chance
it finishes the task, the chance it leaves the agent stuck (no choice from
there gets it on), the progress, frames and points it brings, and the
chance it leaves the agent in danger (dead soon after if it then does
nothing, a way out or not).

A saved network carries its ChoiceSpec (save, load_model), so it can be
loaded and checked against a domain without that domain's code.
"""

from dataclasses import asdict, dataclass, field
from typing import Optional, Sequence

import numpy as np
import torch
from torch import nn

PREDICTIONS = ("death", "infeasible", "finished", "stuck", "progress", "frames", "points", "danger")
# The logits (chances) and the amounts, and the scale the amounts are
# learned at (progress, frames, points).
CHANCES, AMOUNTS = (0, 1, 2, 3, 7), (4, 5, 6)
SCALES = (64.0, 64.0, 5.0)


@dataclass(frozen=True)
class TokenGroup:
    """One group of the state's tokens: ``slots`` tokens of ``width``
    numbers (slots 0: the agent's own single token), each with ``heads``
    choices about it. ``scale``: what each number is divided by to bring it
    to about one. ``present``: the (first, last) columns that are all zero
    in an unused slot (None: never unused)."""

    name: str
    slots: int
    width: int
    heads: int
    scale: tuple
    present: Optional[tuple] = None


@dataclass(frozen=True)
class Kind:
    """One kind of choice: about the things of token group ``group`` (an
    index), its variants (``modifiers`` of them) on that group's heads from
    ``first`` on."""

    name: str
    group: int
    first: int
    modifiers: int


@dataclass(frozen=True)
class ChoiceSpec:
    groups: tuple
    kinds: tuple

    @staticmethod
    def from_plain(plain: dict) -> "ChoiceSpec":
        groups = tuple(TokenGroup(**{**g, "scale": tuple(g["scale"]),
                                     "present": tuple(g["present"]) if g["present"] else None})
                       for g in plain["groups"])
        return ChoiceSpec(groups, tuple(Kind(**k) for k in plain["kinds"]))

    def kind_names(self) -> tuple:
        return tuple(k.name for k in self.kinds)


@dataclass
class OutcomeConfig:
    width: int = 96
    layers: int = 3
    heads: int = 4
    # How many of each group's numbers the network reads (the first ones;
    # empty: all of them). A network trained before a group gained numbers
    # reads fewer.
    reads: tuple = field(default_factory=tuple)
    # How many of the PREDICTIONS the network gives (the first ones: one
    # trained before a prediction was added gives fewer, and `predict` says
    # "no chance" for the rest).
    predicts: int = len(PREDICTIONS)


class OutcomeNetwork(nn.Module):
    def __init__(self, spec: ChoiceSpec, config: OutcomeConfig = OutcomeConfig()):
        super().__init__()
        self.spec, self.config = spec, config
        w, n = config.width, config.predicts
        self.reads = tuple(config.reads) or tuple(g.width for g in spec.groups)
        self.inputs = nn.ModuleList(nn.Linear(read, w) for read in self.reads)
        self.kind = nn.Embedding(len(spec.groups), w)
        layer = nn.TransformerEncoderLayer(w, config.heads, 4 * w, dropout=0.0, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, config.layers)
        self.outputs = nn.ModuleList(nn.Linear(w, g.heads * n) for g in spec.groups)
        for k, g in enumerate(spec.groups):
            self.register_buffer(f"scale_{k}", torch.tensor(g.scale[: self.reads[k]], dtype=torch.float32),
                                 persistent=False)

    def forward(self, *tokens):
        """One tensor per token group ([B, width] for the agent's, [B, slots,
        width] for the others) -> one per group, [B, slots or 1, heads, P],
        P = PREDICTIONS (chances as logits, amounts at SCALES)."""
        b = tokens[0].shape[0]
        embedded, absent = [], []
        for k, (group, t) in enumerate(zip(self.spec.groups, tokens)):
            read = self.inputs[k](t[..., : self.reads[k]] / getattr(self, f"scale_{k}"))
            if not group.slots:
                t, read = t[:, None], read[:, None]
            embedded.append(read + self.kind.weight[k])
            if group.present is None:
                absent.append(torch.zeros(b, t.shape[1], dtype=torch.bool, device=t.device))
            else:
                first, last = group.present
                absent.append(t[..., first:last].abs().sum(-1) == 0)
        h = self.encoder(torch.cat(embedded, dim=1), src_key_padding_mask=torch.cat(absent, dim=1))
        out, start, n = [], 0, self.config.predicts
        for k, group in enumerate(self.spec.groups):
            count = group.slots or 1
            tokens_of = h[:, start:start + count] if group.slots else h[:, start]
            out.append(self.outputs[k](tokens_of).view(b, count, group.heads, n))
            start += count
        return tuple(out)


def pick(spec: ChoiceSpec, outputs, kind, slot, modifier):
    """The predictions [N, P] of the choices (kind, slot, modifier) [N], one
    per row of ``outputs`` (forward's, for those rows' states)."""
    device = kind.device
    group = torch.tensor([k.group for k in spec.kinds], device=device)[kind]
    first = torch.tensor([k.first for k in spec.kinds], device=device)[kind]
    last = torch.tensor([k.modifiers - 1 for k in spec.kinds], device=device)[kind]
    head = first + torch.minimum(modifier.clamp(min=0), last)
    rows = torch.arange(kind.shape[0], device=device)
    picked = None
    for g, out in enumerate(outputs):
        value = out[rows, slot.clamp(min=0, max=out.shape[1] - 1), head.clamp(max=out.shape[2] - 1)]
        picked = value if picked is None else torch.where((group == g)[:, None], value, picked)
    return picked


def predict(model: OutcomeNetwork, tokens: Sequence, choices, device) -> np.ndarray:
    """For one state's tokens (one array per group) and its choices [(kind,
    slot, modifier)]: each choice's PREDICTIONS as chances and amounts in
    their own units."""
    tensors = [torch.as_tensor(t, device=device)[None] for t in tokens]
    with torch.no_grad():
        outputs = model(*tensors)
        c = torch.as_tensor(np.array(choices, np.int64).reshape(-1, 3), device=device)
        outputs = tuple(o.expand(len(c), *o.shape[1:]) for o in outputs)
        p = pick(model.spec, outputs, c[:, 0], c[:, 1], c[:, 2])
        if p.shape[1] < len(PREDICTIONS):  # (an older network: no chance of what it does not predict)
            p = torch.cat([p, torch.full((len(p), len(PREDICTIONS) - p.shape[1]), -30.0, device=device)], dim=1)
        out = p.clone()
        out[:, list(CHANCES)] = torch.sigmoid(p[:, list(CHANCES)])
        out[:, list(AMOUNTS)] = p[:, list(AMOUNTS)] * torch.tensor(SCALES, device=device)
    return out.cpu().numpy()


class NetworkPredictor:
    """An outcome network as the chooser's predictor (interfaces.
    OutcomePredictor), on a domain's choice space."""

    def __init__(self, model: OutcomeNetwork, space, device="cpu"):
        self.model, self.space, self.device = model, space, device

    def predict(self, state, candidates) -> np.ndarray:
        tokens, slots = self.space.tokens(state)
        rows = [(c.kind, slots.get(c.ref, -1) if c.ref is not None else -1, c.modifier) for c in candidates]
        return predict(self.model, tokens, rows, self.device)


def save(model: OutcomeNetwork, path: str, log: Sequence = ()) -> None:
    config = {**asdict(model.config), "reads": list(model.reads)}
    torch.save({"config": config, "spec": asdict(model.spec), "model": model.state_dict(), "log": list(log)}, path)


def load_model(path: str, device="cpu") -> OutcomeNetwork:
    saved = torch.load(path, map_location=device)
    # (a network saved before "danger" was predicted gives the first 7)
    config = OutcomeConfig(**{"predicts": 7, **saved["config"], "reads": tuple(saved["config"]["reads"])})
    model = OutcomeNetwork(ChoiceSpec.from_plain(saved["spec"]), config).to(device)
    model.load_state_dict(saved["model"])
    return model.eval()


def check(model: OutcomeNetwork, spec: ChoiceSpec) -> None:
    """Raise if ``model`` was not made for the choice space ``spec``: the
    same kinds of choice, and token groups of the same names, slots and
    heads, each at least as wide as the network reads."""
    mine, theirs = model.spec, spec
    same = mine.kinds == theirs.kinds and len(mine.groups) == len(theirs.groups) and all(
        (a.name, a.slots, a.heads) == (b.name, b.slots, b.heads) and read <= b.width
        for a, b, read in zip(mine.groups, theirs.groups, model.reads)
    )
    if not same:
        raise ValueError("the outcome network was trained for another choice space than this domain's")
