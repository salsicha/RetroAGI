"""The outcome network: from the state, one pass predicts the outcome of
every candidate choice at once.

It reads the state as tokens in groups (a ChoiceSpec, given by the domain's
choice space): the agent's own token, then a fixed number of slots for each
kind of thing around it (in Super Mario Bros: the surfaces, the enemies and
the items on screen). A small transformer lets every token see the others;
then a head on each token gives the outcomes of the choices about that
thing, and a head on the agent's token those about no thing. Each choice's
outcome is PREDICTIONS: the chance the agent dies during it or soon after,
the chance the controller finds it infeasible, the chance it finishes the
task, the chance it leaves the agent stuck (no choice from there gets it
on), and the progress, frames and points it brings; then, given the choice,
how the task ends: the chance it is finished (in Super Mario Bros: the flag
reached) and the frames lost after the choice on the way to it (frames
spent less progress made at the domain's steady pace; docs/requirements.md,
section 7). The last two are learned by bootstrapping (training.py): a
choice's end is its own outcome joined with the network's own guess from
where it leaves the agent.

The frames lost after a choice come in two parts added together: one for
the state, from all its tokens together, the same for every choice there,
and one for each choice, less the mean of those over the state's choices.
A whole level's frames lost are hundreds, guessed some fifty frames wrong,
and the choices differ by a few tens: guessed whole for each choice, the
differences were lost in the error and Mario stepped back and forth until
the time ran out (2026-11-23). Split so, an error in the state's part is
the same for every choice and does not change which is taken; the choice's
part has only the differences to learn.

A saved network carries its ChoiceSpec (save, load_model), so it can be
loaded and checked against a domain without that domain's code.
"""

from dataclasses import asdict, dataclass, field
from typing import Optional, Sequence

import numpy as np
import torch
from torch import nn

PREDICTIONS = ("death", "infeasible", "finished", "stuck", "progress", "frames", "points", "reach", "lost")
# The logits (chances) and the amounts, and the scale the amounts are
# learned at (progress, frames, points, frames lost to the task's end).
CHANCES, AMOUNTS = (0, 1, 2, 3, 7), (4, 5, 6, 8)
AFTER = PREDICTIONS.index("lost")
SCALES = (64.0, 64.0, 5.0, 512.0)
# How many PREDICTIONS a network gave before "reach" and "lost" were added
# (a saved network that does not say gives these).
FIRST_PREDICTIONS = 7


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
    # (twice as wide and a layer deeper as at first: trained on the same
    # outcomes, it finished 162 of 200 plays of fresh Block SMB levels to 140,
    # 2026-11-12)
    width: int = 192
    layers: int = 4
    heads: int = 4
    # How many of each group's numbers the network reads (the first ones;
    # empty: all of them). A network trained before a group gained numbers
    # reads fewer.
    reads: tuple = field(default_factory=tuple)
    # How many of PREDICTIONS it gives (the first ones; 0: all of them). A
    # network trained before predictions were added gives fewer.
    outputs: int = 0


class OutcomeNetwork(nn.Module):
    def __init__(self, spec: ChoiceSpec, config: OutcomeConfig = OutcomeConfig()):
        super().__init__()
        self.spec, self.config = spec, config
        w = config.width
        self.n = n = config.outputs or len(PREDICTIONS)
        self.reads = tuple(config.reads) or tuple(g.width for g in spec.groups)
        self.inputs = nn.ModuleList(nn.Linear(read, w) for read in self.reads)
        self.kind = nn.Embedding(len(spec.groups), w)
        layer = nn.TransformerEncoderLayer(w, config.heads, 4 * w, dropout=0.0, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, config.layers)
        self.outputs = nn.ModuleList(nn.Linear(w, g.heads * n) for g in spec.groups)
        if n > AFTER:
            # the state's part of the frames lost after a choice (module notes)
            self.shared = nn.Linear(w, 1)
        for k, g in enumerate(spec.groups):
            self.register_buffer(f"scale_{k}", torch.tensor(g.scale[: self.reads[k]], dtype=torch.float32),
                                 persistent=False)

    def forward(self, *tokens):
        """One tensor per token group ([B, width] for the agent's, [B, slots,
        width] for the others) -> one per group, [B, slots or 1, heads, P],
        P = the first ``n`` PREDICTIONS (chances as logits, amounts at
        SCALES)."""
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
        out, start, n = [], 0, self.n
        for k, group in enumerate(self.spec.groups):
            count = group.slots or 1
            tokens_of = h[:, start:start + count] if group.slots else h[:, start]
            out.append(self.outputs[k](tokens_of).view(b, count, group.heads, n))
            start += count
        if n > AFTER:
            out = self._shared_after(h, out, absent)
        return tuple(out)

    def _shared_after(self, h, out, absent):
        """``out`` with each choice's frames lost after it given as the
        state's part (from the mean of the present tokens) plus its own less
        the mean of the choices' own over the present tokens (module
        notes)."""
        present = [(~a).float() for a in absent]
        weight = torch.cat(present, dim=1)[..., None]
        state = self.shared((h * weight).sum(1) / weight.sum(1).clamp(min=1))[:, 0]
        total = sum((o[..., AFTER] * p[..., None]).sum((1, 2)) for o, p in zip(out, present))
        cells = sum(p.sum(1) * o.shape[2] for o, p in zip(out, present))
        mean = total / cells.clamp(min=1)
        return [torch.cat([o[..., :AFTER], (state - mean)[:, None, None, None] + o[..., AFTER:AFTER + 1],
                           o[..., AFTER + 1:]], dim=-1) for o in out]


def pick(spec: ChoiceSpec, outputs, kind, slot, modifier, owner=None):
    """The predictions [N, P] of the choices (kind, slot, modifier) [N], each
    from row ``owner`` [N] of ``outputs`` (forward's; None: one row per
    choice, in order)."""
    device = kind.device
    group = torch.tensor([k.group for k in spec.kinds], device=device)[kind]
    first = torch.tensor([k.first for k in spec.kinds], device=device)[kind]
    last = torch.tensor([k.modifiers - 1 for k in spec.kinds], device=device)[kind]
    head = first + torch.minimum(modifier.clamp(min=0), last)
    rows = torch.arange(kind.shape[0], device=device) if owner is None else owner
    picked = None
    for g, out in enumerate(outputs):
        value = out[rows, slot.clamp(min=0, max=out.shape[1] - 1), head.clamp(max=out.shape[2] - 1)]
        picked = value if picked is None else torch.where((group == g)[:, None], value, picked)
    return picked


def cells_of(spec: ChoiceSpec, kind, slot, modifier) -> np.ndarray:
    """The output each choice (kind, slot, modifier) [N] is read from, as
    one number: the cells of forward's outputs counted group after group,
    slot after slot, head after head (``all_cells``). Two choices read from
    the same cell (a modifier beyond its kind's last) are predicted alike."""
    kind, slot, modifier = (np.asarray(a, np.int64) for a in (kind, slot, modifier))
    heads = np.array([g.heads for g in spec.groups])
    slots = np.array([g.slots or 1 for g in spec.groups])
    offset = np.concatenate([[0], np.cumsum(heads * slots)[:-1]])
    group = np.array([k.group for k in spec.kinds])[kind]
    first = np.array([k.first for k in spec.kinds])[kind]
    last = np.array([k.modifiers - 1 for k in spec.kinds])[kind]
    head = np.minimum(first + np.minimum(np.maximum(modifier, 0), last), heads[group] - 1)
    return offset[group] + np.clip(slot, 0, slots[group] - 1) * heads[group] + head


def cell_count(spec: ChoiceSpec) -> int:
    return int(sum(g.heads * (g.slots or 1) for g in spec.groups))


def all_cells(outputs) -> torch.Tensor:
    """forward's outputs as one [B, cell_count, P] tensor (cells_of's order)."""
    return torch.cat([out.reshape(out.shape[0], -1, out.shape[-1]) for out in outputs], dim=1)


def predict(model: OutcomeNetwork, tokens: Sequence, choices, device) -> np.ndarray:
    """For one state's tokens (one array per group) and its choices [(kind,
    slot, modifier)]: each choice's PREDICTIONS as chances and amounts in
    their own units (NaN for those a network trained before them does not
    give)."""
    tensors = [torch.as_tensor(t, device=device)[None] for t in tokens]
    with torch.no_grad():
        outputs = model(*tensors)
        c = torch.as_tensor(np.array(choices, np.int64).reshape(-1, 3), device=device)
        outputs = tuple(o.expand(len(c), *o.shape[1:]) for o in outputs)
        p = pick(model.spec, outputs, c[:, 0], c[:, 1], c[:, 2])
        out = torch.full((len(c), len(PREDICTIONS)), float("nan"), device=device)
        out[:, : model.n] = p
        chances = [k for k in CHANCES if k < model.n]
        amounts = [(k, scale) for k, scale in zip(AMOUNTS, SCALES) if k < model.n]
        out[:, chances] = torch.sigmoid(p[:, chances])
        out[:, [k for k, _ in amounts]] = p[:, [k for k, _ in amounts]] * torch.tensor(
            [scale for _, scale in amounts], device=device)
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
    config = {**asdict(model.config), "reads": list(model.reads), "outputs": model.n}
    torch.save({"config": config, "spec": asdict(model.spec), "model": model.state_dict(), "log": list(log)}, path)


def load_model(path: str, device="cpu") -> OutcomeNetwork:
    saved = torch.load(path, map_location=device)
    config = OutcomeConfig(**{"outputs": FIRST_PREDICTIONS, **saved["config"],
                              "reads": tuple(saved["config"]["reads"])})
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
