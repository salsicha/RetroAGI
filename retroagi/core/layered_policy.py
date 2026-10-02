"""The four-layer SMB agent: strategy, tactic, skill and action transformers.

Every layer sees the game only through the vision transformer
(smb_observer.PolicyInput) and the agent's own memory:

- SceneEncoder turns one PolicyInput into tokens: one per reported object
  (Mario, each enemy, coin, power-up, moving platform, pipe, block, surface,
  gap), the skill's target, and the 8 + 16 screen-band codes. Absent list
  slots are left out.
- Every position number enters SceneEncoder also as sine and cosine waves of
  several wavelengths (PositionWaves), so a pixel's difference is a clear
  difference of input.
- At the start of every action, before anything decides, Memory, a long
  short-term memory network, steps once with the latest picture's scene (the
  scene encoder's summary of the frame the action starts on). It is never
  told what Mario will do; what happened earlier it carries itself. From its state it predicts the world when the
  coming action ends: the scene numbers the vision transformer will report
  then. It is trained action by action against what was actually reported.
- Each layer reads the current scene's tokens, the expected scene's tokens
  (the memory's prediction, encoded the same way) and the token from the
  layer above, and emits its own token: strategy -> tactic -> skill -> action. The
  action layer's output is an executor plan: an action and a frame count
  (smb_executor). The skill layer points at its target among the scene's
  surfaces, enemies and moving platforms.

Layers decide only when the executor's plan has ended (completed or
interrupted). In Block SMB a layer is trained with explicit tokens from a
teacher for the layer above it (``given``); at play time each token comes only
from the layer above, and the strategy layer is trained only by playing Full
SMB (until then DEFAULT_STRATEGY drives the stack).
"""

import math
from dataclasses import dataclass
from typing import Mapping, Optional

import torch
import torch.nn as nn

from .actions import SMB_ACTIONS
from .smb_executor import FRAME_COUNTS, ActionPlan
from .smb_observer import (
    _SLOT_WIDTH,
    C_SPANS,
    COLUMN_CODES,
    MARIO_WIDTH,
    SCENE_SLOTS,
    SEQ_LEN_A,
    SEQ_LEN_B,
    SEQ_LEN_C,
    TARGET_WIDTH,
)
from .tokens import (
    DEFAULT_STRATEGY,
    NO_TARGET,
    SKILL_WIDTH,
    SKILLS,
    STRATEGIES,
    STRATEGY_WIDTH,
    TACTIC_WIDTH,
    TACTICS,
    TARGET_LISTS,
    TARGETS,
    SkillToken,
    StrategyToken,
    TacticToken,
    encode_skill,
    encode_strategy,
    encode_tactic,
)

LAYERS = ("strategy", "tactic", "skill", "action")
FRAME_BINS = len(FRAME_COUNTS)  # the action layer's frame-count choices, per action


@dataclass(frozen=True)
class PolicySettings:
    width: int = 96
    heads: int = 4
    scene_depth: int = 2
    layer_depth: int = 1
    memory_width: int = 128
    # Each position number also enters as sine and cosine waves of this many
    # wavelengths, halving from 512 pixels (8 waves: down to 4 pixels).
    position_frequencies: int = 8


# ── Scene encoder ─────────────────────────────────────────────────────────────

# Which of each part's numbers (smb_observer.pack_c) are positions: Mario's box
# on the screen; every other box relative to Mario; a surface's left and right
# edge and its top relative to his feet; a gap's two edges.
POSITION_COLUMNS = {
    "mario": (1, 2, 3, 4),
    "target": (1, 2, 3, 4),
    "enemies": (1, 2, 3, 4),
    "coins": (1, 2, 3, 4),
    "power_ups": (1, 2, 3, 4),
    "moving_platforms": (1, 2, 3, 4),
    "pipes": (1, 2, 3, 4),
    "blocks": (1, 2, 3, 4),
    "surfaces": (1, 2, 3),
    "gaps": (1, 2),
}


class PositionWaves(nn.Module):
    """A part's numbers, with each position number also given as sine and cosine
    waves of several wavelengths.

    Positions are fractions of the screen (1 pixel is about 0.004), so on their
    own a pixel's difference is a tiny change of input. As waves of wavelength
    2, 1, 1/2, ... screen widths (512 pixels down to 4 pixels for 8 waves),
    nearby positions differ clearly in the short waves while the long waves
    still tell far from near.
    """

    def __init__(self, columns, frequencies: int):
        super().__init__()
        self.register_buffer("columns", torch.tensor(columns, dtype=torch.long), persistent=False)
        self.register_buffer(
            "scales",
            math.pi * 2.0 ** torch.arange(frequencies, dtype=torch.float32),
            persistent=False,
        )

    def extra_width(self) -> int:
        return 2 * len(self.columns) * len(self.scales)

    def forward(self, numbers):
        angles = numbers[..., self.columns].unsqueeze(-1) * self.scales
        return torch.cat((numbers, angles.sin().flatten(-2), angles.cos().flatten(-2)), dim=-1)


class SceneEncoder(nn.Module):
    """One PolicyInput (batched rows) -> object tokens, a padding mask and a summary."""

    def __init__(self, settings: PolicySettings):
        super().__init__()
        width = settings.width
        self.width = width
        waves = {
            name: PositionWaves(columns, settings.position_frequencies)
            for name, columns in POSITION_COLUMNS.items()
        }
        self.waves = nn.ModuleDict(waves)
        self.mario = nn.Linear(MARIO_WIDTH + waves["mario"].extra_width(), width)
        self.target = nn.Linear(TARGET_WIDTH + waves["target"].extra_width(), width)
        self.lists = nn.ModuleDict(
            {
                name: nn.Linear(_SLOT_WIDTH[name] + waves[name].extra_width(), width)
                for name in SCENE_SLOTS
            }
        )
        self.slot_position = nn.ParameterDict(
            {name: nn.Parameter(torch.zeros(SCENE_SLOTS[name], width)) for name in SCENE_SLOTS}
        )
        self.codes = nn.Embedding(len(COLUMN_CODES), width)
        self.band_position = nn.Parameter(torch.zeros(SEQ_LEN_A + SEQ_LEN_B, width))
        self.summary_token = nn.Parameter(torch.zeros(1, 1, width))
        layer = nn.TransformerEncoderLayer(
            width, settings.heads, width * 2, dropout=0.0, batch_first=True, norm_first=True
        )
        self.encoder = nn.TransformerEncoder(
            layer, settings.scene_depth, enable_nested_tensor=False
        )
        self.norm = nn.LayerNorm(width)
        # Token index of each pointer target (TARGETS), for the skill layer.
        offsets, start = {}, 3  # summary, Mario, target
        for name in SCENE_SLOTS:
            offsets[name] = start
            start += SCENE_SLOTS[name]
        self.target_token_index = torch.tensor(
            [offsets[name] + slot for name, slot in TARGETS], dtype=torch.long
        )
        # Tokens per scene: summary, Mario, target, every list slot, every band.
        self.token_count = start + SEQ_LEN_A + SEQ_LEN_B

    def _objects(self, src_c, with_target: bool = True):
        """Summary, Mario, target and every list slot of a C row, as tokens and a mask."""
        batch = src_c.shape[0]
        if src_c.shape[1] != SEQ_LEN_C:
            raise ValueError(f"src_c must have {SEQ_LEN_C} numbers, got {src_c.shape[1]}")
        everyone = torch.ones(batch, 1, dtype=torch.bool, device=src_c.device)
        tokens, present = [self.summary_token.expand(batch, -1, -1)], [everyone]
        mario = src_c[:, slice(*C_SPANS["c_mario"])]
        tokens.append(self.mario(self.waves["mario"](mario)).unsqueeze(1))
        present.append(everyone)
        target = src_c[:, slice(*C_SPANS["c_target"])]
        tokens.append(self.target(self.waves["target"](target)).unsqueeze(1))
        present.append((target[:, :1] > 0.5) & with_target)
        for name in SCENE_SLOTS:
            slots = src_c[:, slice(*C_SPANS[f"c_{name}"])].view(batch, SCENE_SLOTS[name], -1)
            tokens.append(self.lists[name](self.waves[name](slots)) + self.slot_position[name])
            present.append(slots[..., 0] > 0.5)
        return tokens, present

    def forward(self, src_a, src_b, src_c):
        tokens, present = self._objects(src_c)
        codes = torch.cat((src_a, src_b), dim=1).long()
        tokens.append(self.codes(codes) + self.band_position)
        present.append(torch.ones(codes.shape, dtype=torch.bool, device=src_c.device))
        tokens, present = torch.cat(tokens, dim=1), torch.cat(present, dim=1)
        return self.norm(self.encoder(tokens, src_key_padding_mask=~present)), present

    def expected(self, src_c):
        """A predicted C row (the memory's expected scene) -> tokens and a mask, encoded
        like a seen scene: summary, Mario and every list slot the prediction says is
        present. It has no screen bands and no skill target."""
        tokens, present = self._objects(src_c, with_target=False)
        tokens, present = torch.cat(tokens, dim=1), torch.cat(present, dim=1)
        return self.norm(self.encoder(tokens, src_key_padding_mask=~present)), present


# ── Memory: what the world will look like when the action ends ────────────────


@dataclass
class MemoryState:
    hidden: torch.Tensor
    cell: torch.Tensor

    def detach(self) -> "MemoryState":
        return MemoryState(self.hidden.detach(), self.cell.detach())


class Memory(nn.Module):
    """A long short-term memory network that steps once per action, from vision only.

    At the start of each action, before anything decides, it takes the latest
    picture's scene (the scene encoder's summary of the frame the action starts
    on) and updates its state, which carries everything earlier. It is never
    told what Mario will do. From that state its expectation part predicts the world when the coming
    action ends: the scene numbers (smb_observer C row) the vision transformer
    will report on that frame. That expected scene is what the decision layers
    read beside the current one.
    """

    def __init__(self, settings: PolicySettings):
        super().__init__()
        self.width = settings.memory_width
        self.cell = nn.LSTM(settings.width, settings.memory_width, batch_first=True)
        self.expectation = nn.Sequential(
            nn.Linear(settings.memory_width, settings.memory_width * 2),
            nn.GELU(),
            nn.Linear(settings.memory_width * 2, SEQ_LEN_C),
        )

    def initial(self, batch: int, device) -> MemoryState:
        zeros = torch.zeros(batch, self.width, device=device)
        return MemoryState(zeros, zeros.clone())

    def forward(self, scene_summary, state: MemoryState) -> MemoryState:
        """One action's start: scene_summary [B, width]."""
        _, (hidden, cell) = self.cell(
            scene_summary.unsqueeze(1), (state.hidden.unsqueeze(0), state.cell.unsqueeze(0))
        )
        return MemoryState(hidden[0], cell[0])

    def sequence(self, scene_summaries) -> torch.Tensor:
        """Every action of whole episodes from their start: scene_summaries
        [B, K, width]. Returns the state after each action's start, [B, K,
        memory_width]. Padding after an episode's last action does not change
        its earlier steps."""
        hidden, _ = self.cell(scene_summaries)
        return hidden

    def expected_scene(self, hidden) -> torch.Tensor:
        """[..., memory_width] -> the scene numbers expected when the action ends."""
        return self.expectation(hidden)


# ── Layers ────────────────────────────────────────────────────────────────────


class _Layer(nn.Module):
    """The current scene's tokens + the expected scene's tokens (the memory's
    prediction of the world when the coming action ends) + the token from above
    -> a decision token."""

    def __init__(self, settings: PolicySettings, above_width: int, outputs: Mapping[str, int]):
        super().__init__()
        width = settings.width
        # Marks the expected scene's tokens apart from the current scene's.
        self.expected_marker = nn.Parameter(torch.zeros(1, 1, width))
        self.above = nn.Linear(above_width, width) if above_width else None
        self.query = nn.Parameter(torch.zeros(1, 1, width))
        layer = nn.TransformerEncoderLayer(
            width, settings.heads, width * 2, dropout=0.0, batch_first=True, norm_first=True
        )
        self.blocks = nn.TransformerEncoder(layer, settings.layer_depth, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(width)
        self.heads = nn.ModuleDict({name: nn.Linear(width, size) for name, size in outputs.items()})
        self.value = nn.Linear(width, 1)

    def encode(self, scene_tokens, present, expected, above=None):
        """Returns the decision token and the current scene's tokens after the blocks."""
        batch = scene_tokens.shape[0]
        expected_tokens, expected_present = expected
        extra = [self.query.expand(batch, -1, -1)]
        if self.above is not None:
            extra.append(self.above(above).unsqueeze(1))
        tokens = torch.cat((*extra, scene_tokens, expected_tokens + self.expected_marker), dim=1)
        mask = torch.cat(
            (
                torch.ones(batch, len(extra), dtype=torch.bool, device=present.device),
                present,
                expected_present,
            ),
            dim=1,
        )
        out = self.norm(self.blocks(tokens, src_key_padding_mask=~mask))
        seen = slice(len(extra), len(extra) + scene_tokens.shape[1])
        return out[:, 0], out[:, seen]

    def forward(self, scene_tokens, present, expected, above=None) -> dict[str, torch.Tensor]:
        decision, _ = self.encode(scene_tokens, present, expected, above)
        out = {name: head(decision) for name, head in self.heads.items()}
        # The estimate of reward to come reads the decision without shaping it.
        out["value"] = self.value(decision.detach()).squeeze(-1)
        return out


class SkillLayer(_Layer):
    """Also points at its target: one of the scene's surfaces, enemies or moving platforms."""

    def __init__(self, settings: PolicySettings):
        super().__init__(
            settings, TACTIC_WIDTH, {"skill": len(SKILLS), "direction": 2, "contact": 1}
        )
        self.pointer_key = nn.Linear(settings.width, settings.width)
        self.no_target = nn.Parameter(torch.zeros(1))

    def forward(self, scene_tokens, present, expected, above=None, target_index=None):
        decision, encoded = self.encode(scene_tokens, present, expected, above)
        out = {name: head(decision) for name, head in self.heads.items()}
        # The estimate of reward to come reads the decision without shaping it.
        out["value"] = self.value(decision.detach()).squeeze(-1)
        keys = self.pointer_key(encoded[:, target_index])
        scores = torch.einsum("bd,btd->bt", decision, keys) / keys.shape[-1] ** 0.5
        scores = scores.masked_fill(~present[:, target_index], float("-inf"))
        out["pointer"] = torch.cat((scores, self.no_target.expand(scores.shape[0], 1)), dim=1)
        return out


class LayeredSMBPolicy(nn.Module):
    """The four layers, the shared scene encoder and the memory.

    When the executor's plan has ended: remember() steps the memory with the
    latest picture, expect() gives its expected scene, and the layers run
    top-down (smb_agent.decide) to choose the next ActionPlan.
    """

    def __init__(self, settings: PolicySettings = PolicySettings()):
        super().__init__()
        self.settings = settings
        self.scene = SceneEncoder(settings)
        self.memory = Memory(settings)
        self.strategy = _Layer(settings, 0, {"strategy": len(STRATEGIES), "direction": 2})
        self.tactic = _Layer(settings, STRATEGY_WIDTH, {"tactic": len(TACTICS), "direction": 2})
        self.skill = SkillLayer(settings)
        self.action = _Layer(
            settings,
            SKILL_WIDTH,
            {"action": len(SMB_ACTIONS), "frames": len(SMB_ACTIONS) * FRAME_BINS},
        )
        # Set once the strategy layer has learned from Full SMB play.
        self.register_buffer("strategy_trained", torch.zeros((), dtype=torch.bool))

    # The parts each training stage changes (everything else stays frozen).
    def parameters_of(self, layer: str):
        if layer == "action":
            modules = (self.scene, self.memory, self.action)
        else:
            modules = (getattr(self, layer),)
        return [p for module in modules for p in module.parameters()]

    def remember(self, now, state: Optional[MemoryState]) -> MemoryState:
        """Step the memory at the start of an action, before anything decides.

        ``now``: the encoded scene (encode_scene) of the frame the action starts
        on; the memory takes its summary token. ``state``: the memory after the
        previous action (None at an episode's start).
        """
        summary = now[0][:, 0]
        if state is None:
            state = self.memory.initial(summary.shape[0], summary.device)
        return self.memory(summary, state)

    def expect(self, state: MemoryState):
        """The memory's expected scene at the end of the coming action: its C row
        numbers and its encoding for the decision layers."""
        numbers = self.memory.expected_scene(state.hidden)
        return numbers, self.scene.expected(numbers)

    def encode_scene(self, inputs):
        """Batched PolicyInput rows (src_a, src_b, src_c) -> scene tokens and their mask."""
        return self.scene(*inputs)

    def run_layer(self, layer: str, encoded, expected, above=None):
        """One layer's raw outputs from the encoded current scene, the encoded
        expected scene and the (encoded) token from the layer above (none for
        the strategy layer)."""
        scene_tokens, present = encoded
        if layer == "skill":
            index = self.scene.target_token_index.to(scene_tokens.device)
            return self.skill(scene_tokens, present, expected, above, target_index=index)
        return getattr(self, layer)(scene_tokens, present, expected, above)

    def layer_outputs(self, inputs, expected, tokens_above: Mapping[str, torch.Tensor]):
        """Raw outputs of each layer whose token from above is given (encoded).

        ``expected``: the encoded expected scene (expect). The strategy layer
        needs no token and always runs; the tactic layer runs given "strategy",
        the skill layer given "tactic", the action layer given "skill". Tokens
        may come from a teacher (training) or from the layer above.
        """
        encoded = self.encode_scene(inputs)
        out = {"strategy": self.run_layer("strategy", encoded, expected)}
        for layer, above in (("tactic", "strategy"), ("skill", "tactic"), ("action", "skill")):
            if above in tokens_above:
                out[layer] = self.run_layer(layer, encoded, expected, tokens_above[above])
        return out


# ── Choosing from the outputs ─────────────────────────────────────────────────
#
# Each layer's choice is a few categorical picks, named by its heads:
# strategy (strategy, direction), tactic (tactic, direction), skill (skill,
# direction, contact, pointer) and action (action, frames: the frame-count bin
# on the chosen action's menu). ``choose`` picks them for one picture, the most
# likely or sampled; ``choice_log_prob`` scores picks for a batch (training).

CHOICES = {
    "strategy": ("strategy", "direction"),
    "tactic": ("tactic", "direction"),
    "skill": ("skill", "direction", "contact", "pointer"),
    "action": ("action", "frames"),
}


def _distributions(layer: str, out: Mapping[str, torch.Tensor], chosen_action=None) -> dict:
    """Batched distributions of a layer's picks; ``frames`` needs the chosen actions."""
    found = {}
    for head in CHOICES[layer]:
        if head == "contact":
            found[head] = torch.distributions.Bernoulli(logits=out["contact"].squeeze(-1))
        elif head == "frames":
            rows = out["frames"].view(-1, len(SMB_ACTIONS), FRAME_BINS)
            index = torch.arange(rows.shape[0], device=rows.device)
            found[head] = torch.distributions.Categorical(logits=rows[index, chosen_action])
        else:
            found[head] = torch.distributions.Categorical(logits=out[head])
    return found


def choose(layer: str, out: Mapping[str, torch.Tensor], *, sample: bool = False):
    """One picture's raw layer outputs -> (its token or ActionPlan, the picks)."""
    batched = {name: value.unsqueeze(0) for name, value in out.items()}
    picks: dict[str, int] = {}
    for head in CHOICES[layer]:
        action = torch.tensor([picks["action"]]) if head == "frames" else None
        if action is not None:
            action = action.to(out["frames"].device)
        distribution = _distributions(layer, batched, action)[head]
        if sample:
            value = distribution.sample()
        elif head == "contact":
            value = (distribution.logits > 0).float()
        else:
            value = distribution.logits.argmax(-1)
        picks[head] = int(value.item())
    return token_from_picks(layer, picks), picks


def token_from_picks(layer: str, picks: Mapping[str, int]):
    direction = (-1, 1)[picks["direction"]] if "direction" in picks else 1
    if layer == "strategy":
        return StrategyToken(STRATEGIES[picks["strategy"]], direction)
    if layer == "tactic":
        return TacticToken(TACTICS[picks["tactic"]], direction)
    if layer == "skill":
        pointer = picks["pointer"]
        return SkillToken(
            SKILLS[picks["skill"]],
            direction,
            bool(picks["contact"]),
            None if pointer == NO_TARGET else TARGETS[pointer],
        )
    return ActionPlan(picks["action"], FRAME_COUNTS[picks["frames"]])


def choice_log_prob(layer: str, out: Mapping[str, torch.Tensor], picks: Mapping[str, torch.Tensor]):
    """Log-probability [B] of batched picks under a layer's outputs, and the entropy [B]."""
    distributions = _distributions(layer, out, picks.get("action"))
    log_prob = entropy = 0.0
    for head in CHOICES[layer]:
        value = picks[head].float() if head == "contact" else picks[head].long()
        log_prob = log_prob + distributions[head].log_prob(value)
        entropy = entropy + distributions[head].entropy()
    return log_prob, entropy


def strategy_token(out) -> StrategyToken:
    return choose("strategy", out)[0]


def tactic_token(out) -> TacticToken:
    return choose("tactic", out)[0]


def skill_token(out) -> SkillToken:
    return choose("skill", out)[0]


def action_plan(out, *, sample: bool = False) -> ActionPlan:
    return choose("action", out, sample=sample)[0]


__all__ = [
    "DEFAULT_STRATEGY",
    "FRAME_BINS",
    "LAYERS",
    "LayeredSMBPolicy",
    "Memory",
    "MemoryState",
    "PolicySettings",
    "SceneEncoder",
    "TARGET_LISTS",
    "CHOICES",
    "action_plan",
    "choice_log_prob",
    "choose",
    "encode_skill",
    "encode_strategy",
    "encode_tactic",
    "skill_token",
    "strategy_token",
    "tactic_token",
]
