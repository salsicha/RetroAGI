"""Three learned layers: tactic, spatial skill, and destination-only action.

Vision and an LSTM scene prediction feed the skill transformer, alongside its
categorical tactic and the last 16 spatial commands. The skill emits a run,
jump or hold command with a destination relative to Mario's feet. Only this
command reaches the action network; it has no scene or memory input.

Tactics remain persistent options with a termination head, critic, strategy
switch and their own recurrent context. The executor applies timed button
plans and uses per-frame vision for closed-loop hold-ground control.
"""

import math
from dataclasses import dataclass
from typing import Mapping, Optional

import torch
import torch.nn as nn

from .smb_executor import EXECUTOR_ACTIONS, FRAME_COUNTS, ActionPlan
from .smb_observer import (
    _SLOT_WIDTH,
    C_SPANS,
    COLUMN_CODES,
    MARIO_WIDTH,
    SCENE_SLOTS,
    SEQ_LEN_A,
    SEQ_LEN_B,
    SEQ_LEN_C,
)
from .tokens import (
    DEFAULT_STRATEGY,
    SKILL_MODES,
    SKILL_WIDTH,
    SKILL_X,
    SKILL_Y,
    STRATEGY_WIDTH,
    TACTIC_WIDTH,
    TACTICS,
    SkillToken,
    encode_skill,
    encode_strategy,
    encode_tactic,
)

# The learned layers, top-down; the strategy above them is a switch.
LAYERS = ("tactic", "skill", "action")
# The layers whose context holds their own last HISTORY choices (the tactic
# layer has its own memory network instead).
HISTORY_LAYERS = ("skill",)
FRAME_BINS = len(FRAME_COUNTS)  # the action layer's frame-count choices, per action
# Each layer's context holds its own last HISTORY choices (the ones used).
HISTORY = 16
ACTION_WIDTH = len(EXECUTOR_ACTIONS) + FRAME_BINS
CHOICE_WIDTH = {"tactic": TACTIC_WIDTH, "skill": SKILL_WIDTH, "action": ACTION_WIDTH}
# A held (or just ended) tactic as numbers: one-hot tactic with a last slot for
# "none", then how long it has been held: log(1 + actions) / log(1 + 64) and
# log(1 + frames) / log(1 + 1024).
HELD_WIDTH = len(TACTICS) + 1 + 2


def held_features(stance: Optional[str], actions: int = 0, frames: int = 0) -> torch.Tensor:
    """[HELD_WIDTH]: a tactic (None: none) and how many actions and frames it was held."""
    vector = torch.zeros(HELD_WIDTH)
    vector[len(TACTICS) if stance is None else TACTICS.index(stance)] = 1.0
    vector[-2] = math.log1p(actions) / math.log1p(64)
    vector[-1] = math.log1p(frames) / math.log1p(1024)
    return vector


def encode_plan(plan: ActionPlan) -> torch.Tensor:
    """[ACTION_WIDTH]: one-hot button action, then one-hot frame count (1 to 32)."""
    vector = torch.zeros(ACTION_WIDTH)
    vector[plan.action] = 1.0
    vector[len(EXECUTOR_ACTIONS) + FRAME_COUNTS.index(plan.frames)] = 1.0
    return vector


def encode_choice(layer: str, choice) -> torch.Tensor:
    """A layer's choice (token or ActionPlan) as numbers: what its history holds."""
    return {"tactic": encode_tactic, "skill": encode_skill, "action": encode_plan}[layer](choice)


@dataclass(frozen=True)
class PolicySettings:
    width: int = 96
    heads: int = 4
    scene_depth: int = 2
    layer_depth: int = 1
    memory_width: int = 128
    tactic_memory_width: int = 128
    # Each position number also enters as sine and cosine waves of this many
    # wavelengths, halving from 512 pixels (8 waves: down to 4 pixels).
    position_frequencies: int = 8


# ── Scene encoder ─────────────────────────────────────────────────────────────

# Which of each part's numbers (smb_observer.pack_c) are positions: Mario's box
# on the screen; every other box relative to Mario; a surface's left and right
# edge and its top relative to his feet; a gap's two edges.
POSITION_COLUMNS = {
    "mario": (1, 2, 3, 4),
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
        # Tokens per scene: summary, Mario, every list slot, every band.
        self.token_count = 2 + sum(SCENE_SLOTS.values()) + SEQ_LEN_A + SEQ_LEN_B

    def _objects(self, src_c):
        """Summary, Mario and every list slot of a C row, as tokens and a mask."""
        batch = src_c.shape[0]
        if src_c.shape[1] != SEQ_LEN_C:
            raise ValueError(f"src_c must have {SEQ_LEN_C} numbers, got {src_c.shape[1]}")
        everyone = torch.ones(batch, 1, dtype=torch.bool, device=src_c.device)
        tokens, present = [self.summary_token.expand(batch, -1, -1)], [everyone]
        mario = src_c[:, slice(*C_SPANS["c_mario"])]
        tokens.append(self.mario(self.waves["mario"](mario)).unsqueeze(1))
        present.append(everyone)
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
        present. It has no screen bands."""
        tokens, present = self._objects(src_c)
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
    """A transformer whose context holds the current scene's tokens, the expected
    scene's tokens (the memory's prediction of the world when the coming action
    ends), the token from above, and its own last HISTORY choices, each marked
    with how many decisions ago it was made -> a decision token."""

    def __init__(
        self,
        settings: PolicySettings,
        above_width: int,
        outputs: Mapping[str, int],
        choice_width: int,
        choice_positions: tuple = (),
    ):
        super().__init__()
        width = settings.width
        # Its own previous choices (any position numbers in them also as sine and
        # cosine waves), and how many decisions ago each was made; none when
        # ``choice_width`` is 0.
        if choice_width:
            self.history_waves = PositionWaves(choice_positions, settings.position_frequencies)
            self.history = nn.Linear(choice_width + self.history_waves.extra_width(), width)
            self.history_age = nn.Parameter(torch.zeros(HISTORY, width))
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

    def encode(self, scene_tokens, present, expected, above=None, history=None, more=()):
        """Returns the decision token and the current scene's tokens after the blocks.

        ``history``: (choices [B, HISTORY, choice width], present [B, HISTORY]),
        the layer's own previous choices, the most recent first; None for none.
        ``more``: further (tokens [B, n, width], present [B, n]) groups.
        """
        batch = scene_tokens.shape[0]
        expected_tokens, expected_present = expected
        extra = [self.query.expand(batch, -1, -1)]
        if self.above is not None:
            extra.append(self.above(above).unsqueeze(1))
        parts = [*extra, scene_tokens, expected_tokens + self.expected_marker]
        masks = [
            torch.ones(batch, len(extra), dtype=torch.bool, device=present.device),
            present,
            expected_present,
        ]
        if history is not None:
            choices, made = history
            parts.append(self.history(self.history_waves(choices)) + self.history_age)
            masks.append(made)
        for tokens, made in more:
            parts.append(tokens)
            masks.append(made)
        tokens = torch.cat(parts, dim=1)
        mask = torch.cat(masks, dim=1)
        out = self.norm(self.blocks(tokens, src_key_padding_mask=~mask))
        seen = slice(len(extra), len(extra) + scene_tokens.shape[1])
        return out[:, 0], out[:, seen]

    def forward(
        self, scene_tokens, present, expected, above=None, history=None
    ) -> dict[str, torch.Tensor]:
        decision, _ = self.encode(scene_tokens, present, expected, above, history)
        out = {name: head(decision) for name, head in self.heads.items()}
        # The estimate of reward to come reads the decision without shaping it.
        out["value"] = self.value(decision.detach()).squeeze(-1)
        return out


class TacticMemory(nn.Module):
    """The tactic layer's own long short-term memory network: it steps once per
    tactic, when a tactic starts (at an episode's start, and whenever the held
    tactic ends).

    Each step takes the picture (the scene encoder's summary of the frame the
    new tactic starts on), the tactic that just ended and how long it lasted
    (held_features; "none" at an episode's start), and the action memory's
    state then, which carries what happened while that tactic was held. From
    its state it predicts the scene when the coming tactic will end. It is
    never told which tactic comes next.
    """

    def __init__(self, settings: PolicySettings):
        super().__init__()
        self.width = settings.tactic_memory_width
        self.inputs = nn.Linear(settings.width + HELD_WIDTH + settings.memory_width, settings.width)
        self.cell = nn.LSTM(settings.width, self.width, batch_first=True)
        self.expectation = nn.Sequential(
            nn.Linear(self.width, self.width * 2),
            nn.GELU(),
            nn.Linear(self.width * 2, SEQ_LEN_C),
        )

    def initial(self, batch: int, device) -> MemoryState:
        zeros = torch.zeros(batch, self.width, device=device)
        return MemoryState(zeros, zeros.clone())

    def step_inputs(self, scene_summary, ended, action_memory):
        """[B, width]: one step's input from the picture's summary [B, width], the
        tactic that ended [B, HELD_WIDTH] and the action memory's state [B,
        memory_width]."""
        return self.inputs(torch.cat((scene_summary, ended, action_memory), dim=-1))

    def forward(self, inputs, state: MemoryState) -> MemoryState:
        """One tactic's start: ``inputs`` [B, width] (step_inputs)."""
        _, (hidden, cell) = self.cell(
            inputs.unsqueeze(1), (state.hidden.unsqueeze(0), state.cell.unsqueeze(0))
        )
        return MemoryState(hidden[0], cell[0])

    def sequence(self, inputs) -> torch.Tensor:
        """Every tactic start of whole episodes: inputs [B, K, width] -> the state
        after each, [B, K, tactic memory width]."""
        hidden, _ = self.cell(inputs)
        return hidden

    def expected_scene(self, hidden) -> torch.Tensor:
        """[..., tactic memory width] -> the scene numbers expected when the tactic ends."""
        return self.expectation(hidden)


class TacticLayer(_Layer):
    """The option-critic tactic layer.

    Its context: the current scene's tokens, the action memory's expected scene
    (end of the coming action), the strategy switch, the tactic it holds and
    for how long (held_features; "none" when choosing), its memory's state, and
    that memory's expected scene when the current tactic ends. Outputs, per
    picture:

    - ``tactic`` [7]: scores of the tactics to choose, when choosing;
    - ``end`` []: the end check, the log-odds that the held tactic is finished;
    - ``values`` [7]: the critic, for each tactic the reward expected from
      holding it from here (rewards scaled as in training).
    """

    def __init__(self, settings: PolicySettings):
        super().__init__(
            settings,
            STRATEGY_WIDTH,
            # ("worth" is returned as "values"; a module dictionary has a values().)
            {"tactic": len(TACTICS), "end": 1, "worth": len(TACTICS)},
            0,
        )
        width = settings.width
        self.held = nn.Linear(HELD_WIDTH, width)
        self.recalled = nn.Linear(settings.tactic_memory_width, width)
        # Marks the tokens of the scene expected when the tactic ends.
        self.end_marker = nn.Parameter(torch.zeros(1, 1, width))

    def forward(self, scene_tokens, present, expected, switch, held, recalled, expected_end):
        """``held`` [B, HELD_WIDTH]; ``recalled`` [B, tactic memory width] (the
        tactic memory's state); ``expected_end``: the encoded scene it expects
        when the tactic ends (tokens, present)."""
        batch = scene_tokens.shape[0]
        everyone = torch.ones(batch, 1, dtype=torch.bool, device=present.device)
        end_tokens, end_present = expected_end
        decision, _ = self.encode(
            scene_tokens,
            present,
            expected,
            switch,
            more=(
                (self.held(held).unsqueeze(1), everyone),
                (self.recalled(recalled).unsqueeze(1), everyone),
                (end_tokens + self.end_marker, end_present),
            ),
        )
        out = {name: head(decision) for name, head in self.heads.items()}
        out["end"] = out["end"].squeeze(-1)
        out["values"] = out.pop("worth")
        return out


class DestinationActionLayer(nn.Module):
    """Map a relative destination to a plan, without vision or recurrent inputs."""

    def __init__(self, settings):
        super().__init__()
        self.waves = PositionWaves(
            (SKILL_WIDTH - 2, SKILL_WIDTH - 1), settings.position_frequencies
        )
        self.network = nn.Sequential(
            nn.Linear(SKILL_WIDTH + self.waves.extra_width(), settings.width),
            nn.GELU(),
            nn.Linear(settings.width, settings.width),
            nn.GELU(),
        )
        self.heads = nn.ModuleDict(
            {
                "action": nn.Linear(settings.width, len(EXECUTOR_ACTIONS)),
                "frames": nn.Linear(settings.width, len(EXECUTOR_ACTIONS) * FRAME_BINS),
            }
        )
        self.value = nn.Linear(settings.width, 1)

    def forward(self, destination):
        hidden = self.network(self.waves(destination))
        out = {name: head(hidden) for name, head in self.heads.items()}
        out["value"] = self.value(hidden.detach()).squeeze(-1)
        return out


class LayeredSMBPolicy(nn.Module):
    """The three layers, scene encoder and two memories.

    When the executor's plan has ended: remember() steps the action memory with
    the latest picture, expect() gives its expected scene, the tactic layer
    checks its held tactic (and, when it ends, recall() steps the tactic memory
    and the layer chooses the next), and the action layer runs (smb_agent) to
    choose the next ActionPlan.
    """

    def __init__(self, settings: PolicySettings = PolicySettings()):
        super().__init__()
        self.settings = settings
        self.scene = SceneEncoder(settings)
        self.memory = Memory(settings)
        self.tactic = TacticLayer(settings)
        self.tactic_memory = TacticMemory(settings)
        self.skill = _Layer(
            settings,
            TACTIC_WIDTH,
            {"mode": len(SKILL_MODES), "x": len(SKILL_X), "y": len(SKILL_Y)},
            SKILL_WIDTH,
            choice_positions=(len(SKILL_MODES), len(SKILL_MODES) + 1),
        )
        self.action = DestinationActionLayer(settings)

    def parameters_of(self, layer: str):
        modules = {
            "action": (self.action,),
            "skill": (self.scene, self.memory, self.skill),
            "tactic": (self.tactic, self.tactic_memory),
        }[layer]
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

    def recall(self, now, ended, action_memory, state: Optional[MemoryState]) -> MemoryState:
        """Step the tactic memory when a tactic starts.

        ``now``: the encoded scene of the frame it starts on (its summary token
        is taken); ``ended`` [B, HELD_WIDTH]: the tactic that just ended and how
        long it was held (held_features); ``action_memory`` [B, memory width]:
        the action memory's state now; ``state``: the tactic memory before (None
        at an episode's start).
        """
        summary = now[0][:, 0]
        if state is None:
            state = self.tactic_memory.initial(summary.shape[0], summary.device)
        return self.tactic_memory(
            self.tactic_memory.step_inputs(summary, ended, action_memory), state
        )

    def tactic_outputs(self, encoded, expected, switch, held, recalled):
        """The tactic layer's outputs (TacticLayer) from the encoded scene, the
        action memory's encoded expected scene, the encoded switch [B,
        STRATEGY_WIDTH], the held tactic [B, HELD_WIDTH] and the tactic memory's
        hidden state [B, tactic memory width]."""
        scene_tokens, present = encoded
        expected_end = self.scene.expected(self.tactic_memory.expected_scene(recalled))
        return self.tactic(scene_tokens, present, expected, switch, held, recalled, expected_end)

    def encode_scene(self, inputs):
        """Batched PolicyInput rows (src_a, src_b, src_c) -> scene tokens and their mask."""
        return self.scene(*inputs)

    def run_skill(self, encoded, expected, tactic, history=None):
        """Choose a destination from vision, memory, tactic and 16 prior choices."""
        scene_tokens, present = encoded
        return self.skill(scene_tokens, present, expected, tactic, history)

    def run_action(self, destination):
        """Only the skill's spatial command reaches this network."""
        return self.action(destination)


# ── Choosing from the outputs ─────────────────────────────────────────────────
#
# Each layer's choice is a few categorical picks, named by its heads: tactic
# (tactic; its direction follows from it) and action (action, frames: the
# frame-count bin on the chosen action's menu). ``choose`` picks them for one picture, the most likely or
# sampled; ``choice_log_prob`` scores picks for a batch (training).

CHOICES = {
    "tactic": ("tactic",),
    "action": ("action", "frames"),
    "skill": ("mode", "x", "y"),
}


def _distributions(layer: str, out: Mapping[str, torch.Tensor], chosen_action=None) -> dict:
    """Batched distributions of a layer's picks; ``frames`` needs the chosen actions."""
    found = {}
    for head in CHOICES[layer]:
        if head == "frames":
            rows = out["frames"].view(-1, len(EXECUTOR_ACTIONS), FRAME_BINS)
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
        else:
            value = distribution.logits.argmax(-1)
        picks[head] = int(value.item())
    return token_from_picks(layer, picks), picks


def token_from_picks(layer: str, picks: Mapping[str, int]):
    """The token or plan picked; a tactic is its name (its direction follows
    from it, tokens.tactic_token)."""
    if layer == "tactic":
        return TACTICS[picks["tactic"]]
    if layer == "skill":
        return SkillToken(SKILL_MODES[picks["mode"]], SKILL_X[picks["x"]], SKILL_Y[picks["y"]])
    return ActionPlan(picks["action"], FRAME_COUNTS[picks["frames"]])


def choice_log_prob(layer: str, out: Mapping[str, torch.Tensor], picks: Mapping[str, torch.Tensor]):
    """Log-probability [B] of batched picks under a layer's outputs, and the entropy [B]."""
    distributions = _distributions(layer, out, picks.get("action"))
    log_prob = entropy = 0.0
    for head in CHOICES[layer]:
        value = picks[head].long()
        log_prob = log_prob + distributions[head].log_prob(value)
        entropy = entropy + distributions[head].entropy()
    return log_prob, entropy


def action_plan(out, *, sample: bool = False) -> ActionPlan:
    return choose("action", out, sample=sample)[0]


__all__ = [
    "DEFAULT_STRATEGY",
    "FRAME_BINS",
    "HELD_WIDTH",
    "LAYERS",
    "LayeredSMBPolicy",
    "Memory",
    "MemoryState",
    "PolicySettings",
    "SceneEncoder",
    "CHOICES",
    "TacticLayer",
    "TacticMemory",
    "action_plan",
    "choice_log_prob",
    "choose",
    "encode_strategy",
    "encode_tactic",
    "held_features",
]
