"""Two learned layers: tactic and spatial skill, followed by predictive control.

Vision and an LSTM scene prediction feed the skill transformer, alongside its
categorical tactic and the last 16 spatial commands. The skill emits a run,
jump or hold command with a destination relative to Mario's feet, chosen in
order: the mode, then the destination's x knowing the mode, then its y knowing
both. This command goes directly to the visual predictive executor.

Tactics remain persistent options with a termination head, critic, strategy
switch and their own recurrent context. The executor chooses buttons with per-frame visual feedback.
"""

import math
from dataclasses import dataclass
from typing import Mapping, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

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
    EXECUTION_WIDTH,
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
LAYERS = ("tactic", "skill")
# The layers whose context holds their own last HISTORY choices (the tactic
# layer has its own memory network instead).
HISTORY_LAYERS = ("skill",)
# Each layer's context holds its own last HISTORY choices (the ones used).
HISTORY = 16
MEMORY_INTERVAL = 4
CHOICE_WIDTH = {"tactic": TACTIC_WIDTH, "skill": SKILL_WIDTH}
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


def encode_choice(layer: str, choice) -> torch.Tensor:
    """A layer's token as numbers: what its history holds."""
    return {"tactic": encode_tactic, "skill": encode_skill}[layer](choice)


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
    """One scene LSTM, refreshed every four frames and at skill decisions.

    The same hidden state predicts the next action endpoint: its scene,
    platform and enemy world displacements, and remaining duration in physical
    frames. No second recurrent model is used. Elapsed time and observed camera scroll distinguish
    object motion from camera motion and irregular decision intervals.
    """

    def __init__(self, settings: PolicySettings):
        super().__init__()
        self.width = settings.memory_width
        self.cell = nn.LSTM(settings.width, settings.memory_width, batch_first=True)
        self.timing = nn.Linear(2, settings.width, bias=False)
        nn.init.zeros_(self.timing.weight)
        self.platform_prediction = _forecast_readout(
            settings.memory_width, SCENE_SLOTS["moving_platforms"]
        )
        # Where each enemy seen now will be when the next action ends.
        self.enemy_prediction = _forecast_readout(settings.memory_width, SCENE_SLOTS["enemies"])
        self.end_time_embedding = nn.Linear(2, settings.width, bias=False)
        nn.init.zeros_(self.end_time_embedding.weight)
        self.end_duration = nn.Linear(settings.memory_width, 2)
        nn.init.zeros_(self.end_duration.weight)
        nn.init.constant_(self.end_duration.bias, math.log(math.expm1(31 / 32)))
        self.expectation = nn.Sequential(
            nn.Linear(settings.memory_width, settings.memory_width * 2),
            nn.GELU(),
            nn.Linear(settings.memory_width * 2, SEQ_LEN_C),
        )

    def initial(self, batch: int, device) -> MemoryState:
        zeros = torch.zeros(batch, self.width, device=device)
        return MemoryState(zeros, zeros.clone())

    def forward(self, scene_summary, state: MemoryState, timing=None) -> MemoryState:
        """One observation tick: scene_summary [B, width], optional elapsed/scroll."""
        if timing is not None:
            scene_summary = scene_summary + self.timing(timing)
        _, (hidden, cell) = self.cell(
            scene_summary.unsqueeze(1), (state.hidden.unsqueeze(0), state.cell.unsqueeze(0))
        )
        return MemoryState(hidden[0], cell[0])

    def sequence(self, scene_summaries, timing=None) -> torch.Tensor:
        """Every memory tick of whole episodes from their start: scene_summaries
        [B, K, width]. Returns the state after each action's start, [B, K,
        memory_width]. Padding after an episode's last action does not change
        its earlier steps."""
        if timing is not None:
            scene_summaries = scene_summaries + self.timing(timing)
        hidden, _ = self.cell(scene_summaries)
        return hidden

    def action_end(self, hidden):
        """Predict time until the next action ends, not a preselected horizon.

        Positive continuous frame count and uncertainty, with no upper duration
        cap. Round only at consumption if an integer frame index is needed.
        """
        values = 1 + torch.nn.functional.softplus(self.end_duration(hidden)) * 32
        return {"frames": values[..., 0], "sigma": values[..., 1]}

    def platforms(self, hidden):
        """Platform displacement at the SAME predicted next-action endpoint.

        Dimensions: [..., current platform slot, (x,y)]. The duration output is
        shared with the next-scene prediction; this is not a multi-horizon head.
        """
        return self._forecast(self.platform_prediction, SCENE_SLOTS["moving_platforms"], hidden)

    def enemies(self, hidden):
        """Each enemy's displacement at the same next-action endpoint, as platforms."""
        return self._forecast(self.enemy_prediction, SCENE_SLOTS["enemies"], hidden)

    def _forecast(self, readout, slots, hidden):
        raw = readout(hidden).view(*hidden.shape[:-1], slots, 5)
        endpoint = self.action_end(hidden)
        return {
            "frames": endpoint["frames"],
            "frame_sigma": endpoint["sigma"],
            "displacement": raw[..., :2] * 64,
            "sigma": raw[..., 2:4].clamp(0, math.log(64)).exp(),
            "visible": raw[..., 4],
        }

    def expected_scene(self, hidden) -> torch.Tensor:
        """[..., memory_width] -> the scene numbers expected when the action ends."""
        return self.expectation(hidden)


def _forecast_readout(width, slots):
    """Per slot: displacement (x, y), log uncertainty (x, y), visibility logit.

    Starts at no displacement, 32-pixel uncertainty and unlikely visibility:
    an untrained forecast is never confident enough to be used.
    """
    readout = nn.Linear(width, slots * 5)
    nn.init.zeros_(readout.weight)
    with torch.no_grad():
        bias = readout.bias.view(-1, 5)
        bias.zero_()
        bias[:, 2:4] = math.log(32)
        bias[:, 4] = -2
    return readout


# Where the enemy slots sit among the scene tokens: after the summary and Mario.
ENEMY_TOKENS = slice(2, 2 + SCENE_SLOTS["enemies"])
FORECAST_WIDTH = 5


def forecast_features(forecast):
    """[..., slots, FORECAST_WIDTH] numbers from a memory forecast: displacement
    and uncertainty in screen widths, and the chance the object is still seen."""
    return torch.cat(
        (
            forecast["displacement"] / 64,
            forecast["sigma"] / 64,
            forecast["visible"].sigmoid().unsqueeze(-1),
        ),
        dim=-1,
    )


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


class _Given(nn.Module):
    """The decision token adjusted by what the layer has already chosen: the
    token plus a small network of (token, chosen). It starts as the token
    unchanged (its last weights are zero)."""

    def __init__(self, width: int, given_width: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(width + given_width, width), nn.GELU(), nn.Linear(width, width)
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, decision, given):
        return decision + self.net(torch.cat((decision, given), dim=-1))


def _pick(logits, sample: bool):
    """[B] indices: sampled from the logits, or the most likely."""
    if sample:
        return torch.distributions.Categorical(logits=logits).sample()
    return logits.argmax(-1)


class SkillLayer(_Layer):
    """The skill layer: it chooses its command in order.

    First the mode (run, jump or hold); then the destination's x, reading the
    chosen mode; then its y, reading the mode and x. Choosing the three apart
    could pair one command's mode with another's x (a run to where a jump over
    an enemy lands), or one command's x with another's y (a stomp's x with a
    jump over's height).
    """

    def __init__(self, settings: PolicySettings, above_width: int):
        super().__init__(
            settings,
            above_width,
            {"mode": len(SKILL_MODES), "x": len(SKILL_X), "y": len(SKILL_Y)},
            SKILL_WIDTH,
            choice_positions=(len(SKILL_MODES), len(SKILL_MODES) + 1),
        )
        width = settings.width
        # The chosen x as a position number (x / 256) with its waves.
        self.chosen_x_waves = PositionWaves((len(SKILL_MODES),), settings.position_frequencies)
        self.given_mode = _Given(width, len(SKILL_MODES))
        self.given_mode_and_x = _Given(
            width, len(SKILL_MODES) + 1 + self.chosen_x_waves.extra_width()
        )

    def x_logits(self, decision, mode):
        """x's logits [B, len(SKILL_X)] for the chosen modes [B] (indices)."""
        chosen = F.one_hot(mode.long(), len(SKILL_MODES)).to(decision.dtype)
        return self.heads["x"](self.given_mode(decision, chosen))

    def y_logits(self, decision, mode, x):
        """y's logits [B, len(SKILL_Y)] for the chosen modes and x [B] (indices)."""
        chosen = F.one_hot(mode.long(), len(SKILL_MODES)).to(decision.dtype)
        pixels = (x.to(decision.dtype) + SKILL_X[0]).unsqueeze(-1) / 256
        numbers = self.chosen_x_waves(torch.cat((chosen, pixels), dim=-1))
        return self.heads["y"](self.given_mode_and_x(decision, numbers))

    def heads_given(self, decision, picks) -> dict:
        """The outputs read from ``decision`` [B, width] with the mode and x of
        ``picks`` (index tensors [B]) as already chosen: logits of every head."""
        return {
            "mode": self.heads["mode"](decision),
            "x": self.x_logits(decision, picks["mode"]),
            "y": self.y_logits(decision, picks["mode"], picks["x"]),
        }

    def forward(
        self, scene_tokens, present, expected, above=None, history=None, picks=None, sample=False
    ) -> dict[str, torch.Tensor]:
        """Logits of each head, conditioned on the picks it chose (or, given
        ``picks`` as index tensors [B] with mode and x, on those: teaching with
        the teacher's choice, or scoring the layer's own); "picks", the index
        tensors used; "decision", the decision token; and the value estimate."""
        decision, _ = self.encode(scene_tokens, present, expected, above, history)
        if picks is None:
            mode_logits = self.heads["mode"](decision)
            mode = _pick(mode_logits, sample)
            x_logits = self.x_logits(decision, mode)
            x = _pick(x_logits, sample)
            y_logits = self.y_logits(decision, mode, x)
            picks = {"mode": mode, "x": x, "y": _pick(y_logits, sample)}
            out = {"mode": mode_logits, "x": x_logits, "y": y_logits}
        else:
            out = self.heads_given(decision, picks)
        out["picks"] = picks
        out["decision"] = decision
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


class LayeredSMBPolicy(nn.Module):
    """The two learned layers, scene encoder and two memories.

    remember() steps scene memory periodically and at decisions. When the
    executor's plan ends, expect() gives its expected scene, the tactic layer
    checks its held tactic (and, when it ends, recall() steps the tactic memory
    and the layer chooses the next), and the skill layer chooses the executor's next destination.
    """

    def __init__(self, settings: PolicySettings = PolicySettings()):
        super().__init__()
        self.settings = settings
        self.scene = SceneEncoder(settings)
        self.memory = Memory(settings)
        self.tactic = TacticLayer(settings)
        self.tactic_memory = TacticMemory(settings)
        self.skill = SkillLayer(settings, TACTIC_WIDTH + STRATEGY_WIDTH + EXECUTION_WIDTH)
        # Each enemy's forecast (where it will be when the next action ends) is
        # added to that enemy's scene token. Starts with no influence.
        self.skill.forecast = nn.Linear(FORECAST_WIDTH, settings.width)
        nn.init.zeros_(self.skill.forecast.weight)
        nn.init.zeros_(self.skill.forecast.bias)

    def parameters_of(self, layer: str):
        modules = {
            "skill": (self.scene, self.memory, self.skill),
            "tactic": (self.tactic, self.tactic_memory),
        }[layer]
        return [p for module in modules for p in module.parameters()]

    def remember(self, now, state: Optional[MemoryState], timing=None) -> MemoryState:
        """Step scene memory on a periodic observation or decision boundary.

        ``now``: the encoded scene (encode_scene) of the frame the action starts
        on; the memory takes its summary token. ``state``: the memory after the
        previous action (None at an episode's start).
        """
        summary = now[0][:, 0]
        if state is None:
            state = self.memory.initial(summary.shape[0], summary.device)
        return self.memory(summary, state, timing)

    def expect(self, state: MemoryState):
        """The memory's expected scene at the end of the coming action: its C row
        numbers and its encoding for the decision layers."""
        numbers = self.memory.expected_scene(state.hidden)
        tokens, present = self.scene.expected(numbers)
        endpoint = self.memory.action_end(state.hidden)
        timing = torch.stack((endpoint["frames"], endpoint["sigma"]), dim=-1)
        tokens = tokens.clone()
        tokens[:, 0] += self.memory.end_time_embedding(torch.log1p(timing / 32))
        return numbers, (tokens, present)

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

    def run_skill(
        self,
        encoded,
        expected,
        tactic,
        history=None,
        strategy=None,
        feedback=None,
        memory=None,
        picks=None,
        sample=False,
    ):
        """Choose a destination using the tactic and its strategy context.

        Training records pack both tokens together. Direct callers may pass
        them separately; the default switch preserves the old convenience API.
        ``memory``: the scene memory's hidden state at the decision [B, memory
        width]; each enemy's forecast is then added to its scene token.
        ``picks``: the mode and x (index tensors [B]) to read as chosen, instead
        of choosing them (most likely, or sampled when ``sample``); see
        SkillLayer.
        """
        scene_tokens, present = encoded
        if memory is not None:
            features = forecast_features(self.memory.enemies(memory)).detach()
            scene_tokens = scene_tokens.clone()
            scene_tokens[:, ENEMY_TOKENS] = scene_tokens[:, ENEMY_TOKENS] + self.skill.forecast(
                features
            )
        if tactic.shape[-1] == TACTIC_WIDTH:
            if strategy is None:
                strategy = encode_strategy(DEFAULT_STRATEGY).to(tactic).expand(len(tactic), -1)
            tactic = torch.cat((tactic, strategy), dim=-1)
        if feedback is None:
            feedback = tactic.new_zeros((len(tactic), EXECUTION_WIDTH))
        tactic = torch.cat((tactic, feedback), dim=-1)
        return self.skill(
            scene_tokens, present, expected, tactic, history, picks=picks, sample=sample
        )


# ── Choosing from the outputs ─────────────────────────────────────────────────
#
# Each learned layer emits categorical token fields.

CHOICES = {
    "tactic": ("tactic",),
    "skill": ("mode", "x", "y"),
}


def _distributions(layer: str, out: Mapping[str, torch.Tensor]) -> dict:
    return {head: torch.distributions.Categorical(logits=out[head]) for head in CHOICES[layer]}


def choose(layer: str, out: Mapping[str, torch.Tensor], *, sample: bool = False):
    """One picture's raw layer outputs -> (its token, the picks).

    A layer that chose in order (the skill: its outputs carry "picks") has
    already chosen; those picks are its choice.
    """
    if "picks" in out:
        picks = {head: int(out["picks"][head]) for head in CHOICES[layer]}
        return token_from_picks(layer, picks), picks
    batched = {name: value.unsqueeze(0) for name, value in out.items()}
    picks: dict[str, int] = {}
    for head in CHOICES[layer]:
        distribution = _distributions(layer, batched)[head]
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
    raise ValueError(f"unknown learned layer {layer!r}")


def choice_log_prob(layer: str, out: Mapping[str, torch.Tensor], picks: Mapping[str, torch.Tensor]):
    """Log-probability [B] of batched picks under a layer's outputs, and the entropy [B].

    For the skill, ``out`` must be read with these picks as chosen (each
    head's logits are then conditional on the heads before it, and the sum is
    the whole command's log-probability).
    """
    distributions = _distributions(layer, out)
    log_prob = entropy = 0.0
    for head in CHOICES[layer]:
        value = picks[head].long()
        log_prob = log_prob + distributions[head].log_prob(value)
        entropy = entropy + distributions[head].entropy()
    return log_prob, entropy


__all__ = [
    "DEFAULT_STRATEGY",
    "HELD_WIDTH",
    "LAYERS",
    "LayeredSMBPolicy",
    "Memory",
    "MemoryState",
    "PolicySettings",
    "SceneEncoder",
    "SkillLayer",
    "CHOICES",
    "TacticLayer",
    "TacticMemory",
    "choice_log_prob",
    "choose",
    "encode_strategy",
    "encode_tactic",
    "held_features",
]
