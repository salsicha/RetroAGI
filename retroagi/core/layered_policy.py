"""Two learned layers: tactic and skill, followed by predictive control.

Vision feeds the skill transformer, alongside its categorical tactic, the
strategy switch, its last 16 actions and the executor's feedback; it reads
no prediction (user, 2026-10-15). The skill chooses an action: a verb, then
the object it is aimed at, one of the objects the vision reports
(core/action_tokens). In Block SMB training the teacher carries the action
out; without the teacher the world model (``memory``, core/world_model),
run once at the action's start, predicts where it ends, and that end is the
predictive executor's destination.

Tactics remain persistent options with a termination head, critic, strategy
switch and their own recurrent context. The executor chooses buttons with per-frame visual feedback.
"""

import math
from dataclasses import dataclass
from typing import Mapping, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .action_tokens import (
    ACTION_WIDTH,
    POINTER_TOKENS,
    VERBS,
    ActionToken,
    allowed_lists,
    encode_action,
    pointer_index,
    pointer_target,
)
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
from .smb_kinematics import MARIO_STATE
from .smb_scene_labels import OBJECT_TYPE_SLOTS
from .world_model import SPRITE_MOTION, WorldModel, add_motion
from .tokens import (
    DEFAULT_STRATEGY,
    EXECUTION_WIDTH,
    STRATEGY_WIDTH,
    TACTIC_WIDTH,
    TACTICS,
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
CHOICE_WIDTH = {"tactic": TACTIC_WIDTH, "skill": ACTION_WIDTH}
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
    if layer == "skill":
        # (the action, or None for a command given without one; the
        # executor's destination)
        return encode_action(*choice)
    return encode_tactic(choice)


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
    # Dropout in the decision layers' transformer blocks (skill and tactic),
    # active only while training: it keeps them from leaning on single inputs.
    decision_dropout: float = 0.1


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
        # Each enemy's object type as vision tells it (smb_observer's
        # "c_enemy_types": its number plus one, 0 not told): what each number
        # means is learned; it starts with no influence.
        self.enemy_type = nn.Embedding(OBJECT_TYPE_SLOTS + 1, width)
        nn.init.zeros_(self.enemy_type.weight)
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
            token = self.lists[name](self.waves[name](slots)) + self.slot_position[name]
            if name == "enemies":
                told = src_c[:, slice(*C_SPANS["c_enemy_types"])].round().long()
                token = token + self.enemy_type(told.clamp(0, OBJECT_TYPE_SLOTS))
            tokens.append(token)
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


# The first object slot's scene token (action_tokens.pointer_index).
POINTER_START = 2


# ── Layers ────────────────────────────────────────────────────────────────────


class _Layer(nn.Module):
    """A transformer whose context holds the current scene's tokens, the token
    from above, and its own last HISTORY choices, each marked with how many
    decisions ago it was made -> a decision token. It reads no prediction: the
    world model predicts the outcome of what it chooses, after it chooses
    (user, 2026-10-15)."""

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
        self.above = nn.Linear(above_width, width) if above_width else None
        self.query = nn.Parameter(torch.zeros(1, 1, width))
        layer = nn.TransformerEncoderLayer(
            width,
            settings.heads,
            width * 2,
            dropout=settings.decision_dropout,
            batch_first=True,
            norm_first=True,
        )
        self.blocks = nn.TransformerEncoder(layer, settings.layer_depth, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(width)
        self.heads = nn.ModuleDict({name: nn.Linear(width, size) for name, size in outputs.items()})
        self.value = nn.Linear(width, 1)
        # Motion (user, 2026-10-21): the executor's estimates from what vision
        # reported over the frames before (each sprite's velocity, Mario's),
        # added to the scene's sprite and Mario tokens, as the world model
        # reads them; starting with no influence.
        self.sprite_motion = nn.Linear(SPRITE_MOTION.stop - SPRITE_MOTION.start, width)
        self.mario_motion = nn.Linear(len(MARIO_STATE), width)
        for layer in (self.sprite_motion, self.mario_motion):
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)

    def encode(self, scene_tokens, present, above=None, history=None, more=(), motion=None):
        """Returns the decision token and the current scene's tokens after the blocks.

        ``history``: (choices [B, HISTORY, choice width], present [B, HISTORY]),
        the layer's own previous choices, the most recent first; None for none.
        ``more``: further (tokens [B, n, width], present [B, n]) groups.
        ``motion``: (sprite kinematics [B, SPRITE_SLOTS, 6], Mario's [B, 3]),
        added to their tokens; None for none.
        """
        if motion is not None:
            scene_tokens = add_motion(scene_tokens, *motion, self.sprite_motion, self.mario_motion)
        batch = scene_tokens.shape[0]
        extra = [self.query.expand(batch, -1, -1)]
        if self.above is not None:
            extra.append(self.above(above).unsqueeze(1))
        parts = [*extra, scene_tokens]
        masks = [
            torch.ones(batch, len(extra), dtype=torch.bool, device=present.device),
            present,
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

    def forward(self, scene_tokens, present, above=None, history=None) -> dict[str, torch.Tensor]:
        decision, _ = self.encode(scene_tokens, present, above, history)
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
    """The skill layer: it chooses an action, a verb (run, jump or wait) and
    then, for a run or a jump, the object it ends on (core/action_tokens).

    The verb first (a run or a jump with nothing in view it can end on is
    excluded); then the pointer, reading the chosen verb: the decision
    token, adjusted by the verb, scored against each object's scene token
    after the blocks (absent objects, and kinds the verb cannot aim at, are
    excluded; a wait aims at nothing). Where exactly the action ends and how
    long it takes (the executor's destination) is the teacher's ground truth
    in training, the world model's prediction later; the layer reads
    neither.
    """

    def __init__(self, settings: PolicySettings, above_width: int):
        super().__init__(
            settings,
            above_width,
            {"verb": len(VERBS)},
            ACTION_WIDTH,
            choice_positions=(len(VERBS), len(VERBS) + 1),
        )
        width = settings.width
        self.given_verb = _Given(width, len(VERBS))
        self.pointer_query = nn.Linear(width, width)
        self.pointer_key = nn.Linear(width, width)
        allowed = torch.zeros(len(VERBS), POINTER_TOKENS, dtype=torch.bool)
        for v, verb in enumerate(VERBS):
            for name in allowed_lists(verb):
                start = pointer_index((name, 0)) - POINTER_START
                allowed[v, start : start + SCENE_SLOTS[name]] = True
        self.register_buffer("allowed", allowed, persistent=False)

    def verb_logits(self, decision, present):
        """The verb's logits [B, len(VERBS)]; ``present`` [B, POINTER_TOKENS]:
        the objects in view."""
        usable = (present[:, None, :] & self.allowed[None]).any(-1)
        usable[:, VERBS.index("wait")] = True
        return self.heads["verb"](decision).masked_fill(~usable, -1e4)

    def pointer_logits(self, decision, objects, present, verb):
        """The pointer's logits [B, POINTER_TOKENS] over the object tokens
        ``objects`` [B, POINTER_TOKENS, width] (the scene's, after the
        blocks), for the chosen verbs [B]: absent objects, and kinds the verb
        cannot aim at, are excluded (a wait aims at nothing: all excluded)."""
        chosen = F.one_hot(verb.long(), len(VERBS)).to(decision.dtype)
        query = self.pointer_query(self.given_verb(decision, chosen))
        logits = torch.einsum("bw,bnw->bn", query, self.pointer_key(objects))
        logits = logits / math.sqrt(query.shape[-1])
        usable = present & self.allowed[verb.long()]
        return logits.masked_fill(~usable, -1e4)

    def heads_given(self, decision, picks, objects, present) -> dict:
        """The logits of both heads read from ``decision`` [B, width] with the
        verb of ``picks`` (index tensors [B]) as already chosen; ``objects`` and
        ``present``: the forward's "objects" and "object_present"."""
        return {
            "verb": self.verb_logits(decision, present),
            "pointer": self.pointer_logits(decision, objects, present, picks["verb"]),
        }

    def forward(
        self, scene_tokens, present, above=None, history=None, picks=None, sample=False, motion=None
    ) -> dict[str, torch.Tensor]:
        """Logits of each head, the pointer's for the verb chosen (or, given
        ``picks`` as index tensors [B] with a verb, for that one: teaching with
        the teacher's choice, or scoring the layer's own); "picks", the index
        tensors used; "decision", the decision token; "objects" and
        "object_present", the object tokens read; and the value estimate."""
        decision, seen = self.encode(scene_tokens, present, above, history, motion=motion)
        span = slice(POINTER_START, POINTER_START + POINTER_TOKENS)
        objects, object_present = seen[:, span], present[:, span]
        verb_logits = self.verb_logits(decision, object_present)
        verb = (
            picks["verb"] if picks is not None and "verb" in picks else _pick(verb_logits, sample)
        )
        pointer_logits = self.pointer_logits(decision, objects, object_present, verb)
        pointer = (
            picks["pointer"]
            if picks is not None and "pointer" in picks
            else _pick(pointer_logits, sample)
        )
        return {
            "verb": verb_logits,
            "pointer": pointer_logits,
            "picks": {"verb": verb, "pointer": pointer},
            "decision": decision,
            "objects": objects,
            "object_present": object_present,
            "value": self.value(decision.detach()).squeeze(-1),
        }


class TacticMemory(nn.Module):
    """The tactic layer's own long short-term memory network: it steps once per
    tactic, when a tactic starts (at an episode's start, and whenever the held
    tactic ends).

    Each step takes the picture (the scene encoder's summary of the frame the
    new tactic starts on) and the tactic that just ended and how long it
    lasted (held_features; "none" at an episode's start). From its state it
    predicts the scene when the coming tactic will end. It is never told
    which tactic comes next.
    """

    def __init__(self, settings: PolicySettings):
        super().__init__()
        self.width = settings.tactic_memory_width
        self.inputs = nn.Linear(settings.width + HELD_WIDTH, settings.width)
        self.cell = nn.LSTM(settings.width, self.width, batch_first=True)
        self.expectation = nn.Sequential(
            nn.Linear(self.width, self.width * 2),
            nn.GELU(),
            nn.Linear(self.width * 2, SEQ_LEN_C),
        )

    def initial(self, batch: int, device) -> MemoryState:
        zeros = torch.zeros(batch, self.width, device=device)
        return MemoryState(zeros, zeros.clone())

    def step_inputs(self, scene_summary, ended):
        """[B, width]: one step's input from the picture's summary [B, width] and
        the tactic that ended [B, HELD_WIDTH]."""
        return self.inputs(torch.cat((scene_summary, ended), dim=-1))

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

    Its context: the current scene's tokens, the strategy switch, the tactic it
    holds and for how long (held_features; "none" when choosing), its memory's
    state, and that memory's expected scene when the current tactic ends.
    Outputs, per picture:

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
        # How the last action went (the executor's report, as the skill reads
        # it; user, 2026-10-21), read with the held tactic; starting with no
        # influence.
        self.feedback = nn.Linear(EXECUTION_WIDTH, width)
        nn.init.zeros_(self.feedback.weight)
        nn.init.zeros_(self.feedback.bias)
        # Marks the tokens of the scene expected when the tactic ends.
        self.end_marker = nn.Parameter(torch.zeros(1, 1, width))

    def forward(
        self, scene_tokens, present, switch, held, recalled, expected_end, motion=None, feedback=None
    ):
        """``held`` [B, HELD_WIDTH]; ``recalled`` [B, tactic memory width] (the
        tactic memory's state); ``expected_end``: the encoded scene it expects
        when the tactic ends (tokens, present); ``motion``: the kinematics
        (_Layer.encode); ``feedback`` [B, EXECUTION_WIDTH]: how the last action
        went (None: none)."""
        batch = scene_tokens.shape[0]
        everyone = torch.ones(batch, 1, dtype=torch.bool, device=present.device)
        end_tokens, end_present = expected_end
        held_token = self.held(held)
        if feedback is not None:
            held_token = held_token + self.feedback(feedback)
        decision, _ = self.encode(
            scene_tokens,
            present,
            switch,
            motion=motion,
            more=(
                (held_token.unsqueeze(1), everyone),
                (self.recalled(recalled).unsqueeze(1), everyone),
                (end_tokens + self.end_marker, end_present),
            ),
        )
        out = {name: head(decision) for name, head in self.heads.items()}
        out["end"] = out["end"].squeeze(-1)
        out["values"] = out.pop("worth")
        return out


class LayeredSMBPolicy(nn.Module):
    """The two learned layers, the scene encoder, the tactic memory, and the
    world model's LSTM (``memory``).

    When the executor's plan ends, the tactic layer checks its held tactic
    (and, when it ends, recall() steps the tactic memory and the layer
    chooses the next), and the skill layer chooses the next action. Neither
    reads a prediction (user, 2026-10-15): the world model runs once at the
    start of an action, after it is chosen, to predict its outcome. It is
    trained from recorded ground truth and not used while the skills and
    tactics train; until then every "prediction" is the ground truth.
    """

    def __init__(self, settings: PolicySettings = PolicySettings()):
        super().__init__()
        self.settings = settings
        self.scene = SceneEncoder(settings)
        # The world model (core/world_model): run once at the start of each
        # action, after the skill picks it; not used while the layers train.
        self.memory = WorldModel(settings.width, settings.memory_width, settings.heads)
        self.tactic = TacticLayer(settings)
        self.tactic_memory = TacticMemory(settings)
        self.skill = SkillLayer(settings, TACTIC_WIDTH + STRATEGY_WIDTH + EXECUTION_WIDTH)

    def parameters_of(self, layer: str):
        modules = {
            "skill": (self.scene, self.skill),
            "tactic": (self.tactic, self.tactic_memory),
        }[layer]
        return [p for module in modules for p in module.parameters()]

    def recall(self, now, ended, state: Optional[MemoryState]) -> MemoryState:
        """Step the tactic memory when a tactic starts.

        ``now``: the encoded scene of the frame it starts on (its summary token
        is taken); ``ended`` [B, HELD_WIDTH]: the tactic that just ended and how
        long it was held (held_features); ``state``: the tactic memory before
        (None at an episode's start).
        """
        summary = now[0][:, 0]
        if state is None:
            state = self.tactic_memory.initial(summary.shape[0], summary.device)
        return self.tactic_memory(self.tactic_memory.step_inputs(summary, ended), state)

    def tactic_outputs(self, encoded, switch, held, recalled, motion=None, feedback=None):
        """The tactic layer's outputs (TacticLayer) from the encoded scene, the
        encoded switch [B, STRATEGY_WIDTH], the held tactic [B, HELD_WIDTH],
        the tactic memory's hidden state [B, tactic memory width], the
        kinematics (``motion``) and how the last action went (``feedback``)."""
        scene_tokens, present = encoded
        expected_end = self.scene.expected(self.tactic_memory.expected_scene(recalled))
        return self.tactic(
            scene_tokens, present, switch, held, recalled, expected_end, motion=motion, feedback=feedback
        )

    def encode_scene(self, inputs):
        """Batched PolicyInput rows (src_a, src_b, src_c) -> scene tokens and their mask."""
        return self.scene(*inputs)

    def run_skill(
        self,
        encoded,
        tactic,
        history=None,
        strategy=None,
        feedback=None,
        picks=None,
        sample=False,
        motion=None,
    ):
        """Choose an action using the tactic and its strategy context.

        Training records pack both tokens together. Direct callers may pass
        them separately; the default switch preserves the old convenience API.
        ``picks``: the verb (and pointer) (index tensors [B]) to read as
        chosen, instead of choosing them (most likely, or sampled when
        ``sample``); see SkillLayer.
        """
        scene_tokens, present = encoded
        if tactic.shape[-1] == TACTIC_WIDTH:
            if strategy is None:
                strategy = encode_strategy(DEFAULT_STRATEGY).to(tactic).expand(len(tactic), -1)
            tactic = torch.cat((tactic, strategy), dim=-1)
        if feedback is None:
            feedback = tactic.new_zeros((len(tactic), EXECUTION_WIDTH))
        tactic = torch.cat((tactic, feedback), dim=-1)
        return self.skill(scene_tokens, present, tactic, history, picks=picks, sample=sample, motion=motion)


# ── Choosing from the outputs ─────────────────────────────────────────────────
#
# Each learned layer emits categorical token fields.

CHOICES = {
    "tactic": ("tactic",),
    # The verb, then the pointer: an object slot (0 for the first: scene
    # token POINTER_START).
    "skill": ("verb", "pointer"),
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
        verb = VERBS[picks["verb"]]
        if verb == "wait":
            return ActionToken("wait")
        return ActionToken(verb, pointer_target(picks["pointer"] + POINTER_START))
    raise ValueError(f"unknown learned layer {layer!r}")


def choice_log_prob(layer: str, out: Mapping[str, torch.Tensor], picks: Mapping[str, torch.Tensor]):
    """Log-probability [B] of batched picks under a layer's outputs, and the entropy [B].

    For the skill, ``out`` must be read with these picks as chosen (the
    pointer's logits are then those for the picked verb, and the sum is the
    whole action's log-probability); a wait's pointer is not a choice.
    """
    distributions = _distributions(layer, out)
    log_prob = entropy = 0.0
    for head in CHOICES[layer]:
        value = picks[head].long()
        head_log_prob = distributions[head].log_prob(value)
        head_entropy = distributions[head].entropy()
        if layer == "skill" and head == "pointer":
            aimed = (picks["verb"].long() != VERBS.index("wait")).to(head_log_prob.dtype)
            head_log_prob, head_entropy = head_log_prob * aimed, head_entropy * aimed
        log_prob = log_prob + head_log_prob
        entropy = entropy + head_entropy
    return log_prob, entropy


__all__ = [
    "DEFAULT_STRATEGY",
    "HELD_WIDTH",
    "LAYERS",
    "LayeredSMBPolicy",
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
