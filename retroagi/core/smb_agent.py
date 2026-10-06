"""Vision-only agent: strategy -> tactic -> spatial skill -> action -> executor.

At each action boundary, memory reads the scene, the option-critic checks the
held tactic, and skill chooses a relative destination from scene, prediction,
tactic and its 16 previous commands. Action reads only that spatial command.
Execution uses per-frame vision for motion, destination and hold feedback.
Landing and destination arrival can end a plan early, and waits are rechecked
each frame. Simulator state is never passed to these networks.

Training may supply teacher tokens for selected layers; runtime playback
uses only the policy's own choices and the externally set strategy switch.
"""

from dataclasses import dataclass, field
from typing import Callable, Mapping, Optional, Sequence

import numpy as np
import torch

from .actions import SMBAction
from .layered_policy import (
    CHOICE_WIDTH,
    CHOICES,
    HISTORY,
    HISTORY_LAYERS,
    LayeredSMBPolicy,
    MemoryState,
    choice_log_prob,
    choose,
    encode_choice,
    held_features,
)
from .smb_executor import ActionPlan, SMBExecutor
from .smb_observer import SEQ_LEN_A, SEQ_LEN_B, VisionObserver, column_codes, pack_c
from .smb_scene_labels import SceneObservation
from .smb_spatial_feedback import SpatialFeedback
from .tokens import (
    DEFAULT_STRATEGY,
    SkillToken,
    StrategyToken,
    TacticToken,
    encode_skill,
    encode_strategy,
    encode_tactic,
    tactic_token,
)

NOOP = int(SMBAction.NOOP)


@dataclass
class TacticStep:
    """What the tactic layer did at one decision point (when it ran)."""

    held: Optional[str]  # the tactic held coming in (None at an episode's start)
    actions: int  # how many actions it had been held
    frames: int  # ... and how many frames
    end_probability: Optional[float]  # its end check's chance that the held tactic is finished
    own_end: bool  # whether the layer itself ended it (always, with none held)
    own: str  # the tactic the layer itself would use now: its held one or its new choice
    started: bool  # the tactic used from here started here (the tactic memory stepped)
    choice_log_prob: float  # the log-probability of its new choice (0 when it chose none)


@dataclass
class Decision:
    """What the layers decided for one copy at one decision point."""

    strategy: StrategyToken  # the copy's strategy switch
    tactic: Optional[TacticToken]
    plan: ActionPlan
    chosen: dict  # the policy's own choice at each layer it ran, before any given token
    picks: dict = field(default_factory=dict)  # layer -> its own picks (layered_policy.choose)
    log_prob: dict = field(default_factory=dict)  # layer -> log-probability of its own picks
    value: dict = field(default_factory=dict)  # layer -> its estimate of the return to come
    tactic_step: Optional[TacticStep] = None  # when the tactic layer ran
    skill: Optional[SkillToken] = None


def scene_rows(scenes: Sequence[SceneObservation]) -> list[tuple]:
    """(src_a, src_b, src_c) for each scene."""
    return [(column_codes(s, SEQ_LEN_A), column_codes(s, SEQ_LEN_B), pack_c(s)) for s in scenes]


def stack_rows(rows: Sequence[tuple], device):
    a, b, c = zip(*rows)
    return (
        torch.as_tensor(np.stack(a), dtype=torch.long, device=device),
        torch.as_tensor(np.stack(b), dtype=torch.long, device=device),
        torch.as_tensor(np.stack(c), dtype=torch.float32, device=device),
    )


def _one(out: Mapping[str, torch.Tensor], index: int) -> dict:
    return {name: value[index] for name, value in out.items()}


def _encoded(encode, tokens, device) -> torch.Tensor:
    return torch.stack([encode(token) for token in tokens]).to(device)


@torch.no_grad()
def decide(
    policy: LayeredSMBPolicy,
    scenes: Sequence[SceneObservation],
    rows: Sequence[tuple],
    expected,
    device,
    switches: Sequence[StrategyToken],
    tactics: Sequence[Optional[TacticToken]],
    given: Optional[Mapping[str, Sequence]] = None,
    run_given: Sequence[str] = (),
    sample: Sequence[str] = (),
    encoded_scene=None,
    histories: Optional[Mapping[str, tuple]] = None,
    tactic_steps: Optional[Sequence[Optional[TacticStep]]] = None,
) -> list[Decision]:
    """Skill destinations and action plans for pictures under ``tactics``
    (each held by the tactic layer, or given).

    ``expected``: the action memory's expected scene at the end of the coming
    action, encoded (LayeredSMBPolicy.expect). ``encoded_scene``: the pictures'
    scenes already encoded, when the caller has them. ``histories``: the
    skill's own previous destinations (choice_histories).

    ``given["action"][i]``, when not None, replaces the policy's ActionPlan for
    picture i: training only. When it is given for every picture the action
    layer does not run, unless "action" is listed in ``run_given`` (to compare
    its choice with the given one). With "action" in ``sample`` it samples its
    choice (training by reward); otherwise it takes the most likely.
    """
    count = len(scenes)
    given = given or {}

    def missing(layer: str) -> bool:
        values = given.get(layer)
        return values is None or any(value is None for value in values)

    def runs(layer: str) -> bool:
        return layer in run_given or missing(layer)

    def replaced(layer: str, chosen: Sequence) -> list:
        values = given.get(layer)
        return [
            chosen[i] if values is None or values[i] is None else values[i] for i in range(count)
        ]

    plain = stack_rows(rows, device)
    cache: dict = {} if encoded_scene is None else {"plain": encoded_scene}

    def encoded():
        if "plain" not in cache:
            cache["plain"] = policy.encode_scene(plain)
        return cache["plain"]

    chosen: dict[str, list] = {}
    picked: dict[str, list] = {}
    scores: dict[str, tuple] = {}

    def run(layer: str, above) -> None:
        out = (
            policy.run_action(above)
            if layer == "action"
            else policy.run_skill(
                encoded(),
                expected,
                above,
                (histories or {}).get("skill"),
                strategy=_encoded(encode_strategy, switches, device),
            )
        )
        made = [choose(layer, _one(out, i), sample=layer in sample) for i in range(count)]
        chosen[layer] = [token for token, _ in made]
        picked[layer] = [picks for _, picks in made]
        stacked = {
            head: torch.tensor([picks[head] for picks in picked[layer]], device=device)
            for head in CHOICES[layer]
        }
        log_prob, _ = choice_log_prob(layer, out, stacked)
        scores[layer] = (log_prob.tolist(), out["value"].tolist())

    if (runs("action") or "skill" in run_given) and runs("skill"):
        run("skill", _encoded(encode_tactic, tactics, device))
    destinations = replaced("skill", chosen.get("skill", [None] * count))
    if runs("action"):
        run("action", _encoded(encode_skill, destinations, device))
    plans = replaced("action", chosen.get("action", [None] * count))
    steps = list(tactic_steps) if tactic_steps is not None else [None] * count
    decisions = []
    for i in range(count):
        own = {layer: values[i] for layer, values in chosen.items()}
        if steps[i] is not None:
            own["tactic"] = tactic_token(steps[i].own)
        decisions.append(
            Decision(
                strategy=switches[i],
                tactic=tactics[i],
                plan=plans[i],
                chosen=own,
                picks={layer: values[i] for layer, values in picked.items()},
                log_prob={layer: values[0][i] for layer, values in scores.items()},
                value={layer: values[1][i] for layer, values in scores.items()},
                tactic_step=steps[i],
                skill=destinations[i],
            )
        )
    return decisions


@dataclass
class AgentStep:
    """One copy's frame: what it saw, why its plan ended, what it decided and pressed."""

    scene: SceneObservation
    rows: tuple
    ended: Optional[str]
    decision: Optional[Decision]
    button: int


@dataclass
class LandingWatch:
    """Watches the vision transformer's land detector for Mario landing, the one
    event that ends an action early: its "feet on something" output (the ground,
    a moving platform or an enemy) turns on after it was off. Pictures without
    Mario change nothing."""

    airborne: bool = False

    def landed(self, scene: SceneObservation) -> bool:
        if scene.mario.box is None:
            return False
        landed = self.airborne and scene.mario.on_something
        self.airborne = not scene.mario.on_something
        return landed


@dataclass
class _Copy:
    executor: SMBExecutor = field(default_factory=SMBExecutor)
    landing: LandingWatch = field(default_factory=LandingWatch)
    spatial: SpatialFeedback = field(default_factory=SpatialFeedback)
    hidden: Optional[torch.Tensor] = None
    cell: Optional[torch.Tensor] = None
    button: int = NOOP
    # Each layer's choices used at earlier decisions, the most recent first.
    choices: dict = field(default_factory=lambda: {layer: [] for layer in HISTORY_LAYERS})
    switch: StrategyToken = DEFAULT_STRATEGY  # the strategy switch
    held: Optional[str] = None  # the tactic held (None before the first decision)
    held_since: tuple[int, int] = (0, 0)  # (decisions, frames) when it started
    decisions: int = 0  # decisions made this episode
    frame: int = 0  # frames played this episode
    tactic_hidden: Optional[torch.Tensor] = None  # the tactic memory
    tactic_cell: Optional[torch.Tensor] = None


def choice_histories(copies: Sequence[_Copy], device) -> dict:
    """Each history layer's previous choices for a batch of copies: (choices [B,
    HISTORY, width], present [B, HISTORY]), the most recent first."""
    histories = {}
    for layer in HISTORY_LAYERS:
        choices = torch.zeros(len(copies), HISTORY, CHOICE_WIDTH[layer])
        present = torch.zeros(len(copies), HISTORY, dtype=torch.bool)
        for i, copy in enumerate(copies):
            for age, vector in enumerate(copy.choices[layer][:HISTORY]):
                choices[i, age] = vector
                present[i, age] = True
        histories[layer] = (choices.to(device), present.to(device))
    return histories


def remember_choices(copy: _Copy, decision: Decision) -> None:
    """Keep the skill's last 16 used destinations, including teacher choices."""
    if decision.skill is not None:
        vector = encode_choice("skill", decision.skill)
        copy.choices["skill"] = [vector, *copy.choices["skill"]][:HISTORY]


class SMBAgents:
    """Copies of the agent sharing one observer and one policy."""

    def __init__(
        self,
        observer: VisionObserver,
        policy: LayeredSMBPolicy,
        device,
        copies: int = 1,
        switch: StrategyToken = DEFAULT_STRATEGY,
    ):
        self.observer = observer
        self.policy = policy
        self.device = torch.device(device)
        self.switch = switch  # the strategy switch a copy starts with, unless told otherwise
        self.copies = [_Copy() for _ in range(copies)]
        for index in range(copies):
            self.reset(index)

    def reset(self, copy: int, switch: Optional[StrategyToken] = None) -> None:
        """Start a new episode for one copy: empty memories, no plan, no tactic,
        and its strategy switch (``switch``, else the agents' own)."""
        zeros = torch.zeros(self.policy.memory.width, device=self.device)
        tactic = torch.zeros(self.policy.tactic_memory.width, device=self.device)
        self.copies[copy] = _Copy(
            hidden=zeros,
            cell=zeros.clone(),
            switch=switch or self.switch,
            tactic_hidden=tactic,
            tactic_cell=tactic.clone(),
        )

    @torch.no_grad()
    def _tactics(self, starting, now, expected, action_memory, given, run_given, sample):
        """The tactic each deciding copy uses for its next action, and what the
        tactic layer did (TacticStep, None where it did not run).

        Where the tactic is given for every copy (training a layer below it), the
        tactic layer does not run. Otherwise, for each copy holding a tactic, the
        end check runs; a copy whose tactic ended (or that holds none) steps its
        tactic memory and chooses the next. A given tactic (the teacher's, while
        the tactic layer trains) replaces the layer's own; it starts when it
        differs from the one held.
        """
        count = len(starting)
        given = given or {}
        given_tactics = given.get("tactic") or [None] * count
        given_plans = given.get("action") or [None] * count
        given_skills = given.get("skill") or [None] * count
        needs_skill = "skill" in run_given or (
            ("action" in run_given or any(p is None for p in given_plans))
            and any(s is None for s in given_skills)
        )
        if "tactic" not in run_given and (
            not needs_skill or all(t is not None for t in given_tactics)
        ):
            return list(given_tactics), [None] * count
        policy, device = self.policy, self.device
        explore = "tactic" in sample

        def part(index, encoded):
            tokens, present = encoded
            return tokens[index], present[index]

        switches = torch.stack([encode_strategy(c.switch) for c in starting]).to(device)
        held_in = [
            held_features(c.held, c.decisions - c.held_since[0], c.frame - c.held_since[1])
            for c in starting
        ]
        held_rows = torch.stack(held_in).to(device)
        hidden = torch.stack([c.tactic_hidden for c in starting])
        cell = torch.stack([c.tactic_cell for c in starting])

        # The end check, for every copy holding a tactic.
        end_probability: list[Optional[float]] = [None] * count
        own_end = [c.held is None for c in starting]
        holding = [i for i, c in enumerate(starting) if c.held is not None]
        if holding:
            index = torch.tensor(holding, device=device)
            out = policy.tactic_outputs(
                part(index, now),
                part(index, expected),
                switches[index],
                held_rows[index],
                hidden[index],
            )
            chance = torch.sigmoid(out["end"])
            ends = torch.bernoulli(chance) > 0.5 if explore else chance > 0.5
            for k, i in enumerate(holding):
                end_probability[i] = float(chance[k])
                own_end[i] = bool(ends[k])

        # Where a tactic starts, the tactic memory steps.
        started = [
            (starting[i].held is None or given_tactics[i].stance != starting[i].held)
            if given_tactics[i] is not None
            else own_end[i]
            for i in range(count)
        ]
        stepping = [i for i in range(count) if started[i]]
        if stepping:
            index = torch.tensor(stepping, device=device)
            stepped = policy.recall(
                part(index, now),
                held_rows[index],
                action_memory[index],
                MemoryState(hidden[index], cell[index]),
            )
            hidden, cell = hidden.clone(), cell.clone()
            hidden[index], cell[index] = stepped.hidden, stepped.cell

        # The layer's own choice, wherever it ended its tactic.
        own = [c.held for c in starting]
        log_prob = [0.0] * count
        choosing = [i for i in range(count) if own_end[i]]
        if choosing:
            index = torch.tensor(choosing, device=device)
            none = held_features(None).to(device).expand(len(choosing), -1)
            out = policy.tactic_outputs(
                part(index, now), part(index, expected), switches[index], none, hidden[index]
            )
            for k, i in enumerate(choosing):
                stance, picks = choose("tactic", {"tactic": out["tactic"][k]}, sample=explore)
                own[i] = stance
                chance, _ = choice_log_prob(
                    "tactic",
                    {"tactic": out["tactic"][k : k + 1]},
                    {"tactic": torch.tensor([picks["tactic"]], device=device)},
                )
                log_prob[i] = float(chance[0])

        tactics, steps = [], []
        for i, c in enumerate(starting):
            used = given_tactics[i].stance if given_tactics[i] is not None else own[i]
            steps.append(
                TacticStep(
                    held=c.held,
                    actions=c.decisions - c.held_since[0],
                    frames=c.frame - c.held_since[1],
                    end_probability=end_probability[i],
                    own_end=own_end[i],
                    own=own[i],
                    started=started[i],
                    choice_log_prob=log_prob[i],
                )
            )
            if started[i]:
                c.held, c.held_since = used, (c.decisions, c.frame)
                c.tactic_hidden, c.tactic_cell = hidden[i], cell[i]
            tactics.append(tactic_token(used))
        return tactics, steps

    @torch.no_grad()
    def act(
        self,
        screens: Sequence[np.ndarray],
        copies: Sequence[int],
        given: Optional[Callable[[list[int], list[SceneObservation]], Mapping]] = None,
        run_given: Sequence[str] = (),
        sample: Sequence[str] = (),
    ) -> list[AgentStep]:
        """One frame for the listed copies, each from its own screen only.

        ``given`` (training only) is called with the copies deciding this frame
        and their scenes, and returns tokens that replace the policy's own
        (see decide and _tactics, also for ``run_given`` and ``sample``).
        """
        playing = [self.copies[c] for c in copies]
        scenes = self.observer.observe(np.stack(screens))
        rows = scene_rows(scenes)
        ended = []
        for copy, scene in zip(playing, scenes):
            copy.spatial.observe(scene)
            landed = copy.landing.landed(scene)
            reason = None
            if not copy.executor.idle:
                copy.executor.plan = copy.spatial.calibrate_launch(
                    scene, copy.executor.plan, copy.executor.pressed
                )
                reason = (
                    "landed"
                    if landed
                    else "done"
                    if copy.executor.finished
                    else "hold_recheck"
                    if copy.executor.reconsider
                    else "arrived"
                    if copy.spatial.arrived()
                    else None
                )
                if reason is not None:
                    copy.executor.end(reason)
            ended.append(reason)
        deciding = [k for k, copy in enumerate(playing) if copy.executor.idle]
        decisions: list[Optional[Decision]] = [None] * len(playing)
        if deciding:
            starting = [playing[k] for k in deciding]
            picked = [scenes[k] for k in deciding]
            picked_rows = [rows[k] for k in deciding]
            extra = given([copies[k] for k in deciding], picked) if given is not None else None
            supplied = extra or {}
            skills = supplied.get("skill") or [None] * len(starting)
            needs_context = (
                "tactic" in run_given or "skill" in run_given or any(s is None for s in skills)
            )
            remembered = MemoryState(
                torch.stack([copy.hidden for copy in starting]),
                torch.stack([copy.cell for copy in starting]),
            )
            now = expected = None
            if needs_context:
                now = self.policy.encode_scene(stack_rows(picked_rows, self.device))
                remembered = self.policy.remember(now, remembered)
                _, expected = self.policy.expect(remembered)
            tactics, tactic_steps = self._tactics(
                starting, now, expected, remembered.hidden, extra, run_given, sample
            )
            made = decide(
                self.policy,
                picked,
                picked_rows,
                expected,
                self.device,
                [copy.switch for copy in starting],
                tactics,
                extra,
                run_given,
                sample,
                now,
                choice_histories(starting, self.device),
                tactic_steps,
            )
            for j, (k, decision) in enumerate(zip(deciding, made)):
                given_actions = supplied.get("action")
                destination = (
                    None if given_actions and given_actions[j] is not None else decision.skill
                )
                decision.plan = playing[k].spatial.begin(destination, scenes[k], decision.plan)
                decisions[k] = decision
                playing[k].hidden, playing[k].cell = remembered.hidden[j], remembered.cell[j]
                remember_choices(playing[k], decision)
                playing[k].decisions += 1
                playing[k].executor.start(decision.plan)
        steps = []
        for k, copy in enumerate(playing):
            copy.button = copy.executor.press(scenes[k])
            copy.frame += 1
            steps.append(AgentStep(scenes[k], rows[k], ended[k], decisions[k], copy.button))
        return steps
