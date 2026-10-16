"""Vision-only agent: strategy -> tactic -> skill -> predictive executor.

At each action boundary the option-critic checks the held tactic, and the
skill chooses an action (a verb aimed at an object it sees) from the scene,
the tactic, the strategy, its 16 previous actions and the executor's
feedback. Neither reads a prediction (user, 2026-10-15). Where the action
ends, the executor's destination, is the teacher's ground truth in training
(``carry_out``); later the world model will predict it, run once at the
start of the action after it is chosen. The target tracker, when given,
observes every frame.
Execution uses per-frame vision for motion, destination and hold feedback.
Landing and destination arrival can end a plan early, and waits are rechecked
each frame unless given a length. Simulator state is never passed to these
networks.

Training may supply teacher tokens for selected layers; runtime playback
uses only the policy's own choices and the externally set strategy switch.
"""

from dataclasses import dataclass, field
from typing import Callable, Mapping, Optional, Sequence

import numpy as np
import torch

from .action_tokens import VERBS, ActionToken, pointer_index
from .actions import SMBAction
from .layered_policy import (
    CHOICE_WIDTH,
    CHOICES,
    HISTORY,
    HISTORY_LAYERS,
    POINTER_START,
    LayeredSMBPolicy,
    MemoryState,
    choice_log_prob,
    choose,
    encode_choice,
    held_features,
)
from .smb_executor import ActionPlan, SMBExecutor
from .smb_observer import SEQ_LEN_A, SEQ_LEN_B, VisionObserver, column_codes, pack_c
from .smb_scene_labels import STOMP_ROWS, SceneObservation
from .smb_spatial_feedback import SpatialFeedback
from .tokens import (
    DEFAULT_STRATEGY,
    EXECUTION_WIDTH,
    SkillToken,
    StrategyToken,
    TacticToken,
    encode_strategy,
    encode_tactic,
    tactic_token,
)

NOOP = int(SMBAction.NOOP)
# Rows Mario's feet must move while he is in the air for contact to be a
# landing (LandingWatch).
LANDING_MOVE = 2


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
    plan: Optional[ActionPlan]
    chosen: dict  # the policy's own choice at each layer it ran, before any given token
    picks: dict = field(default_factory=dict)  # layer -> its own picks (layered_policy.choose)
    log_prob: dict = field(default_factory=dict)  # layer -> log-probability of its own picks
    value: dict = field(default_factory=dict)  # layer -> its estimate of the return to come
    tactic_step: Optional[TacticStep] = None  # when the tactic layer ran
    skill: Optional[SkillToken] = None
    execution_feedback: list = field(default_factory=lambda: [0.0] * EXECUTION_WIDTH)
    # The skill's own action, whose end (the teacher's version, in training)
    # is the destination ``skill``.
    action: Optional[ActionToken] = None


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
    return {
        name: _one(value, index) if isinstance(value, Mapping) else value[index]
        for name, value in out.items()
    }


def _encoded(encode, tokens, device) -> torch.Tensor:
    return torch.stack([encode(token) for token in tokens]).to(device)


@torch.no_grad()
def decide(
    policy: LayeredSMBPolicy,
    scenes: Sequence[SceneObservation],
    rows: Sequence[tuple],
    device,
    switches: Sequence[StrategyToken],
    tactics: Sequence[Optional[TacticToken]],
    given: Optional[Mapping[str, Sequence]] = None,
    run_given: Sequence[str] = (),
    sample: Sequence[str] = (),
    encoded_scene=None,
    histories: Optional[Mapping[str, tuple]] = None,
    tactic_steps: Optional[Sequence[Optional[TacticStep]]] = None,
    execution_feedback=None,
) -> list[Decision]:
    """Skill destinations and action plans for pictures under ``tactics``
    (each held by the tactic layer, or given).

    ``encoded_scene``: the pictures' scenes already encoded, when the caller
    has them. ``histories``: the skill's own previous destinations
    (choice_histories).

    ``given["action"]`` can replay an exact teacher button plan in tests and
    teacher diagnostics. Learned policies never produce button plans.
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
        out = policy.run_skill(
            encoded(),
            above,
            (histories or {}).get("skill"),
            strategy=_encoded(encode_strategy, switches, device),
            feedback=execution_feedback,
            sample=layer in sample,
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

    if (missing("action") or "skill" in run_given) and runs("skill"):
        run("skill", _encoded(encode_tactic, tactics, device))
    # The skill's own choice is an action: its destination is the teacher's
    # version of it (SMBAgents.carry_out); a given command is its own
    # destination.
    own_actions = chosen.get("skill", [None] * count)
    given_skills = given.get("skill")
    destinations = replaced("skill", [None] * count)
    plans = replaced("action", [None] * count)
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
                action=(
                    own_actions[i] if given_skills is None or given_skills[i] is None else None
                ),
                execution_feedback=(
                    execution_feedback[i].tolist()
                    if execution_feedback is not None
                    else [0.0] * EXECUTION_WIDTH
                ),
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
    execution_status: str = "unplanned"


@dataclass
class LandingWatch:
    """Use vision's contact signal; the executor does not rebuild terrain contacts.

    A landing is contact after Mario was in the air and moved up or down there
    (his feet by at least LANDING_MOVE rows). As he walks off an edge, vision's
    contact can flicker on for a frame at the height he left (his box still
    touching the edge's corner): that is no landing, and he is still in the
    air (it had ended an action there, and the next decision was made in the
    air). Nor is contact while his feet still rise, except on a moving
    platform (which may rise under him): rising past the top corner of a
    pipe as tall as his jump, his box touched it 1 row below his feet, and
    the jump was ended on the way up. Nor is coming down past an enemy
    stomped before: it stays drawn flattened, and its top within STOMP_ROWS
    of his feet is contact, as on the frame he stomps it (when it was still
    alive the frame before); falling back past it after the bounce ended the
    next action in the air."""

    airborne: bool = False
    # The lowest and highest feet row seen since he left the ground (None:
    # not seen, as when the watch starts in the air).
    feet: Optional[tuple] = None
    # The feet row on the frame before (None: not seen yet).
    last: Optional[int] = None
    # The defeated enemies' boxes on the frame before.
    defeated: tuple = ()

    def landed(self, scene: SceneObservation) -> bool:
        if scene.mario.box is None:
            return False
        bottom = scene.mario.box[3]
        rising = self.last is not None and bottom < self.last
        self.last = bottom
        before, self.defeated = self.defeated, tuple(
            e.box for e in scene.enemies if e.kind == "defeated"
        )
        if not scene.mario.on_something or (
            self.airborne
            and (
                (rising and scene.mario.support != "moving_platform")
                or _only_old_defeated(scene, before)
            )
        ):
            if not self.airborne:
                self.airborne, self.feet = True, (bottom, bottom)
            elif self.feet is not None:
                self.feet = (min(self.feet[0], bottom), max(self.feet[1], bottom))
            return False
        if (
            self.airborne
            and self.feet is not None
            and max(self.feet[1], bottom) - min(self.feet[0], bottom) < LANDING_MOVE
        ):
            return False
        landed = self.airborne
        self.airborne, self.feet = False, None
        return landed


# A defeated enemy's box this many pixels from where one was on the frame
# before, sideways (the camera scrolls), is the same one.
DEFEATED_MATCH = 4


def _only_old_defeated(scene: SceneObservation, before: tuple) -> bool:
    """Whether vision's contact is only a stomp contact (no surface at his
    feet) with enemies that were already defeated on the frame before."""
    box = scene.mario.box
    if scene.mario.support != "air":
        return False
    touched = [
        e
        for e in scene.enemies
        if e.box[0] < box[2] and e.box[2] > box[0] and abs(e.box[1] - box[3]) <= STOMP_ROWS
    ]
    return bool(touched) and all(
        e.kind == "defeated"
        and any(abs(b[0] - e.box[0]) <= DEFEATED_MATCH and b[1] == e.box[1] for b in before)
        for e in touched
    )


@dataclass
class _Copy:
    executor: SMBExecutor = field(default_factory=SMBExecutor)
    landing: LandingWatch = field(default_factory=LandingWatch)
    spatial: SpatialFeedback = field(default_factory=SpatialFeedback)
    button: int = NOOP
    # Each layer's choices used at earlier decisions, the most recent first.
    choices: dict = field(default_factory=lambda: {layer: [] for layer in HISTORY_LAYERS})
    switch: StrategyToken = DEFAULT_STRATEGY  # the strategy switch
    held: Optional[str] = None  # the tactic held (None before the first decision)
    held_since: tuple[int, int] = (0, 0)  # (decisions, frames) when it started
    decisions: int = 0  # decisions made this episode
    frame: int = 0  # frames played this episode
    # Pre-episode frames still to watch (prefilled history): the agent observes
    # them, its tracks (and target tracker) update, but it decides nothing and
    # presses nothing until they are over.
    watch: int = 0
    tactic_hidden: Optional[torch.Tensor] = None  # the tactic memory
    tactic_cell: Optional[torch.Tensor] = None
    # With collision trackers (core/target_tracker.CollisionTrackers): stepped
    # every frame on the executor's kinematics and the action in progress.
    trackers: object = None
    # The action in progress as the trackers read it (target_tracker.
    # ActionState), set when one starts.
    action_now: object = None
    # This frame's kinematics (smb_kinematics: per sprite slot, Mario's), the
    # sprite slots' identities, and what vision shows of the enemies.
    kinematics: tuple = ()


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
    """Keep the skill's last 16 used actions (each with where the executor was
    sent), including the teacher's commands."""
    if decision.skill is not None:
        vector = encode_choice("skill", (decision.action, decision.skill))
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
        tracker=None,
        carry_out=None,
    ):
        self.observer = observer
        self.policy = policy
        self.device = torch.device(device)
        self.switch = switch  # the strategy switch a copy starts with, unless told otherwise
        # Collision trackers (core/target_tracker.CollisionTracker), when
        # given, are stepped every frame.
        self.tracker = tracker
        # Training: carry_out(copy number, action) gives the executor's
        # destination for the skill's own action (the teacher's version of
        # it: the ground truth an outcome prediction will stand for), or None
        # when there is none (then the copy holds; its episode is ended as
        # lost).
        self.carry_out = carry_out
        self.copies = [_Copy() for _ in range(copies)]
        for index in range(copies):
            self.reset(index)

    def reset(self, copy: int, switch: Optional[StrategyToken] = None, watch: int = 0) -> None:
        """Start a new episode for one copy: an empty tactic memory, no plan, no
        tactic, and its strategy switch (``switch``, else the agents' own).
        ``watch``: pre-episode frames to observe before the first decision."""
        tactic = torch.zeros(self.policy.tactic_memory.width, device=self.device)
        from .target_tracker import ActionState, CollisionTrackers

        self.copies[copy] = _Copy(
            switch=switch or self.switch,
            tactic_hidden=tactic,
            tactic_cell=tactic.clone(),
            watch=watch,
            trackers=(
                CollisionTrackers(self.tracker, self.device) if self.tracker is not None else None
            ),
            action_now=ActionState(),
        )

    def _track(self, copy: _Copy, scene) -> None:
        """Step the copy's collision trackers on this frame: the executor's
        kinematics, what vision shows of the enemies, and the action in
        progress."""
        from .smb_kinematics import enemy_context, kinematics, slot_identities
        from .target_tracker import frame_features

        sprites, mario = kinematics(scene, copy.spatial)
        identities = slot_identities(scene, copy.spatial)
        context = enemy_context(scene)
        copy.kinematics = (sprites, mario, identities, context)
        copy.trackers.observe(
            frame_features(
                sprites,
                mario,
                context,
                identities,
                self._position(copy, scene),
                copy.frame,
                copy.action_now,
            ),
            identities,
        )

    @staticmethod
    def _position(copy: _Copy, scene) -> tuple:
        """Mario's world displacement (the executor's) and his feet's row."""
        feet = scene.mario.box[3] if scene.mario.box is not None else 0.0
        return (copy.spatial.displacement, feet)

    def _began(self, copy: _Copy, decision: Decision, scene) -> None:
        """A new action began: the trackers read it from now on."""
        from .target_tracker import action_state, frame_features
        from .tokens import command_row

        action = decision.action
        verb = VERBS.index(action.verb) if action is not None else -1
        pointer = (
            pointer_index(action.target) - POINTER_START
            if action is not None and action.target is not None
            else -1
        )
        sprites, mario, identities, context = copy.kinematics
        position = self._position(copy, scene)
        copy.action_now = action_state(
            verb, pointer, command_row(decision.skill), sprites, identities, position, copy.frame
        )
        copy.trackers.refresh(
            frame_features(
                sprites, mario, context, identities, position, copy.frame, copy.action_now
            ),
            identities,
        )

    @torch.no_grad()
    def _actions(self, starting, made, own, numbers) -> None:
        """For the deciding copies playing their own skill choice (``own``):
        the executor's destination for the skill's action, the teacher's
        version of it (``carry_out``; ``numbers``: the deciding copies'
        numbers). Without ``carry_out`` a hold holds; any other action needs
        the world model's predicted outcome, not built yet."""
        chosen = [i for i in own if made[i].action is not None]
        if self.carry_out is not None:
            # Holds too: the teacher's version of a hold lasts as long as its
            # true wait (a wait is one action; teacher_actions.carry_out).
            for i in chosen:
                destination = self.carry_out(numbers[i], made[i].action)
                made[i].skill = destination or SkillToken("hold", 0, 0)
            return
        for i in [i for i in chosen if made[i].action.verb == "hold"]:
            made[i].skill = SkillToken("hold", 0, 0)
        if any(made[i].action.verb != "hold" for i in chosen):
            raise RuntimeError(
                "the skill chose an action: playing it without the teacher needs the world "
                "model's predicted outcome (not built yet; docs/design-decisions.md 2026-10-15)"
            )

    @torch.no_grad()
    def _tactics(self, starting, now, given, run_given, sample):
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
            (
                (starting[i].held is None or given_tactics[i].stance != starting[i].held)
                if given_tactics[i] is not None
                else own_end[i]
            )
            for i in range(count)
        ]
        stepping = [i for i in range(count) if started[i]]
        if stepping:
            index = torch.tensor(stepping, device=device)
            stepped = policy.recall(
                part(index, now),
                held_rows[index],
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
            out = policy.tactic_outputs(part(index, now), switches[index], none, hidden[index])
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
            if copy.trackers is not None:
                self._track(copy, scene)
            landed = copy.landing.landed(scene)
            reason = None
            if not copy.executor.idle:
                reason = (
                    "landed"
                    if landed
                    else (
                        ("threat" if copy.executor.plan.threat else "done")
                        if copy.executor.finished
                        else (
                            "hold_recheck"
                            if copy.executor.reconsider
                            else "arrived"
                            if copy.spatial.arrived()
                            else None
                        )
                    )
                )
                if copy.spatial.stalled and (
                    reason is not None
                    or copy.spatial.observed_frames >= copy.spatial.positions.maxlen
                ):
                    # Preserve prior stall feedback for the new decision, but
                    # let a new movement attempt accelerate before cancelling
                    # it against the previous command's stationary history.
                    reason = "no_progress"
                if reason is not None:
                    copy.executor.end(reason)
                    copy.spatial.ended = reason
            ended.append(reason)
        deciding = [k for k, copy in enumerate(playing) if copy.executor.idle and not copy.watch]
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
            # The scene is encoded at decisions only (nothing reads it between).
            now = (
                self.policy.encode_scene(stack_rows(picked_rows, self.device))
                if needs_context
                else None
            )
            tactics, tactic_steps = self._tactics(starting, now, extra, run_given, sample)
            made = decide(
                self.policy,
                picked,
                picked_rows,
                self.device,
                [copy.switch for copy in starting],
                tactics,
                extra,
                run_given,
                sample,
                now,
                choice_histories(starting, self.device),
                tactic_steps,
                torch.tensor(
                    [copy.spatial.report(scene) for copy, scene in zip(starting, picked)],
                    device=self.device,
                ),
            )
            own = [
                j
                for j in range(len(deciding))
                if skills[j] is None
                and not (supplied.get("action") and supplied["action"][j] is not None)
            ]
            self._actions(starting, made, own, [copies[k] for k in deciding])
            for j, (k, decision) in enumerate(zip(deciding, made)):
                given_actions = supplied.get("action")
                destination = (
                    None if given_actions and given_actions[j] is not None else decision.skill
                )
                decision.plan = playing[k].spatial.begin(destination, scenes[k], decision.plan)
                if playing[k].trackers is not None:
                    self._began(playing[k], decision, scenes[k])
                decisions[k] = decision
                remember_choices(playing[k], decision)
                playing[k].decisions += 1
                playing[k].executor.start(
                    decision.plan,
                    flight=playing[k].spatial.flight,
                    travel=playing[k].spatial.travel,
                )
        steps = []
        for k, copy in enumerate(playing):
            if copy.watch:
                copy.watch -= 1
                copy.button = NOOP
            else:
                copy.button = copy.executor.press(scenes[k])
            copy.spatial.executed(copy.button, scenes[k])
            copy.frame += 1
            maneuver = copy.executor.flight or copy.executor.travel
            status = maneuver.status if maneuver else copy.spatial.status
            steps.append(AgentStep(scenes[k], rows[k], ended[k], decisions[k], copy.button, status))
        return steps
