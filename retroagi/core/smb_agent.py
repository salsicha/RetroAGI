"""The four-layer agent playing from screens alone, in either game.

SMBAgents holds several copies of the agent, one per game being played side
by side, sharing one vision observer and one policy. Each frame, act() takes
only the screens and returns one button action per copy:

1. the vision transformer reports each screen's scene (smb_observer);
2. each copy's action ends if it has pressed all its frames, or if the vision
   transformer's land detector says Mario has just landed on something
   (LandingWatch) - the executor itself reads nothing but its plan
   (smb_executor);
3. for each copy whose action ended, a new action starts: first the memory
   steps, taking in only the latest picture's scene, and gives its expected
   scene for when the coming action ends; then
   the layers decide top-down (decide: strategy, tactic, skill, then the
   action plan) reading the current scene and that expected scene; and the
   executor starts the action;
4. each executor presses this frame's button.

Nothing else about the game reaches the agent. In training only, a collector
may pass ``given``: for the copies deciding this frame, tokens that replace
the policy's own at chosen layers (a teacher's token for the layer above the
one being trained, say). Play never passes it.
"""

from dataclasses import dataclass, field
from typing import Callable, Mapping, Optional, Sequence

import numpy as np
import torch

from .actions import SMBAction
from .layered_policy import (
    CHOICES,
    LayeredSMBPolicy,
    MemoryState,
    choice_log_prob,
    choose,
)
from .smb_executor import ActionPlan, SMBExecutor
from .smb_observer import (
    C_SPANS,
    SEQ_LEN_A,
    SEQ_LEN_B,
    VisionObserver,
    column_codes,
    pack_c,
    target_box,
    target_row,
)
from .smb_scene_labels import SceneObservation
from .tokens import (
    DEFAULT_STRATEGY,
    SkillToken,
    StrategyToken,
    TacticToken,
    encode_skill,
    encode_strategy,
    encode_tactic,
)

TARGET_SPAN = slice(*C_SPANS["c_target"])
NOOP = int(SMBAction.NOOP)


@dataclass
class Decision:
    """What the layers decided for one copy at one decision point."""

    strategy: Optional[StrategyToken]  # None when no layer below needed one
    tactic: Optional[TacticToken]
    skill: SkillToken
    plan: ActionPlan
    chosen: dict  # the policy's own choice at each layer it ran, before any given token
    target: np.ndarray  # the c_target numbers the action layer read
    picks: dict = field(default_factory=dict)  # layer -> its own picks (layered_policy.choose)
    log_prob: dict = field(default_factory=dict)  # layer -> log-probability of its own picks
    value: dict = field(default_factory=dict)  # layer -> its estimate of the return to come


def scene_rows(scenes: Sequence[SceneObservation]) -> list[tuple]:
    """(src_a, src_b, src_c) for each scene, with no skill target filled in."""
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
    given: Optional[Mapping[str, Sequence]] = None,
    run_given: Sequence[str] = (),
    sample: Sequence[str] = (),
    encoded_scene=None,
) -> list[Decision]:
    """The layers' decisions, top-down, for a batch of pictures.

    ``expected``: the memory's expected scene at the end of the coming action,
    encoded (LayeredSMBPolicy.expect). ``encoded_scene``: the pictures' scenes
    already encoded (without a skill target), when the caller has them.

    ``given[layer][i]``, when not None, replaces the policy's choice at that
    layer ("strategy", "tactic", "skill": tokens; "action": an ActionPlan) for
    picture i: training only. A layer whose choice is given for every picture
    does not run, unless it is listed in ``run_given`` (to compare its choice
    with the given one). Layers in ``sample`` sample their choice (training
    by reward); the others take the most likely. Until the strategy layer has
    learned from Full SMB play, the strategy is DEFAULT_STRATEGY.
    """
    count = len(scenes)
    given = given or {}

    def missing(layer: str) -> bool:
        values = given.get(layer)
        return values is None or any(value is None for value in values)

    # Top-down, a layer runs when it is listed in run_given, or when its choice
    # is not given for every picture and the layer below needs it: the action
    # layer always runs or is given; each layer above runs only for a layer
    # below that runs.
    needed = {"action": True}
    needed["skill"] = True  # the action layer reads the skill token, run or given
    needed["tactic"] = "skill" in run_given or missing("skill")
    needed["strategy"] = "tactic" in run_given or (needed["tactic"] and missing("tactic"))

    def runs(layer: str) -> bool:
        return layer in run_given or (needed[layer] and missing(layer))

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

    def run(layer: str, inputs, above=None) -> None:
        out = policy.run_layer(layer, inputs, expected, above)
        made = [choose(layer, _one(out, i), sample=layer in sample) for i in range(count)]
        chosen[layer] = [token for token, _ in made]
        picked[layer] = [picks for _, picks in made]
        stacked = {
            head: torch.tensor([picks[head] for picks in picked[layer]], device=device)
            for head in CHOICES[layer]
        }
        log_prob, _ = choice_log_prob(layer, out, stacked)
        scores[layer] = (log_prob.tolist(), out["value"].tolist())

    if runs("strategy") and bool(policy.strategy_trained):
        run("strategy", encoded())
    default = DEFAULT_STRATEGY if needed["strategy"] else None
    strategies = replaced("strategy", chosen.get("strategy", [default] * count))
    if runs("tactic"):
        run("tactic", encoded(), _encoded(encode_strategy, strategies, device))
    tactics = replaced("tactic", chosen.get("tactic", [None] * count))
    if runs("skill"):
        run("skill", encoded(), _encoded(encode_tactic, tactics, device))
    skills = replaced("skill", chosen.get("skill", [None] * count))
    targets = [target_row(s, target_box(s, skill.target)) for s, skill in zip(scenes, skills)]
    if runs("action"):
        with_target = plain[2].clone()
        with_target[:, TARGET_SPAN] = torch.as_tensor(np.stack(targets), device=device)
        run(
            "action",
            policy.encode_scene((plain[0], plain[1], with_target)),
            _encoded(encode_skill, skills, device),
        )
    plans = replaced("action", chosen.get("action", [None] * count))
    return [
        Decision(
            strategy=strategies[i],
            tactic=tactics[i],
            skill=skills[i],
            plan=plans[i],
            chosen={layer: values[i] for layer, values in chosen.items()},
            target=targets[i],
            picks={layer: values[i] for layer, values in picked.items()},
            log_prob={layer: values[0][i] for layer, values in scores.items()},
            value={layer: values[1][i] for layer, values in scores.items()},
        )
        for i in range(count)
    ]


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
    hidden: Optional[torch.Tensor] = None
    cell: Optional[torch.Tensor] = None
    button: int = NOOP


class SMBAgents:
    """Copies of the agent sharing one observer and one policy."""

    def __init__(self, observer: VisionObserver, policy: LayeredSMBPolicy, device, copies: int = 1):
        self.observer = observer
        self.policy = policy
        self.device = torch.device(device)
        self.copies = [_Copy() for _ in range(copies)]
        for index in range(copies):
            self.reset(index)

    def reset(self, copy: int) -> None:
        """Start a new episode for one copy: empty memory, no plan."""
        zeros = torch.zeros(self.policy.memory.width, device=self.device)
        self.copies[copy] = _Copy(hidden=zeros, cell=zeros.clone())

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
        (see decide, also for ``run_given`` and ``sample``).
        """
        playing = [self.copies[c] for c in copies]
        scenes = self.observer.observe(np.stack(screens))
        rows = scene_rows(scenes)
        ended = []
        for copy, scene in zip(playing, scenes):
            landed = copy.landing.landed(scene)
            reason = None
            if not copy.executor.idle:
                reason = "landed" if landed else ("done" if copy.executor.finished else None)
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
            now = self.policy.encode_scene(stack_rows(picked_rows, self.device))
            # The memory steps at each action's start, from the latest picture
            # only, and gives the scene it expects when the action ends.
            remembered = self.policy.remember(
                now,
                MemoryState(
                    torch.stack([copy.hidden for copy in starting]),
                    torch.stack([copy.cell for copy in starting]),
                ),
            )
            _, expected = self.policy.expect(remembered)
            made = decide(
                self.policy,
                picked,
                picked_rows,
                expected,
                self.device,
                extra,
                run_given,
                sample,
                now,
            )
            for j, (k, decision) in enumerate(zip(deciding, made)):
                decisions[k] = decision
                playing[k].hidden, playing[k].cell = remembered.hidden[j], remembered.cell[j]
                playing[k].executor.start(decision.plan)
        steps = []
        for k, copy in enumerate(playing):
            copy.button = copy.executor.press()
            steps.append(AgentStep(scenes[k], rows[k], ended[k], decisions[k], copy.button))
        return steps
