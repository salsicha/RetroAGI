"""Block SMB training for the four-layer agent (core.layered_policy).

Every policy input comes from the Block vision transformer: workers play
episodes, show each screen to the vision transformer, and pack what it
reports (core.smb_observer). The simulator is read only by the teachers
(teacher_tokens), by the rewards, and by the scoring of episodes.

Layers are trained bottom-up, one at a time, with the others frozen. Each
learner is given the explicit token of the layer above it, from a teacher:

- action: given the teacher's skill token, choose the action and frame count
  (this also trains the scene encoder and the memory, whose world-model part
  learns to predict the next frame's scene numbers);
- skill: given the teacher's tactic token, choose the skill and its target;
- tactic: given the teacher's strategy token, choose the tactic.

Each learner is taught by imitation with corrections: workers play with the
policy, a teacher labels every decision with what it would have done from that
exact state, and the learner trains on every labelled decision so far. In the
first round every decision plays the teacher's choice; later the chance
shrinks to zero, so the policy learns to recover from its own mistakes.

Workers run on the graphics card: each plays several episodes side by side
and shows their screens to the vision transformer together.
"""

import dataclasses
import hashlib
import json
import multiprocessing
import random
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from retroagi.core.actions import SMB_ACTIONS, SMB_JUMP_ACTIONS, SMBAction
from retroagi.core.layered_policy import (
    CHOICE_WIDTH,
    CHOICES,
    FRAME_BINS,
    HISTORY,
    LayeredSMBPolicy,
    MemoryState,
    PolicySettings,
    choice_log_prob,
    encode_choice,
)
from retroagi.core.smb_agent import SMBAgents
from retroagi.core.smb_executor import FRAME_COUNTS, ActionPlan
from retroagi.core.smb_observer import (
    C_SPANS,
    SEQ_LEN_A,
    SEQ_LEN_B,
    SEQ_LEN_C,
    VisionObserver,
    observation_layout,
)
from retroagi.core.smb_physics import NES_JUMP_FRAMES
from retroagi.core.tokens import (
    SKILLS,
    TACTICS,
    SkillToken,
    TacticToken,
    encode_skill,
    encode_strategy,
    encode_tactic,
    token_layout,
)

from .monte_carlo import BLOCK_SMB_MC_DIFFICULTY_BINS as DIFFICULTIES
from .monte_carlo import BLOCK_SMB_MC_FAMILIES

LEARNERS = ("action", "skill", "tactic")
# The token each learner is given from above.
GIVEN = {"action": "skill", "skill": "tactic", "tactic": "strategy"}
TARGET_SPAN = slice(*C_SPANS["c_target"])
NOOP = int(SMBAction.NOOP)


@dataclass
class LayeredTrainConfig:
    learner: str = "action"
    families: tuple[str, ...] = tuple(BLOCK_SMB_MC_FAMILIES)
    rounds: int = 12
    train_layouts_per_family: int = 8
    # Extra layouts next round for a family that lost validation episodes:
    # this many times its share of losses.
    focus_layouts: int = 8
    # Held-out layouts per family and difficulty (easy, medium, hard), fixed for the run.
    validation_layouts_per_difficulty: int = 3
    # Training layouts' difficulty: easy, medium and hard in these proportions.
    difficulty_weights: tuple[float, float, float] = (2.0, 2.0, 1.0)
    teacher_share_start: float = 1.0  # chance a decision plays the teacher's choice, first round
    teacher_share_end: float = 0.0  # ... and last round
    epochs_per_round: int = 8
    batch_frames: int = 8192
    learning_rate: float = 3e-4
    expectation_weight: float = 1.0  # the memory's expected scene, against imitation
    replay_episodes: int = 6000
    workers: int = 12
    lanes: int = 8  # episodes each worker plays side by side
    episode_frames: int = 600  # an episode not won by then ends as a timeout
    family_gate: float = 0.9
    # After the imitation rounds, rounds in which the learner samples its own
    # choices and learns from the rewards the families give (plus the teacher's
    # labels, weighted by imitation_weight).
    reward_rounds: int = 0
    discount: float = 0.995  # per frame
    advantage_smoothing: float = 0.95
    clip: float = 0.2  # how far one update may move a choice's probability ratio
    entropy_weight: float = 0.01
    value_weight: float = 0.5
    imitation_weight: float = 1.0
    reward_scale: float = 0.05  # rewards are multiplied by this before crediting decisions
    reward_learning_rate: float = 1e-4
    seed: int = 0
    device: str = "cuda"
    vision_checkpoint: str = "data/block_vit/block_vit_scene.pth"
    init: Optional[str] = None  # a checkpoint whose lower layers are already trained
    # The size of a new policy (ignored with init).
    policy_width: int = PolicySettings.width
    policy_scene_depth: int = PolicySettings.scene_depth
    policy_layer_depth: int = PolicySettings.layer_depth
    policy_memory_width: int = PolicySettings.memory_width
    output: str = "artifacts/block_smb/layered"

    def __post_init__(self):
        if self.learner not in LEARNERS:
            raise ValueError(f"learner must be one of {LEARNERS}, got {self.learner!r}")
        unknown = set(self.families) - set(BLOCK_SMB_MC_FAMILIES)
        if unknown:
            raise ValueError(f"unknown families: {sorted(unknown)}")


@dataclass(frozen=True)
class EpisodeTask:
    index: int
    family: str
    split: str
    seed: int
    sample_index: int
    teacher_share: float  # chance per decision that the learner plays the teacher's choice
    label: bool  # ask the teacher what it would do at every decision
    frames: int = 600  # the episode ends as a timeout after this many frames
    scenario: Optional[dict] = None  # the layout, when already made (else made from the above)
    difficulty: Optional[str] = None  # easy, medium or hard (None: the split's own mix)
    explore: bool = False  # the learner samples its choices (reward rounds)


@dataclass
class EpisodeRecord:
    """One played episode: what the policy saw and pressed, and its labelled decisions."""

    family: str
    split: str
    sample_index: int
    won: bool
    difficulty: Optional[str]
    end: str
    src_a: np.ndarray  # [T, 8] int8
    src_b: np.ndarray  # [T, 16] int8
    src_c: np.ndarray  # [T, SEQ_LEN_C] float16 (no target)
    buttons: np.ndarray  # [T] int8, pressed at each frame
    decision_frames: np.ndarray  # [D] int32
    targets: np.ndarray  # [D, 5] float32, the executed skill's target row
    given: np.ndarray  # [D, width] float32, the token the learner was given
    labels: dict  # name -> [D] arrays, with "valid" [D] bool
    played_teacher: np.ndarray  # [D] bool
    agreed: np.ndarray  # [D] bool, the policy chose what the teacher would have
    rewards: Optional[np.ndarray] = None  # [T] float32, the reward after each frame
    terminal: bool = False  # ended by death or the goal (not a timeout)
    explored: bool = False  # the learner sampled its choices
    picks: dict = field(default_factory=dict)  # head -> [D] the learner's own picks
    old_log_prob: Optional[np.ndarray] = None  # [D] their log-probability when played
    old_value: Optional[np.ndarray] = None  # [D] the learner's estimate of the return then
    used: Optional[np.ndarray] = None  # [D, choice width] the learner's choice used, per decision

    @property
    def frames(self) -> int:
        return int(len(self.buttons))


# ── Labels ────────────────────────────────────────────────────────────────────


def jump_frame_label(teacher_frames: int, holds: Sequence[int]) -> int:
    """The frame count a jump is taught: the middle of the longest run of
    certified holds (holds the teacher tested, local_traversal.safe_jump_holds,
    adjacent in the order it tests them), or the teacher's own hold."""
    menu = list(NES_JUMP_FRAMES)
    certified = sorted(menu.index(hold) for hold in holds if hold in menu)
    if not certified:
        return teacher_frames
    runs, run = [], [certified[0]]
    for index in certified[1:]:
        if index == run[-1] + 1:
            run.append(index)
        else:
            runs.append(run)
            run = [index]
    runs.append(run)
    longest = max(runs, key=len)
    return menu[longest[(len(longest) - 1) // 2]]


def _plan_label(plan: Optional[ActionPlan], holds) -> tuple[int, int, bool]:
    """(action, frame bin, valid) for the action learner."""
    if plan is None:
        return NOOP, 0, False
    frames = plan.frames
    if SMBAction(plan.action) in SMB_JUMP_ACTIONS:
        frames = jump_frame_label(frames, holds)
    return plan.action, FRAME_COUNTS.index(frames), True


# ── Workers ───────────────────────────────────────────────────────────────────

_WORKER: dict = {}


def _initialize_worker(vision_checkpoint: str, settings: dict, device: str) -> None:
    from .vision import load_block_vit_checkpoint

    torch.set_num_threads(1)
    vision = load_block_vit_checkpoint(Path(vision_checkpoint), device=device).model
    policy = LayeredSMBPolicy(PolicySettings(**settings)).to(device).eval()
    _WORKER.update(observer=VisionObserver(vision), policy=policy, device=device, version=None)


def _play_job(job) -> list[EpisodeRecord]:
    weights, version, learner, lanes, tasks = job
    if _WORKER["version"] != version:
        _WORKER["policy"].load_state_dict(
            torch.load(weights, map_location=_WORKER["device"], weights_only=True)
        )
        _WORKER["version"] = version
    return play_episodes(
        _WORKER["observer"], _WORKER["policy"], learner, tasks, _WORKER["device"], lanes
    )


def task_scenario(task: EpisodeTask) -> dict:
    """The layout a task names (made by the family's generator; seconds for bridges)."""
    from .monte_carlo import sample_block_smb_monte_carlo_scenario

    return dict(
        sample_block_smb_monte_carlo_scenario(
            split=task.split,
            seed=task.seed,
            sample_index=task.sample_index,
            family=task.family,
            difficulty=task.difficulty,
        ).scenario
    )


# An episode that makes no progress for this many frames ends as a timeout:
# no new tactic segment or route platform, and Mario no nearer the goal.
STALL_FRAMES = 400


class _Lane:
    """One episode being played: the simulator, its teacher, and the records."""

    def __init__(self, task: EpisodeTask, copy: int):
        from .env import MarioScenarioEnv
        from .teacher_tokens import episode_teacher

        scenario = task.scenario if task.scenario is not None else task_scenario(task)
        self.task = task
        self.copy = copy
        # A long layout gets the frames its teacher's route is allowed.
        self.frame_limit = max(task.frames, int(scenario.get("frame_budget", 0)))
        self.env = MarioScenarioEnv()
        self.screen, _ = self.env.reset(scenario=scenario, seed=0)
        self.teacher = episode_teacher(scenario)
        self.rng = random.Random(task.index * 7919 + task.sample_index)
        self.goal = 0.0
        self.end = ""
        self.frames: dict[str, list] = defaultdict(list)
        self.decisions: dict[str, list] = defaultdict(list)
        self.asked: dict = {}
        self.progress, self.progressed = None, 0

    def stalled(self) -> bool:
        """Whether the episode has made no progress for STALL_FRAMES frames."""
        env = self.env
        nearest = abs(env.mario["x"] - env.goal.centerx) if env.goal is not None else 0.0
        mark = (env._tactic_index, env._route_done)
        if self.progress is None or mark != self.progress[0] or nearest < self.progress[1] - 1:
            self.progress = (mark, min(nearest, self.progress[1]) if self.progress else nearest)
            self.progressed = env.steps
        return env.steps - self.progressed > STALL_FRAMES

    def ask_teacher(self, learner: str, scene) -> dict:
        """The teacher's tokens (and, for the action learner, its plan) here.

        Reads the simulator: training only. The skill's target is matched to
        what the vision transformer reports in ``scene``.
        """
        from .teacher_tokens import teacher_plan, teacher_skill, teacher_strategy, teacher_tactic

        # Only what this learner is given or taught. The skill is decided by
        # the tactic, so the teacher's tactic is worked out for every learner.
        tactic = teacher_tactic(self.env, self.teacher)
        asked = {
            "strategy": teacher_strategy(self.teacher) if learner == "tactic" else None,
            "tactic": tactic,
            "skill": teacher_skill(self.env, scene, self.teacher, tactic)
            if learner != "tactic"
            else None,
            "action": None,
            "holds": (),
            "plays_teacher": self.rng.random() < self.task.teacher_share,
        }
        if learner == "action" and (self.task.label or self.task.teacher_share > 0):
            asked["action"], asked["holds"] = teacher_plan(self.env, self.teacher)
        self.asked = asked
        return asked

    def note_decision(self, learner: str, decision) -> None:
        """Record a decision, the token the learner was given and the teacher's label."""
        d, asked = self.decisions, self.asked
        d["frame"].append(len(self.frames["a"]) - 1)
        d["target"].append(decision.target)
        d["played_teacher"].append(asked["plays_teacher"])
        for head, value in decision.picks[learner].items():
            d[f"pick_{head}"].append(value)
        d["log_prob"].append(decision.log_prob[learner])
        d["value"].append(decision.value[learner])
        used = {
            "strategy": decision.strategy,
            "tactic": decision.tactic,
            "skill": decision.skill,
            "action": decision.plan,
        }[learner]
        d["used"].append(
            encode_choice(learner, used, decision.target if learner == "skill" else None).numpy()
        )
        mine = decision.chosen[learner]
        if learner == "action":
            d["given"].append(encode_skill(decision.skill).numpy())
            action, frame_bin, valid = _plan_label(asked["action"], asked["holds"])
            d["label_action"].append(action)
            d["label_frames"].append(frame_bin)
            d["label_valid"].append(valid)
            agreed = valid and (mine.action, FRAME_COUNTS.index(mine.frames)) == (
                action,
                frame_bin,
            )
        elif learner == "skill":
            d["given"].append(encode_tactic(decision.tactic).numpy())
            token: SkillToken = asked["skill"]
            d["label_skill"].append(SKILLS.index(token.kind))
            d["label_direction"].append(int(token.direction > 0))
            d["label_pointer"].append(token.pointer)
            d["label_valid"].append(True)
            agreed = mine == token
        else:
            d["given"].append(encode_strategy(decision.strategy).numpy())
            stance: TacticToken = asked["tactic"]
            d["label_tactic"].append(TACTICS.index(stance.stance))
            d["label_direction"].append(int(stance.direction > 0))
            d["label_valid"].append(True)
            agreed = mine == stance
        d["agreed"].append(bool(agreed))

    def record(self) -> EpisodeRecord:
        d = self.decisions
        count = len(d["frame"])
        labels = {
            name[len("label_") :]: np.asarray(d[name]) for name in d if name.startswith("label_")
        }
        labels.setdefault("valid", np.zeros(count, bool))
        return EpisodeRecord(
            family=self.task.family,
            split=self.task.split,
            sample_index=self.task.sample_index,
            won=self.goal > 0,
            difficulty=self.task.difficulty,
            end=self.end,
            src_a=np.asarray(self.frames["a"], np.int8).reshape(-1, SEQ_LEN_A),
            src_b=np.asarray(self.frames["b"], np.int8).reshape(-1, SEQ_LEN_B),
            src_c=np.asarray(self.frames["c"], np.float16).reshape(-1, SEQ_LEN_C),
            buttons=np.asarray(self.frames["button"], np.int8),
            decision_frames=np.asarray(d["frame"], np.int32),
            targets=np.asarray(d["target"], np.float32).reshape(count, -1),
            given=np.asarray(d["given"], np.float32).reshape(count, -1),
            labels=labels,
            played_teacher=np.asarray(d["played_teacher"], bool),
            agreed=np.asarray(d["agreed"], bool),
            rewards=np.asarray(self.frames["reward"], np.float32),
            terminal=self.end in ("goal", "death", "off_route", "missed_objective"),
            explored=self.task.explore,
            picks={
                name[len("pick_") :]: np.asarray(d[name], np.int64)
                for name in d
                if name.startswith("pick_")
            },
            old_log_prob=np.asarray(d["log_prob"], np.float32),
            old_value=np.asarray(d["value"], np.float32),
            used=np.asarray(d["used"], np.float32).reshape(count, -1),
        )


def _teacher_given(learner: str, lanes: dict):
    """For the deciding copies: the teacher's token for the layer above the
    learner, and the teacher's own choice for the learner where it plays the
    teacher this time. (Layers higher up do not run: nothing below needs them.)"""
    above = GIVEN[learner]

    def given(copies, scenes):
        tokens = {above: [], learner: []}
        for copy, scene in zip(copies, scenes):
            asked = lanes[copy].ask_teacher(learner, scene)
            tokens[above].append(asked[above])
            tokens[learner].append(asked[learner] if asked["plays_teacher"] else None)
        return tokens

    return given


@torch.no_grad()
def play_episodes(
    observer: VisionObserver,
    policy: LayeredSMBPolicy,
    learner: Optional[str],
    tasks: Sequence[EpisodeTask],
    device,
    lanes: int,
) -> list[EpisodeRecord]:
    """Play ``tasks``, ``lanes`` at a time. The agent sees only the screens.

    With a ``learner``, the teachers give it its token from above and label
    its decisions. With none, the agent plays exactly as deployed: every token
    is its own and nothing is given or labelled.
    """
    agents = SMBAgents(observer, policy, device, lanes)
    explore = {task.explore for task in tasks}
    if len(explore) > 1 or (learner is None and True in explore):
        raise ValueError("one call either explores, for a learner, or does not")
    sample = (learner,) if True in explore else ()
    pending = list(tasks)
    playing: dict[int, _Lane] = {}
    done: dict[int, EpisodeRecord] = {}
    given = _teacher_given(learner, playing) if learner is not None else None
    while pending or playing:
        for copy in range(lanes):
            if copy not in playing and pending:
                playing[copy] = _Lane(pending.pop(0), copy)
                agents.reset(copy)
        live = list(playing.values())
        if learner is not None:
            for lane in live:
                lane.teacher.observe_frame(lane.env)
        steps = agents.act(
            [lane.screen for lane in live],
            [lane.copy for lane in live],
            given=given,
            run_given=(learner,) if learner is not None else (),
            sample=sample,
        )
        for lane, step in zip(live, steps):
            for name, value in zip("abc", step.rows):
                lane.frames[name].append(value)
            if step.decision is not None and learner is not None:
                lane.note_decision(learner, step.decision)
            lane.frames["button"].append(step.button)
            lane.screen, reward, terminated, truncated, info = lane.env.step(step.button)
            lane.frames["reward"].append(float(reward))
            lane.goal += float(info["reward_terms"].get("goal", 0.0))
            if len(lane.frames["button"]) >= lane.frame_limit or lane.stalled():
                truncated = True
            if terminated or truncated:
                lane.end = (
                    "goal"
                    if lane.goal > 0
                    else (
                        "death"
                        if info.get("death")
                        else (
                            "off_route"
                            if info.get("off_route")
                            else ("missed_objective" if info.get("objective_missed") else "timeout")
                        )
                    )
                )
                done[lane.task.index] = lane.record()
                lane.env.close()
                del playing[lane.copy]
    return [done[task.index] for task in tasks]


class EpisodePool:
    """Worker processes on the graphics card that play episodes with the published policy."""

    def __init__(self, config: LayeredTrainConfig, settings: PolicySettings):
        self.config = config
        self.weights = Path(config.output) / "published_policy.pt"
        self.version = 0
        context = multiprocessing.get_context("spawn")
        self.pool = ProcessPoolExecutor(
            config.workers,
            mp_context=context,
            initializer=_initialize_worker,
            initargs=(config.vision_checkpoint, asdict(settings), config.device),
        )

    def publish(self, policy: LayeredSMBPolicy) -> None:
        torch.save(policy.state_dict(), self.weights)
        self.version += 1

    def play(
        self, tasks: Sequence[EpisodeTask], *, as_deployed: bool = False
    ) -> list[EpisodeRecord]:
        """Play the tasks for the learner, or (``as_deployed``) with no teacher at all.

        The tasks go out shuffled, one batch of lanes per job, so slow and fast
        families mix and every worker stays busy to the end; the records come
        back in the tasks' order.
        """
        order = list(range(len(tasks)))
        random.Random(self.version).shuffle(order)
        size = self.config.lanes
        chunks = [order[i : i + size] for i in range(0, len(order), size)]
        jobs = [
            (
                str(self.weights),
                self.version,
                None if as_deployed else self.config.learner,
                self.config.lanes,
                [tasks[i] for i in chunk],
            )
            for chunk in chunks
        ]
        records: list = [None] * len(tasks)
        for chunk, played in zip(chunks, self.pool.map(_play_job, jobs)):
            for i, record in zip(chunk, played):
                records[i] = record
        return records

    def with_scenarios(self, tasks: Sequence[EpisodeTask]) -> list[EpisodeTask]:
        """The tasks with their layouts made once (for sets played every round)."""
        order = list(range(len(tasks)))
        random.Random(len(tasks)).shuffle(order)  # slow families spread over the workers
        made = dict(zip(order, self.pool.map(task_scenario, [tasks[i] for i in order])))
        return [dataclasses.replace(t, scenario=made[i]) for i, t in enumerate(tasks)]

    def close(self) -> None:
        self.pool.shutdown(cancel_futures=True)


# ── Learning ──────────────────────────────────────────────────────────────────


def _batches(episodes: Sequence[EpisodeRecord], batch_frames: int, rng: random.Random):
    order = list(range(len(episodes)))
    rng.shuffle(order)
    batch, frames = [], 0
    for index in order:
        episode = episodes[index]
        if batch and frames + episode.frames > batch_frames:
            yield batch
            batch, frames = [], 0
        batch.append(episode)
        frames += episode.frames
    if batch:
        yield batch


def _padded(episodes: Sequence[EpisodeRecord], device):
    count, length = len(episodes), max(e.frames for e in episodes)
    a = torch.zeros(count, length, SEQ_LEN_A, dtype=torch.long)
    b = torch.zeros(count, length, SEQ_LEN_B, dtype=torch.long)
    c = torch.zeros(count, length, SEQ_LEN_C)
    buttons = torch.full((count, length), NOOP, dtype=torch.long)
    valid = torch.zeros(count, length, dtype=torch.bool)
    for i, e in enumerate(episodes):
        t = e.frames
        a[i, :t] = torch.from_numpy(e.src_a.astype(np.int64))
        b[i, :t] = torch.from_numpy(e.src_b.astype(np.int64))
        c[i, :t] = torch.from_numpy(e.src_c.astype(np.float32))
        buttons[i, :t] = torch.from_numpy(e.buttons.astype(np.int64))
        valid[i, :t] = True
    before = torch.cat((torch.full((count, 1), NOOP, dtype=torch.long), buttons[:, :-1]), dim=1)
    return [x.to(device) for x in (a, b, c, buttons, before, valid)]


def reward_advantages(
    episode: EpisodeRecord, discount: float, smoothing: float, scale: float = 1.0
) -> tuple[np.ndarray, np.ndarray]:
    """Per decision: how much better than expected it turned out (advantage), and
    the return the learner's estimate should have been.

    A decision earns the rewards of the frames until the next decision,
    discounted per frame; the estimate after the last decision is zero when
    the episode ended (death or goal) and the last estimate after a timeout.
    Rewards are multiplied by ``scale`` first.
    """
    frames, values = episode.decision_frames, episode.old_value
    count = len(frames)
    stops = np.append(frames[1:], episode.frames)
    advantages = np.zeros(count, np.float32)
    following = 0.0
    for d in reversed(range(count)):
        span = episode.rewards[frames[d] : stops[d]] * scale
        earned = float((span * discount ** np.arange(len(span))).sum())
        if d + 1 < count:
            after = values[d + 1]
        else:
            after = 0.0 if episode.terminal else values[d]
        carry = discount ** len(span)
        delta = earned + carry * after - values[d]
        following = delta + carry * smoothing * (following if d + 1 < count else 0.0)
        advantages[d] = following
    return advantages, advantages + values


def _decisions(
    episodes: Sequence[EpisodeRecord],
    device,
    learner: str,
    discount=0.995,
    smoothing=0.95,
    scale=1.0,
):
    """Every decision in the batch, flattened, with masks for those the teacher
    labelled and those the learner chose by sampling (reward rounds)."""
    parts = defaultdict(list)
    for i, e in enumerate(episodes):
        count = len(e.decision_frames)
        if not count:
            continue
        parts["episode"].append(np.full(count, i))
        parts["frame"].append(e.decision_frames)
        parts["target"].append(e.targets)
        parts["given"].append(e.given)
        parts["labelled"].append(e.labels["valid"].astype(bool))
        parts["explored"].append(np.full(count, e.explored) & ~e.played_teacher)
        for name, value in e.labels.items():
            if name != "valid":
                parts[f"label_{name}"].append(np.asarray(value))
        if e.explored:
            advantages, returns = reward_advantages(e, discount, smoothing, scale)
        else:
            advantages = returns = np.zeros(count, np.float32)
        parts["advantage"].append(advantages)
        parts["return"].append(returns)
        parts["old_log_prob"].append(
            e.old_log_prob if e.old_log_prob is not None else np.zeros(count, np.float32)
        )
        width = CHOICE_WIDTH[learner]
        parts["used"].append(e.used if e.used is not None else np.zeros((count, width), np.float32))
        for head, value in e.picks.items():
            parts[f"pick_{head}"].append(value)
    if not parts:
        return None
    floats = ("target", "given", "advantage", "return", "old_log_prob", "used")
    return {
        name: torch.as_tensor(
            np.concatenate(values),
            dtype=torch.float32
            if name in floats
            else (torch.bool if name in ("labelled", "explored") else torch.long),
            device=device,
        )
        for name, values in parts.items()
    }


def action_memory(policy, a, b, c, d):
    """Replay the memory over every action of a batch of episodes.

    At each decision (an action's start), before anything decides, the memory
    takes the latest picture: the summary of that frame's encoded scene. It is
    never told the action.
    Returns, per decision, the memory's state after that step, from which it
    predicts the scene at the action's end.
    """
    e, f = d["episode"], d["frame"]
    summaries = policy.scene(a[e, f], b[e, f], c[e, f])[0][:, 0]
    # Arrange the decisions of each episode in order: [episodes, actions, width].
    count = int(e.max()) + 1
    per_episode = torch.bincount(e, minlength=count)
    first = torch.cumsum(per_episode, 0) - per_episode
    position = torch.arange(len(e), device=e.device) - first[e]
    window = summaries.new_zeros(count, int(per_episode.max()), summaries.shape[-1])
    window[e, position] = summaries
    return policy.memory.sequence(window)[e, position]


def choice_history(used, episode):
    """For every decision of a batch (episode after episode, in order): the
    learner's choices used at its previous HISTORY decisions in the same
    episode, the most recent first, and which of them exist. ``used``: [D,
    width]; ``episode``: [D] episode numbers. Returns ([D, HISTORY, width],
    [D, HISTORY])."""
    count = int(episode.max()) + 1
    per_episode = torch.bincount(episode, minlength=count)
    first = torch.cumsum(per_episode, 0) - per_episode
    index = torch.arange(len(episode), device=episode.device)
    position = index - first[episode]
    ages = torch.arange(1, HISTORY + 1, device=episode.device)
    exists = position[:, None] - ages[None, :] >= 0
    earlier = (index[:, None] - ages[None, :]).clamp_min(0)
    return used[earlier] * exists.unsqueeze(-1), exists


def learner_losses(policy, learner: str, episodes, expectation_weight: float, device, rl=None):
    """The learner's losses on a batch of episodes.

    - imitation, on every decision the teacher labelled: the teacher's choice;
    - reward (``rl``, a LayeredTrainConfig, on decisions the learner sampled):
      the clipped policy-gradient objective, its estimate of the return, and
      an entropy bonus;
    - the memory's expectation (action learner only: it alone trains the
      scene encoder, the window and the memory): at each action's start, the
      scene the vision transformer reports when that action ends.
    """
    a, b, c, _, _, _ = _padded(episodes, device)
    d = _decisions(
        episodes,
        device,
        learner,
        *((rl.discount, rl.advantage_smoothing, rl.reward_scale) if rl is not None else ()),
    )
    if d is None:
        return {}, {}
    e, f = d["episode"], d["frame"]
    # Reward rounds change only the learner's own layer: what imitation built of
    # the scene encoder, the window and the memory stays as it is.
    trains_memory = learner == "action" and rl is None
    with torch.set_grad_enabled(trains_memory):
        state = action_memory(policy, a, b, c, d)
        numbers, expected = policy.expect(MemoryState(state, torch.zeros_like(state)))
    losses = {}
    # An action ends on the frame the next action starts on (same episode); the
    # last action of an episode that ended in death or the goal has no picture.
    ends = e[1:] == e[:-1]
    if trains_memory and ends.any():
        scene_numbers = slice(0, C_SPANS["c_target"][0])
        predicted = numbers[:-1][ends][:, scene_numbers]
        actual = c[e[1:][ends], f[1:][ends]][:, scene_numbers]
        losses["expectation"] = expectation_weight * (predicted - actual).pow(2).mean()
    rows_c = c[e, f]
    if learner == "action":
        rows_c = rows_c.clone()
        rows_c[:, TARGET_SPAN] = d["target"]
    out = policy.layer_outputs(
        (a[e, f], b[e, f], rows_c),
        expected,
        {GIVEN[learner]: d["given"]},
        {learner: choice_history(d["used"], e)},
    )
    out = out[learner]
    stats = {"decisions": int(len(e))}
    imitation = 1.0 if rl is None else rl.imitation_weight
    m = d["labelled"]
    if m.any() and imitation > 0:
        label = {
            name[len("label_") :]: value[m]
            for name, value in d.items()
            if name.startswith("label_")
        }
        if learner == "action":
            losses["action"] = imitation * F.cross_entropy(out["action"][m], label["action"])
            frames = out["frames"][m].view(-1, len(SMB_ACTIONS), FRAME_BINS)
            chosen = frames[torch.arange(len(frames)), label["action"]]
            losses["frames"] = imitation * F.cross_entropy(chosen, label["frames"])
            agree = out["action"][m].argmax(-1) == label["action"]
            stats["action_accuracy"] = float(agree.float().mean())
            same_length = chosen.argmax(-1) == label["frames"]
            stats["frames_accuracy"] = float(same_length.float().mean())
        elif learner == "skill":
            losses["skill"] = imitation * F.cross_entropy(out["skill"][m], label["skill"])
            losses["direction"] = imitation * F.cross_entropy(
                out["direction"][m], label["direction"]
            )
            pointer = out["pointer"][m]
            reachable = torch.isfinite(pointer.gather(1, label["pointer"][:, None])).squeeze(1)
            if reachable.any():
                losses["pointer"] = imitation * F.cross_entropy(
                    pointer[reachable], label["pointer"][reachable]
                )
            agree = out["skill"][m].argmax(-1) == label["skill"]
            stats["skill_accuracy"] = float(agree.float().mean())
        else:
            losses["tactic"] = imitation * F.cross_entropy(out["tactic"][m], label["tactic"])
            losses["direction"] = imitation * F.cross_entropy(
                out["direction"][m], label["direction"]
            )
            agree = out["tactic"][m].argmax(-1) == label["tactic"]
            stats["tactic_accuracy"] = float(agree.float().mean())
    x = d["explored"]
    if rl is not None and x.any():
        picks = {head: d[f"pick_{head}"][x] for head in CHOICES[learner]}
        log_prob, entropy = choice_log_prob(learner, {k: v[x] for k, v in out.items()}, picks)
        advantage = d["advantage"][x]
        advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-6)
        ratio = torch.exp(log_prob - d["old_log_prob"][x])
        clipped = torch.clamp(ratio, 1 - rl.clip, 1 + rl.clip)
        losses["reward"] = -torch.minimum(ratio * advantage, clipped * advantage).mean()
        losses["estimate"] = rl.value_weight * F.mse_loss(out["value"][x], d["return"][x])
        losses["entropy"] = -rl.entropy_weight * entropy.mean()
        stats["mean_return"] = float(d["return"][x].mean())
    return losses, stats


# ── Rounds ────────────────────────────────────────────────────────────────────


def learner_families(learner: str, families: Sequence[str]) -> tuple[str, ...]:
    """The families a learner trains and is tested on. The tactic layer leaves
    out the clones, whose tactic is given rather than decided by the scene."""
    from .tactic_families import CLONE_FAMILIES

    if learner == "tactic":
        return tuple(f for f in families if f not in CLONE_FAMILIES)
    return tuple(families)


def _tasks(
    config,
    split: str,
    layouts: int,
    round_index: int,
    share: float,
    label: bool,
    explore: bool = False,
    extra: Optional[dict] = None,
):
    """Training tasks: ``layouts`` per family (plus ``extra``), each layout's
    difficulty drawn by config.difficulty_weights. Held-out tasks: ``layouts``
    per family at each difficulty."""
    tasks = []
    rng = random.Random(f"{config.seed}|{split}|{round_index}")
    for family in learner_families(config.learner, config.families):
        if split == "train":
            count = layouts + (extra or {}).get(family, 0)
            difficulties = rng.choices(DIFFICULTIES, weights=config.difficulty_weights, k=count)
        else:
            difficulties = [d for d in DIFFICULTIES for _ in range(layouts)]
        for i, difficulty in enumerate(difficulties):
            tasks.append(
                EpisodeTask(
                    index=len(tasks),
                    family=family,
                    split=split,
                    seed=config.seed,
                    sample_index=round_index * 1000 + i,
                    difficulty=difficulty,
                    teacher_share=share,
                    label=label,
                    frames=config.episode_frames,
                    explore=explore,
                )
            )
    return tasks


def family_success(episodes: Sequence[EpisodeRecord], by_difficulty: bool = False) -> dict:
    """Share of episodes won per family (or per family and difficulty)."""
    wins = defaultdict(list)
    for e in episodes:
        wins[f"{e.family}:{e.difficulty}" if by_difficulty else e.family].append(e.won)
    return {name: float(np.mean(values)) for name, values in sorted(wins.items())}


def _file_digest(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_layered_checkpoint(path, policy, config, trained_layers, history) -> None:
    torch.save(
        {
            "settings": asdict(policy.settings),
            "state_dict": policy.state_dict(),
            "observation_layout": observation_layout(),
            "token_layout": token_layout(),
            "vision_checkpoint": config.vision_checkpoint,
            "vision_sha256": _file_digest(config.vision_checkpoint),
            "trained_layers": list(trained_layers),
            "config": asdict(config),
            "history": history,
        },
        path,
    )


def load_layered_checkpoint(path, device="cpu"):
    """A saved policy and its checkpoint; refuses one built for other inputs or tokens."""
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    if checkpoint["observation_layout"] != observation_layout():
        raise ValueError(f"{path} was trained on a different observation layout")
    if checkpoint["token_layout"] != token_layout():
        raise ValueError(f"{path} was trained with different tokens")
    policy = LayeredSMBPolicy(PolicySettings(**checkpoint["settings"])).to(device)
    policy.load_state_dict(checkpoint["state_dict"])
    return policy, checkpoint


def train_layer(config: LayeredTrainConfig) -> dict:
    """Train one layer of the agent in Block SMB; returns the run summary."""
    output = Path(config.output)
    output.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(config.seed)
    rng = random.Random(config.seed)
    device = torch.device(config.device)
    trained_layers: list[str] = []
    if config.init:
        policy, checkpoint = load_layered_checkpoint(config.init, device)
        trained_layers = list(checkpoint["trained_layers"])
    else:
        policy = LayeredSMBPolicy(
            PolicySettings(
                width=config.policy_width,
                scene_depth=config.policy_scene_depth,
                layer_depth=config.policy_layer_depth,
                memory_width=config.policy_memory_width,
            )
        ).to(device)
    below = LEARNERS[: LEARNERS.index(config.learner)]
    missing = [layer for layer in below if layer not in trained_layers]
    if missing:
        raise ValueError(f"train the {', '.join(missing)} layer(s) first (pass --init)")

    # Only the learner changes; every other part stays exactly as it was. In
    # reward rounds only the learner's own layer changes.
    def learn_only(parameters):
        for parameter in policy.parameters():
            parameter.requires_grad_(False)
        for parameter in parameters:
            parameter.requires_grad_(True)
        return parameters

    learning = learn_only(policy.parameters_of(config.learner))
    optimizer = torch.optim.AdamW(learning, lr=config.learning_rate)
    pool = EpisodePool(config, policy.settings)
    validation_tasks = pool.with_scenarios(
        _tasks(config, "validation", config.validation_layouts_per_difficulty, 0, 0.0, False)
    )
    replay: list[EpisodeRecord] = []
    focus: dict[str, int] = {}  # extra layouts per family, from the last validation
    history: list[dict] = []
    best = best_passed = -1.0
    try:
        for round_index in range(config.rounds + config.reward_rounds):
            started = time.time()
            # Imitation rounds, the teacher's share shrinking; then reward rounds,
            # in which the learner samples its own choices and learns from what
            # they earn (on that round's episodes only).
            by_reward = round_index >= config.rounds
            if by_reward and round_index == config.rounds:
                learning = learn_only(list(getattr(policy, config.learner).parameters()))
                optimizer = torch.optim.AdamW(learning, lr=config.reward_learning_rate)
            fraction = min(1.0, round_index / max(1, config.rounds - 1))
            share = (
                0.0
                if by_reward
                else config.teacher_share_start
                + fraction * (config.teacher_share_end - config.teacher_share_start)
            )
            pool.publish(policy)
            tasks = _tasks(
                config,
                "train",
                config.train_layouts_per_family,
                round_index,
                share,
                True,
                explore=by_reward,
                extra=focus,
            )
            played = pool.play(tasks)
            replay = (replay + played)[-config.replay_episodes :]
            play_time = time.time() - started

            started = time.time()
            policy.train()
            totals: dict[str, list] = defaultdict(list)
            for _ in range(config.epochs_per_round):
                for batch in _batches(played if by_reward else replay, config.batch_frames, rng):
                    losses, stats = learner_losses(
                        policy,
                        config.learner,
                        batch,
                        config.expectation_weight,
                        device,
                        rl=config if by_reward else None,
                    )
                    if not losses:
                        continue
                    loss = sum(losses.values())
                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(learning, 1.0)
                    optimizer.step()
                    for name, value in losses.items():
                        totals[name].append(float(value))
                    for name, value in stats.items():
                        totals[name].append(float(value))
            policy.eval()
            learn_time = time.time() - started

            started = time.time()
            pool.publish(policy)
            validation = pool.play(validation_tasks)
            evaluate_time = time.time() - started
            per_family = family_success(validation)
            focus = {
                family: round(config.focus_layouts * (1.0 - success))
                for family, success in per_family.items()
            }
            mean = float(np.mean(list(per_family.values())))
            agreed = [bool(x) for e in played for x in e.agreed[e.labels["valid"].astype(bool)]]
            entry = {
                "round": round_index,
                "by_reward": by_reward,
                "teacher_share": share,
                "train_success": float(np.mean([e.won for e in played])),
                "agreement": float(np.mean(agreed)) if agreed else None,
                "validation_success": mean,
                "validation_families": per_family,
                "validation_difficulties": family_success(validation, by_difficulty=True),
                "losses": {name: float(np.mean(values)) for name, values in totals.items()},
                "replay_episodes": len(replay),
                "seconds": {
                    "play": round(play_time, 1),
                    "learn": round(learn_time, 1),
                    "evaluate": round(evaluate_time, 1),
                },
            }
            if config.learner == LEARNERS[-1]:
                # The top layer trained in Block SMB: also the whole agent as it plays,
                # every token its own (the strategy is the default until Full SMB).
                deployed = family_success(pool.play(validation_tasks, as_deployed=True))
                entry["deployed_validation_success"] = float(np.mean(list(deployed.values())))
                entry["deployed_validation_families"] = deployed
            history.append(entry)
            print(_round_line(config.learner, entry), flush=True)
            gate_met = all(value >= config.family_gate for value in per_family.values())
            layers = trained_layers + ([config.learner] if gate_met else [])
            save_layered_checkpoint(output / "last.pt", policy, config, layers, history)
            if mean > best:
                best = mean
                save_layered_checkpoint(output / "best.pt", policy, config, layers, history)
            if gate_met and mean > best_passed:
                # The best round that met the gate: the next layer starts from it.
                best_passed = mean
                save_layered_checkpoint(output / "passed.pt", policy, config, layers, history)
            (output / "history.json").write_text(json.dumps(history, indent=2))
    finally:
        pool.close()
    return {"best_validation_success": best, "history": history}


def _round_line(learner: str, entry: dict) -> str:
    losses = " ".join(f"{name}={value:.3f}" for name, value in entry["losses"].items())
    weakest = sorted(entry["validation_families"].items(), key=lambda item: item[1])[:4]
    agreement = entry["agreement"]
    return (
        f"[{learner}] round {entry['round']:02d} teacher share {entry['teacher_share']:.2f}: "
        f"train wins {entry['train_success']:.2f}, "
        f"agrees with teacher {agreement if agreement is None else round(agreement, 3)}, "
        f"validation wins {entry['validation_success']:.3f} "
        f"(weakest {', '.join(f'{f} {v:.2f}' for f, v in weakest)}); {losses}; "
        f"seconds {entry['seconds']}"
    )


def examine_layer(
    checkpoint: str,
    learner: Optional[str],
    *,
    layouts_per_difficulty: int = 6,
    first_layout: int = 100,
    workers: int = 12,
    families: Sequence[str] = tuple(BLOCK_SMB_MC_FAMILIES),
    vision_checkpoint: str = LayeredTrainConfig.vision_checkpoint,
) -> dict:
    """Play fresh held-out layouts (never used for validation) with a saved policy.

    With a ``learner``, the teacher gives it its token from above, as in its
    validation; with none, the whole agent plays as deployed.
    """
    policy, _ = load_layered_checkpoint(checkpoint, "cpu")
    config = LayeredTrainConfig(
        learner=learner or "tactic",
        families=tuple(families),
        workers=workers,
        vision_checkpoint=vision_checkpoint,
        output=str(Path(checkpoint).parent),
    )
    tasks = [
        dataclasses.replace(task, sample_index=first_layout + task.sample_index)
        for task in _tasks(config, "validation", layouts_per_difficulty, 0, 0.0, False)
    ]
    pool = EpisodePool(config, policy.settings)
    pool.weights = Path(checkpoint).with_name("examined_policy.pt")
    try:
        pool.publish(policy)
        played = pool.play(tasks, as_deployed=learner is None)
    finally:
        pool.close()
    families_won = family_success(played)
    return {
        "checkpoint": str(checkpoint),
        "learner": learner,
        "episodes": len(played),
        "success": float(np.mean([e.won for e in played])),
        "families": families_won,
        "difficulties": family_success(played, by_difficulty=True),
        "below_gate": {f: v for f, v in families_won.items() if v < config.family_gate},
    }
