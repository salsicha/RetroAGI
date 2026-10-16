"""Block SMB training for the layered agent (core.layered_policy).

Every policy input comes from the Block vision transformer: workers play
episodes, show each screen to the vision transformer, and pack what it
reports (core.smb_observer). The simulator is read only by the teachers
(teacher_tokens), by the rewards, and by the scoring of episodes.

Layers are trained bottom-up, one at a time, with the others frozen:

- skill: given the tactic, the strategy switch, the scene, its 16 prior
  actions and the executor's feedback, choose an action (a verb aimed at an
  object); the teacher carries it out (the ground truth the world model
  will stand for; core/world_model);
- tactic: reading the layout's strategy switch, hold a tactic over many
  actions and know when it is finished: the tactic layer is an option-critic
  (core.layered_policy.TacticLayer) with its own memory network.

Each learner is taught by imitation with corrections: workers play with the
policy, a teacher labels every decision with what it would have done from that
exact state, and the learner trains on every labelled decision so far. In the
first round every decision plays the teacher's choice; later the chance
shrinks to zero, so the policy learns to recover from its own mistakes.

The tactic learner is taught, at every decision, the teacher's tactic (what to
choose) and whether the tactic it holds is finished (the teacher's tactic is
no longer it); its memory learns to predict the scene when each tactic ends;
and its critic learns the reward to expect from holding each tactic. In
reward rounds the option-critic rules then improve its choices and its end
check, once its critic predicts held-out returns well enough (critic_ready).

Workers run on the graphics card: each plays several episodes side by side
and shows their screens to the vision transformer together.
"""

import dataclasses
import hashlib
import json
import multiprocessing
import random
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from retroagi.core.action_tokens import VERB_TARGETS, VERBS, ActionToken, pointer_index
from retroagi.core.actions import SMBAction
from retroagi.core.layered_policy import (
    CHOICE_WIDTH,
    CHOICES,
    HELD_WIDTH,
    HISTORY,
    POINTER_START,
    LayeredSMBPolicy,
    PolicySettings,
    choice_log_prob,
    encode_choice,
)
from retroagi.core.smb_agent import SMBAgents
from retroagi.core.smb_observer import (
    C_SPANS,
    SEQ_LEN_A,
    SEQ_LEN_B,
    SEQ_LEN_C,
    VisionObserver,
    observation_layout,
)
from retroagi.core.smb_executor import HOLD_FRAMES
from retroagi.core.tokens import (
    COMMAND_FIELDS,
    EXECUTION_WIDTH,
    TACTICS,
    SkillToken,
    TacticToken,
    command_row,
    encode_strategy,
    encode_tactic,
    token_layout,
)

from .monte_carlo import BLOCK_SMB_MC_DIFFICULTY_BINS as DIFFICULTIES
from .monte_carlo import BLOCK_SMB_MC_FAMILIES

LEARNERS = ("skill", "tactic")
# The token each learner is given from above (the tactic layer reads the
# strategy switch, which every episode sets).
GIVEN = {"skill": "tactic", "tactic": None}
# The C row's scene numbers (what the memory predicts), before the unused rest.
SCENE_NUMBERS = slice(0, C_SPANS["c_reserved"][0])
NOOP = int(SMBAction.NOOP)


@dataclass
class LayeredTrainConfig:
    learner: str = "skill"
    families: tuple[str, ...] = tuple(BLOCK_SMB_MC_FAMILIES)
    rounds: int = 12
    train_layouts_per_family: int = 8
    # Extra layouts next round for a family that lost validation episodes:
    # this many times its share of losses.
    focus_layouts: int = 8
    # Extra layouts every round for named families, "family=count,...": more
    # practice where the skill is weakest (extra_layouts_of).
    extra_layouts: Optional[str] = None
    # Optional full parameter sweep for explicitly selected small skill
    # families. Normal skill training samples all skill families with focus replay.
    sweep: bool = False
    # A family and difficulty with more combinations than this is not made: the
    # run stops and names it (its parameters need a smaller space first).
    sweep_limit: int = 20_000
    # Held-out layouts per family and difficulty (easy, medium, hard), fixed for the run.
    validation_layouts_per_difficulty: int = 3
    # Training layouts' difficulty: easy, medium and hard in these proportions.
    difficulty_weights: tuple[float, float, float] = (2.0, 2.0, 1.0)
    # ... and a family's extra layouts' (extra_layouts, focus_layouts): the
    # same by default; more hard ones where the skill fails on hard layouts.
    extra_difficulty_weights: tuple[float, float, float] = (2.0, 2.0, 1.0)
    teacher_share_start: float = 1.0  # chance a decision plays the teacher's choice, first round
    teacher_share_end: float = 0.0  # ... and last round
    epochs_per_round: int = 8
    batch_frames: int = 8192
    learning_rate: float = 3e-4
    # Weight decay of the optimizer (AdamW), regularizing the learned layers.
    weight_decay: float = 0.05
    expectation_weight: float = 1.0  # the tactic memory's expected scene, against imitation
    replay_episodes: int = 6000  # at least the latest round is always kept
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
    # The tactic learner (an option-critic):
    switching_cost: float = 0.01  # subtracted from ending a tactic (scaled reward units)
    end_positive_weight: float = 4.0  # "finished" labels are rare; their weight in the end check
    critic_ready: float = 0.5  # held-out explained variance the critic needs before reward updates
    end_gate: float = 0.7  # agreement with the teacher on where tactics change, to pass
    end_tolerance: int = 2  # decisions a tactic change may be off and still agree
    seed: int = 0
    device: str = "cuda"
    vision_checkpoint: str = "data/block_vit/block_vit_scene.pth"
    # Retired (the action predictor is replaced by the world model): kept so
    # saved configs load; refused when set.
    predictor: Optional[str] = None
    # The collision trackers (tracker_train; None: none), stepped every frame
    # in play; not used while the layers train.
    tracker: Optional[str] = None
    # Skill training: the scene encoder the skill reads learns with it (else
    # only the skill's own transformer does). The world model reads it too:
    # one trained on the old encoder refuses the new one.
    train_reading: bool = False
    init: Optional[str] = None  # a checkpoint whose lower layers are already trained
    # The size of a new policy (ignored with init).
    policy_width: int = PolicySettings.width
    policy_scene_depth: int = PolicySettings.scene_depth
    policy_layer_depth: int = PolicySettings.layer_depth
    policy_memory_width: int = PolicySettings.memory_width
    policy_tactic_memory_width: int = PolicySettings.tactic_memory_width
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
    weight: float = 1.0  # how much its decisions count in learning
    # Record, at each labelled skill decision, the teacher's choice and every
    # jump it could make, as actions with their outcomes (the action
    # predictor's data; teacher_actions.action_rows). Slow: off in training.
    record_actions: bool = False
    # Record the simulator's truth every frame (truth_recording), which true
    # enemy each scene slot shows, and the executor's kinematics: the data the
    # collision trackers' and the world model's labels are made from
    # (core/collision_labels). Training only.
    record_truth: bool = False
    # How the skill's own actions are carried out (EXECUTIONS): by the
    # teacher (training); or, to measure the prediction networks, by the
    # world model, with the collision trackers ending waits and walks and
    # aiming stomps, or both.
    execution: str = "teacher"


# How the skill's own actions are carried out (EpisodeTask.execution):
# teacher: its version of them (training); world_model: the world model's
# predicted outcome (stomps aimed by the simulator's truth, waits as long as
# predicted); trackers: the teacher's, but waits left open and ended by the
# collision trackers, stomps aimed by them; deployed: both networks.
EXECUTIONS = ("teacher", "world_model", "trackers", "deployed")
# Why an action ended (core.smb_agent AgentStep.ended), as recordings code it.
END_REASONS = ("", "landed", "done", "threat", "hold_recheck", "arrived", "no_progress")
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
    src_c: np.ndarray  # [T, SEQ_LEN_C] float16
    buttons: np.ndarray  # [T] int8, pressed at each frame
    decision_frames: np.ndarray  # [D] int32
    given: np.ndarray  # [D, width] float32, the token the learner was given
    labels: dict  # name -> [D] arrays, with "valid" [D] bool
    played_teacher: np.ndarray  # [D] bool
    agreed: np.ndarray  # [D] bool, the policy chose what the teacher would have
    execution_feedback: Optional[np.ndarray] = None  # [D,8], measured prior command outcome
    # Skill learner, per decision: the teacher's verb at that moment (index
    # into VERBS; -1 when it has no named choice) and whether the skill's own
    # choice (verb and object) was the teacher's. Asked at every decision the
    # skill plays itself, labelled or not (carry_out needs it): how its
    # held-out choices are judged (choices_against_teacher).
    teacher_verb: Optional[np.ndarray] = None
    same_as_teacher: Optional[np.ndarray] = None
    # Whether the last action ended before the episode did (else it was cut
    # off by the episode's end: its length is not known).
    last_action_ended: bool = False
    rewards: Optional[np.ndarray] = None  # [T] float32, the reward after each frame
    terminal: bool = False  # ended by death or the goal (not a timeout)
    explored: bool = False  # the learner sampled its choices
    picks: dict = field(default_factory=dict)  # head -> [D] the learner's own picks
    old_log_prob: Optional[np.ndarray] = None  # [D] their log-probability when played
    old_value: Optional[np.ndarray] = None  # [D] the learner's estimate of the return then
    used: Optional[np.ndarray] = None  # [D, choice width] the learner's choice used, per decision
    # [T] the goal-distance shaping's potential after each frame (its weight
    # times Mario's distance to the goal; 0 without shaping).
    potentials: Optional[np.ndarray] = None
    # Tactic learner, per decision (core.smb_agent.TacticStep): "held" (tactic
    # index held coming in, -1 none), "actions" and "frames" (how long),
    # "started", "used" (tactic index), "own_end", "end_probability".
    tactic: dict = field(default_factory=dict)
    weight: float = 1.0  # how much its decisions count in learning (EpisodeTask.weight)
    # [D, ACTIONS_PER_DECISION, len(ACTION_RECORD)] (core/world_model),
    # with EpisodeTask.record_actions; else None.
    actions: Optional[np.ndarray] = None
    # With EpisodeTask.record_truth: the simulator's state each frame and one
    # more after the last button (truth_recording.TruthRecorder.arrays), the
    # true enemy (or moving platform) each sprite slot shows [T, SPRITE_SLOTS]
    # (-1: none), and the executor's kinematics the networks read
    # (core/smb_kinematics): per sprite slot [T, SPRITE_SLOTS, 6], Mario's
    # [T, 3].
    truth: Optional[dict] = None
    slot_match: Optional[np.ndarray] = None
    sprite_kinematics: Optional[np.ndarray] = None
    mario_kinematics: Optional[np.ndarray] = None
    # ... and each sprite slot's track identity [T, SPRITE_SLOTS] (0: none),
    # what vision shows of each enemy slot [T, ENEMY_SLOTS, len(CONTEXT)]
    # (smb_kinematics.enemy_context), and where Mario is [T, 2] (the
    # executor's world displacement and his feet's row): the collision
    # trackers' inputs are made from these (target_tracker.record_features).
    slot_identity: Optional[np.ndarray] = None
    enemy_context: Optional[np.ndarray] = None
    mario_position: Optional[np.ndarray] = None
    # Always: why an action ended at each frame [T] (END_REASONS index; 0
    # none), and per decision of the skill learner the command executed [D, 6]
    # (COMMAND_FIELDS; mode -1 for an exact button plan) and the action it
    # carried out (verb [D], pointer slot [D]; -1 unknown): the teacher's
    # choice as named where the teacher played, else the skill's own.
    ended: Optional[np.ndarray] = None
    commands: Optional[np.ndarray] = None
    action_verb: Optional[np.ndarray] = None
    action_pointer: Optional[np.ndarray] = None

    @property
    def frames(self) -> int:
        return int(len(self.buttons))


# ── Labels ────────────────────────────────────────────────────────────────────


# ── Workers ───────────────────────────────────────────────────────────────────

_WORKER: dict = {}


def _initialize_worker(
    vision_checkpoint: str,
    settings: dict,
    device: str,
    tracker: Optional[str] = None,
) -> None:
    from .vision import load_block_vit_checkpoint

    torch.set_num_threads(1)
    vision = load_block_vit_checkpoint(Path(vision_checkpoint), device=device).model
    policy = LayeredSMBPolicy(PolicySettings(**settings)).to(device).eval()
    _WORKER.update(
        observer=VisionObserver(vision),
        policy=policy,
        device=device,
        version=None,
        **_execution_models(tracker, device),
    )


def _execution_models(tracker: Optional[str], device) -> dict:
    """The collision trackers from their checkpoint (None where none is given)."""
    from .tracker_train import load_tracker

    return {"tracker": load_tracker(tracker, device).eval() if tracker else None}


def _play_job(job) -> list[EpisodeRecord]:
    weights, version, learner, lanes, tasks = job
    if _WORKER["version"] != version:
        _WORKER["policy"].load_state_dict(
            torch.load(weights, map_location=_WORKER["device"], weights_only=True)
        )
        _WORKER["version"] = version
    return play_episodes(
        _WORKER["observer"],
        _WORKER["policy"],
        learner,
        tasks,
        _WORKER["device"],
        lanes,
        tracker=_WORKER["tracker"],
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


def _rows(values, count: int) -> np.ndarray:
    """[count, width] float rows (width 0 when nothing was recorded)."""
    return (
        np.asarray(values, np.float32).reshape(count, -1) if count else np.zeros((0, 0), np.float32)
    )


class _Lane:
    """One episode being played: the simulator, its teacher, and the records."""

    def __init__(self, task: EpisodeTask, copy: int):
        from .env import MarioScenarioEnv
        from .teacher_tokens import episode_teacher, teacher_strategy

        scenario = task.scenario if task.scenario is not None else task_scenario(task)
        self.task = task
        self.copy = copy
        # A long layout gets the frames its teacher's route is allowed.
        self.frame_limit = max(task.frames, int(scenario.get("frame_budget", 0)))
        self.env = MarioScenarioEnv()
        self.screen, _ = self.env.reset(scenario=scenario, seed=0)
        # Pre-episode pictures the agent watches first (prefilled history);
        # they are recorded as the episode's first frames, with no decisions.
        self.watching = list(self.env.watched_screens)
        self.frame_limit += len(self.watching)
        self.teacher = episode_teacher(scenario)
        # The strategy switch the episode is played with: the layout's strategy
        # and the side its goal is on (set by whoever runs the agent).
        self.switch = teacher_strategy(self.teacher)
        self.rng = random.Random(task.index * 7919 + task.sample_index)
        self.last_action_ended = False
        self.goal = 0.0
        self.end = ""
        # The skill chose an action the teacher has no safe version of
        # (carry_out): the episode ends as lost.
        self.unsafe_choice = False
        # The teacher played a decision and had no command: the episode ends
        # there (the decision has no label, so teacher qualification fails the
        # layout either way; held and asked again every frame until the stall
        # limit, one such layout took an hour and a half).
        self.no_teacher_command = False
        self.frames: dict[str, list] = defaultdict(list)
        self.decisions: dict[str, list] = defaultdict(list)
        # The simulator's truth (EpisodeTask.record_truth); the watched
        # pictures' was kept at reset.
        self.truth = None
        if task.record_truth:
            from .truth_recording import TruthRecorder

            self.truth = TruthRecorder()
        self.watched_truth = list(self.env.watched_truth)
        self.asked: dict = {}
        self.progress, self.progressed = None, 0

    def _as_action(self, command, result=None):
        """The teacher's ``command`` (with its trial ``result``, if made) as
        {"action", "outcome"} (teacher_actions.teacher_example), or None."""
        from .teacher_actions import teacher_example

        return teacher_example(self.env, self.teacher, command, result)

    def carry_out(self, action):
        """The executor's destination for the skill's own ``action`` here: the
        teacher's version of it (teacher_actions.carry_out), used instead of
        the world model, which is not used while the layers train. None when the teacher
        has no safe version: the episode then ends as lost.

        The teacher's own choice is asked for first when it was not (an
        unlabelled episode): its command is the version of that action (its
        own search reaches commands, such as a run-up before a committed jump
        or a wait, that carry_out's does not)."""
        from .teacher_actions import carry_out

        if "action_label" not in self.asked:
            from .teacher_tokens import teacher_plan, teacher_skill

            plan = (
                None
                if self.teacher.family == "enemy_patrol"
                else teacher_plan(self.env, self.teacher, certify_holds=False)[0]
            )
            own = teacher_skill(self.env, self.teacher, plan)
            self.asked["skill"] = own
            self.asked["action_label"] = self._as_action(own) if own is not None else None
        label = (self.asked.get("action_label") or {}).get("action")
        if action.verb == "hold":
            from .teacher_tokens import whole_wait

            # The teacher's hold, played: as long as its whole wait.
            self.asked["skill"] = whole_wait(self.env, self.teacher, self.asked.get("skill"))
        command = carry_out(self.env, self.teacher, action, label, self.asked.get("skill"))
        if command is None:
            self.unsafe_choice = True
        return command

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
        """Only compute teacher destinations when consumed or needed as labels.

        Skill evaluation only needs its tactic, unless executing a teacher choice.
        Reads simulator state for teacher inputs and supervision only.
        """
        from .teacher_tokens import teacher_plan, teacher_skill, teacher_tactic

        asked = {
            "tactic": teacher_tactic(self.env, self.teacher),
            "action": None,
            "skill": None,
            "holds": (),
            "plays_teacher": self.rng.random() < self.task.teacher_share,
        }
        if learner == "skill" and (self.task.label or asked["plays_teacher"]):
            if self.teacher.family != "enemy_patrol":
                asked["action"], asked["holds"] = teacher_plan(
                    self.env, self.teacher, certify_holds=False
                )
            asked["skill"] = teacher_skill(self.env, self.teacher, asked["action"])
            if asked["plays_teacher"]:
                from .teacher_tokens import whole_wait

                # Played: a hold as long as its whole wait.
                asked["skill"] = whole_wait(self.env, self.teacher, asked["skill"])
            if self.task.label and asked["skill"] is not None:
                from .teacher_actions import _trial_of

                # The same choice as an action (a verb aimed at a reported
                # object; None when its object is not reported) and where it
                # ends; its trial, made before it is executed, is kept for
                # naming run-ups (_name_run_ups).
                asked["label_result"] = _trial_of(self.env, self.teacher, asked["skill"])
                asked["action_label"] = self._as_action(asked["skill"], asked["label_result"])
            if self.task.record_actions and self.task.label and asked["skill"] is not None:
                from .teacher_actions import action_rows

                asked["actions"] = action_rows(self.env, self.teacher, asked["skill"])
        self.asked = asked
        return asked

    def note_decision(self, learner: str, decision) -> None:
        """Record a decision, the token the learner was given and the teacher's label."""
        d, asked = self.decisions, self.asked
        d["frame"].append(len(self.frames["a"]) - 1)
        d["played_teacher"].append(asked["plays_teacher"])
        if learner == "tactic":
            self._note_tactic(decision)
            return
        for head, value in decision.picks[learner].items():
            d[f"pick_{head}"].append(value)
        d["log_prob"].append(decision.log_prob[learner])
        d["value"].append(decision.value[learner])
        # The skill's: the action used (None for the teacher's command, given)
        # and where the executor was sent, as its history holds it at play.
        used = (decision.action, decision.skill) if learner == "skill" else decision.skill
        d["used"].append(encode_choice(learner, used).numpy())
        mine = decision.chosen[learner]
        if learner == "skill":
            if self.task.record_actions:
                from retroagi.core.world_model import empty_records

                found = asked.get("actions")
                d["actions"].append(found if found is not None else empty_records())
            d["execution_feedback"].append(decision.execution_feedback)
            d["given"].append(
                torch.cat(
                    (encode_tactic(decision.tactic), encode_strategy(decision.strategy))
                ).numpy()
            )
            valid = self.task.label and asked["skill"] is not None
            d["label_valid"].append(valid)
            example = asked.get("action_label")
            action = example["action"] if example is not None else None
            d["label_action_valid"].append(valid and action is not None)
            d["label_verb"].append(VERBS.index(action.verb) if action is not None else 0)
            d["label_pointer"].append(
                pointer_index(action.target) - POINTER_START
                if action is not None and action.target is not None
                else 0
            )
            # Where the teacher's action ends (pixels from Mario's feet).
            outcome = example["outcome"] if example is not None else {"dx": 0, "dy": 0}
            d["label_end_dx"].append(round(outcome["dx"]))
            d["label_end_dy"].append(round(outcome["dy"]))
            d["agreed"].append(valid and action is not None and mine == action)
            d["teacher_verb"].append(VERBS.index(action.verb) if action is not None else -1)
            d["same_as_teacher"].append(action is not None and mine == action)
            # What was executed: the command, and the skill's own action
            # (where the teacher played, its named choice is used: record).
            d["command"].append(command_row(decision.skill))
            own = decision.action
            d["own_verb"].append(VERBS.index(own.verb) if own is not None else -1)
            d["own_pointer"].append(
                pointer_index(own.target) - POINTER_START
                if own is not None and own.target is not None
                else -1
            )
            if self.task.label:
                self._name_run_ups(d, asked, decision)
            return

    def _name_run_ups(self, d, asked, decision) -> None:
        """A run-up of the teacher's (a run that keeps its speed) that its next
        command follows with a jump in the same direction is named after that
        jump: approaching what the
        jump is aimed at, in the run's own scene (teacher_actions.approach_slot,
        the rule the teacher's remembered plans and its executor use too), or
        a plain run to where it ends when the jump lands back on the surface
        the run started on. Without this, whether such a run was labelled an
        approach depended on whether the teacher had planned the jump before
        running (its committed plan), which the scene does not show. Called
        at each decision of a labelled skill episode, after its labels are
        recorded."""
        from .teacher_actions import approach_slot, run_context, surface_slot

        env, command = self.env, asked.get("skill")
        pending, self._pending_run = getattr(self, "_pending_run", None), None
        # The jump's trial made before it started (ask_teacher): its execution
        # has begun by now.
        result = asked.get("label_result")
        if (
            pending is not None
            and command is not None
            and result is not None
            and command.mode == "jump"
            and command.x != 0
            and (command.x > 0) == pending["right"]
            and env.mario["on_ground"]
        ):
            slot = approach_slot(pending["context"], env, result)
            if slot is not None and slot[0] in VERB_TARGETS["approach"]:
                action = ActionToken("approach", slot)
            else:
                action = pending["plain"]
            if action is not None:
                t = pending["index"]
                d["label_verb"][t] = VERBS.index(action.verb)
                d["label_pointer"][t] = pointer_index(action.target) - POINTER_START
                d["agreed"][t] = pending["mine"] == action
        example = asked.get("action_label")
        action = example["action"] if example is not None else None
        if (
            action is not None
            and action.verb in ("run_to", "approach")
            and command is not None
            and command.carry  # a run-up: one that stops is a plain run
            and command.x != 0
            and decision.skill == command
            and env.mario["on_ground"]
        ):
            context = run_context(env, self.teacher)
            plain = action if action.verb == "run_to" else None
            run = asked.get("label_result")
            if plain is None and run is not None:
                # The same run as a plain run: to the surface it ends on.
                m = env.mario
                slot = surface_slot(
                    env,
                    context["scene"],
                    m["x"] + m["w"] / 2 + run.end_dx,
                    m["y"] + m["h"] + run.end_dy,
                )
                plain = ActionToken("run_to", slot) if slot is not None else None
            self._pending_run = {
                "index": len(d["label_verb"]) - 1,
                "context": context,
                "right": command.x > 0,
                "mine": decision.chosen["skill"],
                "plain": plain,
            }

    def _note_tactic(self, decision) -> None:
        """The tactic learner's decision: what its layer did (TacticStep), and the
        teacher's labels: its tactic here, and whether the held tactic is
        finished (the teacher's tactic is no longer it)."""
        d, step = self.decisions, decision.tactic_step
        teacher: TacticToken = self.asked["tactic"]
        d["given"].append(encode_strategy(decision.strategy).numpy())
        d["label_tactic"].append(TACTICS.index(teacher.stance))
        d["label_end"].append(step.held is not None and teacher.stance != step.held)
        d["label_valid"].append(True)
        d["pick_tactic"].append(TACTICS.index(step.own))
        d["log_prob"].append(step.choice_log_prob)
        d["value"].append(0.0)
        d["used"].append(encode_choice("tactic", decision.tactic).numpy())
        d["tactic_held"].append(-1 if step.held is None else TACTICS.index(step.held))
        d["tactic_actions"].append(step.actions)
        d["tactic_frames"].append(step.frames)
        d["tactic_started"].append(step.started)
        d["tactic_used"].append(TACTICS.index(decision.tactic.stance))
        d["tactic_own_end"].append(step.own_end)
        d["tactic_end_probability"].append(
            -1.0 if step.end_probability is None else step.end_probability
        )
        d["agreed"].append(step.own == teacher.stance)

    def _executed_actions(self) -> dict:
        """{"action_verb", "action_pointer"} per skill decision: the action
        carried out (the teacher's choice as named, run-ups renamed, where it
        played; else the skill's own); -1 unknown. Empty when not recorded."""
        d = self.decisions
        if not d["own_verb"]:
            return {}
        verb = np.asarray(d["own_verb"], np.int64)
        pointer = np.asarray(d["own_pointer"], np.int64)
        named = np.asarray(d["label_action_valid"], bool) & np.asarray(d["played_teacher"], bool)
        verb[named] = np.asarray(d["label_verb"], np.int64)[named]
        pointer[named] = np.asarray(d["label_pointer"], np.int64)[named]
        unknown = np.asarray(d["played_teacher"], bool) & ~named
        verb[unknown], pointer[unknown] = -1, -1
        return {"action_verb": verb.astype(np.int8), "action_pointer": pointer.astype(np.int16)}

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
            given=_rows(d["given"], count),
            labels=labels,
            played_teacher=np.asarray(d["played_teacher"], bool),
            agreed=np.asarray(d["agreed"], bool),
            execution_feedback=(
                np.asarray(d["execution_feedback"], np.float32) if d["execution_feedback"] else None
            ),
            teacher_verb=np.asarray(d["teacher_verb"], np.int64) if d["teacher_verb"] else None,
            truth=self.truth.arrays() if self.truth is not None else None,
            slot_match=(
                np.asarray(self.frames["slot_match"], np.int16) if self.truth is not None else None
            ),
            sprite_kinematics=(
                np.asarray(self.frames["sprite_kinematics"], np.float16)
                if self.truth is not None
                else None
            ),
            mario_kinematics=(
                np.asarray(self.frames["mario_kinematics"], np.float16)
                if self.truth is not None
                else None
            ),
            slot_identity=(
                np.asarray(self.frames["slot_identity"], np.int32) if self.truth is not None else None
            ),
            enemy_context=(
                np.asarray(self.frames["enemy_context"], np.float16)
                if self.truth is not None
                else None
            ),
            mario_position=(
                np.asarray(self.frames["mario_position"], np.float32)
                if self.truth is not None
                else None
            ),
            ended=np.asarray(self.frames["ended"], np.int8),
            commands=np.asarray(d["command"], np.int16).reshape(-1, len(COMMAND_FIELDS))
            if d["command"]
            else None,
            **self._executed_actions(),
            same_as_teacher=(
                np.asarray(d["same_as_teacher"], bool) if d["same_as_teacher"] else None
            ),
            last_action_ended=self.last_action_ended,
            rewards=np.asarray(self.frames["reward"], np.float32),
            terminal=self.end
            in ("goal", "death", "off_route", "missed_objective", "failed_attempt", "unsafe_choice"),
            explored=self.task.explore,
            picks={
                name[len("pick_") :]: np.asarray(d[name], np.int64)
                for name in d
                if name.startswith("pick_")
            },
            old_log_prob=np.asarray(d["log_prob"], np.float32),
            old_value=np.asarray(d["value"], np.float32),
            used=_rows(d["used"], count),
            potentials=np.asarray(self.frames["potential"], np.float32),
            weight=self.task.weight,
            actions=np.asarray(d["actions"], np.float32) if self.task.record_actions else None,
            tactic={
                name[len("tactic_") :]: np.asarray(d[name])
                for name in d
                if name.startswith("tactic_")
            },
        )


def _teacher_given(learner: str, lanes: dict, agents=None):
    """For the deciding copies: the teacher's token for the layer above the
    learner (none for the tactic learner, which reads the strategy switch), and
    the teacher's own choice for the learner where it plays the teacher this
    time. (Layers higher up do not run: nothing below needs them.)"""
    above = GIVEN[learner]

    def given(copies, scenes):
        tokens = {learner: [], **({above: []} if above else {})}
        for copy, scene in zip(copies, scenes):
            lanes[copy].teacher.execution = agents.copies[copy].spatial if agents else None
            lanes[copy].teacher.controller = agents.copies[copy].executor if agents else None
            lanes[copy].teacher.scene = scene
            lanes[copy].teacher.visual_observer = agents.observer if agents else None
            asked = lanes[copy].ask_teacher(learner, scene)
            if above:
                tokens[above].append(asked[above])
            choice = asked[learner] if asked["plays_teacher"] else None
            if choice is None and learner == "skill" and asked["plays_teacher"]:
                # The teacher plays this decision but has no command: hold, and
                # the episode ends. The decision has no label, so teacher
                # qualification (which plays only the teacher) counts the
                # layout as failed.
                from retroagi.core.tokens import SkillToken

                choice = SkillToken("hold", 0, 0)
                lanes[copy].no_teacher_command = True
            tokens[learner].append(choice)
        return tokens

    return given


def _potential(env) -> float:
    """The goal-distance shaping's potential now: its weight times Mario's
    (normalised) distance to the goal, which the shaping pays for reducing."""
    weight = env._goal_distance_shaping
    if weight <= 0.0 or env.goal is None or env._prev_goal_distance is None:
        return 0.0
    return float(weight * env._prev_goal_distance)


def play_episodes(
    observer: VisionObserver,
    policy: LayeredSMBPolicy,
    learner: Optional[str],
    tasks: Sequence[EpisodeTask],
    device,
    lanes: int,
    tracker=None,
) -> list[EpisodeRecord]:
    """Play ``tasks``, ``lanes`` at a time. The agent sees only the screens.

    With a ``learner``, the teachers give it its token from above and label
    its decisions, and the teacher carries out the skill's own actions (the
    ground truth the world model will stand for). With none, the agent plays
    exactly as deployed: every token is its own and nothing is given or
    labelled. ``tracker``: the collision trackers, stepped every frame
    (SMBAgents).
    """
    playing: dict[int, _Lane] = {}
    executions = {task.execution for task in tasks}
    if len(executions) > 1 or not executions <= set(EXECUTIONS):
        raise ValueError(f"one call plays one execution of {EXECUTIONS}, not {executions}")
    execution = executions.pop() if executions else "teacher"
    carry = None
    if learner is not None and execution in ("teacher", "trackers"):
        def carry(number, action):
            command = playing[number].carry_out(action)
            if execution == "trackers" and command is not None and command.mode == "hold":
                # The wait is left open: the collision trackers end it.
                return SkillToken("hold", 0, 0, frames=HOLD_FRAMES)
            return command

    agents = SMBAgents(
        observer,
        policy,
        device,
        lanes,
        tracker=tracker,
        carry_out=carry,
        world_model=execution in ("world_model", "deployed"),
        watch_threats=execution in ("trackers", "deployed"),
    )
    # The simulator's truth aims stomps where the trackers do not.
    exact = learner is not None and execution in ("teacher", "world_model")
    explore = {task.explore for task in tasks}
    if len(explore) > 1 or (learner is None and True in explore):
        raise ValueError("one call either explores, for a learner, or does not")
    sample = (learner,) if True in explore else ()
    pending = list(tasks)
    done: dict[int, EpisodeRecord] = {}
    given = _teacher_given(learner, playing, agents) if learner is not None else None
    while pending or playing:
        for copy in range(lanes):
            if copy not in playing and pending:
                playing[copy] = _Lane(pending.pop(0), copy)
                agents.reset(copy, playing[copy].switch, watch=len(playing[copy].watching))
                if exact:
                    # Training: the executor aims at walkers where the
                    # simulator moves them, as the teacher's trials do: the
                    # ground truth the target tracker's collision point will
                    # stand for (user, 2026-10-14/15).
                    from .controller_teacher import exact_positions

                    agents.copies[copy].spatial.tracks.exact = exact_positions(playing[copy].env)
        live = list(playing.values())
        if learner is not None:
            for lane in live:
                lane.teacher.observe_frame(lane.env)
        steps = agents.act(
            [lane.watching[0] if lane.watching else lane.screen for lane in live],
            [lane.copy for lane in live],
            given=given,
            run_given=(learner,) if learner is not None else (),
            sample=sample,
        )
        for lane, step in zip(live, steps):
            for name, value in zip("abc", step.rows):
                lane.frames[name].append(value)
            spatial = agents.copies[lane.copy].spatial
            if lane.truth is not None:
                from retroagi.core.smb_kinematics import (
                    enemy_context,
                    kinematics,
                    slot_identities,
                )

                from .truth_recording import match_slots

                # The state behind this picture (a watched one's was kept at reset).
                frame = lane.watched_truth.pop(0) if lane.watching else lane.env.true_frame()
                lane.truth.add(frame)
                sprites, mario = kinematics(step.scene, spatial)
                lane.frames["sprite_kinematics"].append(sprites)
                lane.frames["mario_kinematics"].append(mario)
                lane.frames["slot_match"].append(match_slots(step.scene, frame))
                lane.frames["slot_identity"].append(slot_identities(step.scene, spatial))
                lane.frames["enemy_context"].append(enemy_context(step.scene))
                feet = step.scene.mario.box[3] if step.scene.mario.box is not None else 0.0
                lane.frames["mario_position"].append((spatial.displacement, feet))
            lane.frames["ended"].append(
                END_REASONS.index(step.ended) if step.ended in END_REASONS else 0
            )
            if lane.watching:
                # A pre-episode picture, already played by the simulator at reset.
                lane.watching.pop(0)
                lane.frames["button"].append(step.button)
                lane.frames["reward"].append(0.0)
                lane.frames["potential"].append(_potential(lane.env))
                continue
            if step.decision is not None and learner is not None:
                lane.note_decision(learner, step.decision)
            lane.frames["button"].append(step.button)
            lane.screen, reward, terminated, truncated, info = lane.env.step(step.button)
            lane.frames["reward"].append(float(reward))
            lane.frames["potential"].append(_potential(lane.env))
            lane.goal += float(info["reward_terms"].get("goal", 0.0))
            if (
                len(lane.frames["button"]) >= lane.frame_limit
                or lane.stalled()
                or lane.unsafe_choice
                or lane.no_teacher_command
            ):
                truncated = True
            if terminated or truncated:
                lane.end = (
                    "goal"
                    if lane.goal > 0
                    else "unsafe_choice"
                    if lane.unsafe_choice
                    else "no_teacher_command"
                    if lane.no_teacher_command
                    else (
                        "death"
                        if info.get("death")
                        else (
                            "off_route"
                            if info.get("off_route")
                            else (
                                "missed_objective"
                                if info.get("objective_missed")
                                else "failed_attempt"
                                if info.get("attempt_failed")
                                else "timeout"
                            )
                        )
                    )
                )
                # A time-budget cutoff leaves the last action's length
                # unknown, unless the executor also ended it.
                executor = agents.copies[lane.copy].executor
                lane.last_action_ended = bool(
                    terminated or executor.finished or executor.reconsider
                )
                if lane.truth is not None:
                    # The state the last button left: the stomp or death it made.
                    lane.truth.add(lane.env.true_frame())
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
            initargs=(
                config.vision_checkpoint,
                asdict(settings),
                config.device,
                config.tracker,
            ),
        )

    def publish(self, policy: LayeredSMBPolicy) -> None:
        torch.save(policy.state_dict(), self.weights)
        self.version += 1

    def play(
        self, tasks: Sequence[EpisodeTask], *, as_deployed: bool = False, teacher_only: bool = False
    ) -> list[EpisodeRecord]:
        """Play the tasks for the learner, or (``as_deployed``) with no teacher at all.

        The tasks go out shuffled, one batch of lanes per job, so slow and fast
        families mix and every worker stays busy to the end; the records come
        back in the tasks' order.
        """
        if teacher_only and as_deployed:
            raise ValueError("teacher qualification cannot run as a deployed policy")
        learner = "skill" if teacher_only else (None if as_deployed else self.config.learner)
        order = list(range(len(tasks)))
        random.Random(self.version).shuffle(order)
        size = self.config.lanes
        chunks = [order[i : i + size] for i in range(0, len(order), size)]
        jobs = [
            (
                str(self.weights),
                self.version,
                learner,
                self.config.lanes,
                [tasks[i] for i in chunk],
            )
            for chunk in chunks
        ]
        records: list = [None] * len(tasks)
        pending = {self.pool.submit(_play_job, job): chunk for chunk, job in zip(chunks, jobs)}
        completed, reported = 0, time.monotonic()
        for future in as_completed(pending):
            chunk, played = pending[future], future.result()
            for i, record in zip(chunk, played):
                records[i] = record
            completed += len(chunk)
            if time.monotonic() - reported >= 60 or completed == len(tasks):
                print(
                    f"[{self.config.learner}] rollouts {completed}/{len(tasks)} episodes",
                    flush=True,
                )
                reported = time.monotonic()
        return records

    def with_scenarios(self, tasks: Sequence[EpisodeTask]) -> list[EpisodeTask]:
        """The tasks with their layouts made once (for sets played every round)."""
        order = list(range(len(tasks)))
        random.Random(len(tasks)).shuffle(order)  # slow families spread over the workers
        pending = {
            self.pool.submit(task_scenario, tasks[i]): i for i in order if tasks[i].scenario is None
        }
        made = {i: task.scenario for i, task in enumerate(tasks) if task.scenario is not None}
        reported = time.monotonic()
        for future in as_completed(pending):
            made[pending[future]] = future.result()
            if time.monotonic() - reported >= 60 or len(made) == len(tasks):
                print(
                    f"[{self.config.learner}] layouts {len(made)}/{len(tasks)} prepared", flush=True
                )
                reported = time.monotonic()
        return [dataclasses.replace(t, scenario=made[i]) for i, t in enumerate(tasks)]

    def close(self) -> None:
        self.pool.shutdown(cancel_futures=True)


# ── Learning ──────────────────────────────────────────────────────────────────


def kept_episodes(replay, played, capacity: int) -> list:
    """The episodes learned from: the newest ``capacity``, but never fewer than
    the latest round, all of which is kept (a full sweep plays every
    combination each round, in family order; cutting it would drop whole
    families)."""
    return (list(replay) + list(played))[-max(capacity, len(played)) :]


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
        parts["given"].append(e.given)
        parts["execution_feedback"].append(
            e.execution_feedback
            if e.execution_feedback is not None
            else np.zeros((count, EXECUTION_WIDTH), np.float32)
        )
        parts["labelled"].append(e.labels["valid"].astype(bool))
        parts["explored"].append(np.full(count, e.explored) & ~e.played_teacher)
        parts["weight"].append(np.full(count, e.weight, np.float32))
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
    floats = (
        "given",
        "advantage",
        "return",
        "old_log_prob",
        "used",
        "weight",
        "execution_feedback",
    )
    return {
        name: torch.as_tensor(
            np.concatenate(values),
            dtype=(
                torch.float32
                if name in floats
                else (torch.bool if name in ("labelled", "explored") else torch.long)
            ),
            device=device,
        )
        for name, values in parts.items()
    }


def _weighted_mean(values, weights):
    """The mean of ``values`` with each counted by its weight."""
    return (values * weights).sum() / weights.sum().clamp_min(1e-12)


def action_duration_summary(episodes):
    """Measured completed command durations; censored tails are not short actions."""
    durations = []
    for episode in episodes:
        boundaries = list(episode.decision_frames)
        if episode.last_action_ended and boundaries:
            boundaries.append(episode.frames)
        durations.extend(int(value) for value in np.diff(boundaries))
    return {
        "completed": len(durations),
        "max_frames": max(durations, default=0),
        "over_64_frames": sum(value > 64 for value in durations),
    }


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
      an entropy bonus.

    The skill reads no prediction (user, 2026-10-15): the world model's LSTM
    is trained apart, from recorded ground truth.

    The tactic learner's losses are tactic_losses (with ``rl`` or the default
    settings).
    """
    if learner == "tactic":
        config = rl or LayeredTrainConfig(learner="tactic", expectation_weight=expectation_weight)
        return tactic_losses(policy, episodes, config, device, by_reward=rl is not None)
    d = _decisions(
        episodes,
        device,
        learner,
        *((rl.discount, rl.advantage_smoothing, rl.reward_scale) if rl is not None else ()),
    )
    if d is None:
        return {}, {}
    e, f = d["episode"], d["frame"]
    losses = {}
    a, b, c, _, _, _ = _padded(episodes, device)
    # The pointer is read with a verb already chosen: the teacher's where it
    # labelled the decision with an action (teaching), the learner's own
    # elsewhere.
    m = d["labelled"]
    acted = m & d["label_action_valid"].bool()
    own_verb = d["pick_verb"].long() if "pick_verb" in d else torch.zeros_like(e)
    teaching = {"verb": torch.where(acted, d["label_verb"].long(), own_verb)}
    out = policy.run_skill(
        policy.encode_scene((a[e, f], b[e, f], c[e, f])),
        d["given"],
        choice_history(d["used"], e),
        feedback=d["execution_feedback"],
        picks=teaching,
    )
    stats = {"decisions": int(len(e))}
    imitation = 1.0 if rl is None else rl.imitation_weight
    if acted.any() and imitation > 0:
        # The action: its verb, and (except hold, which aims at nothing) the
        # object it is aimed at among those in view. A useful next action at
        # a failed command counts more, instead of letting the many easy
        # approach decisions drown out the stall-repair examples.
        weight = d["weight"][acted]
        stalled = d["execution_feedback"][acted, 1] > 0.5
        weight = weight * torch.where(stalled, 3.0, 1.0)
        verb = d["label_verb"][acted].long()
        losses["verb"] = imitation * _weighted_mean(
            F.cross_entropy(out["verb"][acted], verb, reduction="none"), weight
        )
        stats["verb_accuracy"] = float((out["verb"][acted].argmax(-1) == verb).float().mean())
        aimed = verb != VERBS.index("hold")
        if aimed.any():
            pointer = d["label_pointer"][acted][aimed].long()
            logits = out["pointer"][acted][aimed]
            losses["pointer"] = imitation * _weighted_mean(
                F.cross_entropy(logits, pointer, reduction="none"), weight[aimed]
            )
            stats["pointer_accuracy"] = float((logits.argmax(-1) == pointer).float().mean())
    x = d["explored"]
    if rl is not None and x.any():
        picks = {head: d[f"pick_{head}"][x] for head in CHOICES[learner]}
        # Read with the learner's own picks as chosen, as it chose them.
        own = policy.skill.heads_given(
            out["decision"][x], picks, out["objects"][x], out["object_present"][x]
        )
        log_prob, entropy = choice_log_prob(learner, own, picks)
        advantage = d["advantage"][x]
        advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-6)
        ratio = torch.exp(log_prob - d["old_log_prob"][x])
        clipped = torch.clamp(ratio, 1 - rl.clip, 1 + rl.clip)
        losses["reward"] = -torch.minimum(ratio * advantage, clipped * advantage).mean()
        losses["estimate"] = rl.value_weight * F.mse_loss(out["value"][x], d["return"][x])
        losses["entropy"] = -rl.entropy_weight * entropy.mean()
        stats["mean_return"] = float(d["return"][x].mean())
    return losses, stats


# ── The tactic learner: an option-critic ──────────────────────────────────────


def shaped_rewards(episode: EpisodeRecord, discount: float, scale: float) -> np.ndarray:
    """[T] the episode's rewards (times ``scale``) with the goal-distance shaping
    made exactly potential-based for this discount, so it cannot change which
    tactic is best: the simulator pays the weight times the drop in distance
    each frame; adding (1 - discount) times the potential after the frame, and
    the last potential when the episode ended by death or the goal, makes the
    shaping discount * potential(next) - potential(now), with no potential at
    the end."""
    rewards = episode.rewards.astype(np.float64)
    potentials = (
        episode.potentials.astype(np.float64)
        if episode.potentials is not None
        else np.zeros_like(rewards)
    )
    shaped = rewards + (1.0 - discount) * potentials
    if episode.terminal and len(shaped):
        shaped[-1] += discount * potentials[-1]
    return (shaped * scale).astype(np.float32)


def decision_rewards(episode: EpisodeRecord, discount: float, scale: float):
    """Per decision: the discounted (shaped, scaled) reward of its frames, and how
    many frames it lasted (to the next decision or the episode's end)."""
    rewards = shaped_rewards(episode, discount, scale)
    frames = episode.decision_frames
    stops = np.append(frames[1:], episode.frames)
    earned = np.zeros(len(frames), np.float32)
    for d, (start, stop) in enumerate(zip(frames, stops)):
        span = rewards[start:stop]
        earned[d] = float((span * discount ** np.arange(len(span))).sum())
    return earned, (stops - frames).astype(np.int64)


def tactic_batch(episodes: Sequence[EpisodeRecord], config, device):
    """Every decision of a batch of tactic-learner episodes, flattened in order."""
    parts = defaultdict(list)
    for i, e in enumerate(episodes):
        count = len(e.decision_frames)
        if not count:
            continue
        earned, lengths = decision_rewards(e, config.discount, config.reward_scale)
        parts["episode"].append(np.full(count, i))
        parts["frame"].append(e.decision_frames)
        parts["switch"].append(e.given)
        parts["earned"].append(earned)
        parts["length"].append(lengths)
        parts["last"].append(np.arange(count) == count - 1)
        parts["terminal"].append(np.full(count, e.terminal))
        parts["labelled"].append(e.labels["valid"].astype(bool))
        parts["label_tactic"].append(e.labels["tactic"])
        parts["label_end"].append(e.labels["end"].astype(bool))
        parts["teacher"].append(e.played_teacher.astype(bool))
        parts["explored"].append(np.full(count, e.explored))
        parts["old_log_prob"].append(e.old_log_prob)
        for name in ("held", "actions", "frames", "started", "used"):
            parts[name].append(e.tactic[name])
    if not parts:
        return None
    floats = ("switch", "earned", "old_log_prob", "actions", "frames")
    flags = ("last", "terminal", "labelled", "label_end", "teacher", "explored", "started")
    return {
        name: torch.as_tensor(
            np.concatenate(values),
            dtype=(
                torch.float32 if name in floats else (torch.bool if name in flags else torch.long)
            ),
            device=device,
        )
        for name, values in parts.items()
    }


def held_rows(held, actions, frames):
    """[D, HELD_WIDTH] held_features for batches of (tactic index or -1, actions, frames)."""
    rows = torch.zeros(len(held), HELD_WIDTH, device=held.device)
    rows[torch.arange(len(held), device=held.device), held.where(held >= 0, len(TACTICS))] = 1.0
    rows[:, -2] = torch.log1p(actions) / np.log1p(64)
    rows[:, -1] = torch.log1p(frames) / np.log1p(1024)
    return rows


def tactic_memory_replay(policy, inputs, started, episode):
    """Replay the tactic memory over each episode's tactic starts, as play
    stepped it. ``inputs`` [D, width] (TacticMemory.step_inputs, used only where
    ``started``); ``started``, ``episode`` [D], episode after episode in order.

    Returns per decision the memory's hidden state before this decision (after
    the episode's last earlier start; zeros before the first) and after it
    (after this decision's own start, if it is one).
    """
    width = policy.tactic_memory.width
    count = int(episode.max()) + 1
    per_episode = torch.bincount(episode, minlength=count)
    first = torch.cumsum(per_episode, 0) - per_episode
    starts = started.long()
    total = torch.cumsum(starts, 0)
    before_episode = (total - starts)[first][episode]  # starts in earlier episodes
    after_rank = total - before_episode  # starts in this episode up to here, inclusive
    before_rank = after_rank - starts
    steps = started.nonzero().squeeze(1)
    if not len(steps):
        zeros = inputs.new_zeros(len(episode), width)
        return zeros, zeros
    window = inputs.new_zeros(count, int(after_rank.max()), inputs.shape[-1])
    window[episode[steps], after_rank[steps] - 1] = inputs[steps]
    hidden = policy.tactic_memory.sequence(window)
    padded = torch.cat((hidden.new_zeros(count, 1, width), hidden), dim=1)
    return padded[episode, before_rank], padded[episode, after_rank]


def tactic_forward(policy, episodes, config, device, memory_grad: bool):
    """The tactic layer at every decision of a batch, rebuilt as play gave it.

    Returns the batch (tactic_batch) and, per decision: ``choose`` (the layer's
    outputs choosing, with no tactic held, from its memory after any start
    here), ``hold`` (where a tactic was held coming in: outputs with it and its
    age, from the memory before any start here), ``before`` (choosing outputs
    from the memory before any start here, where it differs: at ends),
    ``expected_end`` (the memory's predicted end scene at each start) and the
    frames' scene rows.
    """
    a, b, c, _, _, _ = _padded(episodes, device)
    d = tactic_batch(episodes, config, device)
    if d is None:
        return None
    e, f = d["episode"], d["frame"]
    with torch.no_grad():
        encoded = policy.scene(a[e, f], b[e, f], c[e, f])
    held = held_rows(d["held"], d["actions"], d["frames"])
    with torch.set_grad_enabled(memory_grad and torch.is_grad_enabled()):
        inputs = policy.tactic_memory.step_inputs(encoded[0][:, 0], held)
        before, after = tactic_memory_replay(policy, inputs, d["started"], e)
    none = held_rows(
        torch.full_like(d["held"], -1),
        torch.zeros_like(d["actions"]),
        torch.zeros_like(d["frames"]),
    )

    def run(index, rows, memory):
        """The layer's outputs at the decisions ``index`` (None for none)."""
        if not len(index):
            return None
        return policy.tactic_outputs(
            (encoded[0][index], encoded[1][index]),
            d["switch"][index],
            rows[index],
            memory[index],
        )

    everyone = torch.arange(len(e), device=e.device)
    out = {"choose": run(everyone, none, after)}
    out["holding"] = (d["held"] >= 0).nonzero().squeeze(1)
    out["hold"] = run(out["holding"], held, before)
    out["ends"] = (d["started"] & (d["held"] >= 0)).nonzero().squeeze(1)
    out["before"] = run(out["ends"], none, before)
    out["expected_end"] = policy.tactic_memory.expected_scene(after)
    out["scene"] = c[e, f]
    return d, out


def _state_value(out) -> torch.Tensor:
    """The value of the situation: each tactic's value weighted by the chance of choosing it."""
    return (torch.softmax(out["tactic"], -1) * out["values"]).sum(-1)


@torch.no_grad()
def critic_targets(d, out, config, smoothing: float) -> tuple[torch.Tensor, dict]:
    """The return each decision's tactic should be valued at (blending, by
    ``smoothing``, the critic's own next estimate with the returns that
    followed), and the values the targets and the option-critic updates read.

    The next estimate, for holding tactic w into the next decision, is the
    chance it continues times its value there plus the chance it ends times
    the value of choosing anew (all read before any start there). An episode
    ended by death or the goal is worth nothing after; after a timeout the last
    tactic's value stands in.
    """
    count = len(d["episode"])
    used = d["used"]
    q_after = out["choose"]["values"]
    v_after = _state_value(out["choose"])
    # Before any start at a decision: the same as after, except at ends.
    q_before, v_before = q_after.clone(), v_after.clone()
    if out["before"] is not None:
        q_before[out["ends"]] = out["before"]["values"]
        v_before[out["ends"]] = _state_value(out["before"])
    chance_end = torch.zeros(count, device=used.device)
    if out["hold"] is not None:
        chance_end[out["holding"]] = torch.sigmoid(out["hold"]["end"])
    index = torch.arange(count, device=used.device)
    nxt = (index + 1).clamp_max(count - 1)
    continuing = q_before[nxt, used]  # the same tactic, held into the next decision
    following = (1.0 - chance_end[nxt]) * continuing + chance_end[nxt] * v_before[nxt]
    bootstrap = torch.where(
        d["last"],
        torch.where(d["terminal"], torch.zeros_like(following), q_after[index, used]),
        following,
    )
    # Back from each episode's end (on the processor: one pass of plain numbers).
    bootstrap = bootstrap.cpu().numpy()
    carry = (config.discount ** d["length"].float()).cpu().numpy()
    earned = d["earned"].cpu().numpy()
    last = d["last"].cpu().numpy()
    returns = np.zeros(count, np.float32)
    later = 0.0
    for k in range(count - 1, -1, -1):
        blended = bootstrap[k] if last[k] else (1.0 - smoothing) * bootstrap[k] + smoothing * later
        later = earned[k] + carry[k] * blended
        returns[k] = later
    returns = torch.from_numpy(returns).to(used.device)
    return returns, {"q_before": q_before, "v_before": v_before, "v_after": v_after}


def termination_loss(end_logit, q_held, value, switching_cost: float) -> torch.Tensor:
    """The option-critic rule for the end check: lower the chance of ending
    where holding the tactic is worth more than choosing anew (by more than
    minus the switching cost), raise it where it is worth less."""
    advantage = (q_held - value + switching_cost).detach()
    return (torch.sigmoid(end_logit) * advantage).mean()


def tactic_losses(
    policy, episodes, config, device, *, by_reward: bool = False, critic_ready: bool = False
):
    """The tactic learner's losses on a batch of episodes.

    - imitation, on every labelled decision: the teacher's tactic, for the
      layer choosing there; and, where a tactic was held, whether it is
      finished (the teacher's tactic is no longer it);
    - the tactic memory's expectation (imitation rounds): at each tactic's
      start, the scene the vision transformer reports when it ends;
    - the critic: each decision's tactic valued at its return (critic_targets);
    - reward rounds, once the critic is ready: the clipped policy-gradient rule
      on the tactics the learner chose by sampling (advantage: the return over
      the value of the situation), an entropy bonus, and the option-critic rule
      for its end check (termination_loss).
    """
    made = tactic_forward(policy, episodes, config, device, memory_grad=not by_reward)
    if made is None:
        return {}, {}
    d, out = made
    losses, stats = {}, {"decisions": int(len(d["episode"]))}
    imitation = config.imitation_weight if by_reward else 1.0
    m = d["labelled"]
    choose, hold, holding = out["choose"], out["hold"], out["holding"]
    if m.any() and imitation > 0:
        losses["tactic"] = imitation * F.cross_entropy(choose["tactic"][m], d["label_tactic"][m])
        stats["tactic_accuracy"] = float(
            (choose["tactic"][m].argmax(-1) == d["label_tactic"][m]).float().mean()
        )
        held_labelled = m[holding] if hold is not None else m[:0]
        if held_labelled.any():
            finished = d["label_end"][holding][held_labelled].float()
            losses["end"] = imitation * F.binary_cross_entropy_with_logits(
                hold["end"][held_labelled],
                finished,
                pos_weight=torch.tensor(config.end_positive_weight, device=finished.device),
            )
            said = hold["end"][held_labelled] > 0
            stats["end_accuracy"] = float((said == finished.bool()).float().mean())
    if not by_reward:
        # Each tactic's start predicts the scene at the next start, same episode.
        starts = d["started"].nonzero().squeeze(1)
        nxt = torch.searchsorted(starts, starts, right=True)
        has_next = nxt < len(starts)
        if has_next.any():
            here, there = starts[has_next], starts[nxt[has_next]]
            same = d["episode"][here] == d["episode"][there]
            here, there = here[same], there[same]
            if len(here):
                predicted = out["expected_end"][here][:, SCENE_NUMBERS]
                actual = out["scene"][there][:, SCENE_NUMBERS]
                losses["tactic_expectation"] = (
                    config.expectation_weight * (predicted - actual).pow(2).mean()
                )
    returns, values = critic_targets(d, out, config, config.advantage_smoothing)
    used = d["used"]
    index = torch.arange(len(used), device=used.device)
    losses["estimate"] = config.value_weight * F.mse_loss(choose["values"][index, used], returns)
    stats["mean_return"] = float(returns.mean())
    if by_reward and critic_ready:
        own = d["explored"] & ~d["teacher"]
        chose = own & d["started"]
        if chose.any():
            log_prob, entropy = choice_log_prob(
                "tactic", {"tactic": choose["tactic"][chose]}, {"tactic": used[chose]}
            )
            advantage = returns[chose] - values["v_after"][chose]
            if len(advantage) > 1:
                advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-6)
            ratio = torch.exp(log_prob - d["old_log_prob"][chose])
            clipped = torch.clamp(ratio, 1 - config.clip, 1 + config.clip)
            losses["reward"] = -torch.minimum(ratio * advantage, clipped * advantage).mean()
            losses["entropy"] = -config.entropy_weight * entropy.mean()
        kept = own[holding] if hold is not None else own[:0]
        if kept.any():
            rows = holding[kept]
            losses["ending"] = termination_loss(
                hold["end"][kept],
                values["q_before"][rows, d["held"][rows]],
                values["v_before"][rows],
                config.switching_cost,
            )
    return losses, stats


@torch.no_grad()
def critic_check(policy, episodes, config, device) -> Optional[float]:
    """How well the critic predicts the returns that followed, on held-out
    episodes: the explained variance of each decision's tactic value against
    its actual (shaped, scaled, discounted) return. 1 is perfect; 0 is no
    better than one number for all."""
    predicted, actual = [], []
    rng = random.Random(0)
    for batch in _batches(list(episodes), config.batch_frames, rng):
        made = tactic_forward(policy, batch, config, device, memory_grad=False)
        if made is None:
            continue
        d, out = made
        returns, _ = critic_targets(d, out, config, smoothing=1.0)
        index = torch.arange(len(d["used"]), device=d["used"].device)
        predicted.append(out["choose"]["values"][index, d["used"]].cpu())
        actual.append(returns.cpu())
    if not actual:
        return None
    predicted, actual = torch.cat(predicted), torch.cat(actual)
    spread = float(actual.var())
    if spread <= 1e-12:
        return None
    return 1.0 - float((actual - predicted).var()) / spread


def end_agreement(episodes, tolerance: int) -> dict:
    """Agreement with the teacher on where tactics change, on episodes the
    learner played: each change of the learner's tactic is matched to a change
    of the teacher's within ``tolerance`` decisions. Returns the share matched
    both ways (2 x matched / (learner's + teacher's changes)), overall and per
    family; None where neither changed tactic."""
    counts = defaultdict(lambda: np.zeros(3, np.int64))  # matched, learner's, teacher's
    for e in episodes:
        if "used" not in e.tactic or not len(e.tactic["used"]):
            continue
        used, taught = np.asarray(e.tactic["used"]), np.asarray(e.labels["tactic"])
        mine = list(np.nonzero(used[1:] != used[:-1])[0] + 1)
        theirs = list(np.nonzero(taught[1:] != taught[:-1])[0] + 1)
        free = list(mine)
        matched = 0
        for t in theirs:
            near = [m for m in free if abs(m - t) <= tolerance]
            if near:
                free.remove(min(near, key=lambda m: abs(m - t)))
                matched += 1
        counts[e.family] += (matched, len(mine), len(theirs))

    def share(c):
        return None if c[1] + c[2] == 0 else float(2 * c[0] / (c[1] + c[2]))

    total = sum(counts.values(), np.zeros(3, np.int64))
    return {
        "overall": share(total),
        "families": {family: share(c) for family, c in sorted(counts.items())},
    }


# ── Rounds ────────────────────────────────────────────────────────────────────


def learner_families(learner: str, families: Sequence[str]) -> tuple[str, ...]:
    """The families a learner trains and is tested on. Each layer trains only on
    families at its own level:

    - Skill layer: local maneuvers under one supplied tactic, with spatial
      destination labels. Families that switch tactics are excluded.
    - Tactic layer: strategy families, scene-driven route/response families,
      composed levels, waiting/proceeding decisions and within-episode tactic switches.
    """
    from .tactic_families import TACTIC_TRAINING_FAMILIES

    if learner not in LEARNERS:
        raise ValueError(f"unknown learned layer {learner!r}; use skill or tactic")
    tactic_families = set(TACTIC_TRAINING_FAMILIES)
    own = set(BLOCK_SMB_MC_FAMILIES) - tactic_families if learner == "skill" else tactic_families
    return tuple(f for f in families if f in own)


def extra_layouts_of(config) -> dict:
    """LayeredTrainConfig.extra_layouts as {family: count}."""
    out = {}
    for part in (config.extra_layouts or "").split(","):
        if part.strip():
            family, count = part.split("=")
            if family.strip() not in BLOCK_SMB_MC_FAMILIES:
                raise ValueError(f"extra_layouts: unknown family {family.strip()!r}")
            out[family.strip()] = int(count)
    return out


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
    difficulty drawn by config.difficulty_weights (its extra layouts:
    extra_difficulty_weights). Held-out tasks: ``layouts``
    per family at each difficulty.

    Each family draws its difficulties from its own random sequence, so a
    family's layouts in a round are the same whatever extra layouts the other
    families get, and its extra layouts follow its usual ones: every layout a
    round can use is known in advance (and can be qualified before training).
    """
    tasks = []
    for family in learner_families(config.learner, config.families):
        if split == "train":
            rng = random.Random(f"{config.seed}|{split}|{round_index}|{family}")
            difficulties = rng.choices(DIFFICULTIES, weights=config.difficulty_weights, k=layouts)
            difficulties += rng.choices(
                DIFFICULTIES,
                weights=config.extra_difficulty_weights,
                k=(extra or {}).get(family, 0),
            )
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


def _prefix_job(job):
    family, difficulty, limit = job
    from .monte_carlo import combination_count, combination_prefixes

    count = combination_count(family, difficulty)
    if count > limit:
        return count, []
    return count, combination_prefixes(family, difficulty)


def _code_fingerprint() -> str:
    from .layout_cache import code_fingerprint

    return code_fingerprint()


def _combination_cache() -> Path:
    import os
    import tempfile

    return Path(os.environ.get("RETROAGI_COMBINATION_CACHE") or tempfile.gettempdir()) / (
        "retroagi_combinations"
    )


def _combination_job(job):
    """One part of a family's combinations (its prefix), from the disk cache when
    the same code made it before."""
    import pickle

    from .monte_carlo import block_smb_parameter_combinations

    family, difficulty, prefix, fingerprint = job
    name = f"{family}.{difficulty}.{'-'.join(map(str, prefix))}.{fingerprint}.pkl"
    path = _combination_cache() / name
    try:
        return family, difficulty, *pickle.loads(path.read_bytes())
    except (OSError, ValueError, EOFError, pickle.UnpicklingError):
        pass
    layouts, dropped = block_smb_parameter_combinations(family, difficulty, prefix=prefix)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        spare = path.with_suffix(".tmp")
        spare.write_bytes(pickle.dumps((layouts, dropped)))
        spare.replace(path)
    except OSError:
        pass
    return family, difficulty, layouts, dropped


def combination_tasks(config, pool) -> tuple[list[EpisodeTask], dict]:
    """An optional skill curriculum sweep: every combination of each family's
    drawn parameters at every difficulty (monte_carlo.block_smb_parameter_
    combinations), made once by the pool's workers. Each family weighs the
    same in learning: an episode's weight is the layouts per family on average
    over its own family's count. Returns the tasks (in order, with their
    layouts) and per family and difficulty how many layouts there are and how
    many combinations had no verified route."""
    parts = [
        (family, difficulty, config.sweep_limit)
        for family in learner_families(config.learner, config.families)
        for difficulty in DIFFICULTIES
    ]
    fingerprint = _code_fingerprint()
    # Each family and difficulty is split into parts by its first draws, so slow
    # families are made by many workers side by side.
    split = list(pool.pool.map(_prefix_job, parts))
    too_many = {
        f"{family}:{difficulty}": count
        for (family, difficulty, _), (count, _) in zip(parts, split)
        if count > config.sweep_limit
    }
    if too_many:
        raise ValueError(
            "a full sweep of these families would make more layouts than "
            f"sweep_limit ({config.sweep_limit}); their parameters need a smaller "
            f"space first: {too_many}"
        )
    jobs = [
        (family, difficulty, prefix, fingerprint)
        for (family, difficulty, _), (_, prefixes) in zip(parts, split)
        for prefix in prefixes
    ]
    jobs.sort(key=lambda job: job[0] not in ("moving_bridge", "bridge_wait", "wait_timing"))
    pieces = defaultdict(list)
    for family, difficulty, layouts, dropped in pool.pool.map(_combination_job, jobs):
        pieces[(family, difficulty)].append((layouts, dropped))
    made = []
    for family, difficulty, _ in parts:
        seen, layouts, dropped = set(), [], 0
        for part, missing in pieces[(family, difficulty)]:
            dropped += missing
            for layout in part:
                key = repr({k: v for k, v in layout.items() if k != "metadata"})
                if key not in seen:
                    seen.add(key)
                    layouts.append(layout)
        made.append((family, difficulty, layouts, dropped))
    counts = defaultdict(int)
    for family, _, layouts, _ in made:
        counts[family] += len(layouts)
    average = sum(counts.values()) / max(1, len(counts))
    tasks, summary = [], {}
    for family, difficulty, layouts, dropped in made:
        summary[f"{family}:{difficulty}"] = {"layouts": len(layouts), "no_route": dropped}
        for k, scenario in enumerate(layouts):
            tasks.append(
                EpisodeTask(
                    index=len(tasks),
                    family=family,
                    split="train",
                    seed=config.seed,
                    sample_index=k,
                    teacher_share=1.0,
                    label=True,
                    frames=config.episode_frames,
                    scenario=scenario,
                    difficulty=difficulty,
                    weight=average / counts[family],
                )
            )
    return tasks, summary


# A family's choices collapse on one value when that value covers more than
# COLLAPSE_SHARE of them (destinations within COLLAPSE_PIXELS count as one
# value), with at least COLLAPSE_MIN choices.
COLLAPSE_SHARE = 0.4
COLLAPSE_PIXELS = 2
COLLAPSE_MIN = 6


def top_share(values) -> tuple:
    """(value, share) of the most common of ``values``: destination x pixels
    (or (x, y) pairs), where values within COLLAPSE_PIXELS count as one, or
    plain choices such as tactic names."""
    if not values:
        return None, 0.0

    def same(v, w):
        if isinstance(v, tuple):
            return all(abs(a - b) <= COLLAPSE_PIXELS for a, b in zip(v, w))
        if isinstance(v, (int, float)):
            return abs(v - w) <= COLLAPSE_PIXELS
        return v == w

    best, count = None, 0
    for v in set(values):
        n = sum(1 for w in values if same(v, w))
        if n > count:
            best, count = v, n
    return best, count / len(values)


def _choices(episode, learner: str, source: str) -> dict:
    """An episode's choices by kind: for the skill, where each action sent
    Mario (pixels from his feet, (x, y)) by verb (a hold has no end); tactic
    names for the tactic layer. ``source``: "picks" (the learner's own, where
    it played its own choice) or "labels" (the teacher's, where valid)."""
    out: dict = defaultdict(list)
    if learner == "tactic":
        if source == "labels":
            valid = np.asarray(episode.labels.get("valid", ()), bool)
            values = np.asarray(episode.labels.get("tactic", ()))[valid]
        else:
            values = np.asarray(episode.picks.get("tactic", ()))
        out["tactic"] = [TACTICS[int(i)] for i in values]
        return out
    if source == "labels":
        if "end_dx" not in episode.labels:
            return out
        acted = np.asarray(episode.labels["valid"], bool) & np.asarray(
            episode.labels["action_valid"], bool
        )
        rows = zip(
            np.asarray(episode.labels["verb"])[acted],
            np.asarray(episode.labels["end_dx"])[acted],
            np.asarray(episode.labels["end_dy"])[acted],
        )
    else:
        if episode.used is None or "verb" not in episode.picks:
            return out
        own = ~np.asarray(episode.played_teacher, bool)
        if getattr(episode, "end", "") == "unsafe_choice" and own.any():
            # Its last own choice had no safe version (the teacher carried
            # nothing out; Mario held): it sent him nowhere.
            own[np.flatnonzero(own)[-1]] = False
        used = np.asarray(episode.used)[own]
        rows = zip(
            np.asarray(episode.picks["verb"])[own],
            np.round(used[:, len(VERBS)] * 256),
            np.round(used[:, len(VERBS) + 1] * 256),
        )
    for verb, dx, dy in rows:
        name = VERBS[int(verb)]
        if name != "hold":
            out[name].append((int(dx), int(dy)))
    return out


def label_collapse(episodes, learner: str, source: str = "labels") -> dict:
    """The kinds of choice whose labels collapse on one value across the whole
    curriculum: {"all families": {kind: (value, share, count)}} (top_share;
    at least COLLAPSE_MIN).

    A single family may teach one value where its scenes call for it (a pit
    nearly as wide as the longest jump is crossed by the longest jump: the
    value follows from the scene); what must not happen is one value for a
    kind of action whatever the scene, so the share is judged over every
    family's labels together. Whether a family's concentration is explained
    by its scenes is measured on the world model, which learns ends from
    scenes (world_model_train.explained)."""
    gathered: dict = defaultdict(list)
    for e in episodes:
        for kind, values in _choices(e, learner, source).items():
            gathered[kind].extend(values)
    found: dict = {}
    for kind, values in gathered.items():
        value, share = top_share(values)
        if len(values) >= COLLAPSE_MIN and share > COLLAPSE_SHARE:
            found.setdefault("all families", {})[kind] = (value, round(share, 3), len(values))
    return found


def _against_teacher(episodes):
    """Per family, the skill's own choices with the teacher's at the same
    moments: (own verb, teacher verb, the same choice) for each decision the
    skill played itself where the teacher had a named choice."""
    found: dict = defaultdict(list)
    for e in episodes:
        if e.teacher_verb is None or e.same_as_teacher is None or "verb" not in e.picks:
            continue
        own = ~np.asarray(e.played_teacher, bool) & (np.asarray(e.teacher_verb) >= 0)
        for verb, taught, same in zip(
            np.asarray(e.picks["verb"])[own],
            np.asarray(e.teacher_verb)[own],
            np.asarray(e.same_as_teacher)[own],
        ):
            found[e.family].append((VERBS[int(verb)], VERBS[int(taught)], bool(same)))
    return found


def choices_against_teacher(episodes) -> dict:
    """How the skill's own choices on these episodes compare with the
    teacher's choice at the same moments, per family: {family: {"decisions":
    n, "agreed": share choosing the teacher's action (verb and object),
    "differ": {"<own verb> for <teacher verb>" or "<verb> at another
    object": count}}}.

    The skill picks an action (a verb and an object); how far it goes is the
    teacher's version of it (carry_out), so its choices are judged here, at
    the same moments, not by where its actions ended."""
    report = {}
    for family, rows in sorted(_against_teacher(episodes).items()):
        differ = Counter(
            f"{own} at another object" if own == taught else f"{own} for {taught}"
            for own, taught, same in rows
            if not same
        )
        report[family] = {
            "decisions": len(rows),
            "agreed": round(sum(same for _, _, same in rows) / len(rows), 3),
            "differ": dict(differ.most_common()),
        }
    return report


def output_collapse(validation, played, learner: str) -> dict:
    """Families where the learner's own choices on held-out layouts collapse on
    one value much more than the teacher's do: {family: {kind: (value, own
    share, teacher share)}}. The learner collapses where its share is above
    COLLAPSE_SHARE and more than 0.2 above the teacher's.

    The skill: a verb it picks at those decisions against the teacher's
    choices at the same moments (value None; its run lengths are the
    teacher's, carry_out). The tactic layer: a tactic against the teacher's
    labels in this round's training episodes."""
    if learner == "skill":
        found: dict = {}
        for family, rows in _against_teacher(validation).items():
            own = Counter(verb for verb, _, _ in rows)
            taught = Counter(verb for _, verb, _ in rows)
            for verb, count in own.items():
                share, teacher_share = count / len(rows), taught[verb] / len(rows)
                if (
                    count >= COLLAPSE_MIN
                    and share > COLLAPSE_SHARE
                    and share > teacher_share + 0.2
                ):
                    found.setdefault(family, {})[verb] = (
                        None,
                        round(share, 3),
                        round(teacher_share, 3),
                    )
        return found
    teacher: dict = defaultdict(lambda: defaultdict(list))
    for e in played:
        for kind, values in _choices(e, learner, "labels").items():
            teacher[e.family][kind].extend(values)
    own: dict = defaultdict(lambda: defaultdict(list))
    for e in validation:
        for kind, values in _choices(e, learner, "picks").items():
            own[e.family][kind].extend(values)
    found: dict = {}
    for family, kinds in own.items():
        for kind, values in kinds.items():
            value, share = top_share(values)
            _, taught = top_share(teacher[family][kind])
            if len(values) >= COLLAPSE_MIN and share > COLLAPSE_SHARE and share > taught + 0.2:
                found.setdefault(family, {})[kind] = (value, round(share, 3), round(taught, 3))
    return found


def family_success(episodes: Sequence[EpisodeRecord], by_difficulty: bool = False) -> dict:
    """Share of episodes won per family (or per family and difficulty)."""
    wins = defaultdict(list)
    for e in episodes:
        wins[f"{e.family}:{e.difficulty}" if by_difficulty else e.family].append(e.won)
    return {name: float(np.mean(values)) for name, values in sorted(wins.items())}


def _file_digest(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_layered_checkpoint(
    path, policy, config, trained_layers, history, world_model=None
) -> None:
    """``world_model``: what its training measured ({"reads": the scene
    encoder's digest it was trained on, "report", "passed"}; None:
    untrained)."""
    torch.save(
        {
            "memory_version": 5,
            "world_model": world_model,
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
    """A saved policy and its checkpoint; refuses one built for other inputs or tokens.

    Tactic and skill both read strategy context. Legacy spatial checkpoints
    discard action weights; remaining learned tensors load strictly. Untrained
    strategy inputs may migrate, while qualified layers require matching tokens.
    """
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    if checkpoint["observation_layout"] != observation_layout():
        raise ValueError(f"{path} was trained on a different observation layout")
    saved, now = checkpoint["token_layout"], token_layout()
    old_executor = saved.get("executor", "predictive_spatial_v1") in (
        "predictive_spatial_v1",
        "predictive_spatial_v2",
        "goal_following_v1",
    )
    if old_executor:
        # Old qualifications do not certify the changed executor and target
        # semantics. Preserve compatible weights only as a warm start.
        checkpoint["trained_layers"] = []
        checkpoint.setdefault("load_migrations", []).append(
            "maneuver_targets_require_requalification"
        )
    saved_skill = dict(saved.get("skill") or {})
    legacy_skill = "strategy_context" not in saved_skill
    saved_feedback = saved_skill.get("execution_feedback")
    # No feedback at all, or fewer numbers than now (new ones are appended,
    # as "collision_due" was on 2026-10-15): the skill's reading of them
    # widens, the new ones starting with no influence.
    legacy_feedback = saved_feedback is None or (
        len(saved_feedback) < len(now["skill"]["execution_feedback"])
        and list(saved_feedback) == now["skill"]["execution_feedback"][: len(saved_feedback)]
    )
    if legacy_feedback:
        saved_skill["execution_feedback"] = now["skill"]["execution_feedback"]
        checkpoint["trained_layers"] = []
    if legacy_skill:
        saved_skill["strategy_context"] = now["skill"]["strategy_context"]
    if saved_skill.get("executed_by") == "action_predictor_v1":
        # Its actions were executed by the action predictor or, in training,
        # the teacher; now always the ground truth until the world model
        # predicts it (user, 2026-10-15).
        saved_skill["executed_by"] = now["skill"]["executed_by"]
    # A skill that chose destinations (a mode, x and y) instead of actions:
    # its outputs start untrained and the layer must be trained again.
    destination_skill = "modes" in saved_skill
    if destination_skill:
        saved_skill = dict(now["skill"])
        checkpoint["trained_layers"] = []
    other_strategies = saved.get("strategies") != now["strategies"]
    if (
        saved_skill != now["skill"]
        or (not old_executor and saved.get("executor") != now["executor"])
        or saved.get("executor_actions") != now["executor_actions"]
        or saved.get("tactics") != now["tactics"]
        or (
            other_strategies
            and (
                "tactic" in checkpoint["trained_layers"]
                or (not legacy_skill and "skill" in checkpoint["trained_layers"])
            )
        )
    ):
        raise ValueError(f"{path} was trained with different tokens")
    policy = LayeredSMBPolicy(PolicySettings(**checkpoint["settings"])).to(device)
    state = dict(checkpoint["state_dict"])
    if any(name.startswith("action.") for name in state):
        if saved.get("action_input") != "skill_only_v1":
            raise ValueError(f"{path} has an unsupported legacy action architecture")
        state = {name: value for name, value in state.items() if not name.startswith("action.")}
        checkpoint.setdefault("load_migrations", []).append("removed_action_network")
    checkpoint["trained_layers"] = [n for n in checkpoint["trained_layers"] if n in LEARNERS]
    if legacy_skill or legacy_feedback:
        # Preserve every learned tactic input exactly. New context starts at
        # zero influence, and is learned during the next skill training run.
        old = state["skill.above.weight"]
        widened = policy.state_dict()["skill.above.weight"].clone().zero_()
        widened[:, : old.shape[1]] = old
        state["skill.above.weight"] = widened
        checkpoint.setdefault("load_migrations", []).append("skill_context_zero_initialized")
    if other_strategies:
        fresh = policy.state_dict()
        state = {
            name: fresh[name] if name.startswith(("tactic.", "tactic_memory.")) else value
            for name, value in state.items()
        }
        old = state["skill.above.weight"]
        widened = fresh["skill.above.weight"].clone().zero_()
        widened[:, : len(TACTICS)] = old[:, : len(TACTICS)]
        state["skill.above.weight"] = widened
    fresh = policy.state_dict()
    if "skill.given_mode.net.0.weight" not in state:
        # The skill's x reading its chosen mode, and y reading both, start with
        # no influence: the same outputs as the skill that chose them apart.
        for name in fresh:
            if name.startswith(("skill.given_mode.", "skill.given_mode_and_x.")):
                state[name] = fresh[name]
        checkpoint.setdefault("load_migrations", []).append("skill_ordered_choice_zero_initialized")
    if checkpoint.get("memory_version", 0) < 4:
        # The skill and tactic layers read no prediction (user, 2026-10-15):
        # the skill's reading of the enemy forecasts, the expected-scene
        # markers and the tactic memory's action-memory input go; the LSTM's
        # own weights stay, unused, for the world model.
        for name in (
            "skill.forecast.weight",
            "skill.forecast.bias",
            "skill.expected_marker",
            "tactic.expected_marker",
        ):
            state.pop(name, None)
        inputs = state.get("tactic_memory.inputs.weight")
        width = fresh["tactic_memory.inputs.weight"].shape[1]
        if inputs is not None and inputs.shape[1] > width:
            state["tactic_memory.inputs.weight"] = inputs[:, :width].clone()
        checkpoint["trained_layers"] = []
        checkpoint.setdefault("load_migrations", []).append("layers_read_no_prediction")
    if checkpoint.get("memory_version", 0) < 5:
        # The LSTM is the world model now (core/world_model; user,
        # 2026-10-15/16), run once per action with the chosen action: the
        # old scene memory's weights go and it starts untrained. No layer
        # reads it, so their qualifications stand.
        state = {name: value for name, value in state.items() if not name.startswith("memory.")}
        state.update({name: value for name, value in fresh.items() if name.startswith("memory.")})
        checkpoint["world_model"] = None
        checkpoint.setdefault("load_migrations", []).append("world_model_reshaped")
    if destination_skill or "skill.heads.verb.weight" not in state:
        # The skill's action heads (a verb and the object it is aimed at) and
        # its history of actions are new: every skill tensor that is no longer
        # there, or has another shape, is dropped; new ones start untrained.
        state = {
            name: value
            for name, value in state.items()
            if name in fresh and value.shape == fresh[name].shape
        }
        for name, value in fresh.items():
            state.setdefault(name, value)
        checkpoint.setdefault("load_migrations", []).append("skill_actions_initialized")
    policy.load_state_dict(state)
    return policy, checkpoint


def train_layer(config: LayeredTrainConfig) -> dict:
    """Train one layer of the agent in Block SMB; returns the run summary."""
    if config.learner not in LEARNERS:
        raise ValueError("only skill and tactic are learned layers")
    if config.predictor is not None:
        raise ValueError(
            "the action predictor is replaced by the world model, trained apart from recorded "
            "ground truth: skills and tactics train with the teacher carrying out their "
            "actions (docs/design-decisions.md 2026-10-15)"
        )
    output = Path(config.output)
    output.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(config.seed)
    rng = random.Random(config.seed)
    device = torch.device(config.device)
    trained_layers: list[str] = []
    world_model = None
    if config.init:
        policy, checkpoint = load_layered_checkpoint(config.init, device)
        trained_layers = list(checkpoint["trained_layers"])
        # Kept as it was trained; its digest says whether the scene encoder
        # it read has changed since (core/world_model.scene_digest).
        world_model = checkpoint.get("world_model")
    else:
        policy = LayeredSMBPolicy(
            PolicySettings(
                width=config.policy_width,
                scene_depth=config.policy_scene_depth,
                layer_depth=config.policy_layer_depth,
                memory_width=config.policy_memory_width,
                tactic_memory_width=config.policy_tactic_memory_width,
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

    if config.learner == "skill" and config.train_reading:
        # The scene encoder learns with the skill (train_reading).
        learning = learn_only(policy.parameters_of("skill"))
    elif config.learner == "skill":
        learning = learn_only(list(policy.skill.parameters()))
    else:
        learning = learn_only(policy.parameters_of(config.learner))
    optimizer = torch.optim.AdamW(
        learning, lr=config.learning_rate, weight_decay=config.weight_decay
    )
    pool = EpisodePool(config, policy.settings)
    validation_tasks = pool.with_scenarios(
        _tasks(config, "validation", config.validation_layouts_per_difficulty, 0, 0.0, False)
    )
    sweep = config.learner == "skill" and config.sweep
    combinations: list[EpisodeTask] = []
    if sweep:
        combinations, made = combination_tasks(config, pool)
        (output / "combinations.json").write_text(json.dumps(made, indent=2))
        per_family = defaultdict(int)
        for name, entry in made.items():
            per_family[name.split(":")[0]] += entry["layouts"]
        print(
            f"[{config.learner}] full sweep: {len(combinations)} layouts, every combination of "
            "every value of each family's parameters: "
            + ", ".join(f"{f} {n}" for f, n in per_family.items()),
            flush=True,
        )
    replay: list[EpisodeRecord] = []
    focus: dict[str, int] = {}  # extra layouts per family, from the last validation
    fixed = extra_layouts_of(config)  # and every round, for named families
    history: list[dict] = []
    best = best_passed = -1.0
    tactic = config.learner == "tactic"
    critic_ready = False  # the tactic critic predicts held-out returns well enough
    try:
        from .teacher_qualification import qualify_teachers

        pool.publish(policy)
        qualify_teachers(
            pool, validation_tasks, output / "teacher_qualification" / "validation.json"
        )
        for round_index in range(config.rounds + config.reward_rounds):
            started = time.time()
            # Imitation rounds, the teacher's share shrinking; then reward rounds,
            # in which the learner samples its own choices and learns from what
            # they earn (on that round's episodes only).
            by_reward = round_index >= config.rounds
            if by_reward and round_index == config.rounds:
                learning = learn_only(list(getattr(policy, config.learner).parameters()))
                optimizer = torch.optim.AdamW(
                    learning, lr=config.reward_learning_rate, weight_decay=config.weight_decay
                )
            fraction = min(1.0, round_index / max(1, config.rounds - 1))
            share = (
                0.0
                if by_reward
                else config.teacher_share_start
                + fraction * (config.teacher_share_end - config.teacher_share_start)
            )
            pool.publish(policy)
            if sweep:
                # Every combination, every round.
                tasks = [
                    dataclasses.replace(
                        task,
                        sample_index=round_index * 1_000_000 + task.sample_index,
                        teacher_share=share,
                        explore=by_reward,
                    )
                    for task in combinations
                ]
            else:
                tasks = _tasks(
                    config,
                    "train",
                    config.train_layouts_per_family,
                    round_index,
                    share,
                    True,
                    explore=by_reward,
                    extra={
                        f: focus.get(f, 0) + fixed.get(f, 0) for f in set(focus) | set(fixed)
                    },
                )
            tasks = pool.with_scenarios(tasks)
            demonstrated = qualify_teachers(
                pool, tasks, output / "teacher_qualification" / f"train_{round_index:02d}.json"
            )
            # Round zero already plays only the teacher. Reuse the exact
            # qualified records instead of executing those episodes twice.
            played = (
                demonstrated
                if config.learner == "skill" and share == 1.0 and not by_reward
                else pool.play(tasks)
            )
            replay = kept_episodes(replay, played, config.replay_episodes)
            play_time = time.time() - started

            started = time.time()
            policy.train()
            totals: dict[str, list] = defaultdict(list)
            for _ in range(config.epochs_per_round):
                for batch in _batches(played if by_reward else replay, config.batch_frames, rng):
                    if tactic:
                        losses, stats = tactic_losses(
                            policy,
                            batch,
                            config,
                            device,
                            by_reward=by_reward,
                            critic_ready=critic_ready,
                        )
                    else:
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
            collapsed = output_collapse(validation, played, config.learner)
            entry = {
                "round": round_index,
                "by_reward": by_reward,
                "teacher_share": share,
                "train_success": float(np.mean([e.won for e in played])),
                "train_families": family_success(played),
                "agreement": float(np.mean(agreed)) if agreed else None,
                "validation_success": mean,
                "validation_families": per_family,
                "validation_difficulties": family_success(validation, by_difficulty=True),
                "validation_outcomes": {
                    family: dict(Counter(e.end for e in validation if e.family == family))
                    for family in per_family
                },
                "train_label_coverage": {
                    family: float(
                        np.mean(
                            [
                                valid
                                for e in played
                                if e.family == family
                                for valid in e.labels["valid"]
                            ]
                        )
                    )
                    for family in per_family
                    if any(e.family == family and len(e.labels["valid"]) for e in played)
                },
                "action_durations": {
                    "train": action_duration_summary(played),
                    "validation": action_duration_summary(validation),
                },
                "losses": {
                    name: float(
                        max(values) if name == "max_end_duration_frames" else np.mean(values)
                    )
                    for name, values in totals.items()
                },
                "replay_episodes": len(replay),
                # The skill's own held-out choices against the teacher's at
                # the same moments.
                **(
                    {"against_teacher": choices_against_teacher(validation)}
                    if config.learner == "skill"
                    else {}
                ),
                # Families whose own held-out choices collapse on one value.
                "collapsed": {
                    family: {kind: list(v) for kind, v in kinds.items()}
                    for family, kinds in collapsed.items()
                },
                "seconds": {
                    "play": round(play_time, 1),
                    "learn": round(learn_time, 1),
                    "evaluate": round(evaluate_time, 1),
                },
            }
            ends_agree = None
            if tactic:
                # The critic's held-out accuracy decides whether reward rounds may
                # change choices and the end check; agreement on where tactics
                # change is part of the gate.
                explained = critic_check(policy, validation, config, device)
                critic_ready = explained is not None and explained >= config.critic_ready
                agreement = end_agreement(validation, config.end_tolerance)
                ends_agree = agreement["overall"]
                entry["critic_explained_variance"] = explained
                entry["critic_ready"] = critic_ready
                entry["end_agreement"] = ends_agree
                entry["end_agreement_families"] = agreement["families"]
            # The whole agent without the teacher (every token its own) needs
            # the world model to carry out its actions: not built yet
            # (docs/design-decisions.md 2026-10-15).
            history.append(entry)
            print(_round_line(config.learner, entry), flush=True)
            # The gate also needs no family whose choices collapse on one value.
            gate_met = not collapsed and all(
                value >= config.family_gate for value in per_family.values()
            )
            if tactic:
                gate_met = gate_met and ends_agree is not None and ends_agree >= config.end_gate
            layers = trained_layers + ([config.learner] if gate_met else [])
            save_layered_checkpoint(
                output / "last.pt", policy, config, layers, history, world_model
            )
            # Of equally good rounds the latest is kept: it has learned from
            # more episodes, more of them its own play.
            if mean >= best:
                best = mean
                save_layered_checkpoint(
                    output / "best.pt", policy, config, layers, history, world_model
                )
            if gate_met and mean >= best_passed:
                # The best round that met the gate: the next layer starts from it.
                best_passed = mean
                save_layered_checkpoint(
                    output / "passed.pt", policy, config, layers, history, world_model
                )
            (output / "history.json").write_text(json.dumps(history, indent=2))
    finally:
        pool.close()
    return {"best_validation_success": best, "history": history}


def _round_line(learner: str, entry: dict) -> str:
    losses = " ".join(f"{name}={value:.3f}" for name, value in entry["losses"].items())
    weakest = sorted(entry["validation_families"].items(), key=lambda item: item[1])[:4]
    agreement = entry["agreement"]
    tactic = ""
    if "end_agreement" in entry:
        ends, explained = entry["end_agreement"], entry["critic_explained_variance"]
        tactic = (
            f", tactic changes agree {ends if ends is None else round(ends, 3)}, "
            f"critic explains {explained if explained is None else round(explained, 3)}"
        )
    against = entry.get("against_teacher") or {}
    differing = sorted(
        ((f, v) for f, v in against.items() if v["differ"]), key=lambda item: item[1]["agreed"]
    )
    if differing:
        # Where the skill's own held-out choices were not the teacher's at
        # the same moments (worst four families).
        tactic += "; differs from teacher " + ", ".join(
            f"{family} {round(v['agreed'] * v['decisions'])}/{v['decisions']} ("
            + ", ".join(f"{pair} x{n}" for pair, n in list(v["differ"].items())[:2])
            + ")"
            for family, v in differing[:4]
        )
    collapsed = entry.get("collapsed") or {}
    if collapsed:
        def value(v):
            if v[0] is None:
                return ""
            return f" {tuple(v[0]) if isinstance(v[0], (list, tuple)) else v[0]}"

        tactic += "; COLLAPSED " + ", ".join(
            f"{family} {kind}{value(v)} {v[1]:.0%} (teacher {v[2]:.0%})"
            for family, kinds in collapsed.items()
            for kind, v in kinds.items()
        )
    return (
        f"[{learner}] round {entry['round']:02d} teacher share {entry['teacher_share']:.2f}: "
        f"train wins {entry['train_success']:.2f}, "
        f"agrees with teacher {agreement if agreement is None else round(agreement, 3)}, "
        f"validation wins {entry['validation_success']:.3f} "
        f"(weakest {', '.join(f'{f} {v:.2f}' for f, v in weakest)}){tactic}; {losses}; "
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
    tracker: Optional[str] = None,
    execution: str = "teacher",
) -> dict:
    """Play fresh held-out layouts (never used for validation) with a saved policy.

    With a ``learner``, the teacher gives it its token from above, as in its
    validation; with none, the whole agent plays as deployed (``execution``
    "deployed"). ``execution`` (EXECUTIONS): who carries out the skill's own
    actions, the same held-out layouts in each, so what each prediction
    network costs is measured apart (the world model's: ``checkpoint``'s
    memory; ``tracker``: the collision trackers, default the run's).
    """
    policy, saved = load_layered_checkpoint(checkpoint, "cpu")
    trained = saved.get("config", {})
    if execution not in EXECUTIONS:
        raise ValueError(f"execution must be one of {EXECUTIONS}")
    if learner is None and execution != "deployed":
        raise ValueError("an agent played without the teacher is the deployed execution")
    if execution in ("world_model", "deployed") and saved.get("world_model") is None:
        raise ValueError(f"{checkpoint} has no trained world model (world_model_train)")
    if execution in ("trackers", "deployed") and not (tracker or trained.get("tracker")):
        raise ValueError("this execution needs the collision trackers (--tracker)")
    config = LayeredTrainConfig(
        learner=learner or "tactic",
        families=tuple(families),
        workers=workers,
        vision_checkpoint=vision_checkpoint,
        output=str(Path(checkpoint).parent),
        tracker=tracker or trained.get("tracker"),
    )
    tasks = [
        dataclasses.replace(
            task, sample_index=first_layout + task.sample_index, execution=execution
        )
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
    threat = END_REASONS.index("threat")
    return {
        "checkpoint": str(checkpoint),
        "learner": learner,
        "execution": execution,
        "episodes": len(played),
        "ends": dict(Counter(e.end for e in played)),
        # Actions the collision trackers (or the teacher's truth) ended
        # because a coming enemy needed a move.
        "threat_ends": int(sum(int((e.ended == threat).sum()) for e in played)),
        "success": float(np.mean([e.won for e in played])),
        "families": families_won,
        "difficulties": family_success(played, by_difficulty=True),
        "below_gate": {f: v for f, v in families_won.items() if v < config.family_gate},
        **(
            {"against_teacher": choices_against_teacher(played)} if learner == "skill" else {}
        ),
    }
