"""Recorded Full SMB playthroughs: the data the collision trackers and the
world model keep learning from in Full SMB, and the rewards the skill and
tactic layers will learn from (user, 2026-10-15: Full SMB played many times,
learning on every playthrough; speed mode rewarded by time to each
checkpoint, points mode by points).

play_recorded plays one level from its start with copy 0 of the agents
(core.smb_agent.SMBAgents, as deployed: no teacher), and keeps, as Block SMB's
recordings do with record_truth (stages/block_smb/layered_train), what the
networks' trainers read: the game's truth every frame (full_smb/truth), which
true enemy each scene slot shows, the executor's kinematics, each decision's
scene rows, action, command and feedback; and per frame the level's progress
(Mario's distance into it, score, time) for the rewards. Nothing recorded
reaches the agent while it plays.
"""

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from retroagi.core.action_tokens import VERBS, pointer_index
from retroagi.core.layered_policy import POINTER_START
from retroagi.core.smb_observer import SCENE_SLOTS
from retroagi.core.tokens import command_row

from .play import FullSMBGame
from .truth import FullTruthRecorder, match_slots, true_frame

# A checkpoint every this many pixels of Mario's distance into the level
# (speed mode is rewarded by the time to each).
CHECKPOINT_PIXELS = 500
END_REASONS = ("", "landed", "done", "threat", "hold_recheck", "arrived", "no_progress")


@dataclass
class Playthrough:
    """One recorded playthrough, with the fields the trainers read
    (tracker_train.track_sequences, world_model_train.episode_data)."""

    family: str  # the level
    split: str
    sample_index: int  # which playthrough
    difficulty: Optional[str]
    won: bool  # left the level (the flagpole, a pipe to the next)
    end: str  # "finished", "death" or "timeout"
    decision_frames: np.ndarray
    truth: dict
    slot_match: np.ndarray
    sprite_kinematics: np.ndarray
    mario_kinematics: np.ndarray
    slot_identity: np.ndarray
    enemy_context: np.ndarray
    mario_position: np.ndarray
    action_verb: np.ndarray
    action_pointer: np.ndarray
    commands: np.ndarray
    ended: np.ndarray
    execution_feedback: np.ndarray
    played_teacher: np.ndarray
    rows: tuple  # the scene rows at each decision (src_a, src_b, src_c)
    actions: Optional[np.ndarray] = None  # no teacher: no tried actions
    picks: dict = field(default_factory=dict)
    log_prob: Optional[np.ndarray] = None
    value: Optional[np.ndarray] = None
    predicted: Optional[np.ndarray] = None  # the world model's outcome per decision

    @property
    def checkpoint_frames(self) -> list:
        """The frame Mario first reached each CHECKPOINT_PIXELS of the level."""
        level_x = self.truth["level_x"]
        out = []
        for mark in range(CHECKPOINT_PIXELS, int(level_x.max()) + 1, CHECKPOINT_PIXELS):
            out.append(int(np.argmax(level_x >= mark)))
        return out


def play_recorded(agents, level: str, *, frames: int, index: int = 0, split: str = "full") -> Playthrough:
    """Play ``level`` from its start with copy 0 of ``agents`` until Mario
    dies, leaves the level or ``frames`` pass, recording it (module notes)."""
    game = FullSMBGame(level)
    recorder = FullTruthRecorder()
    per_frame = {name: [] for name in ("match", "sprites", "mario", "identity", "context", "position", "ended")}
    decisions = {name: [] for name in ("frame", "verb", "pointer", "command", "feedback", "rows", "predicted")}
    picks, log_prob, value = {}, [], []
    end = "timeout"
    try:
        screen = game.reset()
        agents.reset(0)
        copy = agents.copies[0]
        for frame in range(frames):
            step = agents.act([screen], [0])[0]
            # The game's state behind this picture (before its button).
            truth = true_frame(game.env.get_ram(), game.start)
            ids = recorder.add(truth)
            scene = step.scene
            # The kinematics the trackers and the world model read this frame.
            sprites, mario, identities, context = copy.kinematics
            per_frame["match"].append(match_slots(scene, truth, ids))
            per_frame["sprites"].append(sprites)
            per_frame["mario"].append(mario)
            per_frame["identity"].append(identities)
            per_frame["context"].append(context)
            feet = scene.mario.box[3] if scene.mario.box is not None else 0.0
            per_frame["position"].append((copy.spatial.displacement, feet))
            per_frame["ended"].append(END_REASONS.index(step.ended) if step.ended in END_REASONS else 0)
            decision = step.decision
            if decision is not None:
                action = decision.action
                decisions["frame"].append(frame)
                decisions["verb"].append(VERBS.index(action.verb) if action is not None else -1)
                decisions["pointer"].append(
                    pointer_index(action.target) - POINTER_START
                    if action is not None and action.target is not None
                    else -1
                )
                decisions["command"].append(command_row(decision.skill))
                decisions["feedback"].append(decision.execution_feedback)
                decisions["rows"].append(step.rows)
                predicted = decision.predicted
                decisions["predicted"].append(
                    predicted["mean"].detach().cpu().numpy() if predicted else np.full(5, np.nan)
                )
                for head, pick in decision.picks.get("skill", {}).items():
                    picks.setdefault(head, []).append(pick)
                log_prob.append(decision.log_prob.get("skill", 0.0))
                value.append(decision.value.get("skill", 0.0))
            screen, outcome = game.step(step.button)
            if outcome.finished:
                end = "finished"
                break
            if outcome.dying:
                end = "death"
                break
        recorder.add(true_frame(game.env.get_ram(), game.start))
    finally:
        game.close()
    truth = recorder.arrays()
    columns = recorder.enemy_columns()
    lifts = [c for c, k in enumerate(recorder.kinds) if k[2]]
    lift_index = {c: i for i, c in enumerate(lifts)}
    match = np.asarray(per_frame["match"], np.int64)
    # Recorder columns to enemy (or lift) indices, as truth_recording's are.
    enemy_slots = SCENE_SLOTS["enemies"]
    as_index = np.full(match.shape, -1, np.int16)
    for t, row in enumerate(match):
        for s, c in enumerate(row):
            if c < 0:
                continue
            if s < enemy_slots and columns[c] >= 0:
                as_index[t, s] = columns[c]
            elif s >= enemy_slots and c in lift_index:
                as_index[t, s] = lift_index[c]
    rows = decisions["rows"]
    count = len(decisions["frame"])
    return Playthrough(
        family=level,
        split=split,
        sample_index=index,
        difficulty=None,
        won=end == "finished",
        end=end,
        decision_frames=np.asarray(decisions["frame"], np.int32),
        truth=truth,
        slot_match=as_index,
        sprite_kinematics=np.asarray(per_frame["sprites"], np.float16),
        mario_kinematics=np.asarray(per_frame["mario"], np.float16),
        slot_identity=np.asarray(per_frame["identity"], np.int32),
        enemy_context=np.asarray(per_frame["context"], np.float16),
        mario_position=np.asarray(per_frame["position"], np.float32),
        action_verb=np.asarray(decisions["verb"], np.int8),
        action_pointer=np.asarray(decisions["pointer"], np.int16),
        commands=np.asarray(decisions["command"], np.int16).reshape(count, 6),
        ended=np.asarray(per_frame["ended"], np.int8),
        execution_feedback=np.asarray(decisions["feedback"], np.float32),
        played_teacher=np.zeros(count, bool),
        rows=tuple(np.asarray([r[i] for r in rows]) for i in range(3)) if rows else ((), (), ()),
        picks={head: np.asarray(v) for head, v in picks.items()},
        log_prob=np.asarray(log_prob, np.float32),
        value=np.asarray(value, np.float32),
        predicted=np.asarray(decisions["predicted"], np.float32) if count else None,
    )
