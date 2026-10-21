"""Recorded Full SMB playthroughs: what every network keeps learning from in
Full SMB (user, 2026-10-17: all models engaged and continuing to train;
played in max points and speed run modes; training continues on the success
in the strategy mode being played).

play_recorded plays one level from its start with copy 0 of the agents
(core.smb_agent.SMBAgents, as deployed: no teacher; the world model carries
out the skill's actions, the collision trackers end waits and aim stomps)
in one strategy mode, and records, through Block SMB's EpisodeRecorder
(stages/block_smb/layered_train), an episode for the skill layer and one for
the tactic layer: every frame's scene rows, button and reward, the game's
truth read from its memory (full_smb/truth), which true enemy each scene
slot shows and the executor's kinematics (the world model's and the
trackers' data), and each decision's choices (the reward learning's data).
It also labels every ``label_every``-th frame from game memory for the
vision network (pixel_labels.label_frame), and, when the level is finished,
can capture the next level's start. Nothing recorded reaches the agent while
it plays.

Each frame's reward follows the mode, in Block SMB's units (block_smb/env
STRATEGY_REWARDS): progress into the level, a cost per frame (five times as
much in speed run), points (max points only), a loss at death, and the goal
at the level's end (in speed run, again times the share of the game's timer
left).
"""

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from retroagi.core.smb_kinematics import MEMORY_SLOTS
from retroagi.core.smb_observer import SCENE_SLOTS

from .play import FullSMBGame
from .truth import FullTruthRecorder, match_slots, true_frame

# A checkpoint every this many pixels of Mario's distance into the level: the
# speed run is measured by the frames to each, max points by the points
# gained between them.
CHECKPOINT_PIXELS = 500
MODES = ("speed_run", "max_points")
# Block SMB's reward terms (env.BlockSMBRewardConfig) and each mode's
# multipliers (env.STRATEGY_REWARDS); points at Block SMB's rate (a stomp,
# 100 points, pays as a Block SMB stomp; a coin, 200 points, as a coin).
PROGRESS_PER_PIXEL = 0.05
FRAME_PENALTY = -0.01
DEATH = -10.0
GOAL = 50.0
POINT = 0.125
MODE_REWARDS = {
    "speed_run": {"frame_penalty": 5.0, "points": 0.0, "time_bonus": 1.0},
    "max_points": {"frame_penalty": 1.0, "points": 1.0, "time_bonus": 0.0},
}
# A playthrough that gets no further into the level for this many frames
# ends as a timeout (Block SMB's STALL_FRAMES).
STALL_FRAMES = 400


@dataclass
class Playthrough:
    """One recorded playthrough: the skill's and the tactic layer's episodes
    (layered_train.EpisodeRecord), what it achieved, the vision labels made
    from it, and the next level's start when it finished the level (its
    name and emulator state: FullSMBGame.next_level_state)."""

    skill: object
    tactic: Optional[object]
    summary: dict
    labelled: list = field(default_factory=list)
    next_state: Optional[tuple] = None


def frame_reward(mode: str, before: dict, after: dict, furthest: int, start_time: int) -> tuple:
    """The reward for one frame (module notes) from the game's truth before
    and after it, and Mario's furthest distance so far: (reward, furthest)."""
    weights = MODE_REWARDS[mode]
    reward = FRAME_PENALTY * weights["frame_penalty"]
    if after["level_x"] > furthest:
        reward += PROGRESS_PER_PIXEL * (after["level_x"] - furthest)
        furthest = after["level_x"]
    reward += POINT * weights["points"] * max(0, after["score"] - before["score"])
    if after["dying"] and not before["dying"]:
        reward += DEATH
    if after["finished"]:
        reward += GOAL * (1.0 + weights["time_bonus"] * after["time"] / max(1, start_time))
    return reward, furthest


def _enemy_indices(
    recorder: FullTruthRecorder, per_frame: list, enemy_slots: int = SCENE_SLOTS["enemies"]
) -> list:
    """Each frame's matches (recorder columns) as enemy (or lift) indices, as
    truth_recording's are: the first ``enemy_slots`` of each row are enemies
    (the scene's enemy slots, or every remembered enemy), the rest lifts."""
    columns = recorder.enemy_columns()
    lifts = [c for c, k in enumerate(recorder.kinds) if k[2]]
    lift_index = {c: i for i, c in enumerate(lifts)}
    out = []
    for row in per_frame:
        converted = np.full(len(row), -1, np.int16)
        for s, c in enumerate(row):
            if c < 0:
                continue
            if s < enemy_slots and columns[c] >= 0:
                converted[s] = columns[c]
            elif s >= enemy_slots and c in lift_index:
                converted[s] = lift_index[c]
        out.append(converted)
    return out


def play_recorded(
    agents,
    level: str,
    *,
    mode: str,
    frames: int,
    index: int = 0,
    start_state: Optional[bytes] = None,
    explore: bool = False,
    label_every: int = 0,
    capture_next: bool = False,
) -> Playthrough:
    """Play ``level`` (from its saved start, or ``start_state``) in ``mode``
    with copy 0 of ``agents`` (its switch set to the mode, the goal to the
    right), sampling the layers' choices when ``explore``, until Mario dies,
    leaves the level, stalls or ``frames`` pass; recorded (module notes)."""
    from retroagi.core.tokens import StrategyToken
    from retroagi.stages.block_smb import layered_train as lt

    from .pixel_labels import Unexplained, label_frame

    game = FullSMBGame(level, start_state)
    recorder = FullTruthRecorder()
    task = lt.EpisodeTask(
        index=index,
        family=level,
        split="full",
        seed=0,
        sample_index=index,
        teacher_share=0.0,
        label=False,
        frames=frames,
        difficulty=mode,  # the strategy mode played
        explore=explore,
        record_truth=True,
    )
    shared = defaultdict(list)
    skill = lt.EpisodeRecorder(task, shared)
    tactic = lt.EpisodeRecorder(task, shared)
    labelled, refused = [], 0
    end, next_state = "timeout", None
    sample = ("skill", "tactic") if explore else ()
    last_action_ended = False
    try:
        screen = game.reset()
        agents.reset(0, StrategyToken(mode, 1))
        copy = agents.copies[0]
        before = true_frame(game.env.get_ram(), game.start)
        start_time, start_score = before["time"], before["score"]
        furthest, progressed = before["level_x"], 0
        earlier_ram = None
        known = {}  # track identity -> the true enemy it last showed
        for frame in range(frames):
            step = agents.act([screen], [0], sample=sample)[0]
            ids = recorder.add(before)
            scene = step.scene
            for name, value in zip("abc", step.rows):
                shared[name].append(value)
            # The kinematics the trackers and the world model read this frame.
            lt.record_kinematics(shared, copy, match_slots(scene, before, ids), known, scene)
            shared["ended"].append(
                lt.END_REASONS.index(step.ended) if step.ended in lt.END_REASONS else 0
            )
            decision = step.decision
            if decision is not None:
                skill.note_decision("skill", decision, lt.NO_TEACHER)
                if decision.tactic_step is not None:
                    tactic.note_decision("tactic", decision, lt.NO_TEACHER)
            shared["button"].append(step.button)
            label = bool(label_every) and frame % label_every == 0
            ram_now = game.env.get_ram().copy()
            state_before = game.env.em.get_state() if label else None
            screen, outcome = game.step(step.button)
            if label:
                try:
                    labelled.append(
                        label_frame(
                            game.raw,
                            state_before,
                            game.env.em.get_state(),
                            family=level,
                            earlier_ram=earlier_ram,
                        )
                    )
                except Unexplained:
                    refused += 1
            earlier_ram = ram_now
            after = true_frame(game.env.get_ram(), game.start)
            reward, reached = frame_reward(mode, before, after, furthest, start_time)
            if reached > furthest:
                progressed = frame
            furthest = reached
            shared["reward"].append(float(reward))
            shared["potential"].append(0.0)
            before = after
            if outcome.finished:
                end = "goal"
                if capture_next:
                    next_state = game.next_level_state()
                break
            if outcome.dying:
                end = "death"
                break
            if frame - progressed > STALL_FRAMES:
                break
        recorder.add(before)  # the state the last button left
        executor = copy.executor
        last_action_ended = bool(end != "timeout" or executor.finished or executor.reconsider)
    finally:
        game.close()
    truth = recorder.arrays()
    shared["slot_match"] = _enemy_indices(recorder, shared["slot_match"])
    shared["memory_match"] = _enemy_indices(recorder, shared["memory_match"], MEMORY_SLOTS)
    won = end == "goal"
    records = {
        name: rec.record(won=won, end=end, truth=truth, last_action_ended=last_action_ended)
        for name, rec in (("skill", skill), ("tactic", tactic))
    }
    level_x, score = truth["level_x"], truth["score"]
    checkpoints = (
        [
            int(np.argmax(level_x >= mark))
            for mark in range(CHECKPOINT_PIXELS, int(level_x.max()) + 1, CHECKPOINT_PIXELS)
        ]
        if len(level_x)
        else []
    )
    marks = [0] + checkpoints + [max(0, len(score) - 1)]
    summary = {
        "level": level,
        "mode": mode,
        "index": index,
        "finished": won,
        "end": end,
        "frames": len(shared["button"]),
        "furthest_x": int(level_x.max()) if len(level_x) else 0,
        "points": int(score[-1] - start_score) if len(score) else 0,
        # Frames to reach each checkpoint, and the points gained before each
        # (the last: from the last checkpoint to the end).
        "checkpoint_frames": checkpoints,
        "checkpoint_points": [int(score[b] - score[a]) for a, b in zip(marks, marks[1:])]
        if len(score)
        else [],
        "decisions": len(records["skill"].decision_frames),
        "return": float(np.sum(records["skill"].rewards)),
        "vision_labels": len(labelled),
        "vision_refused": refused,
    }
    return Playthrough(
        skill=records["skill"],
        tactic=records["tactic"] if len(records["tactic"].decision_frames) else None,
        summary=summary,
        labelled=labelled,
        next_state=next_state,
    )
