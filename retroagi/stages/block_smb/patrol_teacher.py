"""Patrol destinations certified through the real controller, training only."""

from copy import deepcopy
from dataclasses import dataclass

from retroagi.core.smb_agent import LandingWatch
from retroagi.core.smb_coaching import probe_state
from retroagi.core.smb_executor import SMBExecutor
from retroagi.core.smb_scene_labels import scene_from_labels
from retroagi.core.smb_spatial_feedback import SpatialFeedback
from retroagi.core.tokens import SkillToken

from .controller_teacher import _box_gaps, _measured


@dataclass(frozen=True)
class Trial:
    safe: bool
    progress: float
    frames: int
    clearance: float = 0
    won: bool = False
    margin: float = 999.0  # controller_teacher.Result.margin: enemy gap and edge margin


def rollout(env, destination, feedback=None, scene=None):
    """Measure a spatial command, including braking, tracking and first landing.

    Probes restore the simulator and copy execution feedback; they never change
    the running controller or substitute a button route for its output.
    """
    with probe_state(env):
        scene = scene or scene_from_labels(env.scene_labels())
        spatial = deepcopy(feedback) if feedback is not None else SpatialFeedback()
        if feedback is None:
            spatial.motion = deepcopy(env.motion)
            spatial.observe(scene)
        executor = SMBExecutor(last_button=2 if spatial.motion.previous_jump else 0)
        landing = LandingWatch(airborne=not scene.mario.on_something)
        plan = spatial.begin(destination, scene)
        executor.start(plan, flight=spatial.flight, travel=spatial.travel)
        start_x, start_step = env.mario["x"], env.steps
        alive = [i for i, e in enumerate(env.enemies) if not e["dead"]]
        mark = env._tactic_index, env._route_done
        gaps: dict = {}
        _box_gaps(env, gaps)

        def margin():
            enemy_gap, edge, _ = _measured(env, gaps, alive, mark)
            return min(enemy_gap, 999.0 if edge is None else edge)

        for _ in range(160):
            button = executor.press(scene)
            spatial.executed(button, scene)
            _, _, done, truncated, _ = env.step(button)
            _box_gaps(env, gaps)
            if done or truncated:
                return Trial(
                    env._goal_credited,
                    env.mario["x"] - start_x,
                    env.steps - start_step,
                    won=env._goal_credited,
                    margin=margin(),
                )
            scene = scene_from_labels(env.scene_labels())
            spatial.observe(scene)
            ended = landing.landed(scene) or executor.finished or spatial.arrived()
            stalled = spatial.stalled and (
                ended or spatial.observed_frames >= spatial.positions.maxlen
            )
            if ended or stalled:
                m = env.mario
                clearance = min(
                    (
                        max(e["x"] - m["x"] - m["w"], m["x"] - e["x"] - e["w"])
                        for e in env.enemies
                        if not e["dead"]
                    ),
                    default=999,
                )
                reached = spatial.report(scene)[2] > 0.5
                return Trial(
                    bool((m["on_ground"] and reached) or env.stomped),
                    m["x"] - start_x,
                    env.steps - start_step,
                    clearance,
                    margin=margin(),
                )
        return Trial(False, 0, 160)


def destination(env, feedback=None, scene=None):
    """Choose a progressing approach, stomp or bypass on the patrol's flat floor."""
    m = env.mario
    grounded = m["on_ground"]
    center, feet = m["x"] + m["w"] / 2, m["y"] + m["h"]
    floor_y = env.platforms[0]["rect"].top - feet
    candidates = [SkillToken("run", round(env.goal.centerx - center), round(floor_y))]
    candidates += [SkillToken("run", x, round(floor_y)) for x in (8, 16, 24, 32, 48, 64)]
    if not grounded:
        # A visual contact can end the preceding command just before physical
        # touchdown. Finishing that descent is useful even with little x travel.
        candidates.append(SkillToken("run", 4, round(floor_y)))
    if grounded:
        candidates += [SkillToken("jump", x, 0) for x in range(32, 97, 8)]
        candidates += [
            SkillToken("jump", round(e["x"] + e["w"] / 2 - center), round(e["y"] - feet))
            for e in env.enemies
            if not e["dead"] and e["x"] > m["x"]
        ]
    best, score = None, float("-inf")
    floor = env.platforms[0]["rect"]
    inset = m["w"] / 2 + 2
    candidates = [
        c
        for c in candidates
        if abs(c.x) <= 256 and floor.left + inset <= center + c.x <= floor.right - inset
    ]
    winner, most = None, float("-inf")
    for candidate in dict.fromkeys(candidates):
        result = rollout(env, candidate, feedback, scene)
        if result.won:
            # Of the commands that finish, the one with the most margin.
            if result.margin > most:
                winner, most = candidate, result.margin
            continue
        minimum_progress = 4 if grounded else 0
        if not result.safe or result.progress < minimum_progress or result.clearance < 8:
            continue
        value = result.progress / result.frames
        if value > score:
            best, score = candidate, value
    return winner if winner is not None else best
