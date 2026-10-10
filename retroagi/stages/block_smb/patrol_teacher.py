"""Patrol destinations certified through the real controller, training only."""

from copy import copy, deepcopy
from dataclasses import dataclass

from retroagi.core.smb_agent import LandingWatch
from retroagi.core.smb_coaching import probe_state
from retroagi.core.smb_executor import SMBExecutor
from retroagi.core.smb_scene_labels import scene_from_labels
from retroagi.core.smb_spatial_feedback import SpatialFeedback
from retroagi.core.tokens import SkillToken

from .controller_teacher import _measured, _Watch


@dataclass(frozen=True)
class Trial:
    safe: bool
    progress: float
    frames: int
    clearance: float = 0
    won: bool = False
    margin: float = (
        999.0  # controller_teacher.Result.margin: the closest enemy, pit edge or landing edge
    )


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
        watch = _Watch(env)

        def margin():
            # The closest threat: enemy, pit edge or landing edge.
            enemy_gap, edge, _, distances = _measured(env, watch, alive, mark)
            return min([enemy_gap, *distances.values()])

        for _ in range(160):
            button = executor.press(scene)
            spatial.executed(button, scene)
            _, _, done, truncated, _ = env.step(button)
            watch.observe(env)
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
    """Choose a progressing approach, stomp or bypass on the patrol's flat floor.

    Every destination is considered: a run to the goal and to every pixel
    from MIN_RUN to 64 ahead, and every jump Mario can make (each destination
    from 1 to 96 pixels ahead, at the floor's height and on each enemy ahead,
    maps to one jump by the executor's takeoff rule; each jump is tried once).
    """
    from collections import defaultdict

    from retroagi.core.smb_trajectory import VisualTracks, best_holds, hold_paths

    from .controller_teacher import MIN_RUN, _nearest, _path_landing, _plan_goal

    m = env.mario
    grounded = m["on_ground"]
    center, feet = m["x"] + m["w"] / 2, m["y"] + m["h"]
    floor = env.platforms[0]["rect"]
    floor_y = round(floor.top - feet)
    inset = m["w"] / 2 + 2
    candidates = [SkillToken("run", round(env.goal.centerx - center), floor_y)]
    candidates += [SkillToken("run", x, floor_y) for x in range(MIN_RUN, 65)]
    if not grounded:
        # A visual contact can end the preceding command just before physical
        # touchdown. Finishing that descent is useful even with little x travel.
        candidates += [SkillToken("run", x, floor_y) for x in range(1, MIN_RUN)]
    if grounded and scene is not None and scene.mario.box is not None:
        box = scene.mario.box
        motion = copy(feedback.motion if feedback is not None else env.motion)
        tracks = feedback.tracks if feedback is not None else VisualTracks()
        heights = {floor_y} | {
            round(e["y"] - feet) for e in env.enemies if not e["dead"] and e["x"] > m["x"]
        }
        destinations = [(x, y) for y in sorted(heights) for x in range(1, 97)]
        paths = hold_paths(scene, motion, box, 1)
        goals = [_plan_goal(tracks, box, x, y) for x, y in destinations]
        groups = defaultdict(list)
        for (x, y), (hold, _, _) in zip(
            destinations, best_holds(scene, motion, box, goals, 1, paths)
        ):
            groups[hold].append(SkillToken("jump", x, y))
        start = (m["x"], m["y"], m["x"] + m["w"], m["y"] + m["h"])
        rects = [p["rect"] for p in env.platforms]
        for hold, members in groups.items():
            world = [tuple(start[k] + b[k] - box[k] for k in range(4)) for b in paths[hold - 1]]
            landing = _path_landing([start, *world], rects)
            candidates.append(
                _nearest(members, landing[1], landing[2]) if landing else members[len(members) // 2]
            )
    best, score = None, float("-inf")
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
