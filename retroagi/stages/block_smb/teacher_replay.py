"""Replay spatial teaching with the production controllers, without a learner."""

from functools import lru_cache

from retroagi.core.smb_agent import LandingWatch
from retroagi.core.smb_executor import SMBExecutor
from retroagi.core.smb_scene_labels import scene_from_labels
from retroagi.core.smb_spatial_feedback import SpatialFeedback

from .env import MarioScenarioEnv
from .teacher_tokens import episode_teacher, teacher_plan, teacher_skill


def replay(
    scenario,
    *,
    family="",
    max_frames=None,
    failure=None,
    extend_limit=False,
    progress=None,
    observer=None,
):
    env = MarioScenarioEnv()
    env.reset(scenario=scenario)
    if extend_limit and max_frames is not None:
        env.max_steps = max(env.max_steps, max_frames)
    env.render = lambda: None
    state = episode_teacher(scenario)
    state.family = family or state.family
    state.visual_observer = observer
    spatial, executor, landing = SpatialFeedback(), SMBExecutor(), LandingWatch()
    # Watch the pre-episode pictures first, as the agent does.
    watched = (
        observer.observe(env.watched_screens)
        if observer is not None and env.watched_screens
        else [scene_from_labels(labels) for labels in env.watched_labels]
    )
    for scene in watched:
        spatial.observe(scene)
        landing.landed(scene)
        spatial.executed(0, scene)
    actions, commands = [], []
    limit = max_frames or env.max_steps
    try:
        for _ in range(limit):
            state.observe_frame(env)
            scene = (
                observer.observe([type(env).render(env)])[0]
                if observer is not None
                else scene_from_labels(env.scene_labels())
            )
            spatial.observe(scene)
            landed = landing.landed(scene)
            if not executor.idle:
                reason = (
                    "landed"
                    if landed
                    else (
                        "done"
                        if executor.finished
                        else (
                            "hold_recheck"
                            if executor.reconsider
                            else "arrived" if spatial.arrived() else None
                        )
                    )
                )
                if spatial.stalled and (
                    reason is not None or spatial.observed_frames >= spatial.positions.maxlen
                ):
                    reason = "no_progress"
                if reason:
                    executor.end(reason)
            if executor.idle:
                state.execution, state.controller, state.scene = spatial, executor, scene
                plan = (
                    None
                    if state.family == "enemy_patrol"
                    else teacher_plan(env, state, certify_holds=False)[0]
                )
                goal = teacher_skill(env, state, plan)
                if goal is None:
                    break
                commands.append(
                    {
                        "frame": env.steps,
                        "x": env.mario["x"],
                        "y": env.mario["y"],
                        "skill": goal.__dict__,
                        "mark": [env._tactic_index, env._route_done],
                    }
                )
                if progress is not None:
                    progress(commands[-1])
                plan = spatial.begin(goal, scene)
                executor.start(plan, flight=spatial.flight, travel=spatial.travel)
            button = executor.press(scene)
            spatial.executed(button, scene)
            actions.append(button)
            _, _, done, truncated, _ = env.step(button)
            if done or truncated:
                break
        if failure is not None and not env._goal_credited:
            from .env_state import snapshot_env_state

            failure.extend((scenario, snapshot_env_state(env), spatial))
        return {
            "won": env._goal_credited,
            "frames": env.steps,
            "points": env.points(),
            "actions": actions,
            "commands": commands,
        }
    finally:
        env.close()


@lru_cache(maxsize=1)
def reference_vision():
    """Use the same fixed ViT as the benchmark when measuring a deadline."""
    import hashlib

    from retroagi.core.smb_observer import VisionObserver

    from .vision import DEFAULT_BLOCK_VIT_CHECKPOINT, load_block_vit_checkpoint

    checkpoint = DEFAULT_BLOCK_VIT_CHECKPOINT
    observer = VisionObserver(load_block_vit_checkpoint(checkpoint, device="cpu").model)
    return observer, hashlib.sha256(checkpoint.read_bytes()).hexdigest()


def calibrate_budget(scenario, family):
    """Measure the selected route through spatial execution before setting its deadline.

    The geometry and route are fixed. A failed demonstration is an error, not
    permission to silently redraw the layout or relax its route requirements.
    """
    import copy
    import hashlib
    import json
    import os
    import tempfile
    from pathlib import Path

    from .layout_cache import code_fingerprint
    from .strategy_families import SPEED_SLACK

    scenario.setdefault("teacher_route_budget", scenario.get("frame_budget", 320))
    observer, vision_hash = reference_vision()
    key = hashlib.sha256(
        (
            code_fingerprint() + vision_hash + json.dumps(scenario, sort_keys=True, default=str)
        ).encode()
    ).hexdigest()
    path = Path(tempfile.gettempdir()) / "retroagi_spatial_deadlines" / (key + ".json")
    try:
        result = json.loads(path.read_text())
    except (OSError, ValueError):
        reference = copy.deepcopy(scenario)
        if family.startswith("speed_run_"):
            reference["strategy_objective"] = {}
        result = replay(
            reference,
            family=family,
            max_frames=max(3000, reference.get("frame_budget", 0)),
            extend_limit=True,
            observer=observer,
        )
        if not result["won"]:
            raise RuntimeError(
                f"{family}: spatial reference failed after {result['frames']} frames"
            )
        path.parent.mkdir(exist_ok=True)
        spare = path.with_suffix(f".{os.getpid()}.tmp")
        spare.write_text(json.dumps(result))
        os.replace(spare, path)
    scenario["strategy_reference"] = {
        "execution": "spatial",
        "frames": result["frames"],
        "points": result["points"],
        "vision_sha256": vision_hash,
    }
    budget = int(result["frames"] * SPEED_SLACK) + 1
    if family.startswith("speed_run_"):
        scenario["strategy_objective"] = {"deadline": budget - 1}
    scenario["frame_budget"] = max(scenario.get("frame_budget", 0), budget)
