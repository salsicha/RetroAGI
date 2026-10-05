"""Policies observe the game only through the vision transformer.

The rule: what a policy receives for a frame may depend on that frame's
pixels (and the policy's own memory) and on nothing else. Each test takes one
picture, then changes the game's hidden state without changing the picture —
Block SMB's simulator fields, Full SMB's emulator memory — and asks the stage
for the policy's input again. Every part of the input that changes is game
state leaking into the policy; the failure message names each one.

These tests describe the target of the vision-only redesign and fail on the
current code, which still feeds game state to the policy. They are marked
expected-to-fail (strictly) until the redesign closes every route; run them
with ``--runxfail`` to see the current list of leaks.
"""

from typing import Any

import numpy as np
import pytest
import torch

from retroagi.stages.block_smb.monte_carlo import sample_block_smb_monte_carlo_scenario
from retroagi.stages.block_smb.vision import BlockVisionTransformer

LEAKS_REMAIN = pytest.mark.xfail(
    strict=True, reason="game state still reaches the policy (vision-only redesign pending)"
)
# Metadata the action executor or the model call reads at play time.
GAME_STATE_METADATA = ("smb_geometry", "info", "episode")


def policy_view(batch) -> dict[str, Any]:
    """Every named part of what a policy receives from one call."""
    parts = {"src_a": batch.src_a, "src_b": batch.src_b}
    spans = (batch.metadata or {}).get("vision_fusion", {})
    for name, (start, stop) in spans.items():
        if name.startswith("c_"):
            parts[name] = batch.src_c[:, start:stop]
    parts["src_c"] = batch.src_c
    return parts


def leaks(honest, tampered) -> list[str]:
    """Names of the parts that differ, plus game-state metadata handed over."""
    a, b = policy_view(honest), policy_view(tampered)
    found = [
        name
        for name in a
        if name != "src_c" and not torch.equal(torch.as_tensor(a[name]), torch.as_tensor(b[name]))
    ]
    if (
        "src_c" in a
        and not torch.equal(a["src_c"], b["src_c"])
        and not any(name.startswith("c_") for name in found)
    ):
        found.append("src_c")
    found += [
        f"metadata[{key!r}]" for key in GAME_STATE_METADATA if key in (tampered.metadata or {})
    ]
    return found


class Recorder:
    """Stands in for an environment and records every attribute the caller touches."""

    def __init__(self, target):
        object.__setattr__(self, "_target", target)
        object.__setattr__(self, "touched", [])

    def __getattr__(self, name):
        self.touched.append(name)
        return getattr(self._target, name)

    def __setattr__(self, name, value):
        self.touched.append(name)
        setattr(self._target, name, value)


# ── Block SMB ─────────────────────────────────────────────────────────────────


def scramble_block_state(env) -> None:
    """Change every simulator field the picture does not show."""
    mario = env.mario
    mario.update(
        x=mario["x"] + 37.0,
        y=mario["y"] - 23.0,
        vx=float(mario.get("vx", 0.0)) + 2.5,
        vy=float(mario.get("vy", 0.0)) - 3.0,
        on_ground=not mario["on_ground"],
        facing=-mario.get("facing", 1),
        skidding=not mario.get("skidding", False),
        _platform=None,
    )
    for enemy in env.enemies:
        enemy.update(
            x=enemy["x"] + 19.0,
            y=enemy["y"] - 11.0,
            speed=float(enemy.get("speed", 0.0)) + 1.3,
            direction=-enemy.get("direction", 1),
        )
    for platform in env.platforms:
        platform["rect"] = platform["rect"].move(5, -3)
        if "move_x" in platform:
            platform["move_x"] += 13.0
            platform["move_speed"] = float(platform.get("move_speed", 0.0)) + 0.7
            platform["move_dir"] = -platform.get("move_dir", 1)
    for coin in env.coins:
        coin["collected"] = not coin["collected"]
    if env.goal is not None:
        env.goal = env.goal.move(-60, 0)
    env._terrain_left = not env._terrain_left
    env._goal_credited = not env._goal_credited
    env.camera_x += 9.0
    env.steps += 50


BLOCK_FAMILIES = ("enemy_stomp", "moving_bridge")
BLOCK_ACTIONS = [1] * 18 + [2] * 8 + [1] * 18


# ── Full SMB ──────────────────────────────────────────────────────────────────


def _cartridge_available() -> bool:
    try:
        from retroagi.stages.full_smb.pixel_labels import cartridge

        cartridge()
    except Exception:  # noqa: BLE001 - any failure to find or read the file means skip
        return False
    return True


needs_cartridge = pytest.mark.skipif(
    not _cartridge_available(), reason="Super Mario Bros cartridge file not installed"
)


# ── The layered agent ─────────────────────────────────────────────────────────

AGENT_MODULES = (
    "retroagi.core.smb_agent",
    "retroagi.core.layered_policy",
    "retroagi.core.smb_observer",
    "retroagi.core.smb_executor",
    "retroagi.core.smb_scene_labels",
    "retroagi.core.tokens",
)


def test_agent_modules_import_no_game():
    """The agent's own modules import nothing that could reach a game."""
    import ast
    import importlib

    for name in AGENT_MODULES:
        source = importlib.import_module(name).__file__
        tree = ast.parse(open(source).read())
        imported = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported |= {alias.name for alias in node.names}
            elif isinstance(node, ast.ImportFrom):
                imported.add(("." * node.level) + (node.module or ""))
        bad = sorted(
            module
            for module in imported
            if "stages" in module or module.split(".")[0] in ("retro", "pygame", "gym")
        )
        assert not bad, f"{name} imports {bad}"


def test_agent_act_takes_only_screens():
    import inspect

    from retroagi.core.smb_agent import SMBAgents

    parameters = list(inspect.signature(SMBAgents.act).parameters)
    assert parameters == ["self", "screens", "copies", "given", "run_given", "sample"]


def scramble_hidden_block_state(env) -> None:
    """Change simulator fields that are never drawn (an enemy's direction is: its feet)."""
    for enemy in env.enemies:
        enemy.update(speed=float(enemy.get("speed", 0.0)) + 1.3)
        for key in ("patrol_min", "patrol_max"):
            if key in enemy:
                enemy[key] = enemy[key] + 40
    for platform in env.platforms:
        if "move_x" in platform:
            platform["move_speed"] = float(platform.get("move_speed", 0.0)) + 0.7
            platform["move_dir"] = -platform.get("move_dir", 1)
    mario = env.mario
    mario.update(vx=float(mario.get("vx", 0.0)) + 2.5, vy=float(mario.get("vy", 0.0)) - 3.0)
    if env.goal is not None:
        env.goal = env.goal.move(-60, 0)
    env._goal_credited = not env._goal_credited
    env.steps += 50


def same_decision(a, b) -> bool:
    if a is None or b is None:
        return a is b
    return (a.strategy, a.tactic, a.plan, a.chosen) == (b.strategy, b.tactic, b.plan, b.chosen)


@pytest.mark.parametrize("family", BLOCK_FAMILIES)
def test_agent_decisions_depend_only_on_the_picture(family):
    """Same screens, any hidden simulator state: identical scenes, memory and plans."""
    from retroagi.core.layered_policy import LayeredSMBPolicy
    from retroagi.core.smb_agent import SMBAgents
    from retroagi.core.smb_observer import VisionObserver
    from retroagi.stages.block_smb.env import MarioScenarioEnv

    sample = sample_block_smb_monte_carlo_scenario(
        split="validation", seed=0, sample_index=0, family=family, difficulty="medium"
    )
    torch.manual_seed(0)
    observer = VisionObserver(BlockVisionTransformer(dim=16, depth=1, heads=4).eval())
    policy = LayeredSMBPolicy().eval()
    honest, tampered = MarioScenarioEnv(), MarioScenarioEnv()
    honest_screen, _ = honest.reset(scenario=dict(sample.scenario), seed=0)
    tampered.reset(scenario=dict(sample.scenario), seed=0)
    # Two agents, so each computes exactly as the other (a shared batch may round
    # differently). The hidden state is scrambled just before the last picture.
    agents = [SMBAgents(observer, policy, "cpu"), SMBAgents(observer, policy, "cpu")]
    frames = 14
    for frame in range(frames):
        if frame == frames - 1:
            scramble_hidden_block_state(tampered)
        tampered_screen = tampered.render()
        assert np.array_equal(honest_screen, tampered_screen), "hidden state changed the picture"
        (mine,) = agents[0].act([honest_screen], [0])
        (theirs,) = agents[1].act([tampered_screen], [0])
        assert mine.scene == theirs.scene
        assert all(np.array_equal(x, y) for x, y in zip(mine.rows, theirs.rows))
        assert mine.button == theirs.button
        assert same_decision(mine.decision, theirs.decision)
        assert torch.equal(agents[0].copies[0].hidden, agents[1].copies[0].hidden)
        honest_screen, *_ = honest.step(mine.button)
        tampered.step(theirs.button)
