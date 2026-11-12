"""The layers stay apart (docs/framework.md): the framework imports no
domain and no retired code, a domain imports no retired code, and Super
Mario Bros meets the framework's interfaces. And what the project must be
(docs/requirements.md): no network inside the controller, and nothing of
the real game inside Block SMB."""

import ast
from pathlib import Path

from retroagi.framework import registry
from retroagi.framework.interfaces import Candidate, Practice, Step, World
from retroagi.framework.outcome_network import ChoiceSpec

DEEP_LEARNING = ("torch", "tensorflow", "jax")


def imported(*places: str) -> set:
    """Every module imported anywhere in the files of ``places`` (folders,
    or single files; function-level imports too), relative ones resolved."""
    found = set()
    for path in (p for place in places for p in ([Path(place)] if place.endswith(".py") else Path(place).rglob("*.py"))):
        package = list(path.with_suffix("").parts[:-1])
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.Import):
                found |= {a.name for a in node.names}
            elif isinstance(node, ast.ImportFrom):
                base = package[: len(package) - (node.level - 1)] if node.level else []
                module = ".".join(base + ([node.module] if node.module else []))
                found |= {module} | {f"{module}.{a.name}" for a in node.names}
    return found


def test_the_framework_imports_no_domain_and_no_retired_code():
    modules = imported("retroagi/framework")
    assert not [m for m in modules if m.startswith(("retroagi.domains", "retroagi.legacy"))]


def test_no_domain_imports_the_retired_code():
    assert not [m for m in imported("retroagi/domains") if m.startswith("retroagi.legacy")]


def test_no_network_runs_inside_the_controller():
    # The controller, its jump planner, the motion model and the models of
    # how other things move: plain arithmetic, fit for a fast control loop.
    modules = imported("retroagi/domains/smb/controller.py", "retroagi/domains/smb/planner.py",
                       "retroagi/domains/smb/physics.py", "retroagi/framework/observed_dynamics.py")
    assert not [m for m in modules if m.split(".")[0] in DEEP_LEARNING]
    assert not [m for m in modules if m.startswith(("retroagi.framework.outcome_network", "retroagi.domains.smb.perception"))]


def test_block_smb_has_nothing_of_the_real_game_in_it():
    # Fast and cheap: no emulator, no network, none of the game's pictures or levels.
    modules = imported("retroagi/domains/smb/sim")
    assert not [m for m in modules if m.split(".")[0] in (*DEEP_LEARNING, "retro")]
    assert not [m for m in modules if m.startswith("retroagi.domains.smb.nes")]
    code = "".join(path.read_text() for path in Path("retroagi/domains/smb/sim").rglob("*.py"))
    assert "pygame.image" not in code and "full_smb" not in code
    # Its practice worlds are drawn at random from the game's pieces: none is a real level's layout.
    from retroagi.domains.smb.sim.levels import SCENE_KINDS

    assert registry.load("smb").practices == ("level", *SCENE_KINDS)


def test_super_mario_bros_meets_the_interfaces():
    domain = registry.load("smb")
    assert isinstance(domain.space.spec, ChoiceSpec) and domain.ends.index("infeasible") >= 0
    world = domain.practice(3)
    assert isinstance(world, World) and isinstance(world, Practice)
    world.reset()
    source = domain.truth_source()
    source.reset()
    state = source.read(world)
    assert state.decidable
    options = domain.space.candidates(state)
    assert options and all(isinstance(c, Candidate) for c in options)
    tokens, slots = domain.space.tokens(state)
    assert [t.shape[-1] for t in tokens] == [g.width for g in domain.space.spec.groups]
    assert all(c.ref in slots for c in options if c.ref is not None)
    # Each kind of choice names a token group and heads that group has.
    for kind in domain.space.spec.kinds:
        assert kind.first + kind.modifiers <= domain.space.spec.groups[kind.group].heads
    controller = domain.controller()
    controller.start(options[0].choice, state)
    saved = world.snapshot()
    assert isinstance(world.step(controller.press(state)), Step)
    world.restore(saved)
    assert source.read(world).body == state.body
