"""Seeing the real Pong (retroagi/domains/pong/vision.py, seeing.py): the
agent plays from the screen alone, when the cartridge is imported and the
vision network has been trained (data/pong_vision/vision.pt)."""

from pathlib import Path

import pytest

retro = pytest.importorskip("retro")
VISION = Path(__file__).parents[2] / "data" / "pong_vision" / "vision.pt"


def test_the_agent_returns_the_ball_from_the_screen():
    from retroagi.domains.pong.domain import DOMAIN
    from retroagi.domains.pong.game import GAME
    from retroagi.framework.agent import Agent

    try:
        retro.data.get_romfile_path(GAME)
    except FileNotFoundError:
        pytest.skip("the Pong cartridge is not imported")
    if not VISION.exists():
        pytest.skip("the Pong vision network is not trained")
    world = DOMAIN.real_world()
    source = DOMAIN.source("vision", vision=str(VISION))
    agent = Agent(source, DOMAIN.behaviour, DOMAIN.controller, DOMAIN.idle)
    agent.reset(0, "points")
    world.reset()
    try:
        for _ in range(3000):
            world.step(agent.act(world))
        assert world.returns >= 8 and world.points[1] == 0, (world.returns, world.points)
        assert source.seeing.best is not None
    finally:
        world.close()
