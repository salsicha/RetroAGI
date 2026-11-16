"""Pong as a domain (retroagi/domains/pong): its controller carries out the
choices in Block Pong, its practice worlds play, outcomes can be sampled,
and in the real game, from its memory, the controller returns the ball."""

import time
from collections import Counter

import numpy as np
import pytest

from retroagi.domains.pong.block_pong import step
from retroagi.domains.pong.controller import PARTS, Choice, Controller
from retroagi.domains.pong.world import random_situation
from retroagi.framework import practice, registry, sampling


def _return(state, part):
    """Carry out a return with ``part`` in Block Pong: (the controller's
    event at the start, the ball's vertical speed after our paddle)."""
    controller = Controller()
    controller.start(Choice("return", part), state)
    if controller.event:
        return controller.event, None
    for _ in range(400):
        action = controller.press(state)
        returns = state.returns
        step(state, action)
        if state.returns > returns:
            return None, state.speed
    return None, None


def test_the_controller_meets_the_ball_with_the_chosen_part():
    met, refused, starts = Counter(), 0, []
    for seed in range(60):
        for part in PARTS:
            state = random_situation(seed, "return")
            began = time.perf_counter()
            event, speed = _return(state, part)
            starts.append(time.perf_counter() - began)
            if event == "infeasible":
                refused += 1
            else:
                met[speed == part] += 1
    assert met[False] == 0  # every plan carried out meets the ball as planned
    assert refused <= 0.03 * 300  # and only a ball out of reach is refused


def test_a_rest_goes_to_its_height_and_ends_when_the_ball_comes():
    state = random_situation(1, "level")
    state.ball_x, state.ball_y, state.across, state.wait = 150, 100, -1, 0
    controller = Controller()
    controller.start(Choice("rest", 0), state)
    for _ in range(200):
        if controller.event:
            break
        step(state, controller.press(state))
    assert controller.event == "turned" and abs(state.ours - 70) <= 4


def test_practice_worlds_play_without_losing_a_fair_point():
    domain = registry.load("pong")
    rows = practice.run(domain, range(30), kinds=domain.practices, scenes=range(30))
    ends = Counter((r["kind"], r["end"]) for r in rows)
    assert ends[("return", "passed")] == 30
    assert ends[("level", "died")] == 0


def test_outcomes_are_sampled_for_every_candidate():
    domain = registry.load("pong")
    shard = sampling.sample(domain, 11, "points")
    spec = domain.space.spec
    assert shard["paddles"].shape[1:] == (spec.groups[0].width,)
    assert shard["ball"].shape[1:] == (1, spec.groups[1].width)
    assert len(shard["choices"]) == len(shard["outcomes"]) > 0
    ends = {domain.ends[int(e)] for e in shard["outcomes"][:, -1]}
    assert ends <= set(domain.ends) and "timeout" not in ends


def test_in_the_real_game_the_controller_returns_the_ball():
    retro = pytest.importorskip("retro")
    from retroagi.domains.pong.game import GAME
    from retroagi.framework.agent import Agent, play

    try:
        retro.data.get_romfile_path(GAME)
    except FileNotFoundError:
        pytest.skip("the Pong cartridge is not imported")
    domain = registry.load("pong")
    world = domain.real_world()
    agent = Agent(domain.source("memory"), domain.behaviour, domain.controller, domain.idle)
    agent.reset(0, "points")
    try:
        result = play(world, agent, 3000)
    finally:
        world.close()
    assert result["end"] == "timeout" and world.points[1] == 0 and world.returns >= 10
