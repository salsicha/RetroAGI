"""The agent's loop (framework/agent.py), with a made-up chooser and
controller."""

from dataclasses import dataclass

from retroagi.framework.agent import TRIES, Agent


@dataclass
class State:
    decidable: bool = True


class Chooser:
    """Offers choice 0, 1, 2, ... in turn, and notes what it was told."""

    def __init__(self):
        self.told = []

    def decide(self, state, last_event, last_choice):
        self.told.append((last_event, last_choice))
        return len(self.told) - 1


class Controller:
    """Refuses, as they start, the choices below ``first_feasible``; presses
    the number of the one in progress."""

    def __init__(self, first_feasible):
        self.first_feasible, self.choice, self.event, self.seen = first_feasible, None, None, 0

    def observe(self, state):
        self.seen += 1

    def start(self, choice, state):
        self.choice, self.event = choice, "infeasible" if choice < self.first_feasible else None

    def press(self, state):
        return 100 + self.choice


class Source:
    def reset(self):
        pass


def agent_refusing(first_feasible):
    agent = Agent(Source(), lambda mode: Chooser(), lambda: Controller(first_feasible), idle=0)
    agent.reset(0, "any")
    return agent


def test_a_refused_choice_is_answered_with_the_next_in_the_same_frame():
    agent = agent_refusing(3)
    assert agent.step(State()) == 103  # the fourth choice, pressed at once
    assert agent.chooser.told == [(None, None), ("infeasible", 0), ("infeasible", 1), ("infeasible", 2)]
    assert agent.step(State()) == 103 and len(agent.chooser.told) == 4  # it goes on: nothing is decided
    assert agent.controller.seen == 2  # the controller is shown every state


def test_with_every_choice_refused_the_frame_is_idle_and_the_next_decides_again():
    agent = agent_refusing(10 ** 6)
    assert agent.step(State()) == 0 and len(agent.chooser.told) == TRIES
    assert agent.step(State()) == 0 and len(agent.chooser.told) == 2 * TRIES


def test_nothing_is_decided_while_the_state_is_not_decidable_or_missing():
    agent = agent_refusing(0)
    assert agent.step(None) == 0 and agent.step(State(decidable=False)) == 0 and agent.chooser.told == []
    assert agent.step(State()) == 100
