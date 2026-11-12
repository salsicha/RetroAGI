"""What a domain supplies, and what the framework's parts ask of each other.

The agent is five parts with narrow data between them:

    world --(source)--> state --(choice space)--> candidate choices
          --(outcome predictor)--> each candidate's predicted outcome
          --(chooser)--> one choice --(controller)--> an action a frame,
          and an event when the choice has ended

- A **World** is a place the agent acts in: a simulator or the real thing.
  A **Practice** world can also save its state and put it back, so every
  candidate can be tried from the same moment (framework/sampling.py).
- A **StateSource** turns what the world shows into the state the other
  parts read: from the world's own truth, or from perception.
- A **ChoiceSpace** lists the choices open in a state and turns the state
  into the tokens a learned predictor reads (framework/outcome_network.py).
- An **OutcomePredictor** says what each candidate would lead to: a learned
  network, or anything else with the same answer.
- A **Chooser** picks one (framework/chooser.py: fixed priorities over the
  predictions; a domain may also bring a hand-written one).
- A **Controller** carries a choice out, an action each frame, and reports
  how it ended. It plans with the domain's **Dynamics**: its model of how
  the agent and the other things move, which may be written by hand,
  identified from what is seen, or learned.

A **Domain** bundles these for one game or robot (retroagi/domains/<name>):
the framework's tools (sampling, training, practice, bench) take a domain
by name and know nothing else about it. The state and the choices are the
domain's own types; the framework only passes them between its parts.
"""

from dataclasses import dataclass
from typing import Any, Optional, Protocol, Sequence, runtime_checkable

import numpy as np


@dataclass(frozen=True)
class Step:
    """What one action led to: whether the episode is over, and how."""

    over: bool = False
    died: bool = False
    finished: bool = False


@runtime_checkable
class World(Protocol):
    """A place the agent acts in."""

    progress: float  # how far along the task the agent is (the chooser's second rule)
    score: float  # what it has gathered (the third)
    settled: bool  # at rest, where a choice can be made (standing)

    def reset(self) -> None: ...

    def step(self, action: int) -> Step: ...

    def close(self) -> None: ...


@runtime_checkable
class Practice(World, Protocol):
    """A world whose state can be saved and put back: a whole task, or a
    skill scene, which is ``passed`` once the agent is through it."""

    passed: bool

    def snapshot(self) -> Any: ...

    def restore(self, saved: Any) -> None: ...


class StateSource(Protocol):
    """The state the agent's parts read, from a world."""

    def reset(self) -> None: ...

    def read(self, world) -> Optional[Any]:
        """The state now; None while there is nothing to act on yet."""

    def acted(self, action: int) -> None:
        """The action just taken (an estimator follows the agent's own motion by it)."""


class Controller(Protocol):
    """Carries out one choice at a time."""

    choice: Any  # the choice in progress (None before the first)
    event: Optional[str]  # how it ended (None while it goes on)

    def observe(self, state) -> None:
        """Told every state, a choice in progress or not (for a dynamics
        model that adapts to what is seen)."""

    def another(self) -> "Controller":
        """A new controller that knows what this one's model has learned
        (to try a choice from a saved state as this one would carry it out)."""

    def start(self, choice, state) -> None: ...

    def press(self, state) -> int: ...


class Chooser(Protocol):
    def decide(self, state, last_event: Optional[str], last_choice) -> Any: ...


@dataclass(frozen=True)
class Candidate:
    """One choice open in a state: the domain's choice, its kind (an index
    into the choice space's kinds), the thing it is about (None: no thing)
    and a modifier (which of the kind's variants)."""

    choice: Any
    kind: int
    ref: Any
    modifier: int


class ChoiceSpace(Protocol):
    """The choices of a domain and the state as a predictor reads it."""

    spec: Any  # outcome_network.ChoiceSpec: the token groups and the kinds of choice
    new_ground: float  # progress beyond the furthest so far that makes a place new

    def candidates(self, state) -> Sequence[Candidate]: ...

    def tokens(self, state) -> tuple:
        """(one array per token group of ``spec``, {ref: slot})."""

    def place(self, state) -> tuple:
        """(progress, a key for the spot the agent is at)."""

    def signature(self, candidate: Candidate, state) -> tuple:
        """What a candidate is, the same from one decision to the next."""

    def waits(self, candidate: Candidate) -> bool: ...

    def still(self, candidate: Candidate) -> bool:
        """Whether it is about nothing that moves (found impossible from a
        spot, it stays so)."""

    def changing(self, state) -> bool:
        """Whether anything in the state moves on its own (waiting can
        change what is possible)."""


class OutcomePredictor(Protocol):
    def predict(self, state, candidates: Sequence[Candidate]) -> np.ndarray:
        """[candidates, outcome_network.PREDICTIONS]: chances and amounts in
        their own units."""


class Dynamics(Protocol):
    """The agent's model of how the other things move."""

    def observe(self, things: Sequence, solids: Sequence) -> None:
        """What is seen this frame (a model that adapts learns from it)."""

    def paths(self, things: Sequence, solids: Sequence, frames: int) -> None:
        """Give each thing its predicted box for each of the next ``frames`` frames."""


class Domain(Protocol):
    """One game or robot, as the framework's tools use it."""

    name: str
    modes: tuple  # the ways to play (the chooser's weights differ)
    idle: int  # the action that does nothing
    ends: tuple  # how a sampled choice can end: the controller's events, then "timeout", "died", "finished"
    space: ChoiceSpace
    parts: dict  # the parts with alternatives: part -> {alternative's name: a function making it}
    chosen: dict  # part -> the name of the alternative in use

    def using(self, **choices) -> "Domain":
        """This domain with other alternatives for some parts (part=name)."""

    def controller(self) -> Controller: ...

    practices: tuple  # the kinds of practice world: "level" (a whole task) and its skill scenes

    def practice(self, seed: int, kind: str = "level") -> Practice:
        """A practice world of ``kind`` made from ``seed``: a random level,
        or a skill scene (a short stretch calling for one skill, ``passed``
        when the agent is through it)."""

    def truth_source(self) -> StateSource:
        """The state from a practice world's own truth."""

    def behaviour(self, mode: str) -> Chooser:
        """The chooser whose play visits the states outcomes are sampled at."""

    def explore(self, state, candidates: Sequence[Candidate], rng) -> Optional[Any]:
        """A choice to try instead of the behaviour's, or None."""

    def probes(self, state) -> list:
        """Choices that would take the agent on from ``state``."""

    def escapes(self, state) -> list:
        """Choices that might save it where it would otherwise die."""

    def chooser(self, predictor: OutcomePredictor, mode: str) -> Chooser: ...
