"""Pong as a domain (framework/interfaces.Domain): what the framework's
tools (sampling, training, practice, play) need to know of the game.

- Practice worlds: a point of Block Pong from a random situation, and the
  "return" skill scene (world.py), the state read from Block Pong's own.
- The real game: a whole game to 21 in the emulator (real.py), the state
  from the game's memory, or from the screen alone (vision.py, seeing.py).
- The controller (controller.py), the choices (choices.py), and the
  hand-written chooser as the behaviour that visits the states outcomes are
  sampled at (scripted_chooser.py). One way to play: win points.

Pong has no dead ends: no probe is tried after a choice (sampling.py's
"stuck" is then true of every choice that neither loses nor wins the point,
and the chooser's rule on it never decides anything).
"""

from copy import copy

from retroagi.framework.chooser import OutcomeChooser

from .choices import SPACE
from .controller import Controller
from .game import STAY
from .scripted_chooser import ScriptedChooser
from .world import PRACTICES, BlockPongWorld, TruthSource, random_situation


class Pong:
    name = "pong"
    modes = ("points",)
    idle = STAY
    ends = ("returned", "turned", "scored", "missed", "infeasible", "timeout", "died", "finished")
    # A return lasts from the ball turning toward us until it comes back:
    # 240 frames across and back, and a serve's wait.
    longest = 400
    space = SPACE
    sources = ("memory", "vision")
    parts: dict = {}
    chosen: dict = {}

    def using(self, **choices) -> "Pong":
        for part in choices:
            raise ValueError(f"no {part} part with alternatives in Pong")
        return copy(self)

    def controller(self) -> Controller:
        return Controller()

    def chooser(self, predictor, mode: str) -> OutcomeChooser:
        return OutcomeChooser(predictor, self.space, 0.0)

    def behaviour(self, mode: str) -> ScriptedChooser:
        return ScriptedChooser(mode)

    practices = PRACTICES

    def practice(self, seed: int, kind: str = "level") -> BlockPongWorld:
        return BlockPongWorld(random_situation(seed, kind), kind)

    def truth_source(self) -> TruthSource:
        return TruthSource()

    def explore(self, state, candidates, rng):
        return None

    def probes(self, state) -> list:
        return []

    def escapes(self, state) -> list:
        return []

    def real_world(self, level: str = "Start", start_state: bytes = None):
        from .real import RealPongWorld

        return RealPongWorld(level, start_state)

    def source(self, kind: str = "memory", level: str = "Start", vision: str = None):
        """The state source ``kind`` for the real game: "memory" (the game's
        own memory) or "vision" (the screen read by the vision network saved
        at ``vision``, followed by the tracker, and nothing else)."""
        from .real import DEFAULT_VISION, MemorySource, VisionSource

        if kind == "memory":
            return MemorySource()
        if kind == "vision":
            from .vision import load

            return VisionSource(load(vision or DEFAULT_VISION))
        raise ValueError(f"no state source {kind!r}: one of {self.sources}")


DOMAIN = Pong()
