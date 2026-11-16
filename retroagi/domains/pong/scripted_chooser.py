"""A hand-written chooser for Pong: the behaviour whose play visits the
states outcomes are sampled at (framework/sampling.py), and a baseline.
It returns every ball with the middle of the paddle, falling back to the
nearest part that can be done, and rests at the middle."""

from .controller import PARTS, Choice, coming

ORDER = (0, -1, 1, -2, 2)  # the parts tried, middle first


class ScriptedChooser:
    def __init__(self, mode: str = "points"):
        self.mode = mode
        self.refused = set()

    def decide(self, state, last_event=None, last_choice=None):
        if not coming(state):
            self.refused.clear()
            return Choice("rest", 1)
        if last_event == "infeasible" and last_choice is not None and last_choice.kind == "return":
            self.refused.add(last_choice.part)
        else:
            self.refused.clear()
        for part in ORDER:
            if part not in self.refused:
                return Choice("return", part)
        self.refused.clear()
        return Choice("return", PARTS[2])
