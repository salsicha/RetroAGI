"""The chooser: fixed priorities on predicted outcomes (user, 2026-11-07,
for Super Mario Bros: "1. Don't die, 2. Progress to the right, 3. get
coins and power-ups", in that order).

At a decision the predictor gives every candidate choice's outcome
(interfaces.OutcomePredictor; outcome_network.PREDICTIONS). Then:
1. don't die: of the choices the controller can carry out, keep those whose
   chance of death is at most SAFE, or within SLACK of the safest one when
   none is that safe; of those, the ones not likely to leave the agent in
   danger (dead soon after if it then does nothing: a choice counts as safe
   while some way out is left after it, so with a Goomba a few steps off
   the chooser waited, safely, and waited again, until no way out was
   left), if any and unless the chooser is not ``wary``; and of those, the
   ones not likely to leave it stuck (from where no choice gets it on: a
   valley between staircases, entered for its quick progress, held Mario
   there), if any;
2. progress: of those, the one bringing the most progress a frame (frames
   counted from MIN_FRAMES up, so a short wait is no free win), and above
   all one likely (at least an even chance) to finish the task (weighed by
   its chance, a 1% chance far from the flag outweighed a jump over 1-1's
   first Goomba and Mario ran back into it);
3. points: a point is worth ``point_value`` units of progress a frame, so
   gathering wins when it costs little progress (0: points are worth
   nothing).
Waiting and retreating come out of the rules: when every way on risks
death, a wait or a step back that does not is the one kept.

What was tried is remembered for a short while: a choice taken from a spot
(the choice space's place) is not taken from it again until the agent
reaches new ground (the space's new_ground beyond its furthest so far; then
all is forgotten), unless nothing else is left; waits while things move may
repeat, and forget what was tried from the spot (the things have moved on),
but not the choices about still things the controller could not carry out
from it, infeasible or blocked (by 1-1's third pipe Mario tried the pipe's
top, and ran into the pipe, after every wait). After MAX_WAITS waits at a
spot, waiting there counts as tried too: rule 1 then keeps the least risky
way on. One choice at a time the chooser had tried two failing landings by
turns, and gone round a staircase it could not climb: land on a step, back
down, run into it, land again. Without the forgetting, it waited beside the
third pipe, every way on tried, while a Goomba walked up from behind: the
jump back over it had been tried there before.
"""

from typing import Callable

import numpy as np

from .outcome_network import PREDICTIONS, NetworkPredictor, check, load_model

SAFE = 0.05
# Waits at one spot after which waiting there counts as tried until new
# ground: before 1-1's second pipe every way on was a few points riskier
# than the 5% kept (a hidden plant in any pipe, the random levels taught),
# and Mario waited until the time ran out.
MAX_WAITS = 4
SLACK = 0.02
MIN_FRAMES = 8
DEATH, INFEASIBLE, FINISHED, STUCK, PROGRESS, FRAMES, POINTS, DANGER = range(len(PREDICTIONS))
# The controller's events that say a choice could not be carried out.
REFUSED = ("infeasible", "blocked")


class OutcomeChooser:
    def __init__(self, predictor, space, point_value: float = 0.0, wary: bool = True):
        self.predictor, self.space, self.point_value, self.wary = predictor, space, point_value, wary
        self.last = None  # (candidates, predictions, picked index): for inspection
        self.tried: dict = {}  # spot -> the choices taken from it
        self.refused: dict = {}  # spot -> the choices about still things found impossible there
        self.waits: dict = {}  # spot -> waits taken there
        self.picked = None  # (spot, signature, candidate) of the last choice
        self.furthest = float("-inf")

    def decide(self, state, last_event=None, last=None):
        space = self.space
        options = space.candidates(state)
        p = self.predictor.predict(state, options)
        feasible = p[:, INFEASIBLE] < 0.5
        if last_event in REFUSED and self.picked is not None and space.still(self.picked[2]):
            self.refused.setdefault(self.picked[0], set()).add(self.picked[1])
        progress, spot = space.place(state)
        if progress >= self.furthest + space.new_ground:
            self.tried.clear()
            self.refused.clear()
            self.waits.clear()
        self.furthest = max(self.furthest, progress)
        signatures = [space.signature(o, state) for o in options]
        waited_out = self.waits.get(spot, 0) >= MAX_WAITS
        fresh = np.array([sig not in self.tried.get(spot, ()) and sig not in self.refused.get(spot, ())
                          and not (waited_out and space.waits(o)) for sig, o in zip(signatures, options)])
        if (feasible & fresh).any():
            feasible &= fresh
        if not feasible.any():
            feasible = np.ones(len(p), bool)
        risk = np.where(feasible, p[:, DEATH], np.inf)
        allowed = risk <= max(SAFE, risk.min() + SLACK)
        for unwanted in ((DANGER, STUCK) if self.wary else (STUCK,)):
            if (allowed & (p[:, unwanted] < 0.5)).any():
                allowed &= p[:, unwanted] < 0.5
        value = (p[:, PROGRESS] / np.maximum(p[:, FRAMES], MIN_FRAMES)
                 + self.point_value * p[:, POINTS] + (p[:, FINISHED] >= 0.5) * 100.0)
        best = int(np.argmax(np.where(allowed, value, -np.inf)))
        self.last = (options, p, best)
        self.picked = (spot, signatures[best], options[best])
        waiting = space.waits(options[best])
        if waiting:
            self.waits[spot] = self.waits.get(spot, 0) + 1
        if waiting and space.changing(state):
            self.tried.pop(spot, None)  # the things move on: what failed here may work after
        else:
            self.tried.setdefault(spot, set()).add(signatures[best])
        return options[best].choice


def network_choosers(domain, path: str, device=None) -> Callable:
    """``make_chooser(mode)`` for ``domain`` on the outcome network saved at
    ``path``: the domain's chooser on that network's predictions."""
    from .devices import select_device

    device = device or select_device("auto")
    model = load_model(path, device)
    check(model, domain.space.spec)
    predictor = NetworkPredictor(model, domain.space, device)
    return lambda mode: domain.chooser(predictor, mode)
