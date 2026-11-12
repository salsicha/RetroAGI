"""The agent: a chooser and a controller on a state source, one action a
frame.

Each frame the source gives the state. While a choice is in progress the
controller presses on. When it has ended (its event), and the agent is
settled enough to decide (state.decidable: in Super Mario Bros, standing
with its speed known; a plan made in the air would start from a standing
jump), the chooser picks the next choice and the controller starts it. A
choice found infeasible as it starts leaves this frame idle: the chooser
decides again on the next. (Asking again within the frame, tried on
2026-11-11, finished no more of 100 fresh practice levels.) The controller
is shown every state, a choice in progress or not.
"""

from typing import Callable, Optional


class Agent:
    def __init__(self, source, make_chooser: Callable, make_controller: Callable, idle: int = 0):
        self.source, self.make_chooser, self.make_controller, self.idle = source, make_chooser, make_controller, idle
        self.chooser, self.controller = None, make_controller()
        self.last_event, self.last_choice = None, None

    def reset(self, seed: int, mode: str) -> None:
        self.source.reset()
        self.chooser = self.make_chooser(mode)
        self.controller = self.make_controller()
        self.last_event, self.last_choice = None, None

    def act(self, world) -> int:
        """The action for this frame of ``world``."""
        action = self.step(self.source.read(world))
        self.source.acted(action)
        return action

    def step(self, state: Optional[object]) -> int:
        controller = self.controller
        if state is None:
            return self.idle  # nothing to act on yet
        controller.observe(state)
        if controller.choice is None or controller.event is not None:
            if not state.decidable:
                return self.idle
            self.last_event = controller.event
            self.last_choice = self.chooser.decide(state, self.last_event, self.last_choice)
            controller.start(self.last_choice, state)
            if controller.event is not None:  # infeasible at once: decide again next frame
                return self.idle
        return controller.press(state)


# Frames a play may wait before the agent acts, by seed: play is
# deterministic, so repeated plays differ only by this (the first starts at
# once). The other things in the world are then elsewhere in their motion.
START_WAITS = (0, 6, 12, 18)


def play(world, agent: Agent, frames: int, wait: int = 0) -> dict:
    """``agent`` (already reset) through ``world`` from its start, after
    ``wait`` frames of doing nothing: how it ended ("finished", "passed" for
    a skill scene got through, "died" or "timeout") and the frames taken."""
    world.reset()
    for _ in range(wait):
        world.step(agent.idle)
    for frame in range(frames):
        step = world.step(agent.act(world))
        if step.over:
            return {"end": "died" if step.died else "finished" if step.finished else "timeout", "frames": frame + 1}
        if getattr(world, "passed", False):
            return {"end": "passed", "frames": frame + 1}
    return {"end": "timeout", "frames": frames}
