"""The agent: a chooser and a controller on a state source, one action a
frame.

Each frame the source gives the state. While a choice is in progress the
controller presses on. When it has ended (its event), and the agent is
settled enough to decide (state.decidable: in Super Mario Bros, standing
with its speed known; a plan made in the air would start from a standing
jump), the chooser picks the next choice and the controller starts it. A
choice found infeasible as it starts leaves this frame idle: the chooser
decides again on the next.
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
        if controller.choice is None or controller.event is not None:
            if not state.decidable:
                return self.idle
            self.last_event = controller.event
            self.last_choice = self.chooser.decide(state, self.last_event, self.last_choice)
            controller.start(self.last_choice, state)
            if controller.event is not None:  # infeasible at once: decide again next frame
                return self.idle
        return controller.press(state)


def play(world, agent: Agent, frames: int) -> dict:
    """``agent`` (already reset) through ``world`` from its start: how it
    ended ("finished", "died" or "timeout") and the frames taken."""
    world.reset()
    for frame in range(frames):
        step = world.step(agent.act(world))
        if step.over:
            return {"end": "died" if step.died else "finished" if step.finished else "timeout", "frames": frame + 1}
    return {"end": "timeout", "frames": frames}
