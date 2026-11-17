"""How the real game's lifts meet Mario, measured in its memory
(docs/requirements.md: the real game may be measured so that Block SMB
behaves as it does). World 1-3 is reached by writing its number into the
game's memory (as scripts/dev/game_catalogue.py does) and Mario flown to
its first lift, one going up and down. Mario is then put beside, under or
over the lift as it comes down, with a speed written into memory, and what
follows is printed: where he is against the lift each frame (his x and
y less the lift's, in the game's units: standing on it, his y is 32 above
its top), his speeds, and when his rise stops (his vertical speed set to 1)
or his run stops (his speed across set to 0).

Measured 2026-11-16: rising into its underside he falls from where he is
(speed 1: no push down, nor aside, his speed across kept), his feet 24
above its top after a slow rise, 21 after a fast one, not at 28 or 26; met
from beside, it stops his run as a wall; falling into it, he is set on it
with his feet up to 4 pixels into its top, and passes through it deeper.
Measured 2026-11-17, run right into its side at every depth and vertical
speed (``--side``): with his feet, after the frame's moves of Mario and
lift, from 1 pixel above its top to 4 into it, falling or level, he is set
on it; rising with them there or deeper, or falling with them deeper than
4, his run is stopped; higher, he passes clear. Block SMB follows (sim/env.py
LIFT_*), and the planner foresees the same.

    python scripts/dev/lift_contact.py [--side]
"""

import sys

import numpy as np

from game_catalogue import (AREA, ENEMY_ID, ENEMY_ON, ENEMY_PAGE, ENEMY_X, ENEMY_Y, ENGINE, FLY_HEIGHT, HALFWAY,
                            LEVEL, LIVES, PAGE, PLAYING, SLOTS, STAR, TIMER, WORLD, X, Y, Y_FORCE, Y_PAGE, Y_SPEED,
                            area_of)

STATE, X_SPEED = 0x1D, 0x57  # Player_State (1: in the air), Player_X_Speed
LIFTS = range(0x24, 0x2D)  # the game's lift codes


def _make():
    import retro

    env = retro.make("SuperMarioBros-Nes", state="Level2-1", inttype=retro.data.Integrations.STABLE, render_mode=None)
    env.reset()
    return env


def at_lift(world: int = 1, level: int = 3) -> bytes:
    """The emulator's state with the level's first lift on the screen."""
    env = _make()
    idle = np.zeros(len(env.buttons), np.int8)
    right = idle.copy()
    right[env.buttons.index("RIGHT")] = 1
    write = lambda address, value: env.data.memory.assign(address, "|u1", value)
    for _ in range(5):
        env.step(idle)
    for address, value in ((WORLD, world - 1), (LEVEL, level - 1), (AREA, area_of(world, level)), (HALFWAY, 0),
                           (LIVES, 5), (TIMER, 0), (TIMER + 1, 0), (TIMER + 2, 1)):
        write(address, value)
    started = False
    try:
        for frame in range(6000):
            ram = env.get_ram()
            started = started or (frame > 300 and ram[ENGINE] == PLAYING and (ram[WORLD], ram[LEVEL]) == (world - 1, level - 1))
            if started:
                for address, value in ((Y, FLY_HEIGHT), (Y_PAGE, 1), (Y_SPEED, 0), (Y_FORCE, 0), (STAR, 30), (TIMER, 3)):
                    write(address, value)
            env.step(right if started else idle)
            ram = env.get_ram()
            camera = int(ram[0x71A]) * 256 + int(ram[0x71C])
            if started and ram[ENEMY_ON] and ram[ENEMY_ID] in LIFTS and \
                    camera < int(ram[ENEMY_PAGE]) * 256 + int(ram[ENEMY_X]) < camera + 160:
                return env.em.get_state()
    finally:
        env.close()
    raise RuntimeError("no lift reached")


def trial(state: bytes, dx: int, dy: int, y_speed: int, x_speed: int, buttons=(), frames: int = 16) -> list:
    """Mario put at (dx, dy) from the lift (slot 0) with those speeds: each
    frame (x less the lift's, y less the lift's, y speed, x speed)."""
    env = _make()
    env.em.set_state(state)
    write = lambda address, value: env.data.memory.assign(address, "|u1", value % 256)
    ram = env.get_ram()
    x, y = int(ram[ENEMY_PAGE]) * 256 + int(ram[ENEMY_X]) + dx, int(ram[ENEMY_Y]) + dy
    for address, value in ((PAGE, x // 256), (X, x), (Y, y), (Y_PAGE, 1), (Y_SPEED, y_speed), (Y_FORCE, 0),
                           (X_SPEED, x_speed), (STATE, 1), (STAR, 0)):
        write(address, value)
    pressed = np.zeros(len(env.buttons), np.int8)
    for name in buttons:
        pressed[env.buttons.index(name)] = 1
    rows = []
    try:
        for _ in range(frames):
            env.step(pressed)
            ram = env.get_ram()
            rows.append((int(ram[PAGE]) * 256 + int(ram[X]) - int(ram[ENEMY_PAGE]) * 256 - int(ram[ENEMY_X]),
                         int(ram[Y]) + 256 * (int(ram[Y_PAGE]) - 1) - int(ram[ENEMY_Y]),
                         int(np.int8(ram[Y_SPEED])), int(np.int8(ram[X_SPEED]))))
    finally:
        env.close()
    return rows


def side(state: bytes) -> None:
    """Mario run right into the lift's side, a pixel short of it, with his
    feet ``d`` below its top and vertical speed ``vy``: how it meets him, by
    where his feet are after the frame's moves (the lift's included)."""
    for vy in (-3, -2, -1, 0, 1, 2, 3):
        ends = []
        for d in range(-3, 9):
            rows = trial(state, dx=-13, dy=-32 + d, y_speed=vy, x_speed=24, buttons=("RIGHT",), frames=1)
            x, y, y_speed, x_speed = rows[0]
            end = "stopped" if x_speed == 0 else "on" if y == -32 and y_speed == 0 else "clear"
            ends.append(f"{y + 32:+d} {end}")
        print(f"vertical speed {vy:+d}: " + ", ".join(ends))


def main() -> int:
    state = at_lift()
    if "--side" in sys.argv:
        side(state)
        return 0
    tries = {"rising slowly under it": dict(dx=16, dy=30, y_speed=-4, x_speed=0),
             "rising fast under it": dict(dx=16, dy=40, y_speed=-5, x_speed=0),
             "rising under its left end, running right": dict(dx=-8, dy=30, y_speed=-4, x_speed=16, buttons=("RIGHT",)),
             "beside it, running right into it": dict(dx=-16, dy=-4, y_speed=0, x_speed=24, buttons=("RIGHT",))}
    for depth in (2, 4, 5, 7):  # (falling 3 a frame as the lift comes down about 1: feet this far into its top)
        tries[f"falling, feet {depth} into its top"] = dict(dx=16, dy=-32 + depth - 2, y_speed=3, x_speed=0, frames=4)
    for name, kw in tries.items():
        rows = trial(state, **kw)
        rise = next((i for i, r in enumerate(rows) if r[2] == 1 and kw["y_speed"] < 0), None)
        stop = next((i for i, r in enumerate(rows) if r[3] == 0 and kw["x_speed"]), None)
        print(f"{name}: " + " ".join(f"({x},{y},{vy},{vx})" for x, y, vy, vx in rows))
        if rise is not None:
            print(f"    rise stopped with his feet {rows[rise][1] + 32} above its top (before: {rows[rise - 1][1] + 32})")
        if stop is not None:
            print(f"    run stopped at x {rows[stop][0]} from the lift")
        if kw["y_speed"] > 0:
            print("    " + ("set on it" if rows[-1][1] == -32 else "passes through"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
