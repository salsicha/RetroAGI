"""Whether the real game hurts Mario while he still touches an enemy he has
just stomped, measured in its memory (docs/requirements.md: the real game
may be measured so that Block SMB behaves as it does). World 1-3 is reached
by writing its number into memory and Mario flown on until one of its red
flying Paratroopas is on the screen; big Mario is then dropped onto it from
a range of heights at a range of speeds, and each frame his and its
collision boxes (the game's own table), its code, his size and his hurt
timer are read.

Measured 2026-11-17: stomped, it loses its wings and Mario bounces up at 4
pixels a frame; in the deepest drops he still touches it one or two frames
after, and is never hurt (his size and hurt timer unchanged). Block SMB
follows (sim/env.py: a Koopa just stomped out of its wings is harmless while
he still touches it).

    python scripts/dev/stomp_contact.py
"""

import numpy as np

from game_catalogue import (AREA, ENEMY_ID, ENEMY_ON, ENEMY_PAGE, ENEMY_X, ENEMY_Y, ENGINE, FLY_HEIGHT, HALFWAY, LEVEL,
                            LIVES, PAGE, PLAYING, SLOTS, STAR, TIMER, WORLD, X, Y, Y_FORCE, Y_PAGE, Y_SPEED, area_of)
from lift_contact import STATE, X_SPEED, _make

# PlayerStatus (1: big), PlayerSize (0: big), InjuryTimer, the enemies' and
# Mario's collision boxes (x0, y0, x1, y1 on the screen, four bytes each).
STATUS, SIZE, INJURY, BOXES, MARIO_BOX = 0x756, 0x754, 0x79E, 0x4B0, 0x4AC
PARATROOPA = 0x0F  # a red Paratroopa flying up and down


def at_paratroopa() -> tuple:
    """(the emulator's state, the slot) with a red Paratroopa of 1-3 on the
    screen."""
    env = _make()
    idle = np.zeros(len(env.buttons), np.int8)
    right = idle.copy()
    right[env.buttons.index("RIGHT")] = 1
    write = lambda address, value: env.data.memory.assign(address, "|u1", value)
    for _ in range(5):
        env.step(idle)
    for address, value in ((WORLD, 0), (LEVEL, 2), (AREA, area_of(1, 3)), (HALFWAY, 0), (LIVES, 5), (TIMER, 0),
                           (TIMER + 1, 0), (TIMER + 2, 1)):
        write(address, value)
    started = False
    try:
        for frame in range(6000):
            ram = env.get_ram()
            started = started or (frame > 300 and ram[ENGINE] == PLAYING and (ram[WORLD], ram[LEVEL]) == (0, 2))
            if started:
                for address, value in ((Y, FLY_HEIGHT), (Y_PAGE, 1), (Y_SPEED, 0), (Y_FORCE, 0), (STAR, 30), (TIMER, 3)):
                    write(address, value)
            env.step(right if started else idle)
            ram = env.get_ram()
            camera = int(ram[0x71A]) * 256 + int(ram[0x71C])
            for slot in range(SLOTS):
                x = int(ram[ENEMY_PAGE + slot]) * 256 + int(ram[ENEMY_X + slot])
                if started and ram[ENEMY_ON + slot] and ram[ENEMY_ID + slot] == PARATROOPA and camera + 40 < x < camera + 200:
                    return env.em.get_state(), slot
    finally:
        env.close()
    raise RuntimeError("no Paratroopa reached")


def drop(state: bytes, slot: int, dy: int, y_speed: int, frames: int = 8) -> list:
    """Big Mario put over the Paratroopa (his y ``dy`` from its), falling at
    ``y_speed``: each frame (its code, his vertical speed, his size, his
    hurt timer, touching it)."""
    env = _make()
    env.em.set_state(state)
    write = lambda address, value: env.data.memory.assign(address, "|u1", value % 256)
    ram = env.get_ram()
    x, y = int(ram[ENEMY_PAGE + slot]) * 256 + int(ram[ENEMY_X + slot]), int(ram[ENEMY_Y + slot]) + dy
    for address, value in ((PAGE, x // 256), (X, x), (Y, y), (Y_PAGE, 1), (Y_SPEED, y_speed), (Y_FORCE, 0),
                           (X_SPEED, 0), (STATE, 1), (STAR, 0), (STATUS, 1), (SIZE, 0), (INJURY, 0)):
        write(address, value)
    rows = []
    try:
        for _ in range(frames):
            env.step(np.zeros(len(env.buttons), np.int8))
            ram = env.get_ram()
            m, e = ram[MARIO_BOX:MARIO_BOX + 4].astype(int), ram[BOXES + 4 * slot:BOXES + 4 * slot + 4].astype(int)
            touching = m[0] <= e[2] and e[0] <= m[2] and m[1] <= e[3] and e[1] <= m[3]
            rows.append((int(ram[ENEMY_ID + slot]), int(np.int8(ram[Y_SPEED])), int(ram[STATUS]), int(ram[INJURY]),
                         bool(touching)))
    finally:
        env.close()
    return rows


def main() -> int:
    state, slot = at_paratroopa()
    for y_speed in (1, 2, 3, 4):
        for dy in range(-40, -14, 2):
            rows = drop(state, slot, dy, y_speed)
            stomp = next((i for i, r in enumerate(rows) if r[1] < 0), None)
            if stomp is None:
                continue
            touching = sum(r[4] for r in rows[stomp + 1:])
            hurt = any(r[2] != 1 or r[3] > 0 for r in rows)
            print(f"falling {y_speed}, dropped from {dy}: stomped at frame {stomp}, touching it {touching} frames "
                  f"after, {'HURT' if hurt else 'not hurt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
