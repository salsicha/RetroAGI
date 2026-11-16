"""How the real game treats a ledge met with Mario's feet a little below its
top, measured in its own memory (docs/requirements.md: the real game may be
measured so that Block SMB behaves as it does). In World 1-1 one column of
floor blocks ahead of Mario's start is cleared in the game's block buffer
(the game reads its ground from there: a one-tile pit, two with --width 2);
Mario walks or runs at it, or is let go at points before it to cross slower.
Printed: whether he crosses or falls in, and how far below the far ledge's
top his feet were before he was set on it.

Measured 2026-11-16: walking or running he crosses a one-tile pit, dipping
1 pixel over it; let go to cross slowly, he still crosses with his feet 3
pixels below the far top, and is stopped by the ledge's side and falls in at
4 or more; stopped over the pit, he falls in. A two-tile pit he falls into.

    python scripts/dev/gap_rule.py [--width 1]
"""

import argparse

import numpy as np

BLOCKS, ROWS, FLOOR_ROWS = 0x500, 13, (11, 12)  # Block_Buffer_1; the floor's two rows
PAGE, X, Y, Y_PAGE, X_SPEED = 0x6D, 0x86, 0xCE, 0xB5, 0x57
COLUMN = 4  # the pit: x 64 to 80 (Mario starts at 40)
STANDING = 176  # Mario's y standing on 1-1's floor


def trial(width: int, buttons) -> list:
    """(x, y, speed) each frame until Mario is past the pit or falls."""
    import retro

    env = retro.make("SuperMarioBros-Nes", state="Level1-1", inttype=retro.data.Integrations.STABLE, render_mode=None)
    env.reset()
    for column in range(COLUMN, COLUMN + width):
        for row in FLOOR_ROWS:
            env.data.memory.assign(BLOCKS + column % 16 + 16 * row, "|u1", 0)
    log = []
    try:
        for frame in range(240):
            pressed = np.zeros(len(env.buttons), np.int8)
            for b in buttons(log):
                pressed[env.buttons.index(b)] = 1
            env.step(pressed)
            ram = env.get_ram()
            x, y = int(ram[PAGE]) * 256 + int(ram[X]), int(ram[Y]) + 256 * (int(ram[Y_PAGE]) - 1)
            log.append((x, y, int(np.int8(ram[X_SPEED]))))
            if y > 200 or x > 16 * (COLUMN + width) + 48:
                break
    finally:
        env.close()
    return log


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--width", type=int, default=1)
    args = parser.parse_args(argv)
    ways = {"running": lambda log: ("RIGHT", "B"), "walking": lambda log: ("RIGHT",)}
    for at in range(50, 66):
        ways[f"let go at x {at}"] = lambda log, at=at: ("RIGHT",) if not log or log[-1][0] < at else ()
    for name, buttons in ways.items():
        log = trial(args.width, buttons)
        fell = log[-1][1] > 200
        dips = [y - STANDING for _, y, _ in log if STANDING < y <= 200]
        print(f"{name:16s} {'falls in' if fell else 'crosses'}; feet below the floor's top at most "
              f"{max(dips, default=0)} px{' before falling' if fell else ''}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
