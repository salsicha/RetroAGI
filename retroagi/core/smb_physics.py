"""NES land-player motion in physical frames, with explicit fractional state.

Movement constants/order follow PlayerPhysicsSub, ImposeFriction,
MoveObjectHorizontally and ImposeGravity in the SMB disassembly. Terrain and
object collision resolution is supplied by the environment. Water/climbing are
outside this profile and must not be advertised as supported.
"""

from dataclasses import dataclass

NES_PHYSICS_PROFILE = "nes_land_v1"
LEGACY_PHYSICS_PROFILE = "block_legacy_v1"
NES_JUMP_FRAMES = (1, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 32)


@dataclass
class NESPlayerMotion:
    x_speed: int = 0  # signed 1/16 pixel/frame
    x_accumulator: int = 0  # fractional acceleration, 1/256 speed unit
    x_fraction: int = 0  # fractional position, 1/256 pixel
    y_speed: int = 0
    y_force: int = 0
    y_fraction: int = 0
    force_up: int = 0x20
    force_down: int = 0x28
    force: int = 0
    origin_y: int = 0
    previous_jump: bool = False
    facing: int = 1
    moving: int = 0
    running_speed: int = 0
    running_timer: int = 0

    @classmethod
    def from_ram(cls, ram):
        def signed(n):
            return int(n) if n < 128 else int(n) - 256

        return cls(
            x_speed=signed(ram[0x57]),
            x_accumulator=int(ram[0x705]),
            x_fraction=int(ram[0x400]),
            y_speed=signed(ram[0x9F]),
            y_force=int(ram[0x433]),
            y_fraction=int(ram[0x416]),
            force=int(ram[0x709]),
            force_down=int(ram[0x70A]),
            facing=1 if ram[0x33] == 1 else -1,
            moving={0: 0, 1: 1, 2: -1}[int(ram[0x45])],
            running_speed=int(ram[0x703]),
            running_timer=int(ram[0x783]),
        )

    def advance(self, *, direction, jump, grounded, y, run=True):
        """Return integer (dx, dy, jumped); B/run is held in this profile."""
        self.running_timer = max(0, self.running_timer - 1)
        absolute = abs(self.x_speed)
        jumped = bool(jump and not self.previous_jump and grounded)
        if jumped:
            index = sum(absolute >= n for n in (9, 16, 25, 28))
            self.force_up = (0x20, 0x20, 0x1E, 0x28, 0x28)[index]
            self.force_down = (0x70, 0x70, 0x60, 0x90, 0x90)[index]
            self.force = self.force_up
            self.y_speed = (-4, -4, -4, -5, -5)[index]
            self.y_force = self.y_fraction = 0
            self.origin_y = int(y)
            grounded = False
        if grounded:
            running = direction == self.moving and (run or self.running_timer > 0)
            if direction == self.moving and run:
                self.running_timer = 10
        else:
            running = absolute >= 25
        maximum = 40 if running else 24
        friction = 0xE4 if running else (0xD0 if self.running_speed or absolute >= 33 else 0x98)
        if self.facing != self.moving:
            friction *= 2
        if grounded:
            if absolute >= 28:
                self.running_speed = absolute
            elif direction == self.moving:
                self.running_speed = 0
            elif direction and absolute < 11:
                self.moving = self.facing
                self.x_speed = self.x_accumulator = 0
            if direction:
                self.facing = direction
        if grounded or direction:
            sign = direction or (-1 if self.x_speed > 0 else 1 if self.x_speed < 0 else 0)
            if sign:
                total = self.x_accumulator + sign * friction
                carry, self.x_accumulator = divmod(total, 256)
                self.x_speed += carry
                self.x_speed = (
                    min(maximum, self.x_speed) if sign > 0 else max(-maximum, self.x_speed)
                )
        dx, self.x_fraction = divmod(self.x_fraction + self.x_speed * 16, 256)
        if self.x_speed:
            self.moving = 1 if self.x_speed > 0 else -1
        dy = 0
        if not grounded:
            if self.y_speed >= 0 or (
                not (jump and self.previous_jump) and self.origin_y - int(y) >= 1
            ):
                self.force = self.force_down
            carry, self.y_fraction = divmod(self.y_fraction + self.y_force, 256)
            dy = self.y_speed + carry
            carry, self.y_force = divmod(self.y_force + self.force, 256)
            self.y_speed += carry
            if self.y_speed >= 4 and self.y_force >= 128:
                self.y_speed, self.y_force = 4, 0
        self.previous_jump = bool(jump)
        return dx, dy, jumped

    def wall_contact(self):
        self.x_speed = 0

    def vertical_contact(self):
        self.y_speed = 0
        self.y_force = 0

    def bounce(self):
        self.y_speed = -4  # NES Goomba/Koopa stomp; other enemy types are separate profiles
        self.force = self.force_down
