"""Observable enemy history v1 shared by Block and NES geometry adapters.

No simulator cycle, hidden patrol bounds, or future observations enter these
features. Missing velocity is distinct from a measured zero velocity.

The optional memory extension remembers how tall the most recently seen enemy
has been exposed this episode. A feedforward policy otherwise cannot tell a
tall retracted plant from a short one once both are hidden.
"""

import numpy as np

HAZARD_NAMES = (
    "enemy_visible",
    "enemy_observed_vy",
    "enemy_vy_known",
    "enemy_visibility_change",
    "enemy_visibility_age",
    "enemy_last_seen_age",
)
# Separate from HAZARD_NAMES so six-feature checkpoints keep their exact layout.
HAZARD_MEMORY_NAMES = ("enemy_peak_exposure",)
PEAK_EXPOSURE_SCALE = 64.0


class EnemyObservationHistory:
    def __init__(self):
        self.reset()

    def reset(self):
        self.previous = None
        self.last_seen = None
        self.changed_at = None
        self.last_frame = None
        self.cached = np.zeros(len(HAZARD_NAMES), dtype=np.float32)
        self.peaks = {}
        self.last_key = None
        self.memory = np.zeros(len(HAZARD_MEMORY_NAMES), dtype=np.float32)

    def observe(self, scene, frame):
        if frame == self.last_frame:
            return self.cached.copy()
        candidates = []
        for index, enemy in enumerate(scene.enemies):
            if enemy.get("dead") or enemy["h"] <= 0:
                continue
            x, y, w, h = (float(enemy[k]) for k in ("x", "y", "w", "h"))
            left = getattr(scene, "camera_x", 0.0)
            if x + w <= left or x >= left + getattr(scene, "width", 256):
                continue
            # A pipe can hide a NES collision box. Use only the exposed part.
            for platform in scene.platforms:
                r = platform["rect"]
                if r.left <= x and x + w <= r.right and r.bottom > y:
                    h = min(h, max(0.0, r.top - y))
            if h <= 0:
                continue
            key = (enemy.get("slot", index), enemy.get("kind", "walking"))
            candidates.append((abs(x + w / 2 - scene.mario["x"]), key, y, h))
        current = min(candidates, key=lambda row: row[0]) if candidates else None
        visible = current is not None
        before = self.previous
        same = bool(visible and before is not None and current[1] == before[0])
        consecutive = self.last_frame is not None and frame == self.last_frame + 1
        known = bool(same and consecutive)
        velocity = current[2] - before[1] if known else 0.0
        transition = (1.0 if visible else -1.0) if visible != (before is not None) else 0.0
        if visible and before is not None and not same:
            transition = 1.0
        if self.changed_at is None or transition:
            self.changed_at = frame
        if visible:
            self.last_seen = frame
            # Per-object peak: a re-emerging plant keeps its identity while rising.
            self.peaks[current[1]] = max(self.peaks.get(current[1], 0.0), current[3])
            self.last_key = current[1]
        self.memory = np.array(
            [min(1.0, self.peaks.get(self.last_key, 0.0) / PEAK_EXPOSURE_SCALE)],
            dtype=np.float32,
        )
        unseen_age = 1.0 if self.last_seen is None else min(1.0, (frame - self.last_seen) / 64)
        self.cached = np.array(
            [
                float(visible),
                np.clip(velocity / 8, -1, 1),
                float(known),
                transition,
                min(1.0, (frame - self.changed_at) / 64),
                unseen_age,
            ],
            dtype=np.float32,
        )
        self.previous = (current[1], current[2]) if visible else None
        self.last_frame = frame
        return self.cached.copy()

    def memory_features(self):
        """Tallest exposure of the latest-seen enemy; zero before any sighting."""
        return self.memory.copy()
