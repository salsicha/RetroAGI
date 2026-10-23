# Collision lessons dataset (2026-10-22: plants that wait for Mario)

`collision_20261021c` with its four plant families recorded again under the
plant rule (a plant does not come up while Mario is within 33 pixels of it):
collide_wait_plant (324 lessons, all misses now: a plant due up under a
standing Mario stays down), collide_run_plant (384, all misses),
collide_jump_plant (1,296: 360 hits) and collide_pipe_wait (1,584: 122
hits), from collision_plants_20261022 (commit 7732531). The other twelve
families' lessons are collision_20261021c's. 9,598 lessons.

- `<family>.pkl.gz`, `complete.json`, `failures.json`: as collision_20261021c.
- `plant_report.json`: the plant families' run (its trackers trained one
  pass, measured on held-out layouts).
