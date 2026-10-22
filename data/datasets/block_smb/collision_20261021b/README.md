# Collision lessons dataset (2026-10-21, after the hold fix)

The collision trackers' training data recorded again (user, 2026-10-21:
"apply all your suggested fixes"): the fifteen collision families
(`retroagi/stages/block_smb/collision_families.py`), each one scripted
action of Mario's (run, jump or wait) against a walker, a monster or a
plant, over every parameter combination at three difficulties (7,548
layouts), played through the training pipeline with the simulator's truth
recorded (`collision_train.py`, commit ba22288). It replaces
`collision_20261021`, recorded before a hold stopped holding against a
surface end where an enemy stands (Mario, told to wait, backed away from a
coming monster: 18 hits in 216 monster waits, 126 now). Since the 2026-10-18
lessons:

- the executor remembers enemies out of sight, and each lesson records the
  remembered rows (`memory`, `memory_identity`, `memory_match`): a plant
  down in its pipe keeps its identity and is followed while hidden;
- the plant run and jump lessons also come with the plant seen going down
  into its pipe before the action (`gone` 24);
- each enemy's context has room for its object type (none told in Block
  SMB).

- `<family>.pkl.gz`: the kept lessons of a family, a pickled list of
  `layered_train.EpisodeRecord` (gzip): a contact during the action (a hit
  or a stomp) or a miss (the action ended with none). 7,504 lessons, 15,632
  enemy tracks.
- `complete.json`: lessons kept, failed and contacts per family.
- `failures.json`: the 44 failed layouts (mostly a death with no enemy
  contact the truth shows) with their parameters and layouts.
- `tracker.pt`: the collision trackers (collision_v3) trained on these
  lessons from the `collision_20261021` ones, and `report.json`: their training and
  their measures on freshly drawn layouts of each family (with the
  remembered rows apart: "remembered/all"); `config.json`: the run's
  settings.

Read back with `collision_train.make_dataset` (or `gzip` + `pickle`).

Replaced by `collision_20261021c` (the pipe-wait lessons, late jumps at
walkers, waits recorded aimed at nothing).
