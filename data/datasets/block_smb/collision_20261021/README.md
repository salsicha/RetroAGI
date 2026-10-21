# Collision lessons dataset (2026-10-21)

The collision trackers' training data recorded again (user, 2026-10-21:
"apply all your suggested fixes"): the fifteen collision families
(`retroagi/stages/block_smb/collision_families.py`), each one scripted
action of Mario's (run, jump or wait) against a walker, a monster or a
plant, over every parameter combination at three difficulties (7,548
layouts), played through the training pipeline with the simulator's truth
recorded (`collision_train.py`, commit 1cf1289). Since the 2026-10-18
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
  or a stomp) or a miss (the action ended with none). 7,514 lessons, 16,414
  enemy tracks.
- `complete.json`: lessons kept, failed and contacts per family.
- `failures.json`: the 34 failed layouts with their parameters and layouts.
- `tracker.pt`: the collision trackers (collision_v3) trained on these
  lessons from the 2026-10-18 ones, and `report.json`: their training and
  their measures on freshly drawn layouts of each family (with the
  remembered rows apart: "remembered/all"); `config.json`: the run's
  settings.

Read back with `collision_train.make_dataset` (or `gzip` + `pickle`).
