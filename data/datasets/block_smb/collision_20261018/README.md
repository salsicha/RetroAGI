# Collision lessons dataset (2026-10-18)

The collision trackers' training data (user, 2026-10-18): every layout of
the fifteen collision families (`retroagi/stages/block_smb/collision_families.py`),
each one scripted action of Mario's (run, jump or wait) against a walker,
a monster or a plant, over every parameter combination at three
difficulties (6,708 layouts), played through the training pipeline with the
simulator's truth recorded (`retroagi/stages/block_smb/collision_train.py`,
commit 702b833).

- `<family>.pkl.gz`: the kept lessons of a family, a pickled list of
  `layered_train.EpisodeRecord` (gzip): a contact during the action (a hit
  or a stomp) or a miss (the action ended with none). 6,674 lessons, 18,750
  enemy tracks.
- `complete.json`: lessons kept, failed and contacts (stomps, hits,
  misses) per family.
- `failures.json`: the 34 failed layouts (a contact before the action, a
  death with no enemy contact, a timeout) with their parameters and layouts.
- `tracker.pt`: the collision trackers trained on these lessons (from the
  skill run's), and `report.json`: their training and their measures on
  freshly drawn layouts of each family; `config.json`: the run's settings.

Read back with `collision_train.make_dataset` (or `gzip` + `pickle`).
