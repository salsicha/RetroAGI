# Collision lessons dataset (2026-10-21, third recording)

The collision trackers' training data recorded again (user, 2026-10-21:
"Test hidden enemies properly ... Find out why jumps at walkers got worse"):
the sixteen collision families (`retroagi/stages/block_smb/collision_families.py`),
each one scripted action of Mario's (run, jump or wait) against a walker, a
monster or a plant, over every parameter combination at three difficulties
(9,816 layouts), played through the training pipeline with the simulator's
truth recorded (`collision_train.py`, commit b297f44). It replaces
`collision_20261021b`. Since then:

- the pipe-wait lessons (`collide_pipe_wait`): a plant that, as the NES's
  does, stays down while Mario is within 33 pixels of it; Mario watches it go
  down, waits near it or farther for 1 to 5 seconds, then jumps onto its
  pipe or runs across it and off (122 hits, 1,462 misses);
- late jumps at walkers (12 to 20 pixels: 100 hits in the family, 24 before);
- waits recorded aimed at nothing (they were aimed at enemy slot 0);
- the list of tracked things (commit a61056f) is made again from these
  recordings when they are read.

- `<family>.pkl.gz`: the kept lessons of a family, a pickled list of
  `layered_train.EpisodeRecord` (gzip): a contact during the action (a hit
  or a stomp) or a miss. 9,598 lessons, 13,644 enemy tracks.
- `complete.json`: lessons kept, failed and contacts per family.
- `failures.json`: the 218 failed layouts (mostly pipe waits whose plant
  touched Mario before the action) with their parameters and layouts.
- `tracker.pt`: the collision trackers trained on these lessons, and
  `report.json`: their measures on layouts drawn off the sweep's grid
  (`held_out_layouts`: never trained on); `config.json`: the run's settings.
