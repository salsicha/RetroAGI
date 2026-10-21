# Tactic families dataset (2026-10-18 to 2026-10-21)

The tactic layer's training data (user, 2026-10-18: "Run the skills and
tactics and save the data"; "Teacher plays both"): twelve rounds of the
teacher playing the 42 tactic families' layouts through the training
pipeline, choosing the tactics and carrying out the skill's actions
(tactic_data_20261018: seed 2026101201, 8 layouts a family and difficulty a
round, 336 a round; commits 207b11f to 709fa49), with the simulator's truth
recorded. 3,908 episodes.

- `round_NN/<family>.pkl.gz`: the round's episodes of a family, a pickled
  list of `layered_train.EpisodeRecord` (gzip): every tactic decision's
  teacher label, the teacher's named action, its execution feedback and the
  tactic and strategy it was chosen under; every frame's scene rows,
  buttons, rewards, truth and the executor's kinematics. Only episodes that
  reached their objective are here.
- `round_NN/complete.json`: episodes per family and layouts left out.
- `missed/train_NN.json`: every layout of a round whose episode missed its
  objective, with its layout and how it ended: the record of failures
  (4 to 14 a round).
- `unmade/train_NN.json`: layouts that could not be made (round 6).
- `preflight.json`: the run's source checkpoint and settings.

A round already saved is read back by `layered_train.load_round` instead of
being played again. Episodes recorded before 2026-10-21 hold no remembered
enemies (`memory` None) and eight numbers of context an enemy; they read as
"none remembered" and "no object type told".
