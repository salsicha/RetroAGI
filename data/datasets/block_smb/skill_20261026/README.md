# Skill families dataset (2026-10-18)

The skill layer's training data (user, 2026-10-18: "Run the skills and
tactics and save the data"): twelve rounds of the teacher playing the skill
families' layouts through the training pipeline (skill_data_20261018: seed
2026101201, 8 layouts a family a round plus 16 extra a round for eight
weaker families; commit 05cd522), with the simulator's truth recorded.

- `round_NN/<family>.pkl.gz`: the round's episodes of a family, a pickled
  list of `layered_train.EpisodeRecord` (gzip): every decision's teacher
  label and executed action, every frame's scene rows, buttons, rewards and
  truth. Only episodes the teacher finished are here.
- `round_NN/complete.json`: episodes per family and layouts left out.
- `teacher_qualification/train_NN.json`: every layout of the round, won or
  failed (a failed one with its layout): the record of failures.
- `preflight.json`: the run's source checkpoint and settings.

A round already saved is read back by `layered_train.load_round` instead of
being played again.

## 2026-10-25

skill_20261018 with action_climb_pipe, action_stomp_down from the teacher's recording rec_skill_20261025 (artifacts/block_smb; 12 rounds, 8 layouts a family a round, seed 2026101201): the families found missing from the Full SMB failures, and the hold-ground scene that shows what Mario holds for. Their episodes are added to each round (replacing any the old dataset had).

## 2026-10-25

skill_20261025 with action_climb_pipe from the teacher's recording rec_skill_20261026 (artifacts/block_smb; 12 rounds, 8 layouts a family a round, seed 2026101201): the families found missing from the Full SMB failures, and the hold-ground scene that shows what Mario holds for. Their episodes are added to each round (replacing any the old dataset had).
