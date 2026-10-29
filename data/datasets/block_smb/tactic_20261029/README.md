# Tactic families dataset (2026-10-22: plants that wait for Mario)

`tactic_20261018` with its five plant families recorded again under the
plant rule (user, 2026-10-22: "make every Block SMB plant follow the real
game's rule": a plant does not come up while Mario is within 33 pixels of
it, as the NES's do): full_smb_opening_proxy, mixed_section,
piranha_avoidance, tactics_bridge_sequence and tactics_obstacle_sequence,
the same layouts (tactic_plants_20261022, commit 7732531; 440 episodes over
twelve rounds). The other 37 families' files are tactic_20261018's. 3,936
episodes.

- `round_NN/<family>.pkl.gz`, `round_NN/complete.json`: as tactic_20261018.
- `missed/train_NN.json`: every layout of a round whose episode missed its
  objective (the plant families' from the new recording).
- `unmade/`, `preflight.json`: tactic_20261018's.

The skill also learns from these episodes during tactic training (user,
2026-10-22: `LayeredTrainConfig.skill_learns`).

## 2026-10-25

tactic_20261022 with full_smb_blocks_proxy, max_points_hold_ground, speed_run_hold_ground from the teacher's recording rec_tactic_20261025 (artifacts/block_smb; 12 rounds, 8 layouts a family a round, seed 2026101201): the families found missing from the Full SMB failures, and the hold-ground scene that shows what Mario holds for. Their episodes are added to each round (replacing any the old dataset had).

## 2026-10-25

tactic_20261025 with full_smb_blocks_proxy from the teacher's recording rec_tactic_20261026 (artifacts/block_smb; 12 rounds, 8 layouts a family a round, seed 2026101201): the families found missing from the Full SMB failures, and the hold-ground scene that shows what Mario holds for. Their episodes are added to each round (replacing any the old dataset had).

## 2026-10-25

tactic_20261026 with max_points_block_goomba, speed_run_block_goomba from the teacher's recording rec_tactic_20261027 (artifacts/block_smb; 12 rounds, 8 layouts a family a round, seed 2026101201): the families found missing from the Full SMB failures, and the hold-ground scene that shows what Mario holds for. Their episodes are added to each round (replacing any the old dataset had).

## 2026-10-25

tactic_20261027 with pen_pipe_climb, full_smb_rows_proxy from the teacher's recording rec_tactic_20261028 (artifacts/block_smb; 12 rounds, 8 layouts a family a round, seed 2026101201): the families found missing from the Full SMB failures, and the hold-ground scene that shows what Mario holds for. Their episodes are added to each round (replacing any the old dataset had).

## 2026-10-29

tactic_20261028 with max_points_block_goomba and speed_run_block_goomba recorded again (rec_tactic_20261029, 8 layouts a family a round), replacing their old episodes: in half the layouts Mario arrives running and max points stops short of the block, holding until the Goomba is out from under it. 12 rounds, seed 2026101201; their episodes are added to each round (replacing any the old dataset had).
