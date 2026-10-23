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
