# Required-stomp supervision repair

`enemy_stomp` previously used a one-frame duration correction from the observed
miss. It did not certify the available holds from the actual takeoff state.
Medium layouts exposed two different errors: a short hold when a longer hold
could succeed, and a departure from which no hold could stomp the enemy.
Both could receive the same incremental duration label.

In the epoch-12 production audit, validation seed 50000, sample 250 selected a
7-frame opening hold where only 14–16 frames were safe. Its old target was 8.
Sample 251 also selected 7, but its departure had no valid hold at all; its old
target was again 8, without an initiation penalty.

The Block SMB trajectory collector now uses `training_target` and
`safe_jump_holds` while a required stomp is outstanding. The target remains the
living enemy even after Mario passes it, allowing leftward recovery labels.
Physical probes include the nonterminal stomp bounce and recovery landing and
restore the environment afterward.

- Nonempty safe sets supervise the categorical duration head directly. The
  recorded target hold is a member of the set, and partial arcs can receive
  certified labels before reaching the descending contact window.
- Empty safe sets mark the departure as `primitive_unreachable` and
  `jump_overreach`, even when the chosen hold is short. They receive no
  fabricated duration or release target.
- A certified reachable departure does not receive a contradictory overreach
  penalty from the older geometric fallback.
- Oracle trajectories do not add a competing time-indexed duration label in
  the required-stomp phase. Bounce and finish retain their existing execution
  behavior; the single-contact `stomp_mount` family retains its existing coach.

Failure replay now samples the reported family **and difficulty**, weighted by
failure counts. Thus `enemy_stomp:medium` failures generate fresh medium train
layouts. Seeds remain deterministic per epoch, and held-out layouts are not
copied into training. Legacy family-only failure keys remain supported. The
change applies to all families using this replay sampler; fresh mastery
sampling remains separate.

Regression coverage is in `scripts/tests/test_enemy_stomp_traversal.py` and
`scripts/tests/test_block_smb_training.py`. It covers both production cases,
partial arcs, short and maximum holds at impossible departures, gradient signs
for safe-set learning, successful stomp/bounce/finish behavior, leftward
recovery, and deterministic train-only replay of the failing difficulty.

The frozen-production verification is stored under
`artifacts/block_smb/enemy_stomp_supervision_fix_20260912/`. It compares the new
collector's labels against independently recorded physical safe sets for the
eight failing medium validation episodes and checks that the complete action
sequences remain identical. This establishes a supervision correction, not a
new trained-policy success rate. An already-running trainer needs to load the
updated code on restart, and model weights need further training to benefit.
