# Block SMB policy credit and retention repair — September 6, 2026

The full-volume run `full_volume_20260905_family_revision2_seed20260905_retry1`
was stopped with SIGTERM at the user's request. Its latest saved checkpoint is
epoch 19. No production training restart was launched.

## Evidence

For the same nine held-out tall-pipe layouts, epochs 5 and 10 mounted the pipe
9/9 times but completed 1/9 and 0/9 respectively. Epoch 15 mounted and completed
0/9: every one of the 2,880 executed frames selected LEFT. Thus the first zero
was a finishing failure; the later zero included a regression in approaching
the obstacle. Physical solvability and successful reference tests had not
established stable learned mastery.

The old collector used deterministic ranked-search decisions during stochastic
training, then credited the executed button using the actor's categorical
log probability. Committed controller frames could also receive action-choice
credit despite ignoring that choice. These are mismatches between behavior and
its learning objective, not evidence that the network needs more capacity.

## Changes

- Autonomous training draws a categorical action before generating the motor
  parameters and world-model prediction for that exact action. Search cannot
  veto the draw. The loss retains the original sampled distribution.
- Committed jumps and steady primitives keep their initiating intent for motor
  and world-model conditioning. Continuation frames receive no new action-choice
  credit. Duration credit remains at the sampled initiation. Supplied actions
  and deterministic evaluation do not receive on-policy action credit.
- Optional candidate evaluation measures predicted progress toward a local
  collision region in normalized engine coordinates. Mounts, gaps, enemies,
  bridge boarding/exits, and the final goal use their corresponding region.
  A valid mount may progress locally while moving away from the final goal.
- The full-volume recipe disables candidate search in evaluation. It measures
  the greedy learned actor; ordinary training samples the same actor. Search
  remains an opt-in diagnostic and still depends on learned prediction accuracy.
- Tall-pipe now shares the local traversal path and the set of jump holds
  certified by replaying real collision physics. Mount/finish diagnostics and
  the original episode completion condition remain available.
- Success replay retains the actual successful button sequence as well as its
  layout. A failed live rehearsal can trigger a separate physical demonstration.
  Only a sequence that still completes is admitted, using fresh observations and
  logits with actor imitation weight 0.1. It receives no on-policy credit and
  does not count as a successful live rehearsal. Failed demonstrations are rejected.
- The full-volume rehearsal budget increases from 12 to 42 balanced samples,
  allowing two per family once all 21 families have successful examples.
- CUDA preflight now exercises both three successful demonstrations and three
  autonomous rollouts with optimizer updates, real frozen vision, and the full
  128-dimensional policy. It asserts that autonomous updates have policy credit
  and no oracle supervision.

## Validation

Regression tests cover sampled-action conditioning and likelihood gradients,
controller continuation credit, local progress versus final-goal distance,
geometry-certified tall-pipe coaching at all difficulties, replaying a success
through a different model, and rejecting unsuccessful retention demonstrations.
Existing model, controller, training, and traversal suites also passed. Ruff,
Black, and whitespace checks passed.

CUDA preflight completed three demonstration updates and three on-policy
updates with finite losses and gradients. Autonomous rollouts used all six
actions, zero oracle-supervised steps, and nonzero policy loss (-0.068438).
Peak allocated GPU memory was 2,214,747,648 bytes. These are execution and
learning-contract checks, not a new accuracy or retention benchmark.

Production training remains stopped. A fresh run must establish per-family
mounting, finishing, and retention over repeated evaluations before claiming
that these repairs restore accuracy.
