# Family-only Block SMB training — September 8, 2026

Production training now uses generated families exclusively and runs for 20
full-volume epochs. The previous production process was stopped at the user's
request; its latest completed checkpoint is epoch 36.

The old fixed JSON layouts remain available as explicit legacy diagnostics and
unit-test fixtures. They are absent from default training configuration, the
full-volume recipe, and production evaluation. They do not contribute training
examples or a separate promotion requirement to a family-only checkpoint.
The full-volume launcher rejects a configuration that includes fixed scenes.

## Why removing the old path is more than removing a metric

The fixed-scene audit found that ordinary tasks without Monte Carlo metadata
received different skill requests, primitive labels, and rehearsal coverage.
`tasks.scenario_family` now resolves an explicit `task.family` independently of
dataset provenance, while existing generator metadata remains compatible.
Unlabelled ordinary traversal uses the same geometry-based local objectives.
Special success rules such as mandatory bridge boarding or stomping require an
explicit task/family contract; merely containing an enemy or moving object does
not imply those requirements. Optional overhead obstacles above a continuous
safe floor no longer force an unnecessary mount/clear request.

## Expanded families and controller repairs

`retreat_recovery` includes flat leftward travel, leftward gaps of 36–56
pixels, and leftward elevated landings with 28–52 pixels of rise. Every
variant retains direction-appropriate reward shaping and a physically verified
route. Local target selection, landing credit, duration probes, and route
augmentation handle both directions. The support edge, next platform, and
terrain probes in the existing numeric state slots now follow the task’s initial
travel direction. That orientation stays stable through a small finish
overshoot, and local objectives exclude obstacles beyond the remaining goal.
The diagnostic `support_right_dx` keeps its historical right-edge meaning;
`support_forward_dx` and `terrain_direction` describe the goal-relative inputs.
Checkpoint dimensions are unchanged, but leftward behavior requires rechecking
and training under the revised observations.

`bridge_wait`, `moving_bridge`, and `wait_timing` retain the wide 100-pixel
bridge cases and add 48–60-pixel bridges moving at 0.5–1.1 pixels/frame.
Moving-bridge approach spawns range from x=20 to x=60. The controller recognizes
approach separately and rechecks physical boarding availability at the shore;
one walking action no longer irrevocably ends waiting. Live collection,
batched evaluation, and demonstrations share those phase semantics. The
reference route uses the same physical phases. Actual engine boarding and
far-shore support remain necessary for bridge task success.

Bridge training budgets include the verified route length plus recovery margin,
with a floor of 240 frames. Evaluation still obeys its explicit 320-frame limit.
The new cases enter normal demonstration collection, family-balanced rehearsal,
success replay, mastery scheduling, and failure replay.

## Evaluation and transfer

With no fixed scenes, the overall completion score and action diagnostics come
from held-out generated-family validation. An empty evaluation fails. The gate
requires the aggregate configured rate, every family, every easy/medium/hard
bin, and no missing family or difficulty coverage. Removing a test suite cannot
produce a passing result by vacuous truth.

Family-only checkpoint transfer requires passing Monte Carlo validation and
measured noncollapsed validation actions. Legacy checkpoints that configured
fixed scenes retain their old fixed-scene requirements. Fixed-scene success is
not fabricated for new runs. Validation/test layouts remain reproducible and
separate from demonstration training layouts.

## Full-volume recipe

`scripts/configs/block_smb_full_volume.json` is the family-only recipe: 20
epochs, 512 base generated layouts per epoch, 64 failure-replay samples when
applicable, 42 success-replay rehearsals, and 1,000 balanced demonstration
updates per epoch.
It measures all 32 families at three difficulties, ten cases per difficulty,
after every epoch. The frozen ViT, the shared observation, fixed jump/wait
commitments, per-frame walking, and carried recurrent state match the
demonstrated learning path.

Evaluation episodes, evaluation layouts, training layouts, teacher routes and
demonstration replays are independent, so the recipe runs them on
`parallel_workers` (12) single-threaded CPU worker processes; the results match
the in-process path. Evaluation layouts are generated once per run.

Training rollouts also run on the workers. Policy updates backpropagate through
each step's policy call, which cannot leave the worker that ran it, so workers
play each update batch's episodes without gradients using the current weights
and record every step's policy inputs and decisions. The learner then re-runs
all of the batch's policy calls as one batched forward pass and rebuilds the
terms the losses consume (`rollout_workers.py`). With dropout off, the rebuilt
terms, losses and gradients match in-process rollouts to float rounding; in
training, dropout and Gumbel noise are drawn afresh in the re-run. Every
episode still plays with the weights of the update batch it joins. The learner
runs on the CPU (`online_training_device`), which is faster than the GPU for
these many small per-step losses; demonstration fitting stays on the GPU.

Validation, which steers the curriculum, runs after every epoch; the reported
test split runs every `monte_carlo_test_interval_epochs` (5) epochs and after
the last.

## One observation, one physics (October 1, 2026)

A Full SMB probe of the September 30 checkpoint found the strategy layer
choosing "mount" on ~85% of grounded 1-1 frames, partly because Block SMB fed
the policy inputs the NES cannot supply. Block SMB's purpose is ground truth
for teachers and small composable scenarios, not a richer observation, so Block
and Full SMB now share a single observation and a single physics:

- **Physics.** `MarioScenarioEnv` moves Mario only with the NES motion model
  (`retroagi/core/smb_physics.py`): no coyote time, no jump buffering, no
  rebound on a held jump, a 10×12 body, and jump bins in NES frames
  (`NES_JUMP_FRAMES`). A transferred policy needs no height calibration.
- **Layouts.** Every family is generated once and finished for the NES body
  (`monte_carlo._finish_layout`: feet kept for the shorter body, running
  takeoffs for `pit_leap`/`platform_hop`, floor-span enemy patrols for
  `enemy_stomp`, an explicit stomp task for stomp families). Its route is then
  verified under NES physics (`_verified_route`: the authored route, then a
  terrain search, then a bridge search). `bridge_wait` places the bridge so the
  opening wait under NES walking falls in each difficulty's band.
- **Observation** (`retroagi/core/smb_scene.py`). Each game's geometry observer
  (Block simulator truth or NES RAM) reports the visible scene in screen
  coordinates, and `SMBProjector` lays out the C stream by `C_SPANS`:
  position, semantic class probabilities, support, the 28 `FEATURE_NAMES`,
  enemy history, availability, relative enemy/platform motion, and a coarse
  class layout. Positions and distances are over the 256-pixel screen,
  velocities on the NES scales, and the goal slots hold the visible local
  objective, kept from takeoff while airborne. Features Full SMB cannot supply
  do not exist: coyote time, jump buffer, the episode clock, enemy patrol
  limits and lift travel limits. The scene handed to the policy carries no
  patrol or travel limits (`HIDDEN_FIELDS`), the finish flag is not rendered,
  and segmentations are mapped to the shared classes by meaning; an encoder's
  latent tokens never reach the policy.

Checkpoints record `observation_spec()`; loading or transferring a checkpoint
whose recorded observation differs is refused, so older checkpoints must be
retrained. The runtime contract carries only control settings.

A random initialization receives 10,000 demonstration bootstrap updates. An
explicit `--init-checkpoint` uses weights only, skips bootstrap, and starts a new
optimizer and epoch counter. The launcher records the resolved configuration,
source hashes, commit, and initial checkpoint hash. It never resumes the old
36-epoch run implicitly.

The shared-policy qualification before restart uses refreshed training data for
the four expanded families plus retained data for the other families, then
separate autonomous validation and test rollouts. The shared policy passed all
21 families in two consecutive validation rounds separated by another 1,000
training updates, and passed the independent test. After the final correction
to finish-overshoot geometry and observation orientation, the frozen policy
completed **630/630** additional held-out layouts on seed 937661: 30 per family,
ten at each difficulty. This establishes simulator learnability on these
samples, not full-game transfer or a guarantee on unseen layouts.

Qualification evidence is in
`artifacts/block_smb/joint_learning_20260908_family_only/`, including
`result.json` and `current_runtime_holdout.json`. The qualified policy SHA-256 is
`903e8a82d7ae553c91b6dfa0b62ad0baa8fc82f4f00526d76933cad40a0ebe17`.
The broad regression suite passed 343 tests; targeted recovery and task-contract
checks also passed after the final geometry changes. A CUDA preflight exercised
three demonstration optimizer updates and three autonomous policy updates
with frozen perception and nonzero policy loss.

The production restart uses a copy of that qualified policy, a fresh optimizer,
and epochs 1–30. The run directory is
`artifacts/block_smb/full_volume_20260908_family_only_seed20260908/`:

```sh
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
python -u -m scripts.block_smb_full_volume \
  --output-dir artifacts/block_smb/full_volume_20260908_family_only_seed20260908 \
  --init-checkpoint artifacts/block_smb/full_volume_20260908_family_only_seed20260908/initial_policy.pth
```

The run's `source_manifest.json`, `resolved_config.json`, `training.log`, and
`events.jsonl` record the actual source commit, configuration, and progress.
