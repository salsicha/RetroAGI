# Full SMB transfer contract and emulator audit — 2026-09-08

The transfer plumbing now preserves the qualified Block SMB observation and
control meanings. **The resulting policy is not qualified for full-level play.**
The final adapted checkpoint completed **0/3 Level 1-1 episodes** with held-out
start timing, and failed its one Level 2-1 diagnostic. Source-family accuracy
is an eligibility check for this experiment, not an emulator completion score.

The stopped Block SMB run and its epoch-5 checkpoint were preserved. Only bounded
emulator calibration, demonstration fitting, and evaluation were run. No
full-volume training has been restarted.

## Changes and remaining limits

| Transfer problem | Implemented change | What this establishes / does not establish |
| --- | --- | --- |
| Same tensor shape, different physical meanings | `smb_geometry_v1` defines the same ordered 27 physical features and 8 motion features for both stages. Block feature extraction is shared without changing its order. Full SMB no longer inserts score/lives into those slots. | Structural compatibility is checked before inference. Actual NES dynamics still differ. |
| Different semantic class IDs | Full SMB's 13 classes are mapped by meaning into Block's 7 classes by summing probabilities. | Fixes categorical meaning; it does not align independently learned ViT token spaces. |
| Missing local skill instruction | Current visible geometry supplies the same local objective and goal encoding. Airborne targets remain in world coordinates as the camera scrolls. | Reuses the source local skill interface; this is not a full-game route planner. |
| Different action execution | Checkpoints carry fixed/adaptive duration mode, walking commitments, steady primitives, frame units, run button, recurrent-state policy, critic settings, and engine-support policy. Shared train/eval/play/direct selection use those settings; the extra Full SMB jump bias is removed for this contract. | Corrects execution mismatches. Legacy checkpoints retain their legacy behavior. |
| Incorrect training credit | Sample the action once, execute the same primitive, and suppress repeated actor credit during commitments or engine overrides. Auxiliary dynamics slots use the shared physical layout and actual terminal slots. | Fixes the identified action/feature credit mismatches; it is not evidence of stable long-horizon reinforcement learning. |
| Jump-duration units and physics | Measure both engines, then store an explicit 16-bin mapping to actual emulator frames in a new checkpoint. | Matches stationary jump apex approximately. Horizontal travel, momentum, and flight time remain different and require adaptation. |
| Privileged simulator observations | Read current NES collision buffers, live object boxes and engine support. Compute motion in world coordinates. Declare unavailable patrol/platform reversal bounds and absent simulator mechanics. | This is an explicit RAM-assisted agent, not a pixels-only agent. No future emulator rollouts are used by inference. |
| Incorrect geometry | Merge solid tile bodies vertically as well as horizontally; use the live collision-box control instead of sprite artwork. Handle observed moving-platform motion and stomp recovery. | Fixes observed pipe/body, camera, and bounce errors. Moving platforms and later-game hazards are not broadly qualified. |
| Misleading episode endings | Detect individual death and castle-entry completion from the engine; the backend's default `done` waits for game-over. | A death cannot silently continue into a retry. Reaching the flag base alone is no longer called level completion. |
| Saved-state inconsistencies | Restore observer history with snapshots, discard incompatible frame shapes, repair the opening enemy and pipe fixture recipes, and reject recipes that die before finishing. | Makes the repaired diagnostic starts usable. Other old scripted fixtures are not certified routes and can now fail generation instead of silently saving a dead/respawned scene. |
| Evaluation bypasses | Shared-runtime comparison loads the contract, uses the common forward path, clears old commitments, and rejects comparisons between incompatible contracts. | Comparison remains a logit diagnostic on an externally driven stream, not a completion test. |
| Incompatible legacy warm-start | Reject the old scripted warm-start for shared checkpoints; provide actual-emulator demonstration adaptation with goals, valid duration sets, and decision masks. | Prevents silently training against the old feature and duration meanings. Use `imitation_warm_start=False` for subsequent online training. |

Native Full SMB visual features remain in the adaptation observations. The bounded
experiment updated only the numeric action and duration heads, preserving the
transferred hierarchy and source checkpoint. The `native_adapted` tag records
that training exposure; it does **not** certify visual alignment or successful
transfer. The zero-token ablation also failed, so zeroing features is not a
supported shortcut.

NES has no simulator coyote-time or jump-buffer mechanic, so those features are
zero. Patrol/platform reversal bounds are not inferred from one position: their
neutral encoding is accompanied by explicit availability diagnostics. The current
model does not consume a separate learned availability mask; training with these
missing features is an adaptation attempt, not proof that the mismatch is solved.
Object IDs outside the implemented geometry scope are reported in probe traces.

## Measured physics

The learned Block duration bins 1–16 map to these NES button-hold frames:

```text
1, 2, 4, 6, 8, 10, 12, 15, 17, 19, 22, 24, 26, 28, 28, 28
```

Block's 16-frame stationary hold rises about 68 pixels; the same NES hold rises
about 55. Longer NES holds reach about 66. The longest Block jump takes about
33 frames of flight versus about 53 in NES. Consequently, apex calibration alone
cannot preserve horizontal landing position. The profile records every trial
and its nearest-apex selection criterion. Wait durations are not rescaled by the
jump calibration table.

## Observed results

Artifacts are local under `artifacts/full_smb/transfer_contract_20260908/`.
Maximum x below is the world x of Mario's collision box. These are small,
deterministic diagnostic samples, not confidence bounds on general success.

| Final diagnostic | Episodes | Result |
| --- | ---: | --- |
| Frozen transferred weights + measured duration map (`final_calibrated_baseline`) | 1 | Death at max x=1131; no completion. |
| 5,000 fitting updates, Level 1-1, held-out start perturbations (`final_level_1_1_verified`) | 3 | Deaths at max x=726, 675, 1417; 0/3 completion; official threshold failed. |
| Same adapted weights, Level 2-1 (`final_level_2_1`) | 1 | Death at max x=323; no completion; official threshold failed. |
| Repaired first-enemy start (`repaired_enemy_section`) | 1 | Walked for 31 frames without initiating the required jump, then died at max x=312. |
| Repaired first-pipe section (`repaired_pipe_section`) | 1 | Reached max x=726 and stalled for 180 frames; no completion. |
| Level 1-2 | 0 | Not evaluated: the installed integration has no `Level1-2` start state. This is a coverage gap, not a scored policy failure. |

Collection retained 4,591 frames from three teacher routes. The teacher may branch
emulator snapshots during **training-label collection only** and retains jump
arcs that actually survive and land. Evaluation uses only the policy and runtime
executor. Those teacher routes reached the flag/end sequence under the earlier
flag-base completion criterion; they are not official policy completion results.

The 1,000-update fit had mean loss 1.693; the 5,000-update fit on the same data had
mean loss 1.019. Earlier 1,000-update policy probes also finished 0/3. The runs use
different start timings and some earlier runtime revisions, so they do not support
a controlled claim that the longer fit improves or regresses the policy. They do
establish that lower demonstration loss has not produced reliable play.

The repaired enemy diagnostic makes the remaining behavior concrete: a live,
recoverable approach still produces walking where a jump is needed. This cannot
be explained solely by the old dead fixture. Repeated fitting on a few successful
teacher trajectories leaves policy-visited timing/landing states insufficiently
covered. That distribution shift is a plausible contributor, not a proved sole
cause. Native visual features, missing motion-bound information, remaining flight
physics differences, and goal selection must still be isolated with controlled
ablations before attributing the failure to model capacity.

Next learning work should collect corrections from the policy's actual failure
states, reserve separate starts/layouts for evaluation, measure action and duration
errors separately, and compare numeric-only versus visual adaptation. Expansion
to later hazards should follow reliable local emulator learning. Adding more
source-family epochs or repeating the same teacher data is not supported by these
results. No adapted checkpoint is promoted by this change.

## Reproduce

Run from the repository root using the project Python environment and existing
local emulator/vision assets. Use fresh output paths; scripts refuse to overwrite
adaptation/calibration checkpoints. The source gate remains enabled.

```bash
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
python -m retroagi.stages.full_smb.transfer \
  --block-policy-checkpoint /path/to/qualified-block-policy.pth \
  --output-checkpoint /path/to/transferred.pth --device cuda
python -m scripts.full_smb_calibrate_physics \
  --checkpoint /path/to/transferred.pth --output /path/to/calibrated.pth
python -m scripts.full_smb_transfer_probe \
  --checkpoint /path/to/calibrated.pth --output /path/to/frozen-probe \
  --episodes 3 --steps 2400
python -m scripts.full_smb_adapt_geometry \
  --checkpoint /path/to/calibrated.pth --output /path/to/adaptation \
  --episodes 3 --steps 1800 --updates 1000
python -m scripts.full_smb_transfer_probe \
  --checkpoint /path/to/adaptation/policy.pth --output /path/to/held-out-probe \
  --episodes 3 --steps 2400 --initial-walk-offset 11
```

Probe summaries record checkpoint SHA-256, Python source hashes, runtime contract,
explicit start perturbations, per-episode deaths/progress/completion, unsupported
objects, and the existing full-level success threshold. Per-frame traces include
actual model inputs, goals, chosen actions, executed buttons, and hold frames.
A saved-section probe never claims whole-level qualification. Seeds alone do not
create independent NES layouts; start waits and walks are recorded explicitly.

The commit contains code, tests, and this report. Checkpoints, raw traces,
ROM content, and emulator snapshots remain local.


## Validation

- All Full SMB regression modules plus the new transfer-contract module:
  **161 passed**. The contract module contains 20 regression tests.
- Earlier Block training/family traversal and integration regression selection:
  **137 passed** (overlaps the Full SMB selection; these are not additive totals).
- Real-emulator shared-checkpoint entry points: one three-frame online training
  update with weighted dynamics loss, three-frame evaluation, three-frame play,
  and a three-frame identity comparison all completed. These are plumbing checks,
  not learning or gameplay qualification.
- Black, Ruff, and `git diff --check` passed for the committed changes.
