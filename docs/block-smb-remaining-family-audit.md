# Remaining Block SMB family audit — September 5, 2026

This audit covers the 15 families following the tall-pipe, stomp, and bridge-wait
repairs. Most fixes concern geometry, credit, demonstrations, conditioning, and
episode length; checkpoint tensor shapes do not change.

## Evidence and changes, in review order

Baseline rates are the held-out validation results at epoch 25 of
artifacts/block_smb/full_volume_20260903_seed20260903_retry1/events.jsonl:
nine layouts per family, 160-frame training, and 200-frame evaluation.
These are observations from that run, not estimates of universal task difficulty.
In particular, enemy_hop trained well on its narrow original distribution.

| Family | Old validation | Problem found | Revision 2 repair |
| --- | ---: | --- | --- |
| wait_timing | 8/9 | All three probed tiers could finish with the bridge removed. Fixed scripts and closest-endpoint coaching did not certify departure. | Wide bridge-dependent crossing and collision-based departure windows; the policy chooses whether to wait, with no forced opening action. |
| pit_leap | 8/9 | Coaching toward the center of the wide goal demanded extra distance after clearing the gap. Goal contact could precede landing. | Targets the near part of the far ledge, requires real support, and coaches the set of successful holds. First missed landing ends the attempt in validation and training. |
| pipe_mount | 8/9 | Some oracles received goal credit while ascending beside the pipe, as early as frame 13. A jittered marker rejected otherwise valid pipe support. Horizontal correction missed height shortfalls. | Goal spans the pipe top and requires support there. Oracle and duration labels replay height, collision, and landing together. |
| enemy_hop | 9/9 | Difficulty tiers moved the stationary enemy only a few pixels. A local clearance could be penalized for falling short of the finish. | Wider enemy-position bands, local enemy-clear targets, verified duration sets, and a finish phase. Avoidance remains valid; a stomp is not required. |
| stair_climb | 0/9 | Reported step-height variation was not applied to geometry. Jump coaching targeted the final goal rather than the next step. | Actual tier-dependent heights and sampled widths; objectives advance through real surfaces. Higher safe landings can clear an earlier step. |
| single_gap | 0/9 | Nearly fixed approach geometry and a far-finish target supplied poor initial-jump feedback. | Varies gap position and width, targets the landing surface, then clears the gap request for finishing. |
| retreat_recovery | 0/9 | Rightward progress paid movement away from the left goal. | Removes the conflicting reward, adds goal-distance shaping, and retains leftward recovery conditioning. |
| platform_chain | 0/9 | Mostly coin/goal jitter around fixed terrain; coaching treated each jump as a request to reach the final floor. | Varies gaps and elevations, preserves a wide far shore, targets successive surfaces, and requires support at the finish. |
| moving_bridge | 0/9 | All three original tiers were jumpable with the bridge removed; no verified board/ride/exit objective. | Requires bridge boarding and far-shore support, varies approach position, and uses bridge phases and safe departure windows. |
| mixed_section | 0/9 | Effectively an alias for the enemy gauntlet; 186–189-frame oracles exceeded training's 160-frame limit. | Samples two actual compositions, uses local objectives, and receives a completion-aware budget. Metadata reflects the selected composition. |
| full_smb_opening_proxy | 0/9 | 195–198-frame oracles were cut off. Global jump goals contradicted intermediate clearances; section metadata was inaccurate. | Local enemy/mount/finish phases, sufficient time, varied pipe positions, calibrated demonstrations, and corrected metadata. Remains a Block SMB proxy. |
| enemy_patrol | 0/9 | Small speed jitter around fixed positions/directions; global target and unintended jump input during stomp bounce. | Varies patrol offsets/directions, verifies local holds from current state, and releases jump during automatic bounce recovery. |
| enemy_gap | 0/9 | A persistent enemy request hid the preceding gap; landing-relative enemy placement barely varied. | Varies offsets and approach geometry; switches gap → enemy → finish and verifies duration against the combined scene. |
| chained_obstacles | 0/9 | 195–198-frame oracles exceeded training time; long held-jump scripts and distant-target penalties conflicted with the executor and local successes. | Completion-aware budgets, local objectives, varied pipe positions, and demonstrations normalized to the hold menu and bounce contract. |
| chained_enemy_gauntlet | 0/9 | 186–189-frame oracles exceeded training time; an enemy request did not describe the gap and pipe stages. | Local gap/enemy/mount/finish conditioning, varied pipe placement, repaired demonstrations, and time to finish. |

The original bridge-removal probe used RIGHT for 24 frames, RIGHT_JUMP for 16,
then RIGHT. It bypassed both wait_timing and moving_bridge in all three tiers.
The original pipe_mount probe swept holds 1–16 and observed credit before
support. These are measurement defects even where old success rates were high.

## Shared training repairs

A jump retains its objective from initiation. The teacher snapshots the current
environment and tries each 1–16-frame hold through landing or contact.
Successful holds form a set; categorical coaching rewards probability anywhere
in that set. This accounts for momentum, height, enemies, collisions, and
support without asking one jump to reach a distant finish. A safe later surface
can satisfy an earlier local objective.

These probes generate labels; they do not choose or execute the policy's action.
Actual local support anchors coaching and overrides a disagreeing distance
heuristic. Certified holds prevent erroneous overreach penalties. Scalar release
coaching and independent oracle duration labels do not compete with the safe-set
loss. Local clearance is separate from final episode success.

A stomp closes the current jump on its collision frame. Jump input is released
during automatic bounce recovery; horizontal direction remains the policy's
choice. The next grounded state selects the next objective. Avoiding a live
enemy remains valid in avoidance families.

The simulation snapshot now restores platform support identity, moving goal
position, goal/stomp/bridge credits, failure flags, reward potential, and energy.
Otherwise imagined successes could leak into real credit and rewards.

Generation normalizes demonstrations to the 16-frame menu and bounce contract.
If the script fails the varied layout, a local physics teacher builds a
replacement. The stored sequence is validated again. Samples and oracle
provenance identify family revision 2; schemas reflect actual generated ranges.

Composite live training and rehearsal get at least 120 frames and 1.5 times
validated completion length. Existing stomp/tall-pipe and bridge floors remain.
Oracle validation allows 320 frames. Evaluation honors its explicit budget,
including deliberately short evaluations that must report failure.

Evaluation adds traversal_metrics: episodes with local clearance, selected local
objectives completed, finishes after local clearance, deaths, and timeouts.
These count selected objectives, not mandatory visits to every platform.
Bridge metrics retain boarding, crossing, and event-versus-timer diagnostics.

Goal routing uses explicit engine geometry, and bridge departures retain
engine-assisted events. Scores measure the policy with this controller; they
are not unaided end-to-end vision-policy scores or proof of real SMB mastery.

## Validation and fresh full-volume training

Tests exercise all 15 families across three tiers through the real trajectory
collector, exact landing credit, snapshot preservation, retreat rewards, terrain
variation, mixed compositions, timeout aggregation, and actual optimization.
Successful reference rollouts match oracle lengths without false overreach.
These establish task and demonstration correctness, not learned mastery.

Validation completed across 270 distinct tests in the targeted suites.
The CUDA preflight completed three optimizer updates with finite losses,
using about 2.2 GB peak allocated memory. Generation preflight validated
1,024 Monte Carlo samples across the regular and mastery schedules; the
mastery sample covered all 21 families.

The [checked-in recipe](../scripts/configs/block_smb_full_volume_revision2.json)
starts a fresh policy using the existing frozen Block ViT:

    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m scripts.block_smb_full_volume       --output-dir artifacts/block_smb/full_volume_20260905_family_revision2_seed20260905

It keeps 70 epochs, 512 Monte Carlo samples per epoch, mastery scheduling,
failure replay, and success rehearsal. Evaluation runs every five epochs with
320 frames. Gamma increases from 0.95 to 0.99 to keep distant finishes
influential. Eight episodes per update, down from sixteen, leave memory for
longer graphs. The foundation phase retains its 30-epoch maximum and can
graduate earlier at an evaluation gate.

The launcher's --preflight option tests the full-sized CUDA model, real frozen
perception, and successful pipe/bridge/chained trajectories without creating
or resuming a policy run. The launcher refuses to overwrite an existing event
log. Revised scores require a fresh training/evaluation run; changes to geometry
and credit make some old scores directly incomparable.
