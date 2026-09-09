# SMB sensorimotor repairs — September 9, 2026

This records the initial version 2 repair. The subsequent
[stomp timing and scrolling repair](smb-stomp-scrolling-repair.md) supersedes its
stomp coaching and scene-input contract; vision remains reusable.

The shared pixel tracker, goal selector, executor and coaching collector now use
`observable_traversal_v2` and `canonical_collision_coaching_v2`. These are control
and curriculum changes; the canonical seven-class vision interface and its
10×12 small-player collision labels remain unchanged. Old policy bundles are
rejected by the new component contract. A qualified vision checkpoint can be
reused independently, with its original metrics, provenance and checksum.

## Four repaired families

| Family | Failure | Repair |
| --- | --- | --- |
| `bridge_wait` | Moving objects occluding terrain corrupted camera estimates; integer displacement flickered between velocities; clipped platform edges moved independently of the physical platform; passive carry appeared as Mario's own velocity. | Camera registration excludes old and new dynamic silhouettes and compares static boundary neighborhoods. Object tracks average four displacements, reset on reversal, and use the visible right edge when the left edge is clipped. Mario's locomotion velocity subtracts observed platform carry. Unobservable camera motion remains explicitly unavailable. |
| `platform_hop` | Migrating to NES physics left Mario starting at rest, making an immediate jump unreachable and silently introducing a run-up into a duration task. | Begin with NES running velocity 2.5 pixels/frame. Generation accepts only physically successful frame-zero jumps from the actual 16-bin menu; it cannot fall back to an approach route. The one-attempt, supported-landing goal is preserved. The NES curriculum distribution is version 2. |
| `enemy_stomp` | Passing a live enemy changed the goal to finish. The enemy could disappear one observation before its bounce was visible. | A required stomp targets live enemies on either side, with a signed recovery direction. Pixel tracking remembers a briefly missing target until observed contact followed by an upward bounce establishes completion. The collision teacher uses the same target rule and adds overshoot reset variations, retained only after actual stomp and task completion. |
| `chained_enemy_gauntlet` | A near-floor pixel box was called grounded before collision. A jump press on that frame was consumed by NES and could not trigger on the real landing. | New landings require stable feet and coherent support geometry. Tiny terrain fringes cannot support Mario. Established contact tolerates body-height jitter while the body remains horizontally supported. The executor releases an uncommitted airborne jump request and allows a fresh press after confirmed landing. The collector labels decisions at those same boundaries. |

Confirmed support also aligns the estimated body with the estimated surface.
This prevents a one-pixel fringe from changing a supported gap objective to
finish. The rule is shared between Block and Full SMB pixel playback and reads
no simulator state or RAM. Physics is used only for training labels and offline
verification. The hierarchy, LSTM, critic, controller and tensor interfaces are
unchanged.

## Verification

The regression suite covers camera translation, bridge occlusion, fractional
and clipped motion, platform carry, early landing, body-height jitter, airborne
jump release, required-stomp persistence, delayed bounce detection, physical
turn-back recovery, immediate jumps across three seeds and three difficulties,
and vision-only restart behavior. The broader component, transfer, perception
and physics tests are also run.

Pixel rollouts reuse the qualified perception checkpoint from
`full_volume_20260909_epochs_only/run/block_perception/perception.pth`.
The targeted audit uses validation seed 20260908, indices
10000, 10001, 10002, 10004 and 10005 for each of the four families, spanning all
three difficulties. Separate overshoot recovery rollouts test all three
`enemy_stomp` difficulties. A broader executable-coaching audit checks the
other families as well. Detailed local results are stored under
`artifacts/smb_composable/sensorimotor_repair_20260909/`.

Measured results: 20/20 targeted pixel routes completed, with zero false-ground
observations across 3,100 observations; all five `platform_hop` routes started
with a jump. Three separate pixel overshoot recoveries completed. The broader
63-case pixel audit initially exposed three support-jitter failures; all three
passed after the support fix, as did the other three cases in their two-family
recheck. The 118 focused/transfer/perception/physics tests passed, including the
focused rerun after correcting a synthetic carry test fixture. Ruff and diff
whitespace checks passed. No policy learning updates were run by these audits.

These checks establish executable labels and regression behavior. They are not
99% autonomous policy qualification; numbered-epoch validation measures that.

## Restart with the existing vision component

```bash
python -m scripts.smb_composable_training \
  --config scripts/configs/smb_composable_full_volume.json \
  --perception-checkpoint artifacts/smb_composable/full_volume_20260909_epochs_only/run/block_perception/perception.pth \
  --output-dir <new-run-directory>
```

This checks the NES motion audit, loads and freezes qualified vision, copies it
with a checksum and provenance record, and initializes a new policy and optimizer.
The manifest distinguishes fresh policy initialization from perception reuse.
Training begins at epoch 1 of 30, with 25 layouts per family per epoch and 1,000
policy updates per epoch. Demonstration collection and updates interleave inside
those epochs. There is no bootstrap, preliminary family training or old-policy
resume. Fixed scenes remain excluded.

Omitting `--perception-checkpoint` retains the complete fresh-perception path.
The Full SMB perception adaptation and emulator qualification stages remain
subject to their existing measured transfer requirements.

The subsequent [motion/curriculum repair](smb-motion-curriculum-repair.md) adds
bounded motion memory, per-frame bridge re-observation, phase-balanced replay,
actual-policy miss recovery, and an explicit `wait_timing` alias. Its v4 interface
supersedes the historical contracts and pre-restart status described above.
