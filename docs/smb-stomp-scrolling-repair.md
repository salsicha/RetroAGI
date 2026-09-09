# Enemy stomp: jump timing and scrolling

The epoch-1 policy from `full_volume_20260909_sensorimotor_v2/run/epoch_01`
completed 12/30 enemy-stomp validation layouts. Fifteen failures timed out and
three died. Scrolling was a contributing problem; unsafe jump holds already
occurred before the camera began moving.

## Measured cause

On easy validation seed 20260908, index 10000, the policy initiated its first
jump at frame 24, Mario x=32, with velocity 1.3125 pixels/frame. It chose a
16-frame A hold; collision probes certified only 2, 4, 6, 8, 10, and 12 frames
from that exact state. A medium failure chose 14 frames where only 8 and 10
worked. Replacing only unsafe holds during a diagnostic intervention rescued
all three inspected failures, preserving the successful fourth control.
That intervention used a physical oracle and is evidence about the cause,
not an allowed playback solution.

The stored replay contained 45 right-jump decision rows for this family; only
30 received safe duration predictions. Its 25 left-jump rows began in artificial
stationary recovery states, with no left-walk/braking decision examples.
Real overshoots had forward momentum. All 7,099 decision rows marked absolute
velocity unavailable because the floor had no visible static horizontal
landmarks. Camera-lock could therefore make a moving player look stationary.
The target was also frozen in an incorrectly estimated world frame during flight.

A learning experiment with the tracking/coaching fixes alone still reached only
21/30 original validation cases, 17/30 scrolling translations and 19/30 nearby
cases at a separate seed. Those results were not accepted as a reliable fix.
A subsequent physical intervention isolated invisible patrol limits: at hard
index 10002, frame 45, identical current state needed 14–22 frame holds with the
short patrol, but 8–12 frames with floor-wide movement limits. These safe sets
did not overlap. The reset-memory policy and its four-frame motion tracker have
no input identifying those future invisible reversals. Training more against
those incompatible targets cannot resolve that observational ambiguity.

## Changes

- NES `enemy_stomp` generation uses the full floor span for movement limits;
  enemies cannot reverse at arbitrary invisible points inside the approach.
  Closing speed remains observable during scrolling. Generated routes are
  revalidated in collision physics. The NES distribution is version 3
  and the family's parameter metadata records `enemy_motion="floor_span"`.
- Collision search begins farther away for incoming enemies, adding their
  travel over a 64-frame horizon to the probe distance. Both route generation
  and coaching use this rule. The former fixed 50-pixel cutoff could start
  after the safe window and silently reject fast incoming-enemy layouts.
  Hard-enemy generation and completed routes are checked in both directions.
- The pixel tracker associates enemies relative to Mario and averages the last
  four relative displacements, resetting on motion reversal or missing matches.
  Subtracting the two screen displacements cancels camera translation. Unknown
  absolute velocity stays unknown; a new relative-velocity input and validity
  bit provide the actually observable quantity.
- The canonical projector masks unobservable world velocity to zero and its
  derived facing to a neutral 0.5, alongside the unavailable flag. It cannot
  pass screen-relative speed through the world-velocity slot during scrolling.
  The measured relative-velocity channel remains available independently.
- A required stomp follows its visible enemy during flight. Brief disappearance
  retains the target in Mario-relative coordinates, propagating observed closing
  velocity for at most four frames. Target disappearance alone cannot credit a
  stomp; observed contact/bounce still establishes completion.
- Training collects successful routes at nearby later takeoffs and at the
  policy's proposed takeoffs. Physical probes label every safe hold from that
  exact state. Delayed approaches are permitted only if the next state still
  admits a safe stomp. The policy learns the correction; playback never calls
  these probes or overrides an unsafe model hold with an oracle.
- Scrolling approach variants translate Mario, the enemy, and the goal
  by 80–144 pixels while extending the uninterrupted floor. This puts takeoffs
  under camera lock without changing the local approach physics. Enemy movement
  limits follow the extended floor, so translation does not add a hidden boundary.
- Recovery examples begin 12–20 pixels beyond the enemy with forward velocity
  1.25 pixels/frame for moving enemies or 2.5 for stationary enemies. They teach
  braking, turning, and then a left jump. Only complete collision-verified
  stomp-and-finish routes enter
  replay. One-way scrolling makes some faster/farther overshoots unrecoverable;
  the sampler avoids those reset states. Existing action balancing gives braking
  decisions practice.

The hierarchical transformers, LSTM, controller and 8/16/64 interfaces retain
their architecture. Shared C slots 62 and 63 now carry relative enemy velocity
(scaled by 3 pixels/frame, clipped to [-1, 1]) and its availability bit. Seven
semantic descriptor slots remain. The same implementation serves Block and Full
SMB, with no RAM in the pixel provider. The diagnostic oracle provider expresses
relative velocity in the same units, including passive platform carry.

The scene encoder is `canonical_semantic_relative_v3`, the objective/runtime
contract is `observable_traversal_v3`, and coaching is
`canonical_collision_coaching_v3`. Bundle loading rejects old feature meanings;
new runtime playback also checks the batch's scene encoder. Old policy bundles
and stored replay cannot simply be relabeled. The seven-class vision interface
and collision labels are unchanged, so the qualified frozen perception
checkpoint is reusable without vision training.

## Verification

Regression tests cover scrolling in both directions over featureless floor,
missing targets and resets, airborne target tracking, oracle/pixel relative
velocity meanings, the exact failed 16-frame hold, shorter holds at later
takeoffs, and braking before a recovery jump across all three difficulties.
The existing component, transfer, coaching, perception, physics and 30-epoch
schedule tests are also checked.

A separate development experiment uses a fresh 128-dimensional policy with the
qualified frozen dense vision component. The first experiment used seed
20260909 and layouts starting at index 40000;
its unsuccessful results and hidden-boundary intervention are stored under
`artifacts/smb_composable/stomp_scroll_repair_20260909/`. After correcting enemy
motion, a fresh experiment starts at training index 42000 under distribution v3,
with results in `artifacts/smb_composable/stomp_observable_v3_20260909/`.
New-distribution scores must be distinguished from the earlier v2 layouts.
This development check is not a stage or prerequisite of full-volume training.

The full-volume recipe still collects 25 layouts per family inside each of 30
numbered epochs and applies 1,000 updates per epoch. The added stomp variants
consume that existing update budget. These source changes do not update a
Python training process that already loaded the previous modules. A new full
run must regenerate policy/replay under version 3 and can reuse frozen vision.


## Intermediate result and final velocity correction

All 141 focused, component, transfer, perception, physics and schedule regression
tests pass, along with Ruff and whitespace checks. The code preserves the shared
architecture and keeps collision coaching out of autonomous playback.

The development policy started fresh, reused frozen vision, and initially received 6,000
updates on 36 training layouts with successful route variants. The first 18
layouts preceded the incoming-enemy search correction; the additional 18 use
the corrected generator and include incoming hard enemies. No validation or test
rows entered optimization. Full scenario records, checkpoints, scripts, hashes,
and `summary.json` are under `stomp_observable_v3_20260909`.

| Intermediate evaluation before masking unobservable motion | Completed |
| --- | ---: |
| Standard validation, seed 20260908 | 27/30 |
| 120-pixel scrolling translations of validation | 18/30 |
| Separate test seed 20260915, enemy offsets ±4 pixels and initial Mario velocities 0.25/0.75 | 22/30 |

These results do **not** establish reliable family mastery or 99% performance.
They verify the code repairs and show that the limited learned policy still has
transfer/generalization failures, especially in scrolling scenes. Correcting
camera initialization alone rescued none of the inspected failure subset (the
subset completed 1/8 with a settled initial camera), so those misses cannot be
attributed solely to an artificial initial camera jump. Further policy training
and coverage validation remain necessary; these experimental weights are not
promoted as a qualified shared policy.

The remaining scrolling failures exposed an input inconsistency: an unavailable
world-velocity slot still contained screen displacement. The projector now masks
that surrogate and its derived facing, preserving the independent measured
relative velocity. Recorded training C/next-C inputs were explicitly migrated to
that missing-value rule, with physical action/hold labels and splits preserved.
The same development policy received 3,000 more updates; vision stayed frozen.
The final source pipeline applies this mask during collection and playback.

| Final masked-input evaluation | Completed |
| --- | ---: |
| Standard validation, seed 20260908 | 27/30 |
| 120-pixel scrolling translations of validation | 26/30 |
| New test seed 20260917, nearby enemy positions and starting speeds | 28/30 |

The additional optimization accompanies the input correction; this is not a
fixed-weight causal ablation. The nearby test uses a new seed. The relevant 67
regression tests passed again after the masking change; Ruff and whitespace
checks passed. The final experimental policy is still **not 99% qualified**.
Its remaining misses require more policy learning/coverage validation before
claiming reliable family mastery. The experiment is not a new training stage,
and its extra updates do not alter the full-volume recipe's epoch budget.

The existing full-volume process, PID 696347, was not restarted by this change
and still has the earlier modules loaded. A restart must use fresh policy/replay
under the new contracts, while reusing the existing qualified vision checkpoint.

The subsequent [motion/curriculum repair](smb-motion-curriculum-repair.md) adds
bounded motion memory, per-frame bridge re-observation, phase-balanced replay,
actual-policy miss recovery, and an explicit `wait_timing` alias. Its v4 interface
supersedes the historical contracts and pre-restart status described above.
