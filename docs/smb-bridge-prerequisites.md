# Moving-bridge prerequisites and passive progress

`bridge_mount` and `bridge_dismount` separate two moving-platform jump skills
from full bridge traversal. They use the same hierarchy, LSTM, adaptive controller,
canonical pixel observations, physical executor, and frozen vision as the other
Block families. Explicit task goals also work through the shared Full SMB scene
interface; simulator collision state is used only for training labels and credit.

## Tasks and collision credit

* **bridge_mount:** start on the near shore, wait for an approaching bridge to
  enter jump range, then jump and land on the moving platform. Credit requires
  a jump-initiated airborne interval from the near shore and a physical grounded
  landing supported by the moving bridge, including valid edge contacts. Walking
  onto the bridge or merely overlapping it in midair does not complete the task.
* **bridge_dismount:** start supported on the bridge with zero active velocity.
  Wait as it travels toward the far shore, then jump from bridge support and
  land on that shore.
  Passive travel or walking off cannot complete the task. The bridge stops with
  a gap before the shore, so a jump is necessary.

Widths vary from 88–100 pixels in easy, through 72–84 in medium, to 56–68 in
hard. Speeds, spawn positions, and shore positions also vary. The coach certifies
actual landings using all 16 physical hold durations. The base coach waits for
multiple safe holds and selects an interior hold. Alternate coaching also covers
both ends of the interval when only the longest hold works, and a longer hold
within the base window. The production builder retains all four routes on
every bridge layout so difficulty cannot exclude a timing boundary.
Policy-state correction at unreachable takeoffs supervises the jump decision
without inventing a hold duration. Duration limits use the actual physics menu.
Autonomous validation has no coach.

The tasks declare `task_objective=bridge_mount` or `bridge_dismount`. Their
observable goals use the existing mount/clear-gap skill encodings with magnitude
64, distinguishing an explicitly requested jump from optional walking. Existing
stomp magnitude 128 and other task meanings remain unchanged. Neither new task
exposes future bridge reversal limits to the policy.

## Ordering within the numbered epochs

This section describes the earlier prerequisite-gated recipe. The current
full-volume recipe and its verification are described in
[the production supervision repair](smb-bridge-production-supervision.md).

The full-volume configuration contains 22 independent families and 30 epochs.
`wait_timing` remains an alias of `bridge_wait`. The two prerequisites are listed
before the complete bridge families. While either prerequisite is below the
configured 99% validation gate in any difficulty, `bridge_wait` and
`moving_bridge` are excluded from training collection. All other families,
including both prerequisites, continue training in the ordinary numbered epochs.

With ten validation cases per difficulty, the 99% gate requires all 30 cases in
each prerequisite to succeed. Once both pass, the full bridge families enter
training in the following epoch. Unlocking is persistent; the prerequisite
families remain in replay and continue receiving fresh examples. Validation
continues to report all 22 families (660 cases), including full bridge scores before they
have been unlocked. Logs explicitly record active families and unlock events.

There is one shared policy, no separate prerequisite model, no bootstrap, and no
extra optimization budget outside the numbered epochs. The existing 1,000
updates per epoch and 25 layouts per active family are retained. If the
prerequisites do not pass within the run, the logs and final qualification must
report that fact rather than silently claiming full bridge training occurred.

## Deduplication

The old full-bridge alternate route only changed jump parameters that its
walking coach ignored; every base/alternate pair in the audit was identical.
That duplicate collection call is removed for `bridge_wait` and `moving_bridge`.
The new jump tasks have actual alternative safe-window/hold choices.

Successful bridge routes are fingerprinted across their complete replay tensors.
Repeated routes are omitted from optimization while episode metadata records
`duplicate_of` and zero retained rows. All retained episode offsets are rebuilt.
Physical bridge scenarios are also fingerprinted without metadata or reward coefficients; duplicate
layouts are deterministically resampled. Training keeps this set across chunks
and epochs. Validation/test batches are deduplicated within their own sets.

## Learning from passive progress

The environment already pays rightward high-water progress and potential-based
goal-distance improvement independently of the action token. Mario carried
toward the goal while choosing NOOP or braking can receive positive progress
without choosing RIGHT. This revision does not double-pay that reward.

`bridge_carry_progress` attributes the already-paid high-water progress to
forward platform carry. Returning over previously reached positions cannot earn
it again, and death earns no carry credit. Completed coaching stores this value
in the optional `carry_progress` replay column. Legacy datasets default to zero.

The imitation sampler uses positive carry progress to give successful NOOP/LEFT
braking rows up to three times their original weight before phase/family
normalization. Each family retains its allocated total practice mass. Adaptive
group balancing and retention still apply. This connects observed useful passive
travel to the actual training sampler; it is not an added reinforcement-learning
optimizer or an oracle action override.

## Provenance and verification

Coaching and demonstration provenance advance to v6. The model input dimensions,
canonical motion slots, calibrated NES physics, and vision weights are unchanged.
The restart uses fresh policy weights and reuses the qualified frozen perception
checkpoint byte-for-byte.

Regression coverage includes required jump landings at all difficulties,
walking-credit rejection, carry reward without RIGHT, no repeated carry payment,
probe-state restoration, reward-aware replay allocation, prerequisite unlocking,
physical-layout deduplication, and route-offset preservation. Development learning
checks and their outcomes are stored under
`artifacts/smb_composable/bridge_prerequisites_v6/`; their weights are not used to
initialize production training.

The regression checks cover 193 tests: the focused suite of 188 plus two
Full SMB pixel-adapter tests and two source-surface rejection tests, and an edge-landing credit test. The latter
ensure jumping on the bridge after walking onto it cannot masquerade as a mount,
and jumping from shore cannot masquerade as a dismount.

A bounded check trained one fresh shared policy on six training layouts per new
family for 2,000 total updates, with frozen perception. Under the final collision
credit rules, normal greedy playback completed 5/6 held-out mount cases and 4/6
dismount cases. This is evidence of learning, not 99% qualification; full bridge
training remains dependent on both prerequisites passing the production split.
