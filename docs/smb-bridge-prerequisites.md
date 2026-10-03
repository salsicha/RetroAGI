# Moving-bridge prerequisites and passive progress

`bridge_mount` and `bridge_dismount` separate two moving-platform jump skills
from full bridge traversal. They use the same hierarchy, LSTM, adaptive controller,
shared SMB observation (frozen Block ViT semantics plus simulator-truth
geometry), and physical executor as the other Block families. The Full SMB
adapter accepts the same explicit task objectives.

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
actual landings using all 16 NES hold durations. The base coach waits for
multiple safe holds and departs with the longest certified hold, which tolerates
departure-timing drift across the window. Alternate coaching also covers
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

## Scheduling

When these tasks were introduced, a separate trainer (since removed) kept
`bridge_wait` and `moving_bridge` out of training until both prerequisites
passed a 99% validation gate, and fingerprinted bridge routes to drop
duplicates. Neither mechanism exists in the current production trainer
(`retroagi/stages/block_smb/train.py` with
`scripts/configs/block_smb_full_volume.json`).

There, `bridge_mount` and `bridge_dismount` are ordinary Monte Carlo families
sampled alongside the full bridge families. Together with `wait_timing` they
are the prerequisites of the composed `tactics_bridge_sequence` family
(`FAMILY_PREREQUISITES` in `retroagi/stages/block_smb/hierarchy.py`), which
unlocks only after all three are mastered on held-out layouts.
`tactics_bridge_sequence` is in turn a prerequisite of
`tactics_bridge_then_gap`. Production route coverage for these tasks is
described in [the production supervision repair](smb-bridge-production-supervision.md).

## Learning from passive progress

The environment already pays rightward high-water progress and potential-based
goal-distance improvement independently of the action token. Mario carried
toward the goal while choosing NOOP or braking can receive positive progress
without choosing RIGHT. The carry credit below does not double-pay that reward.

`info["bridge_carry_progress"]` attributes the already-paid high-water progress
to forward platform carry. Returning over previously reached positions cannot
earn it again, and death earns no carry credit.

The demonstration sampler (`demonstration_sample_weights` in
`retroagi/stages/block_smb/demonstrations.py`) can use an optional
`carry_progress` column to give successful NOOP/LEFT braking rows up to three
times their original weight before phase/family normalization, while each
family keeps its allocated total practice mass. That column was filled only by
the removed separate trainer. Production demonstration collection does not
record it, so it defaults to zero and the carry weighting currently has no
effect. It is not a reinforcement-learning optimizer or an oracle action
override.

## Provenance and verification

The model input dimensions, canonical motion slots, NES physics, and vision
weights are unchanged; coaching and demonstrations are recollected. The restart
uses fresh policy weights and reuses the qualified frozen perception
checkpoint byte-for-byte.

`scripts/tests/test_smb_bridge_prerequisites.py` covers walking-credit
rejection, carry reward without RIGHT, no repeated carry payment, probe-state
restoration, carry-weighted replay allocation, source-surface rejection, and
edge-landing credit. The latter checks ensure jumping on the bridge after
walking onto it cannot masquerade as a mount, and jumping from shore cannot
masquerade as a dismount.

The original development learning check ran under the removed separate trainer;
its outputs are stored under `artifacts/smb_composable/bridge_prerequisites_v6/`
and its weights are not used to initialize production training. It trained one
fresh shared policy on six training layouts per new family for 2,000 total
updates, with frozen perception. Under the final collision credit rules, normal
greedy playback completed 5/6 held-out mount cases and 4/6 dismount cases. This
is evidence of learning, not 99% qualification.
