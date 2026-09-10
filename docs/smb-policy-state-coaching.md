# Policy-state coaching and adaptive replay

The v4 epoch-1 audit found a gap between correct coaching labels and autonomous
behavior. Tall-pipe jump labels were all correct at demonstrated takeoffs, but
the actor jumped earlier, where the learned 18-frame hold was too short. Bridge
braking was present in replay but almost never selected. Compound routes and
stomp mounts exposed the same takeoff and recovery coverage gaps.

## Collection inside the numbered epochs

`canonical_collision_coaching_v5` extends actual-policy collection to tall pipes,
pipe mounts, stairs, platform chains, mixed sections, chained obstacles,
opening proxies, enemy stomps and stomp mounts. Each attempts two routes:

* **Takeoff correction:** follow the current policy with the shared physical
  executor. At the selected jump proposal, certify hold durations at the actual
  position and velocity. A proposal with no safe hold receives an approach or
  braking correction immediately. Compound families rotate the selected takeoff
  across the first three proposals to cover later obstacles.
* **Miss recovery:** execute the policy's original actions and holds until an
  attempted local objective is missed at landing or the player stalls against
  a raised obstacle. Continue collision coaching from that exact state, keeping
  the camera, motion, perception tracker and executor history. Successful early
  stomps do not suppress recovery collection at a later pipe or gap.

Bridge families additionally collect the first policy/coach decision disagreement
and a separate braking correction after actual policy boarding. The latter can
reach a bad ride state even if the policy disagreed with waiting earlier.

Only completed coached suffixes become supervised actor/duration replay. The
uncorrected policy prefix is not labeled as successful behavior. Fatal or
unrecoverable prefixes are reported and omitted; normal completed coaching
remains available. There is no policy teacher, veto or duration replacement in
validation or gameplay. Physical state supplies training labels and diagnostics,
while the model continues to receive the shared canonical observations.

Collection reports decision counts, jump proposals, impossible takeoffs, unsafe
holds at otherwise feasible takeoffs, missed braking and missed waits. These
measure the joint policy decisions at the states it visits. They describe the
sampled collection prefixes, not unbiased full validation accuracy.

## Adaptive practice with retention

The full-volume trainer starts from family/phase/action balancing, then adjusts
group allocation from errors observed in training minibatches. Ride decisions
and waiting/braking on approach or boarding receive a larger error-dependent
boost. A group with only a few examples receives a smaller boost than a well
supported group. This avoids giving two unusual shoreline examples the same
additional error allocation as hundreds of missed ride-braking decisions.

Each family retains its original total practice mass. At least 35% of every
original group's allocation is reserved for retention; the remaining 65% can
move toward difficult groups within that family. Error estimates update during
optimization and a group relinquishes its extra allocation as it is learned.
Prioritization still balances examples within the resulting groups. The original
sampler remains available to legacy callers; full-volume training explicitly
enables `adaptive_groups=True` and records `adaptive_decision_groups_v1`.

Logs report the sampled count, error moving average and allocation of each
family/phase/action group. These are training-label diagnostics; autonomous
family completion remains the measure of learned behavior. Preserving practice
mass does not guarantee that a skill's completion score cannot regress.

## Progress, interfaces and restart

Validation continues to execute the normal greedy policy across 600 cases.
Every ten cases it now reports partial family completion/death counts, instead
of remaining silent until the entire split finishes. Partial scores identify
how many examples have actually completed and are not final family scores.

The input representation remains `canonical_semantic_motion_v4`, the runtime
remains `observable_traversal_v4`, and NES scenarios remain `block_smb_nes_land_v4`.
No tensor meanings, component dimensions, CNN/ViT labels, or physical executor
rules change in this revision. Block and Full still share the hierarchy, LSTM,
adaptive controller and canonical interfaces. Only coaching/replay provenance
advances to v5. Existing qualified frozen vision is reused byte-for-byte; see
[the composable segmentation curriculum](smb-segmentation-curriculum.md).

The production restart initializes fresh policy weights and uses the existing
30 numbered epochs, 25 layouts per independent family per epoch and 1,000
optimizer updates per epoch. Coaching and adaptive replay occur within those
epochs. No bootstrap, fixed scenes or preliminary family-training stage is added.

## Verification

Regression tests reproduce an early pipe jump needing a longer hold, an
impossible early stomp-mount jump, failed pipe landing, walking against a pipe,
and recovery at a later raised obstacle after a successful stomp. Bridge tests
check correction at an actual braking state. Sampling tests check increased
practice for a failing brake group, retained family/group allocation, support
for small groups, and withdrawal of extra allocation after errors disappear.
Validation-progress tests execute through the normal playback interface. The
focused coaching, sampling, runtime, composability, transfer, perception and
physics regression suite passes all 162 tests. A separate collection check on
six fresh bridge training layouts retains four completed actual-braking
corrections and rejects two policy prefixes that fail before reaching a
recoverable boarding state; rejected prefixes supply no positive labels.

The bounded development learning check uses a copy of the immutable v4 epoch-1
policy and its training replay, plus newly collected **training** layouts. It
then evaluates normal greedy play on the audited failures and successful control
families. Its weights are development artifacts, not production initialization.
Results and reproduction scripts are under
`artifacts/smb_composable/policy_state_repair_v5/`.

## Moving-bridge prerequisites (v6)

The current curriculum adds jump-on/jump-off bridge prerequisites, deduplicates
bridge scenarios and replay routes, and reinforces productive passive carry.
See [Moving-bridge prerequisites](smb-bridge-prerequisites.md) for task credit,
ordering within the 30 epochs, verification, and component compatibility.
