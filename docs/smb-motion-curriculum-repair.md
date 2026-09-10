# Motion and curriculum repair — September 9, 2026

This revision completes the [stomp/scrolling repair](smb-stomp-scrolling-repair.md).
It addresses measured motion loss, missed bridge windows, and missing recovery
supervision. It does not establish 99% autonomous completion; the restarted
30-epoch run supplies that measurement.

## Observation and compatibility

The shared pixel tracker retains enemy/platform tracks for up to four frames
through body occlusion or camera-registration loss. Velocity estimates carry an
age; lost landmarks no longer immediately erase a useful estimate or turn it
into an observed zero. Hidden tracks support association and motion memory only:
they never create an invisible collision surface. A shared unclipped body edge
measures displacement; subtracting Mario's displacement cancels camera motion.
On a platform, the negative of this relative velocity measures Mario's active
locomotion without mistaking platform carry for walking.

Both Block and Full use `canonical_semantic_motion_v4`, preserving A/B/C shapes
8/16/64 and the hierarchy, LSTM and adaptive controller implementations. C58 is
enemy estimate age, C59–61 are platform-relative velocity, a fresh-measurement
bit and age, and C62–63 are enemy-relative velocity and its fresh-measurement
bit. Velocities divide by 3 and clip to [-1,1]; ages divide by 5 and saturate at
1. C55–57 retain three semantic spatial descriptors. Existing physical and
availability slots retain their meanings. Unobservable world velocity/facing
are masked; stale relative estimates remain distinguishable from fresh ones.

The objective/executor contract is `observable_traversal_v4`, coaching is
`canonical_collision_coaching_v4`, and NES scenarios use `block_smb_nes_land_v4`.
Old core bundles fail compatibility checks. The seven segmentation labels and
collision-body interpretation are unchanged, so qualified CNN/ViT weights remain
reusable. The CNN teacher → domain ViT → canonical scene → shared core curriculum
is described in [the segmentation curriculum](smb-segmentation-curriculum.md).

## Decisions and supervision

Every bridge wait lasts one physical frame before the actor decides again.
This conservative re-observation rule also covers uncertain motion, masked
shore contact and nearby boarding/exit windows without relying on a coarse
phase transition. It never chooses walking for the actor. Other primitive
commitments, including jump holds, retain their physical semantics. The collector
uses the same executor and labels bridge waits with the one-frame bin. Bridge
coaching probes whether walking now completes the relevant transition, then
reobserves if waiting; it no longer teaches a long committed departure delay.
When NES momentum would turn a wait into drifting, coaching teaches a physically
checked braking input. The narrow moving-bridge regression required this to
avoid sliding off before passive friction stopped Mario.

Replay balances approach, wait, board, ride, exit and routine phases within each
family, then actions within each phase. Priority updates preserve each
family/phase/action group's mass. Physical phase labels, including certified exit departures that begin before shore
contact, are training-only sampling metadata: a visually misclassified exit still gets its share of practice, while
the actor receives the unchanged pixel-derived objective and observations.

Stomp collection now has two actual-policy routes in addition to canonical,
nearby, scrolling and feasible momentum-reset routes. One follows the policy's
real approach until a jump proposal, probes safe holds at that exact state, and
coaches the completed suffix. The other executes the policy's original jump and
waits for an actual missed landing, preserving position, velocity, camera and
tracker history, before coaching braking, turning and re-interception. Failed
prefixes are not successful actor labels. Only collision-completed suffixes are
retained. Routes with no miss or an unrecoverable miss are reported and skipped.
These routes run inside each numbered epoch, with no bootstrap stage.

`wait_timing` is explicitly an alias of `bridge_wait` in the canonical NES
curriculum. Alias requests resolve to the same canonical family and scenario ID;
training and evaluation deduplicate aliases. The full configuration lists 20
independent families, with 25 layouts each per epoch and 30 each in validation
and test (600 cases per split). This removes duplicate coverage and training
weight without deleting legacy family names or old artifact provenance.

## Verification and restart

Regression checks cover camera cancellation, clipped edges, motion ages and
expiry, hidden-body association, carry, bridge reconsideration without a phase
change, replay mass under priorities, alias equivalence, and physical recovery
from an executed overshoot with forward momentum. Existing component swapping,
Full pixel input isolation, coaching, landing and numbered-epoch tests also run.
A bounded frozen-vision bridge collection/GPU update check verifies the complete
collector → phase-labelled replay → optimizer path; it is not autonomous policy
qualification. Its results are in
`artifacts/smb_composable/motion_curriculum_v4_check.json`.

Restart uses fresh shared-core weights, the existing qualified frozen vision
checkpoint, 30 numbered epochs and 1,000 updates per epoch. No fixed scenes,
independent-family prerequisites, bootstrap updates or vision retraining are
added. The active run pointer records the committed revision, PID and log.

Verification completed: the 150-test focused suite passed, followed by final
targeted reruns including two additional regressions (36 tests in the last
run). Frozen-vision coaching completed all six original bridge cases across
easy/medium/hard, produced boarding and exit replay strata, and completed ten
CUDA optimizer updates with finite loss. These are executable coaching and
optimizer checks, not a claim of learned autonomous accuracy.

The current [policy-state coaching revision](smb-policy-state-coaching.md) extends
actual takeoff and miss recovery across raised-obstacle and stomp families, adds
actual bridge braking-state collection, and uses error-adaptive replay with
retention. Coaching/replay provenance is v5; the v4 scene/runtime interfaces and
qualified vision weights remain compatible. Validation now reports partial
family results every ten cases.
