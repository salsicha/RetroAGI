# Composable Block SMB → Full SMB transfer and emulator curriculum

The compatible coaching repair and family-by-family audit are documented in
[canonical-smb-coaching.md](canonical-smb-coaching.md). This supersedes the initial
generic collector’s single-duration supervision and incomplete bridge goals.

Updated 2026-09-08. The implemented workflow and current verification evidence are
recorded in [the implementation report](composable-smb-implementation.md). CNN
creation, training, teacher use and component swaps are detailed in the
[segmentation curriculum](smb-segmentation-curriculum.md). This plan
supersedes the adaptation order in
`full-smb-transfer-contract.md`; that report remains the record of the failed
initial experiments. The new pipeline starts from scratch and gates each phase.

## Objective and working hypothesis

Use one implementation and compatible interfaces for the hierarchical actor,
LSTM world model, critic, adaptive controller, and primitive executor in both
stages. Train the transferable core in Block SMB with NES-compatible physics and
observations. Replace the perception module for Full SMB. First test the core
with frozen actor weights; adapt the world model on emulator transitions if
needed. Actor/controller fine-tuning is a measured fallback, not the default
substitute for correcting physics or representation mismatches.

Architecture identity and weight identity are separate requirements. The actor
and controller should initially keep identical weights across the transfer. A
Full SMB world-model checkpoint may contain updated weights while retaining the
same architecture, state format, and input/output meanings. Compatibility alone
does not prove that swapping weights improves behavior.

## Repository findings at plan inception

- Shared classes already exist in `retroagi/core/models.py`:
  `HierarchicalAdaptiveModel`, `WorldModel`, `Critic`, `AdaptiveController`, and
  `MotorPrimitiveController`. Both stages already construct the common core;
  the problem is more than duplicate architecture definitions.
- `VisionOutput`, `StageBatch`, and `VisionHierarchyProjector` provide useful
  interfaces, but the projector flattens and pools each ViT's native latent
  features into C. Equal dimensions do not establish equal latent meanings.
- The current transferred checkpoint disables carried recurrent state. The
  greedy action path bypasses critic-driven action refinement. Thus, better
  LSTM predictions do not automatically imply better action choices. The
  revised implementation must make and test this dependency explicitly.
- `AdaptiveController` itself applies generated w/b parameters; in its current
  implementation it has no independent learned weight matrices. Transformer B
  and its output heads learn the controller parameters. Preserve both the
  controller operation and its learned parameter-producing network.
- Block SMB still explicitly uses coyote time, jump buffering, and jumping again
  when landing with jump held. Its constants and jump trajectories differ from
  NES. The earlier height-only duration calibration does not solve this.

## Existing Full SMB CNN: available, no recovery needed

The legacy CNN implementation and actual checkpoint are present:

- `scripts/segmentation/segment_inference.py`
- `scripts/segmentation/segment_training.py`
- `scripts/segmentation/MarioSegmentationModel.pth`

Checkpoint size: 168,362,296 bytes. SHA-256:
`fdd812476c8715e97b39ca9060a3dc90b015966db17b3d9729e6b27f0da5be62`.
It is a real state dictionary, not an unresolved Git LFS pointer. It loaded
strictly into DeepLabV3/ResNet50 with a six-class main head and produced finite
segmentation logits on two Level 1-1 emulator frames (reset and 100 frames of
rightward movement), without downloading any model weights.

This is a load/inference check, not an accuracy qualification. The recorded
class vocabulary is default, floor, brick, box, enemy, mario. That vocabulary
cannot supply every Full SMB class or collision property. Smoke evidence is in
`artifacts/full_smb/composable_plan_20260908/cnn_inventory.json`.

The current pipeline also already has a Full SMB ViT and a sprite-composition
label generator (`scripts/vit/generate_dataset.py`). A CNN is therefore useful
as a real-frame annotation teacher, not a prerequisite for generating all labels.
The old training/inference scripts remain legacy examples with top-level side
effects. A maintained offline wrapper now exists in
`retroagi/stages/full_smb/segmentation_teacher.py`. The current pipeline audits
the recovered CNN but does not retrain it or consume its proposals automatically;
it uses independent NES collision labels to train the Full dense ViT. The latest
recorded CNN collision audit approved no classes. The manual teacher-training
module and the planned proposal-ingestion path are described separately in the
[segmentation curriculum](smb-segmentation-curriculum.md).

## Phase 1 — Define and enforce interchangeable components

Introduce a versioned component bundle with separately saved perception, shared
scene encoder/projector, actor, world model, critic, and execution settings.
Each manifest records architecture, feature layout, semantic vocabulary,
normalization, coordinate frame, timing, primitive units, recurrent-state
semantics, physics profile, and dependencies on other component versions.

Freeze/train modules explicitly. Verify requested frozen parameters remain
unchanged after updates. Loader compatibility checks must reject mismatched
feature meanings or duration units, rather than merely accepting state shapes.
A different LSTM version invalidates previous critic/policy qualification until
retested. Reset carried memory when changing component weights.

Use a canonical scene interface for terrain, actors, positions/collision-box
estimates, velocity estimates, support, local objectives, elapsed time, and
availability/confidence. Distinguish pixel segmentation from physical collision
geometry; matching sprite outlines is not sufficient for landing/stomp decisions.
Choose units/scales once for both environments. Local goals must be generated
from the same observable scene interface in both stages.

Recommended policy boundary: deterministic spatial features or a shared scene
encoder built from canonical semantic/geometry outputs. Keep domain-specific
ViT latent features inside perception, rather than directly mixing independently
learned embeddings into C. If residual learned visual features are retained,
require explicit alignment training on paired scenes and action-equivalence
checks before allowing a swap. This is a representation migration requiring
Block retraining; the old checkpoint is not silently relabeled as compatible.

All adapters project into the same A/B/C lengths and meanings for the new bundle.
Availability features belong in actual model inputs, not only diagnostics.
Preserve the common architecture family; version any feature allocation or
shape change and migrate/retrain both stages together.

Two explicit observation providers implement this interface:

1. Oracle geometry: simulator state or current NES RAM, for training labels,
   physics tests, and diagnosis.
2. Perceived geometry: ViT outputs plus a shared temporal tracker/estimator, for
   the target pixel-based player. Train Block on this provider too before
   claiming pixel-based transfer; do not hide oracle state in its policy inputs.

The currently implemented RAM-assisted Full player remains a diagnostic lane.
It is not silently presented as the final pixel-based system. Unobservable
patrol limits are unknown in both stages, rather than privileges given only to
Block. Compare the two lanes to separate perception error from dynamics/control.

## Phase 2 — Match Block physics to measured NES behavior

Create a versioned NES-compatible Block physics profile while retaining the old
profile only to reproduce historical results. Measure both engines on matched
local geometry, initial velocity, contact state, and button history.

Build paired traces for:

- acceleration, run speed, release friction, braking, reversals, air steering;
- jump press/release edges, variable height, apex, flight time, landing distance;
- holding jump through landing, early/late presses, ledge departure;
- collision boxes, wall/head contacts, corners, support and collision ordering;
- enemy motion/contact, stomp bounce and release behavior;
- platform carrying and moving-platform contacts;
- supported power states and their collision/movement differences.

Compare full x/y/vx/vy trajectories and contact/death outcomes, not just final
height. Use engine-consistent arithmetic/tables where measurements require them.
Set trajectory tolerances from observed emulator precision; require matching
success/failure/contact outcomes at the curriculum boundaries. Use independent
button sequences and starting states to validate the fitted profile. Report
unimplemented mechanics instead of treating them as matched.

Unify duration bins in physical frames over a range that covers NES maneuvers.
Once both engines use the same profile, remove the compensating height-only
transfer mapping for that new profile. Do not reuse the old 16-frame-duration
weights under different bin meanings without retraining.

Rebuild teacher trajectories and recheck physical feasibility of all 21 Block
families. Re-sample impossible parameter combinations without weakening the
skills' success definitions. Held-jump scripts must explicitly release/repress
when required. Existing family success rates do not qualify the new profile.

## Phase 3 — Create/train the CNN teacher and qualify compatible ViTs

Use the [CNN lifecycle module](smb-segmentation-curriculum.md) to recover or
construct DeepLabV3/ResNet50, prepare native six-class annotations, train or
fine-tune it on the training split, and save a versioned checkpoint plus provenance.
The reference trainer's 45-epoch recipe is distinct from the current 8,000-update
dense-ViT training and the 30 shared-policy epochs. Creation/retraining is an
offline preparation step; the current automated pipeline begins from the recovered
CNN checkpoint and runs its audit during emulator curriculum preparation.

Validate the recovered CNN on independently labeled real emulator clips. Record
per-class errors, small-enemy misses, contact-edge errors and temporal stability.
Use it only for the classes/conditions it passes; confidence filtering alone is
not proof that its labels are correct.

The planned annotation workflow combines three label sources with explicit provenance:

- sprite-composed scenes with known rendering masks;
- CNN proposals on training clips, reviewed/corrected and qualified against independent held-out clips;
- NES instrumentation for collision boxes/support/motion/event labels, distinct
  from visible sprite segmentation labels.

Fill unsupported classes with new annotations/generated labels or a retrained
teacher. Do not invent a one-to-one mapping from the six legacy classes to
the seven canonical classes or the older thirteen-class patch vocabulary. Keep real-frame validation separate from synthetic validation and
split by clips/approaches to avoid adjacent-frame leakage.

The implemented Full ViT training path currently uses instrumentation labels;
CNN proposal ingestion is not yet automated. Train or fine-tune the Full ViT and
its convolutional refinement decoder jointly to emit the canonical interface. Use paired
geometry rendered in Block and Full styles to check interface agreement. The
Block ViT must satisfy the same boundary. Validate perception by its effect on
local objectives and action choice, not only average pixel accuracy. RAM remains
supervision/evaluation evidence in the perceived lane, not an undeclared input.

## Phase 4 — Train and qualify the shared Block core

Start the full generated curriculum with one fresh shared core after perception
preparation. The user's direct-restart instruction supersedes the earlier proposal
to train separate family models as a prerequisite. Start epoch 1 with fresh policy
weights; there is no policy bootstrap stage. Within each of the 30 epochs,
interleave all-family coaching batches and the existing 1,000-update budget,
retaining earlier epochs’ examples for replay.
Measure every family on held-out layouts during that run; low intermediate scores
are diagnostic and do not block the next epoch. Final qualification governs
emulator promotion. Cover varied speeds, button histories, camera positions,
perceptual errors, and transitions between skills. Preserve physical ground truth for labels while
matching the observations actually available during playback.

Train the LSTM on ordered transitions with explicit primitive/button context,
elapsed physical time and episode boundaries. Define and enforce the update
cadence in training and playback. Measure one-step and multi-step motion,
landing/contact, death, and primitive outcomes. Sequence training is necessary
if carried history is enabled; do not train isolated frames with reset memory
and then enable history only in Full SMB.

Explicitly choose the shared world-model usage: first-pass policy for isolation,
then the same validated recurrent-context/refinement mechanism in both stages.
Demonstrate that altered predictions can influence decisions in the intended
path, and that this improves held-out behavior. Do not silently enable a new
search mode during transfer. Keep critic judgments calibrated to model outcomes.

## Phase 5 — Transfer with the actor/controller initially frozen

Assemble the Full bundle from the qualified Block actor/controller, its world
model and critic, the Full ViT, and the common scene encoder/executor. Both
stages use the same runtime contract; initially only the perception checkpoint
and environment backend differ.

Run controlled comparisons on exactly the same emulator starts:

1. Oracle scene provider with frozen Block core: isolates physics/control reuse.
2. Full ViT provider with that same frozen core: measures perception transfer.
3. Full ViT with adapted LSTM/world-model weights, keeping actor/controller
   frozen: tests whether dynamics adaptation is sufficient.

Train the whole necessary world-model component (LSTM plus its input/output and
outcome heads), not arbitrary recurrent matrices in isolation. Use actual NES
transition sequences and a replay set from Block to assess retention. Revalidate
the critic after dynamics changes. Keep architecture and interfaces identical.

Only if local tests still fail after fixing observation and physics errors,
consider a small actor/controller adapter or fine-tuning selected existing heads.
Document that the frozen-core hypothesis failed and compare against a scratch
initialization with the same emulator data/budget. Do not assume actor adaptation
is required, but do not promise that ViT plus LSTM training is sufficient.

## Phase 6 — Individual emulator approaches, then variations

Begin with the recorded first-enemy, pipe-stall and gap failures. For each:

1. Capture live, recoverable starts and declare a stable exit condition.
2. Verify solutions using the actual executor, searching jump initiation,
   duration, waiting/braking and recovery actions where appropriate.
3. Train on a small verified set and prove autonomous mastery of that set.
4. Separate action errors, duration errors, execution mismatches, perception/
   goal errors and world-model prediction errors in the report.
5. Add physically reached variations in distance, speed, enemy phase, prior
   landing, camera position and supported power state. Reserve validation/test
   conditions before fitting; repeated seeds are not independent NES layouts.
6. Collect corrections from policy-visited training states before failure becomes
   irreversible. Do not treat an unsuccessful teacher search as a valid walking
   label, or the last surviving step before death as proof of a safe action.
7. Require the same checkpoint's autonomous playback to pass held-out variations.

Proposed local promotion target: at least 99% success on a suite of hundreds of
distinct held-out approaches, with breakdowns by variation and uncertainty from
finite coverage reported. This is an engineering gate, not proof of universal
reliability. A single repeated snapshot is only a deterministic regression test.

All success checks require stable support/safe continuation beyond the obstacle.
Restore emulator, scene tracker, objective, primitive state, RNG and applicable
LSTM history together; use explicit episode resets for independent trials.
Teacher rollouts remain training-only and unavailable to deployed inference.

## Phase 7 — Composition, full levels and release evidence

Chain learned approaches without resets: enemy→pipe, pipe→gap, then longer
sections. Include states produced by the preceding learned skill, not only
curated entry states. Revisit a local family when its exit breaks the next one.

Expand actual mechanic/object coverage before adding later levels. Keep existing
full-level success criteria; report deaths, completion, progress and stalls over
held-out starts/levels. Missing content/start states are coverage gaps, not policy
failures. Measure data/update savings against the scratch baseline to establish
that Block training provides useful transfer.

Deliverables:

- independently loadable/versioned components and swap/round-trip tests;
- CNN inventory plus teacher qualification, perception datasets and class maps;
- paired physics traces and a validated Block profile;
- per-family Block learnability and transfer-source evidence;
- frozen-core / perceived-input / adapted-dynamics comparison matrix;
- per-approach and chained/full-level emulator results;
- a reproducible bundle manifest naming every component and runtime setting.

No long training run or checkpoint promotion precedes the applicable gates.
