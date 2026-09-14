# Stair final-riser recovery

The September 14 production audit found a repeatable regression: an epoch-25
policy completed the first two steps but walked against the final riser until
timeout on 55 of 60 held-out stair layouts. Every stalled state still admitted
collision-certified jumps. An earlier checkpoint with the same hidden width
of 128 completed 58/60, so the repair targets training coverage and retention.

`stair_climb` now participates in the existing train-only policy recovery
collection, with the configured limit of two trajectories per difficulty bin
per epoch. The original demonstrations and three most recent nonempty recovery
collections continue to share rehearsal using the existing family/phase/action
weights. The policy architecture and observation layout are unchanged.

The staircase collector continues examining the trajectory after finding its
first three repairs. It keeps at most three successful suffixes, prioritizing
the furthest local step in the direction of travel. This prevents repairs at
earlier steps from filling the budget before the final riser is reached.
Within a step it prioritizes:

- recovery after a jump lands without clearing its intended objective;
- the initial arrival on the preceding step;
- a prolonged pause of at least 16 consecutive grounded, nearly stationary
  transitions, followed by the shorter three-frame stall and takeoff corrections.

A pause can be either pushing against the riser or waiting without direction.
Airborne time against a wall does not count toward the grounded pause threshold.
Each selected suffix must finish the entire level. The failed prefix only
reconstructs the actual state and is excluded from supervision. Identical
already-captured pause/stall states do not repeat all collision probes on every
remaining frame. Bridge and other family repair selection retain their existing
first-successful-suffix behavior.

A repeated-learning check also exposed contradictory stair actor labels. The
executor owns the landing-release frame and suppresses another jump on the
following frame. Stair demonstrations walked on these two forced frames, but
frame-walk conversion exposed them as free walking decisions. In the audited
12-layout-per-family cache, this produced 96 mount-phase walking labels alongside
72 jump labels. The migration preserves all 72 jumps and leaves non-stair actor
masks unchanged. Contract version 7 excludes these release/suppression frames
from stair actor supervision while retaining their other training targets.
Existing frame-walk caches migrate these masks without rerendering; bridge
contract version 6 refresh requirements remain unchanged. Episode boundaries
prevent a preceding trajectory's jump from masking a new trajectory's walk.

Regression coverage replays the actual final-riser approach, checks arrival,
walking and NOOP pauses, and a failed short-jump retry, and verifies complete
successful continuations and the repair cap. A real numbered training epoch
also checks that fresh train stair trajectories enter the bounded recovery
collection. Existing tests cover failed-prefix exclusion, train/held-out
separation and balancing against original demonstrations.

The physical audit produced 165 complete successful final-riser suffixes across
all 55 original failed validation/test episodes. Those repaired held-out routes
are diagnostics only; they are not used in learning.

Development artifacts and executable checks are under
`artifacts/block_smb/stair_recovery_fix_20260914/`. The repeated-learning check
starts with a copy of epoch 25 at the original width of 128 and frozen ViT,
with a fresh optimizer and the 58,714-frame, 12-layout-per-family demonstration
cache. Each round runs 75 fresh stochastic training episodes (one per
family/difficulty plus a second per stair difficulty), collects successful
policy-state suffixes, and performs the production 1,000 rehearsal updates.
The rehearsal uses the existing mastery-dependent family weights, updated
from validation after each round. Four rounds exercise eviction from the
three-collection recovery history. Every evaluation uses the same 720
validation layouts across all 24 families plus 30 separate stair test layouts.
This is a bounded retention check, not a replacement for the full 180-layout,
512-fresh-episode production curriculum.

The focused recovery, demonstration and pipe suite passed 58 tests, including
actual-executor landing/release behavior and idempotent cache migration.
Black 26.5.1 and Ruff checks cover all four changed Python files.

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -p pytest_timeout -q \
  scripts/tests/test_block_smb_policy_recovery.py \
  scripts/tests/test_demonstration_learning.py \
  scripts/tests/test_tall_pipe_traversal.py
```

Before the actor-mask correction, the initial four-round stress check omitted the production success-replay
rehearsals and retention-imitation fallbacks. It retained stairs perfectly,
including after recovery-history eviction, but did not establish stable
retention of every other family:

| Round | Stair validation | Stair test | All-family validation |
|---|---:|---:|---:|
| Before | 1/30 | 4/30 | 679/720 |
| 1 | 30/30 | 30/30 | 708/720 |
| 2 | 30/30 | 30/30 | 711/720 |
| 3 | 30/30 | 30/30 | 697/720 |
| 4 | 30/30 | 30/30 | 684/720 |

In round 3 bridge dismount fell to 23/30, then recovered to 28/30 in round 4.
Bridge mount fell to 16/30 in round 4; the gauntlet and mixed section also lost
cases. The original and updated collectors selected identical repairs for
all 27 non-stair policy trajectories audited from round 3. This rules out a
change to those repair labels in that sample, but does not establish that
shared-policy learning is free of interference.

A subsequent collector-only check enabled the 42 production success-replay
rehearsals and retention-imitation fallbacks from the fourth-round checkpoint.
Stairs regressed to 5/30 validation and 12/30 test in its first round (672/720
all-family validation). That failed check motivated the actor-mask correction;
recovery collection alone did not establish retention under success replay.

The corrected-supervision check (`with_release_masks/`) starts from that same
fourth-round checkpoint and uses the same first-round fresh-case seed as the
failed success-replay check. It migrates the base cache and rebuilds the three
inherited recovery collections with corrected masks. Three further rounds
include 75 fresh episodes, 42 configured success-replay rehearsals, the existing
retention-imitation fallbacks, and 1,000 demonstration updates each. Total
training episodes were 120, 120, and 121. The stair rehearsal weight decreases
from 0.75 to 0.5 to the existing minimum of 0.25. The final round has replaced
all three inherited recovery collections.

| Corrected round | Stair weight | Stair validation | Stair test | All-family validation |
|---|---:|---:|---:|---:|
| Before | — | 30/30 | 30/30 | 684/720 |
| 1 | 0.75 | 30/30 | 30/30 | 698/720 |
| 2 | 0.5 | 30/30 | 30/30 | 702/720 |
| 3 | 0.25 | 30/30 | 30/30 | 702/720 |

Each corrected round passes 10/10 easy, medium, and hard stairs on both splits.
Fresh stair recovery collections contain 576, 574, and 600 frames. Thus stairs
retain 60/60 over repeated training, success replay, reduced practice, and
recovery-history turnover at the original hidden width of 128. This bounded
check does not establish full-volume or indefinite retention.

Other-family retention remains imperfect. The final validation failures are
platform chain 4/30, enemy stomp 3/30, bridge mount 1/30, and bridge dismount
10/30; the other 20 families pass 30/30. These results support the stair repair
without claiming that shared-policy interference across all families is solved.
`release_mask_audit.json` and `with_release_masks/retention_summary.json` preserve
label counts, exact source hashes, and the repeated-round results. The current
production process was left running; it must restart to load this code.
