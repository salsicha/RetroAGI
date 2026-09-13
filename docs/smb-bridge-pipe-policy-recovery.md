# Bridge departure and pipe recovery repair

The September 12 audit reproduced bridge jumps outside the feasible departure
window and three related composite failure patterns: walking into a pipe after
a stomp, using a short hold at a tall second pipe, and never retrying after a
failed mount. These failures affected `bridge_mount`, `chained_obstacles`,
`mixed_section`, and `full_smb_opening_proxy`. Every audited pipe stall still
admitted a physically valid jump from its final position.

## Training from actual policy states

The Block SMB full-volume trainer now records up to two policy trajectories
per family/difficulty bin during each numbered epoch. It includes bridge mount
and dismount, pipe teachers, chained obstacles, the enemy gauntlet, mixed
sections, the opening proxy, and required stomps. Collection uses only fresh
train-split layouts. It does not copy validation/test failures into training.

`policy_recovery.py` replays the exact action prefix and identifies:

- bridge waits inside a feasible departure window, impossible takeoffs, and
  feasible takeoffs with an invalid hold;
- short pipe jumps, grounded wall stalls, and mount opportunities after landing
  or stomp-bounce recovery;
- invalid required-stomp departures and holds, including recovery toward an
  enemy that Mario has already passed.

From each selected grounded state, physical coaching attempts a complete
successful continuation. Bridge corrections cover the next feasible departure
and the closing boundary of a safe window. Pipe corrections select certified
holds and can back away for a run-up when the wall state itself is unreachable.
The suffix respects jump holds, release, and automatic stomp bounce. Failed
continuations are discarded.

Only the completed suffix supplies supervision. Its failed policy prefix is
replayed to reconstruct position, velocity, enemies, bridge motion, camera and
phase history, then excluded from the dataset. Suffix boundaries are explicit
so walk targets cannot accidentally point into the following episode.
Validation and autonomous playback still use the learned policy and its normal
executor; physical coaching never substitutes actions there.

## Retention and objective phases

The existing 1,000-update demonstration rehearsal now includes the current
repairs and the two most recent earlier nonempty collections. Within a family, objective
phase and decision action, half the practice mass is reserved for original
routes and half for repairs when both exist. This prevents a small repair set
from disappearing among the 180-layout base curriculum, while preserving
practice on original successes. Family weights and within-group prioritization
remain active. The pool is bounded to three nonempty collections.

Demonstrations now populate their objective-phase groups: enemy approach,
gap/wait, pipe mount/bridge boarding, riding, and exit. Finish walking therefore
does not dilute the jump group for a blocking pipe. The demonstration contract
advances to version 6; fresh demonstrations are built for the restart.

## Observation and supervision corrections

`geometry_features` now retains a blocking platform at wall contact, including
leftward traversal. Previously the strict "ahead" test discarded the pipe
when Mario touched its edge, while the local objective still requested it.
The next-platform distance is now zero at contact and its height remains
visible. Feature dimensions and ordering are unchanged; the frozen perception
checkpoint is reused.

An empty certified hold set now produces an initiation penalty without a
fabricated duration label for local obstacle jumps as well as bridge and
required-stomp jumps. Feasible short pipe jumps continue to receive the full
safe duration set. Required-stomp coaching and difficulty-preserving failure
replay are described in [the stomp repair](smb-enemy-stomp-supervision.md).

Saved-best checkpoint selection also now writes the current evaluated model,
optimizer, epoch and step, instead of copying the preceding epoch's checkpoint.

## Verification and production recipe

Regression tests exercise bridge departure-window continuations, post-stomp
pipe recovery, short second-pipe jumps, left/right wall observations, exclusion
of failed prefixes and held-out data, retention mass, bounded integration into
numbered epochs, and saving the correct best-checkpoint weights.

Development artifacts are under
`artifacts/block_smb/bridge_pipe_recovery_fix_20260912/`. The physical audit
checks complete repaired routes on the prior validation failures; those routes
are diagnostics only. The shared-policy learning check starts from a copy of
the epoch-12 policy, rebuilds demonstrations with 12 train layouts per family,
then exercises two 72-episode on-policy rounds and the production 1,000-update
rehearsal with fresh policy-state corrections. Its held-out evaluations also
include the other families as retention controls. It is a bounded development
check, not a substitute for the full 180-layout production run.

The focused suite passed 220 tests, with two additional suffix-boundary and
unreachable-pipe regressions passing in the final 11-test recovery run. Black
26.5.1 and Ruff checks passed on all changed Python files. The physical audit
produced 99 successful continuations covering all 46 prior validation failures.

After two development learning rounds, each of the five audited families
(`chained_obstacles`, `enemy_stomp`, `bridge_mount`, `mixed_section`, and
`full_smb_opening_proxy`) passed 30/30 validation cases and 30/30 separate test
cases, including 10/10 easy bridge mounts on each split. Retention was imperfect:
bridge dismount went from 30/30 to 28/30 validation cases, while the remaining
18-family control panel improved from 52/54 to 53/54. These bounded results
support the repair but do not establish complete mastery of every family.

The restart uses the existing 30-epoch full-volume recipe, fresh policy and
optimizer, fresh demonstrations, and the same frozen ViT. Recovery collection
occurs within those numbered epochs. There is no runtime safety veto or
additional production training stage.
