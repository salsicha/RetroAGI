# Bridge takeoff coverage repair — 2026-09-11

After the execution-contract repair, epoch 1 still scored 20% on easy mounts
and 0% on every dismount difficulty. The run was stopped before further training.

Replaying the saved policy and probing the same physical departure states found
that these failures were reachable jumps with incorrect hold choices:

| Task | Difficulty | Selected hold | Safe holds at that departure |
|---|---|---:|---|
| mount | easy | 28 | 32 |
| mount | medium | 26 | 26, 28, 32 |
| mount | hard | 28 | 28, 32 |
| dismount | easy | 26 | 32 |
| dismount | medium | 28 | 32 |
| dismount | hard | 28 | 32 |

These are the original validation seed 50000 and sample indices
660, 670, 680, 690, 700, and 710. The model could observe the changing geometry,
but its jump-duration examples began only after at least three holds were safe.
The earlier reachable states were labeled only as waiting. On a slightly early
policy departure, B therefore reused the 26–28-frame holds from later states.

Alternate coaching made this worse: requiring four or five safe holds excluded
some narrow hard dismount windows entirely. Such alternatives were silently
replaced with the original route or omitted.

The revised training-only coach retains the original three-hold teacher and
adds alternatives at both ends of the longest-hold-only interval, plus a longer
hold at the ordinary three-hold interval. Covering both boundaries prevents
interpolation toward the shorter hold before it becomes safe. Every route still executes the actual
physics through the required collision landing. Runtime action selection,
physical geometry, rewards, and completion rules are unchanged.

Demonstration contract version 4 requires refreshing bridge data in old caches.
The full-volume restart builds its demonstrations afresh.

Regression checks cover successful 32-frame onset jumps for both tasks at every
difficulty, preservation of existing execution contracts, and alternatives for a
hard dismount whose window never offers five safe holds. The focused and related
suites passed 91 tests. Diagnostic replays, data, and learned-policy results are
stored under `artifacts/block_smb/bridge_timing_repair_20260911/`.

## Learned-policy verification

A fresh shared policy (seed 101) trained on 24 training layouts per family,
with four physically executed routes per layout and the production frozen
perception checkpoint. The bounded probe used a numeric-head learning rate of
0.003; the full-volume restart retains its established 0.0005 recipe.

After 2,000 supervised updates, autonomous evaluation passed all 60 original
bridge validation layouts (seed 50000, indices 660–719): 10/10 at every
difficulty for both mount and dismount. No oracle chose evaluation actions or
durations. Training on onset examples alone had fixed mounts but left dismount
failures even after 6,000 updates; the final boundary examples resolve those
original failures within 2,000 updates.

The same 2,000-update checkpoint passed 119/120 additional test layouts
(seeds 202 and 303, 10 per difficulty and family per seed). Mount scored
60/60; dismount scored 59/60, with one medium failure at seed 202. Every
family/difficulty/seed met the 90% gate. The original validation remained
60/60 after 3,000 updates.

The probe qualifies this repair on bridge tasks; full-volume evaluation must
still establish retention alongside all other families.
