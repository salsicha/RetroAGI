# Bridge production supervision repair — 2026-09-12

The epoch-2 full-volume policy failed nine of ten easy mounts and nine of ten
hard dismounts despite a successful bridge-only qualification.

Physical replay distinguished two errors. Every failed easy mount departed
3–9 frames before any hold could reach the bridge. Waiting those extra frames
and holding 32 repaired all nine. Every failed hard dismount departed at a valid
time but held 28 frames where only 32 worked; changing only the hold repaired
all nine.

## Correct initiation supervision

For an impossible bridge takeoff, the old correction compared Mario with the
bridge's final horizontal extent. Mario could reach that horizontal boundary
below the platform without triggering the strict undershoot test. The fallback
then used a legacy 16-frame limit, assigning a 16-frame target to a failed
32-frame attempt.

Bridge takeoffs now use the already-certified empty safe-hold set to penalize
the jump decision. Such an attempt has no successful duration or release label,
so neither target is invented. This applies to short and long attempts. When
physical holds exist, their duration supervision remains intact. Completed
spans use the actual executor menu maximum for fallback limits, overreach
thresholds, and hold normalization; standalone legacy callers retain 16.

Replaying the unchanged failing checkpoint confirms that sample 660 now has
one initiation correction and no duration target. Sample 710 retains its safe
32-frame target, normalized to 1 rather than 2.

## Complete production route coverage

The production builder selected difficulty with `i % 3` and bridge route with
`(seed + i) % 3`. At seed 20260908, hard bridges never received the final frame
of the longest-hold-only interval. The bridge-only qualification explicitly
supplied all variants and therefore did not exercise this defect.

When bridge augmentation is enabled, the production builder now retains the
canonical route and all three physically successful alternatives on every
layout. Selection is independent of difficulty and training seed. Other family
augmentation is unchanged. Demonstration contract version 5 invalidates old
bridge caches.

## Verification

Regression tests cover unreachable short and long attempts in both families,
complete route coverage at two seeds, physical duration normalization, and
legacy/NES fallback limits. Exact-checkpoint traces and production-path learning
artifacts are under `artifacts/block_smb/bridge_production_repair_20260912/`.

The bounded production-path probe uses the real builder for all 24 families,
12 layouts per family, fresh shared weights at production seed 20260908,
the production frozen perception, 10,000 bootstrap updates at numeric learning
rate 0.0005, and two cycles of 72 on-policy episodes plus 1,000 rehearsal
updates. It evaluates the original 60 bridge validation layouts and 66 unseen
layouts from the other families before and after the updates. This checks the
production learning paths at bounded volume; the full run still establishes
performance at the complete training scale.

Results: all 202 focused and related regression tests passed. The original
checkpoint replay confirms the erroneous 16-frame target is absent and the
missing initiation penalty is present.

| Complete-cycle evaluation | Mount easy / medium / hard | Dismount easy / medium / hard | Other families |
|---|---|---|---|
| After bootstrap | 100% / 100% / 100% | 60% / 80% / 20% | 66/66 |
| After cycle 1 rehearsal | 100% / 100% / 100% | 80% / 90% / 100% | 66/66 |
| After cycle 2 rehearsal | 100% / 90% / 100% | 80% / 90% / 100% | 66/66 |

The two originally failing bins passed 10/10 after both complete cycles.
The final bridge total was 56/60; one medium mount, two easy dismounts, and
one medium dismount still failed in this small-data probe. Intermediate
on-policy updates reduced mount performance before rehearsal restored it.
These results verify the repaired learning paths and recovery of the affected
bins, not universal mastery or retention at the full 180-layout training scale.
The restart therefore builds fresh version-5 demonstrations and keeps the
full-volume evaluation gates active.

The final model also passed 111/120 additional unseen bridge layouts
(seeds 202 and 303): easy mount 19/20 and hard dismount 20/20. Remaining
unseen failures were one easy mount, five easy dismounts, and three medium
dismounts. These limits are retained in the diagnostic artifacts.
