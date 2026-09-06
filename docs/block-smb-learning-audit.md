# Block SMB learning audit

Pass requires two successive autonomous validations, then at least 9/10 successes at each difficulty on a separate test split. Each family needs all three training seeds. One shared policy must also pass; missing evidence is incomplete.

Demonstration runs use supervised learning. Evaluation supplies no demonstration actions or family-provided A actions. Percentages are observed level-completion rates, not confidence bounds.

| Family | Seed 101 | Seed 202 | Seed 303 | Qualified |
| --- | --- | --- | --- | --- |
| flat_run | 100%/100%/100% | 100%/100%/100% | 100%/100%/100% | Yes |
| single_gap | 100%/100%/100% | 100%/100%/100% | 100%/100%/100% | Yes |
| stair_climb | 100%/100%/100% | 100%/100%/100% | 100%/100%/100% | Yes |
| platform_chain | 100%/100%/100% | 100%/100%/100% | 100%/100%/100% | Yes |
| moving_bridge | 100%/100%/90% | 100%/100%/100% | 100%/100%/100% | Yes |
| enemy_hop | 100%/100%/100% | 100%/100%/100% | 100%/100%/100% | Yes |
| enemy_patrol | 100%/100%/90% | 100%/100%/90% | 100%/100%/100% | Yes |
| enemy_gap | 100%/100%/90% | 100%/100%/100% | 100%/100%/100% | Yes |
| enemy_stomp | 100%/100%/100% | 100%/100%/100% | 100%/100%/100% | Yes |
| retreat_recovery | 100%/100%/100% | 100%/100%/100% | 100%/100%/100% | Yes |
| wait_timing | 100%/100%/100% | 100%/100%/100% | 100%/100%/100% | Yes |
| chained_obstacles | 90%/100%/100% | 100%/100%/100% | 100%/100%/100% | Yes |
| chained_enemy_gauntlet | 100%/100%/100% | 100%/100%/100% | 100%/100%/100% | Yes |
| full_smb_opening_proxy | 90%/100%/100% | 100%/100%/100% | 100%/100%/100% | Yes |
| mixed_section | 100%/100%/100% | 100%/100%/100% | 100%/100%/100% | Yes |
| tall_pipe_jump | 100%/100%/100% | 100%/100%/100% | 100%/100%/100% | Yes |
| pipe_mount | 100%/100%/100% | 100%/100%/100% | 100%/100%/100% | Yes |
| pit_leap | 100%/100%/100% | 100%/100%/100% | 100%/100%/100% | Yes |
| stomp_mount | 100%/100%/90% | 100%/100%/90% | 100%/100%/90% | Yes |
| platform_hop | 100%/100%/100% | 100%/100%/100% | 100%/100%/100% | Yes |
| bridge_wait | 100%/100%/100% | 100%/100%/100% | 100%/100%/100% | Yes |

Rates are easy / medium / hard. Matching current-code rechecks replace historical test rates when available. The JSON companion records both results and run configurations. Isolated families may use different teaching and control recipes; they do not establish that one common recipe is reliable across seeds.

Shared policy qualified: **True**. Full audit qualified: **True**.

Additional frozen-policy holdout seed: **937661**.

| Family | Shared policy: easy / medium / hard |
| --- | --- |
| flat_run | 100%/100%/100% |
| single_gap | 100%/100%/100% |
| stair_climb | 100%/100%/100% |
| platform_chain | 100%/90%/100% |
| moving_bridge | 90%/100%/100% |
| enemy_hop | 100%/100%/100% |
| enemy_patrol | 90%/100%/100% |
| enemy_gap | 100%/100%/100% |
| enemy_stomp | 100%/100%/100% |
| retreat_recovery | 100%/100%/100% |
| wait_timing | 100%/100%/100% |
| chained_obstacles | 100%/100%/100% |
| chained_enemy_gauntlet | 100%/100%/100% |
| full_smb_opening_proxy | 100%/100%/100% |
| mixed_section | 100%/100%/100% |
| tall_pipe_jump | 100%/100%/100% |
| pipe_mount | 100%/100%/100% |
| pit_leap | 100%/100%/100% |
| stomp_mount | 100%/100%/100% |
| platform_hop | 100%/100%/100% |
| bridge_wait | 100%/100%/100% |
