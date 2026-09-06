# Findings by family

All 21 families now pass the documented learning gate across three training
seeds. The shared policy also passes two separate test seeds. Counts and run
configurations are recorded in [the learning matrix](block-smb-learning-audit.md). A passing physics oracle
is not counted as a learned success. A failed short run is not proof that a
neural network cannot represent the skill.

| Family | Finding and repair |
| --- | --- |
| flat_run | No new geometry defect found. Correct policy credit and balanced action-decision supervision let it learn the walking policy. |
| single_gap | Early demonstrations used takeoffs with little margin. Robust routes and per-frame walking improve timing. Two remaining hard failures held for eight frames where nine were required; duration learning now favors safe interiors. |
| stair_climb | Success depends on successive local mounts. Numeric geometry reaches both action selection and duration prediction; local goals and primitive outcomes remain consistent. |
| platform_chain | Small errors compound across landings. Canonical routes alone omit recovery states; successful alternate routes and broader training coverage are necessary. A validation pass did not always survive the larger test set. |
| moving_bridge | The walking approach, boarding, riding, and exit are separate phases. Motion observations and consistent committed-exit goals repair weak observations and contradictory teaching. |
| enemy_hop | Ground decisions and stomp-bounce continuations must be distinguished. Held/recovery frames no longer receive fresh actor labels. |
| enemy_patrol | Position alone is a weak motion cue. Motion observations expose velocity and patrol limits. A locally cleared first enemy can leave an impossible second interception; labels now verify bounce survival and reject nearby next-enemy landings from which no jump is viable. |
| enemy_gap | Landing state and the next takeoff interact. Broad route coverage, correct walk commitments, and primitive-outcome supervision address the mismatch between teacher routes and policy visits. |
| enemy_stomp | Clearing the enemy without stomping is a failure. Duration supervision requires stomp credit, and the bounce/finish handoff uses the same goals in demonstrations and live control. A traced hard failure jumped later than the canonical route and selected an eight-frame hold where the actual takeoff allowed only one to four. Alternate routes now vary takeoff position as well as hold duration. Rare initial walking states were underweighted relative to takeoff examples. Priority sampling preserves action balance while emphasizing these errors, and identical-state successful alternatives share an accepted action set. Missed interceptions remain explicit failures. |
| retreat_recovery | The actor needs the signed goal and examples of leftward decisions. Direction-specific decisions receive balanced supervision; completion remains judged by the environment. |
| wait_timing | Wait continuation frames were mislabeled as new NOOP decisions, and exit frames could receive the wait goal. The demonstration contract now matches committed waits and the latched exit. |
| chained_obstacles | Multi-step completion amplifies small primitive errors. A pipe over continuous lower floor was incorrectly called a gap to the next pipe; local goals now distinguish descent from a pit. Complete-route learning and completion tests remain required. |
| chained_enemy_gauntlet | Combines moving interception, bounces, and further obstacles. Motion, bounce labels, alternative routes, and balanced rehearsal all matter. |
| full_smb_opening_proxy | Short validation sets can hide failures on another opening layout. Larger validation sets and a minimum training budget are being used for failed runs; a passed local skill is not a completed opening. |
| mixed_section | The same compound-error problem appears across gaps and enemies. Larger validation, broader route coverage, and correct walk/jump commitments are required; test failures are not promoted to passes. |
| tall_pipe_jump | Sparse takeoff decisions, weak numeric duration conditioning, incorrect policy credit, and forgetting were shared training problems. The actor, motor prediction, and likelihood now describe the same action; landing and finish remain separate objectives. |
| pipe_mount | Precise duration must depend on continuous geometry. Direct numeric duration conditioning and physics-verified landing labels address the observed constant-hold collapse. |
| pit_leap | The teaching goal previously changed during flight. It now stays fixed through the committed arc, and permitted holds must achieve the actual landing objective. |
| stomp_mount | Avoiding the enemy was incorrectly acceptable to one duration-label path. Labels now require an actual stomp; explicit motion observations improve the moving tiers. |
| platform_hop | The family described a landing task but credited airborne goal contact. It now requires support on the target platform and a single attempt, so the oracle and duration labels agree. |
| bridge_wait | A short wait timer could incorrectly end the opening phase before boarding was safe. Readiness events or actual departure now control that transition. The other phase and duration-contract corrections apply as wait-timing. Qualification additionally removes its supplied opening A action, testing whether the policy chooses it. |

Shared-model retention is a separate requirement. Independent family models do
not establish that a mixed-training policy will retain all of these behaviors.
The current experiments use demonstration-assisted learning with the real frozen
vision model, not oracle-controlled evaluation or a claim that pure RL from
random initialization has been qualified.
