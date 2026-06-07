# Graph Report - RetroAGI  (2026-06-07)

## Corpus Check
- 37 files · ~26,972 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 371 nodes · 507 edges · 32 communities (21 shown, 11 thin omitted)
- Extraction: 98% EXTRACTED · 2% INFERRED · 0% AMBIGUOUS · INFERRED: 10 edges (avg confidence: 0.56)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `fa0d91d2`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- [[_COMMUNITY_Community 0|Community 0]]
- [[_COMMUNITY_Community 1|Community 1]]
- [[_COMMUNITY_Community 2|Community 2]]
- [[_COMMUNITY_Community 3|Community 3]]
- [[_COMMUNITY_Community 4|Community 4]]
- [[_COMMUNITY_Community 5|Community 5]]
- [[_COMMUNITY_Community 6|Community 6]]
- [[_COMMUNITY_Community 7|Community 7]]
- [[_COMMUNITY_Community 8|Community 8]]
- [[_COMMUNITY_Community 9|Community 9]]
- [[_COMMUNITY_Community 10|Community 10]]
- [[_COMMUNITY_Community 11|Community 11]]
- [[_COMMUNITY_Community 12|Community 12]]
- [[_COMMUNITY_Community 13|Community 13]]
- [[_COMMUNITY_Community 14|Community 14]]
- [[_COMMUNITY_Community 15|Community 15]]
- [[_COMMUNITY_Community 16|Community 16]]
- [[_COMMUNITY_Community 17|Community 17]]
- [[_COMMUNITY_Community 18|Community 18]]
- [[_COMMUNITY_Community 19|Community 19]]
- [[_COMMUNITY_Community 20|Community 20]]
- [[_COMMUNITY_Community 21|Community 21]]
- [[_COMMUNITY_Community 24|Community 24]]
- [[_COMMUNITY_Community 25|Community 25]]
- [[_COMMUNITY_Community 26|Community 26]]
- [[_COMMUNITY_Community 27|Community 27]]
- [[_COMMUNITY_Community 28|Community 28]]
- [[_COMMUNITY_Community 29|Community 29]]
- [[_COMMUNITY_Community 30|Community 30]]
- [[_COMMUNITY_Community 31|Community 31]]

## God Nodes (most connected - your core abstractions)
1. `MarioScenarioEnv` - 25 edges
2. `SpriteLoader` - 17 edges
3. `MarioScenarioEnv` - 17 edges
4. `GridGenerator` - 16 edges
5. `TestMarioScenarios` - 10 edges
6. `FrameGenerator` - 9 edges
7. `TrainingUtils` - 9 edges
8. `HierarchicalAdaptiveModel` - 8 edges
9. `BlockSMBStage` - 8 edges
10. `main()` - 8 edges

## Surprising Connections (you probably didn't know these)
- `MarioScenarioEnv` --uses--> `MarioScenarioEnv`  [INFERRED]
  retroagi/stages/block_smb/adapter.py → retroagi/stages/block_smb/env.py
- `ndarray` --uses--> `MarioScenarioEnv`  [INFERRED]
  retroagi/stages/block_smb/adapter.py → retroagi/stages/block_smb/env.py
- `str` --uses--> `MarioScenarioEnv`  [INFERRED]
  retroagi/stages/block_smb/adapter.py → retroagi/stages/block_smb/env.py
- `Any` --uses--> `MarioScenarioEnv`  [INFERRED]
  retroagi/stages/block_smb/adapter.py → retroagi/stages/block_smb/env.py
- `StageBatch` --uses--> `MarioScenarioEnv`  [INFERRED]
  retroagi/stages/block_smb/adapter.py → retroagi/stages/block_smb/env.py

## Import Cycles
- None detected.

## Communities (32 total, 11 thin omitted)

### Community 0 - "Community 0"
Cohesion: 0.09
Nodes (17): AdaptiveController, AgentWorldModelCritic, Critic, generate_hierarchical_data(), HierarchicalAdaptiveModel, PositionalEncoding, Compatibility wrapper for the synthetic 1D curriculum stage., A three-level hierarchy:      Transformer A (Top) -> Transformer B (Mid) -> Adap (+9 more)

### Community 1 - "Community 1"
Cohesion: 0.10
Nodes (9): Configuration, FrameGenerator, Generates a dataset of a given size., Class with different load functtions which return a dictionary with pairs, Super mario frames have a dimension of         256 x 240 x 3 (x,y,c)         The, Initializes the frame generator, This function loads sprites and textures that will be used in the image., Load sprites for mario, enemies and generates their ground truth. (+1 more)

### Community 2 - "Community 2"
Cohesion: 0.06
Nodes (25): float, int, ndarray, _BoxSpace, _DiscreteSpace, MarioScenarioEnv, Compatibility wrapper for the block-SMB scenario environment., Set RNG seed for reproducible procedural generation. (+17 more)

### Community 3 - "Community 3"
Cohesion: 0.13
Nodes (3): SegmenInf, MarioDataset, TrainingUtils

### Community 4 - "Community 4"
Cohesion: 0.06
Nodes (32): BlockSMBStage, Adapter from the block-SMB pygame environment to the shared training contract., Stage adapter for scriptable pygame scenarios., Convert symbolic block-SMB state into the common A/B/C tensor layout.          T, _BoxSpace, _DiscreteSpace, main(), MarioScenarioEnv (+24 more)

### Community 6 - "Community 6"
Cohesion: 0.23
Nodes (9): build_scene(), load_sprites(), main(), make_split(), generate_dataset.py =================== Procedurally compose Super Mario Bros sc, Holds the RGB image and the per-pixel class-id canvas., Alpha-composite `sprite` at (x,y); write `cls` where sprite is opaque., Reduce per-pixel labels to a GH x GW patch-class grid. (+1 more)

### Community 7 - "Community 7"
Cohesion: 0.33
Nodes (8): autocrop_sky(), crop(), load(), main(), extract_sprites.py ================== Slice accurate Super Mario Bros sprites ou, Crop (x,y,w,h) and keep the existing alpha channel., Replace sky-blue with transparency and trim to the content bbox.     Used for ti, save()

### Community 8 - "Community 8"
Cohesion: 0.33
Nodes (6): evaluate(), load_split(), main(), train_vit.py ============ A Vision Transformer that performs PATCH-LEVEL SEMANTI, save_predictions(), ViTSegmenter

### Community 9 - "Community 9"
Cohesion: 0.29
Nodes (6): 1. `extract_sprites.py`, 2. `generate_dataset.py`, 3. `train_vit.py`, Pipeline, Results (1000 held-out scenes, 30 epochs, ~17 min on Apple MPS), Super Mario Bros Vision Transformer

### Community 10 - "Community 10"
Cohesion: 0.29
Nodes (6): Build, Diagram, Project Layout, RetroAGI, Training, Usage

### Community 11 - "Community 11"
Cohesion: 0.33
Nodes (5): mask_to_rgb(), Semantic Segmentation module for MarioScenarioEnv. Segments frames perfectly by, Takes an RGB array (H, W, 3) and returns a 2D class mask (H, W)., Converts a 2D class mask (H, W) back into a colored RGB array (H, W, 3)      for, segment_frame()

### Community 12 - "Community 12"
Cohesion: 0.40
Nodes (4): coins, goal, mario, platforms

### Community 13 - "Community 13"
Cohesion: 0.40
Nodes (4): coins, goal, mario, platforms

### Community 14 - "Community 14"
Cohesion: 0.40
Nodes (4): coins, goal, mario, platforms

### Community 15 - "Community 15"
Cohesion: 0.40
Nodes (4): coins, goal, mario, platforms

### Community 18 - "Community 18"
Cohesion: 0.33
Nodes (3): main(), Entry point for the RetroAGI agent. Sets up the Super Mario Bros retro environme, Compatibility wrapper for the full-SMB emulator runner.

### Community 24 - "Community 24"
Cohesion: 0.11
Nodes (14): Shared interfaces and model components for all training stages., AdaptiveController, AgentWorldModelCritic, Critic, HierarchicalAdaptiveModel, PositionalEncoding, Reusable hierarchical actor, world-model, and critic components., Predicts the next state using episodic LSTM memory plus multi-frequency     sinu (+6 more)

### Community 25 - "Community 25"
Cohesion: 0.11
Nodes (18): bool, AgentStep, Common data contracts shared by every curriculum stage., Describes the resolution and timing contract for a stage., Canonical tensors used by the hierarchical training loop., Outputs from one actor/world-model/critic refinement step., Minimal interface implemented by synthetic, block-SMB, and full-SMB stages., Start a new episode and return a stage-native observation. (+10 more)

### Community 31 - "Community 31"
Cohesion: 0.43
Nodes (5): Stage 1: one-dimensional procedural validation., generate_hierarchical_data(), Generates three levels of data:     Seq A: Slow discrete sequence.     Seq B: Me, Generates three levels of data:     Seq A: Slow discrete sequence.     Seq B: Me, train_and_evaluate()

## Knowledge Gaps
- **41 isolated node(s):** `version`, `configurations`, `allow`, `BeforeTool`, `build.sh script` (+36 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **11 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `MarioScenarioEnv` connect `Community 4` to `Community 2`?**
  _High betweenness centrality (0.167) - this node is a cross-community bridge._
- **Why does `StageSpec` connect `Community 25` to `Community 24`, `Community 4`, `Community 31`?**
  _High betweenness centrality (0.107) - this node is a cross-community bridge._
- **Why does `train_and_evaluate()` connect `Community 31` to `Community 0`, `Community 24`?**
  _High betweenness centrality (0.080) - this node is a cross-community bridge._
- **Are the 7 inferred relationships involving `MarioScenarioEnv` (e.g. with `BlockSMBStage` and `MarioScenarioEnv`) actually correct?**
  _`MarioScenarioEnv` has 7 INFERRED edges - model-reasoned connections that need verification._
- **What connects `version`, `configurations`, `allow` to the rest of the system?**
  _126 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `Community 0` be split into smaller, more focused modules?**
  _Cohesion score 0.08901515151515152 - nodes in this community are weakly interconnected._
- **Should `Community 1` be split into smaller, more focused modules?**
  _Cohesion score 0.10037878787878787 - nodes in this community are weakly interconnected._