# Graph Report - RetroAGI  (2026-06-06)

## Corpus Check
- 23 files · ~26,279 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 237 nodes · 312 edges · 24 communities (17 shown, 7 thin omitted)
- Extraction: 99% EXTRACTED · 1% INFERRED · 0% AMBIGUOUS · INFERRED: 3 edges (avg confidence: 0.7)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `87ace491`
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

## God Nodes (most connected - your core abstractions)
1. `MarioScenarioEnv` - 17 edges
2. `SpriteLoader` - 17 edges
3. `GridGenerator` - 16 edges
4. `FrameGenerator` - 9 edges
5. `TrainingUtils` - 9 edges
6. `TestMarioScenarios` - 9 edges
7. `HierarchicalAdaptiveModel` - 7 edges
8. `SegmenInf` - 6 edges
9. `MarioDataset` - 6 edges
10. `WorldModel` - 6 edges

## Surprising Connections (you probably didn't know these)
- `TestMarioScenarios` --uses--> `MarioScenarioEnv`  [INFERRED]
  scripts/tests/test_scenarios.py → scripts/mario_scenario_env.py

## Import Cycles
- None detected.

## Communities (24 total, 7 thin omitted)

### Community 0 - "Community 0"
Cohesion: 0.09
Nodes (16): AdaptiveController, AgentWorldModelCritic, Critic, generate_hierarchical_data(), HierarchicalAdaptiveModel, PositionalEncoding, A three-level hierarchy:      Transformer A (Top) -> Transformer B (Mid) -> Adap, Generates three levels of data:     Seq A: Slow discrete sequence.     Seq B: Me (+8 more)

### Community 1 - "Community 1"
Cohesion: 0.11
Nodes (8): FrameGenerator, Generates a dataset of a given size., Class with different load functtions which return a dictionary with pairs, Super mario frames have a dimension of         256 x 240 x 3 (x,y,c)         The, Initializes the frame generator, This function loads sprites and textures that will be used in the image., Load sprites for mario, enemies and generates their ground truth., SpriteLoader

### Community 2 - "Community 2"
Cohesion: 0.10
Nodes (16): float, int, ndarray, MarioScenarioEnv, Set RNG seed for reproducible procedural generation., Clean up pygame resources., Reset and return (obs, info) — Gym v26 API.         Passing seed= here is equiva, Advance one frame.         Returns (obs, reward, terminated, truncated, info) — (+8 more)

### Community 3 - "Community 3"
Cohesion: 0.12
Nodes (4): SegmenInf, Configuration, MarioDataset, TrainingUtils

### Community 4 - "Community 4"
Cohesion: 0.14
Nodes (7): _BoxSpace, _DiscreteSpace, mario_scenario_env.py — Lightweight SMB-style platformer for AI training.  Gym v, Tests for the Mario scenario environments., Test suite for the different Mario level configurations., Helper method to load a scenario and validate its instantiation., TestMarioScenarios

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
Cohesion: 0.33
Nodes (5): Build, Diagram, RetroAGI, Training, Usage

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

## Knowledge Gaps
- **33 isolated node(s):** `version`, `configurations`, `allow`, `build.sh script`, `entrypoint.sh script` (+28 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **7 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `GridGenerator` connect `Community 5` to `Community 1`, `Community 3`?**
  _High betweenness centrality (0.029) - this node is a cross-community bridge._
- **Why does `SpriteLoader` connect `Community 1` to `Community 3`?**
  _High betweenness centrality (0.027) - this node is a cross-community bridge._
- **Why does `MarioScenarioEnv` connect `Community 2` to `Community 4`?**
  _High betweenness centrality (0.026) - this node is a cross-community bridge._
- **What connects `version`, `configurations`, `allow` to the rest of the system?**
  _75 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `Community 0` be split into smaller, more focused modules?**
  _Cohesion score 0.09274193548387097 - nodes in this community are weakly interconnected._
- **Should `Community 1` be split into smaller, more focused modules?**
  _Cohesion score 0.11264367816091954 - nodes in this community are weakly interconnected._
- **Should `Community 2` be split into smaller, more focused modules?**
  _Cohesion score 0.10052910052910052 - nodes in this community are weakly interconnected._