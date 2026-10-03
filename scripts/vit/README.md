# Super Mario Bros Vision Transformer Training

Both Super Mario Bros games read the screen through one vision model class,
`retroagi.core.vision.SceneVisionTransformer`. It gives every pixel of the
256x240 screen one of nine types, listed in
`retroagi.core.smb_pixel_types.PIXEL_TYPES`:

`background, mario, ground, brick, question_block, pipe, coin, enemy,
moving_platform`

Each game has its own trained weights for this one class. The model cuts the
screen into 16x16 squares, describes each square, and lets the squares inform
each other over a few rounds. Each square's final description gives scores for
its own 256 pixels and every type, and a small per-pixel step then corrects
those scores using the pixel's colour.

The agent never reads the model's internal values. What it reads is built
from the pixel types by shared rules: `objects_from_types` in
`retroagi/core/smb_scene_labels.py` turns each group of touching pixels of one
type into an object (Mario, a surface, a gap, a pipe, a coin, an enemy), and
`retroagi/core/smb_observer.py` turns those objects into the report the
decision layers read. Because both games use
the same types and the same rules, the two models cannot drift apart in
meaning.

## Trainers (this folder)

| Script | Game | Frames | True labels | Writes |
| --- | --- | --- | --- | --- |
| `train_block_vit.py` | Block SMB | Every Monte Carlo family on the train split, at every difficulty, played with teacher, perturbed, delayed and random routes, plus generated levels (`retroagi/stages/block_smb/vision_frames.py`) | `MarioScenarioEnv.render_labels()`: the simulator draws each frame's own shapes with their types instead of their colours, so the labels are exact | `data/block_vit/block_vit_scene.pth` and `data/block_vit/block_vit_scene.json` |
| `train_full_vit.py` | Full SMB | Real emulator frames from the training levels (`retroagi/stages/full_smb/vision_frames.py`, `TRAIN_LEVELS`), each played from its saved start by a random player that mostly runs right, jumps for random lengths, and is rewound a few seconds after each death | `retroagi/stages/full_smb/pixel_labels.py` (`label_frame`) reads each pixel's type from game memory. A frame is accepted only when the picture rebuilt from memory matches the emulator's picture at every visible pixel; any frame memory cannot fully explain is refused and never used | `data/full_vit/full_vit_scene.pth` and `data/full_vit/full_vit_scene.json` |

```bash
python scripts/vit/train_block_vit.py --epochs 40 --samples-per-epoch 40000
python scripts/vit/train_full_vit.py --epochs 40 --samples-per-epoch 40000
```

Both trainers use the same training loop (`retroagi/core/scene_vision.py`):
per-pixel cross-entropy with rare types (Mario, coins, enemies, question
blocks) weighted up. Worker processes play fresh episodes throughout training,
so frames rarely repeat. Progress is printed each epoch on held-out frames from
the training source: held-out train-split layouts for Block SMB, separate plays
of the training levels for Full SMB.

The Full SMB trainer needs the Super Mario Bros ROM imported into
stable-retro; see [docs/full-smb-content.md](../../docs/full-smb-content.md).

## Evaluators (`scripts/vision/`)

| Script | Frames measured |
| --- | --- |
| `evaluate_block_vision.py` | Every Monte Carlo family at every difficulty on the validation split, each layout played with its teacher route and with a perturbed teacher route. |
| `evaluate_full_vision.py` | The test levels `Level1-1` and `Level5-1` (`vision_frames.TEST_LEVELS`), which the Full SMB model is never trained on, each played several times. Frames memory cannot fully explain are counted by reason and not measured. |

```bash
python scripts/vision/evaluate_block_vision.py --checkpoint data/block_vit/block_vit_scene.pth
python scripts/vision/evaluate_full_vision.py --checkpoint data/full_vit/full_vit_scene.pth
```

Both evaluators take the same measurements
(`retroagi.core.scene_vision.evaluate_scene_vision`) and print the same table:

- pixels correct, and for each type the share of its true pixels the model
  found and the share of pixels given that type that truly are it;
- frames where Mario is found, out of the frames whose true labels show him;
- Mario position error in pixels, between the centres of Mario's pixels in
  the predicted and true labels;
- standing/air agreement: whether Mario stands by the predicted labels, against
  the game's own standing flag;
- enemies seen: the share of drawn enemies with at least one pixel labelled
  enemy.

Each evaluator writes its results beside the checkpoint as
`<checkpoint name>_evaluation.json`.

## Using a trained model

`retroagi.stages.block_smb.vision.load_block_vit_checkpoint` and
`retroagi.stages.full_smb.vision.load_full_vit_checkpoint` load a checkpoint,
check that it was saved for that game, and freeze it by default so policy
training cannot change it.
