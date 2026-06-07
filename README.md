
<a href="">
  <img src="https://media.githubusercontent.com/media/salsicha/RetroAGI/main/mario.gif"
    height="80" align="right" alt="" />
</a><br>


# RetroAGI
General purpose machine learning agent for retro games  

RetroAGI is organized as a three-stage curriculum for training an architecture
to play Super Mario Bros:

1. **Synthetic 1D** validates the hierarchy on procedural sequence data.
2. **Block SMB** uses a scriptable pygame platformer with low-resolution,
   scenario-driven tasks.
3. **Full SMB** connects the same architecture to the full emulator.

The stage code is separated, but all stages share the same core contract:

```text
observation -> hierarchical actor -> action
observation + action -> world model prediction
prediction -> critic -> actor refinement
```

Shared actor/world-model/critic components live in `retroagi/core`. Stage
adapters live in `retroagi/stages/*` and convert stage-native observations into
the common A/B/C timescale tensors.

## Project Layout

```text
retroagi/
  core/
    interfaces.py      # StageSpec, StageBatch, shared adapter protocol
    models.py          # reusable actor, world model, critic, controller
  stages/
    synthetic_1d/      # procedural one-dimensional validation
    block_smb/         # pygame SMB-like scenarios and adapter
    full_smb/          # stable-retro / emulator runner
scripts/               # compatibility wrappers and older experiments
```

## Diagram
![The Brain](https://github.com/salsicha/RetroAGI/blob/main/docs/architecture.html)


## Build
```bash
./build.sh
```


## Usage
1. Start the container environment:
   ```bash
   ./run.sh
   ```
2. Run a curriculum stage:
   ```bash
   python -m retroagi.stages.synthetic_1d.train
   python -m retroagi.stages.full_smb.run --steps 500
   ```

Legacy wrappers still work:
   ```bash
   python scripts/simple_transformer.py
   python scripts/run.py
   ```

## Training
Stage 1 currently trains the shared hierarchical actor/world-model/critic stack
on synthetic data. Stage 2 has the scriptable environment and adapter in place;
training loops can now reuse `retroagi.core.models.AgentWorldModelCritic` and
consume `BlockSMBStage.encode_observation(...)`. Stage 3 keeps the full emulator
runner isolated behind `retroagi/stages/full_smb`.
