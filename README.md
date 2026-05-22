
<a href="">
  <img src="https://media.githubusercontent.com/media/salsicha/RetroAGI/main/mario.gif"
    height="80" align="right" alt="" />
</a><br>


# RetroAGI
General purpose machine learning agent for retro games  


## Diagram
![The Brain](https://github.com/salsicha/RetroAGI/blob/main/diagrams/brain.jpg)


## Build
```bash
./build.sh
```


## Usage
1. Start the container environment:
   ```bash
   ./run.sh
   ```
2. Once inside the environment (via Jupyter terminal or shell), run the agent:
   ```bash
   python scripts/run.py
   ```

## Training

RetroAGI uses an **online learning** approach based on Predictive Coding. The models do not rely on a static, offline dataset. Instead, they learn dynamically as the agent plays the game. 

To "train" the models, simply run the agent as described in the Usage section. 

During gameplay:
- The `Supervisor` module continuously provides prediction error signals (rewards/penalties) to each model (Occipital, Temporal, Hippocampus, Prefrontal, and Motor).
- The models immediately perform weight updates via their `.learn()` methods to minimize surprise.
- Model weights are automatically serialized and saved to `data/checkpoints/` every 1,000 steps.
