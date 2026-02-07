
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
   python3 src/main.py
   ```

## Synthetic Data & Offline Training
To pre-train the models without running the game loop in real-time:

1. **Generate Synthetic Data:**
   Run the generation pipeline. This plays the game using a heuristic agent and generates labeled data (images, text descriptions, objective maps).
   ```bash
   python3 scripts/generate_data.py
   ```
   Data will be saved to `data/synthetic/`.

2. **Train Offline:**
   Train all lobes using the generated dataset.
   ```bash
   python3 scripts/train.py
   ```


# Architecture

The purpose of this project is to create a continuously learning agent with an architecture similiar to the human brain. 

- **Occipital Lobe:** Processes visual input (Mario game), reconstructs the image (autoencoder), and extracts "what" and "where" latent parameters.
- **Temporal Lobe:** Receives "what" information and Parietal input to generate semantic descriptions of events and update its internal state.
- **Parietal Lobe:** Integrates "where" information, Temporal state, and Frontal state to identify short-term objectives and form a spatial/action-oriented latent representation.
- **Frontal Lobe:** Uses Parietal input to determine long-term goals.
- **Motor Lobe:** Translates the Parietal latent representation into specific game actions (key presses).

The system implements a continuous learning loop where the Occipital lobe optimizes for visual reconstruction and the Motor lobe optimizes using reinforcement learning (Policy Gradient) based on game rewards.


