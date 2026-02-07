
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

The purpose of this project is to create a continuously learning agent with an architecture similar to the human brain, composed of five distinct "lobes" that interact to perceive, plan, and act.

### 1. Occipital Lobe (Vision & Perception)
*   **Architecture:** Convolutional Autoencoder.
*   **Function:** Processes raw pixel input from the game.
*   **Output:** Reconstructs the image (self-supervised learning) and splits the latent space into two distinct vectors:
    *   **"What":** Semantic object identity features.
    *   **"Where":** Spatial location features.

### 2. Temporal Lobe (Sequence & Memory)
*   **Architecture:** Recurrent Neural Network (GRU) with embeddings.
*   **Function:** Maintains the short-term memory of events and context.
*   **Input:** "What" vectors from the Occipital lobe and feedback from the Parietal lobe.
*   **Output:** Generates semantic text descriptions of current events (e.g., "Mario jumps") and a hidden state vector representing temporal context.

### 3. Parietal Lobe (Spatial Attention & Objectives)
*   **Architecture:** Multi-modal Fusion Network with Deconvolutional Decoder.
*   **Function:** Acts as the "bridge" between perception, memory, and action. Identifies short-term spatial objectives.
*   **Input:** "Where" vectors (Occipital), Temporal hidden state, and Frontal goals.
*   **Output:** A high-dimensional latent vector for action selection and a 2D "Objective Map" (saliency map) indicating where Mario should go.

### 4. Frontal Lobe (Planning & Strategy)
*   **Architecture:** High-level Planner / RNN.
*   **Function:** Determines long-term goals based on the current situation.
*   **Input:** Parietal latent vector.
*   **Output:** High-level textual goals (e.g., "Reach the castle") and a state vector that influences the Parietal lobe's attention.

### 5. Motor Lobe (Action Execution)
*   **Architecture:** Policy Network (MLP).
*   **Function:** Translates the brain's intent into precise motor commands.
*   **Input:** Parietal latent vector.
*   **Output:** Discrete game actions (NES controller button presses) trained via Reinforcement Learning (Policy Gradient).

---

# Next Steps and Future Improvements

*   **Advanced RL Algorithms:** Transition from basic Policy Gradient/REINFORCE to more stable algorithms like PPO (Proximal Policy Optimization) or SAC (Soft Actor-Critic) for the Motor lobe.
*   **Transformer Integration:** Replace GRUs in Temporal and Frontal lobes with Transformer blocks to handle longer context windows and more complex reasoning.
*   **Hippocampal Memory:** Implement an external memory bank (Vector Database) to store and retrieve successful strategies from past gameplay sessions.
*   **Curriculum Learning:** Design a training curriculum that starts with simple movement tasks before introducing enemies and complex platforming.
*   **Multi-Modal Reinforcement:** Utilize the generated text descriptions (Temporal) and goals (Frontal) as auxiliary rewards or inputs for a more robust Critic network.
*   **Real-Time Brain Visualization:** Create a dashboard to visualize the "thoughts" of the agent in real-time—showing the reconstructed view, attention maps, and generated text stream alongside the gameplay.


