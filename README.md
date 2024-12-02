<a href="">
  <img src="https://media.githubusercontent.com/media/salsicha/RetroAGI/main/mario.gif"
    height="80" align="right" alt="" />
</a><br>

# RetroAGI


### TODO
Train convolution model to sematically segment frames
Create behavior tree for apply user generated constraints
Create progressive perceptive field and occupancy detection for agent
Create tokens and transformer for actions
Implement DQN for filtering action tokens
Create method for clustering sequences
Build system for training new DNNs on sequence clusters
Build system for generating new tokens and retraining transformer


### Examples
python3 /stable-retro-scripts/model_trainer.py --env=Airstriker-Genesis --num_env=8 --num_timesteps=100_000_000 --play
python3 /examples/ppo.py --game='Airstriker-Genesis'
gym_super_mario_bros -e SuperMarioBros-v0 -m human

### References
https://github.com/Farama-Foundation/stable-retro?tab=readme-ov-file
https://huggingface.co/transformers/v3.0.2/training.html
https://github.com/vpulab/Semantic-Segmentation-Boost-Reinforcement-Learning
https://myrient.erista.me/files/No-Intro/Nintendo%20-%20Nintendo%20Entertainment%20System%20(Headered)/
