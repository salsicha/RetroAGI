<a href="">
  <img src="https://media.githubusercontent.com/media/salsicha/RetroAGI/main/mario.gif"
    height="80" align="right" alt="" />
</a><br>


# RetroAGI
General purpose machine learning agent for retro games 


## Build
```bash
./build.sh
```


## Usage
```bash
./run.sh
```


### Example scripts
python3 /stable-retro-scripts/model_trainer.py --env=Airstriker-Genesis --num_env=8 --num_timesteps=100_000_000 --play  
python3 /examples/ppo.py --game='Airstriker-Genesis'  
gym_super_mario_bros -e SuperMarioBros-v0 -m human  


### References
https://github.com/Farama-Foundation/stable-retro?tab=readme-ov-file  
https://huggingface.co/transformers/v3.0.2/training.html  
https://github.com/vpulab/Semantic-Segmentation-Boost-Reinforcement-Learning  
https://myrient.erista.me/files/No-Intro/Nintendo%20-%20Nintendo%20Entertainment%20System%20(Headered)/  
https://link.springer.com/article/10.1007/s11042-022-13695-1
https://arxiv.org/pdf/2309.01140


### Note on CUDA driver
Occasionally the CUDA driver throws an unknown error, this is sometimes a fix:
```bash
sudo rmmod nvidia_uvm
sudo modprobe nvidia_uvm
```

