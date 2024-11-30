

pip3 install stable-retro

sudo apt-get update
sudo apt-get install python3 python3-pip git zlib1g-dev libopenmpi-dev ffmpeg


pip3 install git+https://github.com/Farama-Foundation/stable-retro.git
pip3 install stable_baselines3[extra]

pip3 install transformers


cd retro/examples
python3 ppo.py --game='Airstriker-Genesis'



# import roms
cd roms
python3 -m retro.import .



