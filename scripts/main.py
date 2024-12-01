
from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
env = gym_super_mario_bros.make('SuperMarioBros-v0')
env = JoypadSpace(env, SIMPLE_MOVEMENT)

done = True
for step in range(5000):
    if done:
        state = env.reset()
    state, reward, done, info = env.step(env.action_space.sample())
    env.render()

env.close()

###########################

import retro
def main():
    env = retro.make(game='SuperMarioBros-Nes')
    obs = env.reset()
    while True:
        obs, rew, done, info = env.step(env.action_space.sample())
        env.render()
        if done:
            obs = env.reset()
    env.close()

if __name__ == "__main__":
    main()

###########################




# get constraints


# create constraints


# start playthrough


# predict next token


# search


# end play through


# cluster


# build model


# record tokens


# train

