import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from supervisor import Supervisor
from brain.occipital import OccipitalLobe
from brain.temporal import TemporalLobe
from brain.hippocampus import HippocampusLobe
from brain.prefrontal import PrefrontalLobe
from brain.motor import MotorLobe
import numpy as np

def main():
    # Initialize environment
    env = gym_super_mario_bros.make('SuperMarioBros-v0')
    env = JoypadSpace(env, [['NOOP'], ['right'], ['right', 'A'], ['right', 'B'], ['right', 'A', 'B'], ['A'], ['left']])
    
    # Initialize Lobes
    # Occipital handles perception and reconstruction
    occipital = OccipitalLobe()
    # Temporal handles action sequences and sprite tracking
    temporal = TemporalLobe()
    # Hippocampus handles spatial relationships
    hippocampus = HippocampusLobe()
    # Prefrontal handles high-level strategy (Speedrun vs Max Coins)
    prefrontal = PrefrontalLobe(planning_mode=0.5) # Default balanced
    # Motor handles action execution
    motor = MotorLobe()

    supervisor = Supervisor([occipital, temporal, hippocampus, prefrontal, motor])

    state = env.reset()
    done = False
    step_count = 0

    while not done:
        # 1. Perception (Occipital)
        latent_v, reconstruction = occipital.process(state)
        
        # 2. Sequence/Memory (Temporal)
        latent_t, sprite_pos_pred = temporal.process(latent_v)
        
        # 3. Spatial Mapping (Hippocampus)
        latent_h, spatial_map = hippocampus.process(latent_v, latent_t)
        
        # 4. Planning (Prefrontal)
        plan = prefrontal.process(latent_t, latent_h)
        
        # 5. Action (Motor)
        action = motor.decide(plan)
        
        # Step the environment
        next_state, reward, done, info = env.step(action)
        
        # 6. Supervision & Online Learning
        # The supervisor uses 'info' to provide prediction error signals
        supervisor.update(state, action, next_state, reward, info)
        
        state = next_state
        step_count += 1

        if step_count % 1000 == 0:
            supervisor.checkpoint()

    env.close()

if __name__ == "__main__":
    main()