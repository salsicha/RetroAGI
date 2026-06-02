import numpy as np
import pygame
import math

class MarioScenarioEnv:
    """
    A lightweight, scriptable 2D platformer environment for AI training.
    Mimics the Gym API: reset(), step(action), render()
    """
    def __init__(self, width=256, height=240):
        self.width = width
        self.height = height
        
        # Physics constants
        self.gravity = 0.5
        self.max_fall_speed = 8.0
        self.move_speed = 3.0
        self.jump_power = -8.5
        
        # Actions
        # 0: NOOP, 1: RIGHT, 2: RIGHT+JUMP, 3: LEFT, 4: LEFT+JUMP, 5: JUMP
        self.action_space_n = 6
        
        # Rendering
        pygame.init()
        self.screen = pygame.Surface((self.width, self.height))
        
        # State
        self.mario = None
        self.platforms = []
        self.coins = []
        self.goal = None
        self.score = 0
        self.steps = 0
        self.max_steps = 500

    def reset(self, scenario=None):
        """
        Resets the environment. You can pass a custom 'scenario' dict:
        {
            'mario': [x, y],
            'platforms': [[x, y, w, h], ...],
            'coins': [[x, y, w, h], ...],
            'goal': [x, y, w, h] # Optional end-level goal
        }
        """
        self.steps = 0
        self.score = 0
        
        if scenario is None:
            # Default simple scenario: A floor, one platform, and a coin
            scenario = {
                'mario': [20, 180],
                'platforms': [
                    [0, 220, 256, 20],   # Floor
                    [120, 170, 40, 10],  # Platform
                ],
                'coins': [
                    [130, 140, 10, 10]   # Coin on top of platform
                ]
            }
            
        self.mario = {
            'x': scenario['mario'][0], 'y': scenario['mario'][1],
            'vx': 0.0, 'vy': 0.0,
            'w': 14, 'h': 16,
            'on_ground': False
        }
        
        self.platforms = [pygame.Rect(p[0], p[1], p[2], p[3]) for p in scenario.get('platforms', [])]
        # Store coins as dicts to track if they are collected
        self.coins = [{'rect': pygame.Rect(c[0], c[1], c[2], c[3]), 'collected': False} for c in scenario.get('coins', [])]
        self.goal = pygame.Rect(scenario['goal']) if 'goal' in scenario else None
        
        return self.render()

    def step(self, action):
        self.steps += 1
        reward = 0.0
        done = False
        info = {}

        # Parse Action
        move_x = 0
        jump = False
        
        if action in [1, 2]: move_x = 1
        if action in [3, 4]: move_x = -1
        if action in [2, 4, 5]: jump = True

        # 1. Apply horizontal movement
        self.mario['vx'] = move_x * self.move_speed
        
        # 2. Apply jumping
        if jump and self.mario['on_ground']:
            self.mario['vy'] = self.jump_power
            self.mario['on_ground'] = False

        # 3. Apply gravity
        self.mario['vy'] += self.gravity
        if self.mario['vy'] > self.max_fall_speed:
            self.mario['vy'] = self.max_fall_speed

        # 4. Resolve X collisions
        self.mario['x'] += self.mario['vx']
        mario_rect = pygame.Rect(self.mario['x'], self.mario['y'], self.mario['w'], self.mario['h'])
        
        for plat in self.platforms:
            if mario_rect.colliderect(plat):
                if self.mario['vx'] > 0: # Moving right
                    mario_rect.right = plat.left
                elif self.mario['vx'] < 0: # Moving left
                    mario_rect.left = plat.right
                self.mario['x'] = mario_rect.x
                self.mario['vx'] = 0

        # 5. Resolve Y collisions
        self.mario['y'] += self.mario['vy']
        mario_rect.y = self.mario['y']
        self.mario['on_ground'] = False
        
        for plat in self.platforms:
            if mario_rect.colliderect(plat):
                if self.mario['vy'] > 0: # Falling
                    mario_rect.bottom = plat.top
                    self.mario['on_ground'] = True
                elif self.mario['vy'] < 0: # Hitting head
                    mario_rect.top = plat.bottom
                self.mario['y'] = mario_rect.y
                self.mario['vy'] = 0

        # Prevent walking off screen boundaries
        if self.mario['x'] < 0: self.mario['x'] = 0
        if self.mario['x'] > self.width - self.mario['w']: self.mario['x'] = self.width - self.mario['w']
        
        # Check fall death
        if self.mario['y'] > self.height:
            done = True
            reward -= 10.0 # Penalty for falling in a pit

        # 6. Coin Collection
        for coin in self.coins:
            if not coin['collected'] and mario_rect.colliderect(coin['rect']):
                coin['collected'] = True
                reward += 10.0
                self.score += 10
                
        # Check Goal condition
        if self.goal and mario_rect.colliderect(self.goal):
            done = True
            reward += 50.0

        # Check timeout
        if self.steps >= self.max_steps:
            done = True

        # Small negative reward to encourage speed
        reward -= 0.01 

        obs = self.render()
        return obs, reward, done, info

    def render(self):
        """
        Draws the environment to a surface and returns a NumPy RGB array.
        """
        # Background (Sky Blue)
        self.screen.fill((107, 140, 255))
        
        # Draw Platforms (Brown)
        for plat in self.platforms:
            pygame.draw.rect(self.screen, (139, 69, 19), plat)
            
        # Draw Coins (Gold)
        for coin in self.coins:
            if not coin['collected']:
                pygame.draw.ellipse(self.screen, (255, 215, 0), coin['rect'])
                
        # Draw Goal (Green Flagpole base)
        if self.goal:
            pygame.draw.rect(self.screen, (0, 255, 0), self.goal)

        # Draw Mario (Red)
        mario_rect = pygame.Rect(self.mario['x'], self.mario['y'], self.mario['w'], self.mario['h'])
        pygame.draw.rect(self.screen, (255, 0, 0), mario_rect)
        
        # Extract RGB array. Pygame surface is (W, H, 3). We transpose to (H, W, 3).
        rgb_array = pygame.surfarray.array3d(self.screen)
        rgb_array = np.transpose(rgb_array, (1, 0, 2))
        return rgb_array


# --- Interactive Demo ---
if __name__ == "__main__":
    env = MarioScenarioEnv()
    
    # Scripted scenario: Parkour
    custom_scenario = {
        'mario': [20, 180],
        'platforms': [
            [0, 220, 60, 20],     # Start platform
            [90, 180, 40, 10],    # First jump
            [160, 140, 40, 10],   # Second jump
            [220, 100, 36, 140]   # End tower
        ],
        'coins': [
            [105, 150, 10, 10],
            [175, 110, 10, 10]
        ],
        'goal': [230, 80, 16, 20]
    }
    
    obs = env.reset(scenario=custom_scenario)
    
    # Setup a display just to watch it play out
    display = pygame.display.set_mode((env.width, env.height))
    pygame.display.set_caption("Mario AI Scenario Simulator")
    clock = pygame.time.Clock()
    
    running = True
    while running:
        # Check for user closing the window
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                
        # Random action agent for demo
        # (0: NOOP, 1: R, 2: R+J, 3: L, 4: L+J, 5: J)
        action = np.random.choice([0, 1, 1, 2, 2, 5])
        
        obs, reward, done, info = env.step(action)
        
        # Blit array to screen for visualizing
        surface = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        display.blit(surface, (0, 0))
        pygame.display.flip()
        clock.tick(30) # 30 FPS
        
        if done:
            print("Scenario finished! Score:", env.score)
            env.reset(scenario=custom_scenario)
            
    pygame.quit()