import numpy as np
import pygame
import math
import json
import os

class MarioScenarioEnv:
    """
    A lightweight, scriptable 2D platformer environment for AI training.
    Mimics the Gym API: reset(), step(action), render()

    All game objects (Mario, platforms, coins, enemies) use WORLD coordinates.
    The viewport is `width` x `height` pixels; `camera_x` tracks the left edge
    of the viewport in world space.  The camera scrolls right following Mario
    and never scrolls back left (classic SMB behaviour).
    """
    def __init__(self, width=256, height=240, world_width=None):
        self.width = width
        self.height = height
        # Default world width equals viewport; overridden per-scenario.
        self.world_width = world_width if world_width is not None else width

        # Physics constants
        self.gravity = 0.5
        self.max_fall_speed = 8.0
        self.jump_power = -8.5

        # Horizontal momentum (SMB-style)
        self.max_walk_speed = 3.0   # top speed
        self.accel = 0.3            # ground acceleration per frame
        self.decel = 0.2            # friction when no input
        self.skid_decel = 0.5       # extra-fast decel when reversing direction

        # Actions
        # 0: NOOP, 1: RIGHT, 2: RIGHT+JUMP, 3: LEFT, 4: LEFT+JUMP, 5: JUMP
        self.action_space_n = 6

        # Rendering — off-screen surface, same size as viewport
        pygame.init()
        self.screen = pygame.Surface((self.width, self.height))

        # State
        self.mario = None
        self.platforms = []
        self.coins = []
        self.enemies = []
        self.goal = None
        self.camera_x = 0.0
        self.score = 0
        self.steps = 0
        self.max_steps = 1000

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------
    def reset(self, scenario=None):
        """
        Resets the environment. You can pass a custom 'scenario' dict:
        {
            'world_width': int,                          # optional, defaults to self.width
            'mario': [x, y],
            'platforms': [[x, y, w, h], ...],
            'coins': [[x, y, w, h], ...],
            'enemies': [[x, y, patrol_min_x, patrol_max_x], ...],
            'goal': [x, y, w, h]                        # optional end-level goal
        }
        All coordinates are in world space.
        """
        self.steps = 0
        self.score = 0
        self.camera_x = 0.0

        if scenario is None:
            scenario = {
                'mario': [20, 180],
                'platforms': [
                    [0, 220, 256, 20],
                    [120, 170, 40, 10],
                ],
                'coins': [
                    [130, 140, 10, 10]
                ]
            }

        # Allow scenario to override world width
        self.world_width = scenario.get('world_width', self.width)

        self.mario = {
            'x': float(scenario['mario'][0]),
            'y': float(scenario['mario'][1]),
            'vx': 0.0, 'vy': 0.0,
            'w': 14, 'h': 16,
            'on_ground': False,
            'facing': 1,       # 1 = right, -1 = left
            'skidding': False,
        }

        self.platforms = [pygame.Rect(p[0], p[1], p[2], p[3]) for p in scenario.get('platforms', [])]
        self.coins = [
            {'rect': pygame.Rect(c[0], c[1], c[2], c[3]), 'collected': False}
            for c in scenario.get('coins', [])
        ]
        self.enemies = [
            {
                'x': float(e[0]), 'y': float(e[1]),
                'w': 12, 'h': 14,
                'speed': 1.5,
                'direction': 1,
                'patrol_min': float(e[2]),
                'patrol_max': float(e[3]),
            }
            for e in scenario.get('enemies', [])
        ]
        self.goal = pygame.Rect(*scenario['goal']) if 'goal' in scenario else None

        return self.render()

    # ------------------------------------------------------------------
    # Step
    # ------------------------------------------------------------------
    def step(self, action):
        self.steps += 1
        reward = 0.0
        done = False
        info = {}

        # --- Parse action ---
        move_x = 0
        jump = False
        if action in [1, 2]: move_x = 1
        if action in [3, 4]: move_x = -1
        if action in [2, 4, 5]: jump = True

        # 1. Horizontal momentum (SMB-style acceleration / skid)
        vx = self.mario['vx']
        self.mario['skidding'] = False

        if move_x != 0:
            # Update facing direction
            self.mario['facing'] = move_x
            # Skidding: pressing opposite to current motion
            if move_x > 0 and vx < 0 or move_x < 0 and vx > 0:
                self.mario['skidding'] = True
                vx += move_x * self.skid_decel
            else:
                vx += move_x * self.accel
            # Clamp to top speed
            vx = max(-self.max_walk_speed, min(self.max_walk_speed, vx))
        else:
            # No input — apply friction toward zero
            if vx > 0:
                vx = max(0.0, vx - self.decel)
            elif vx < 0:
                vx = min(0.0, vx + self.decel)

        self.mario['vx'] = vx

        # 2. Jump
        if jump and self.mario['on_ground']:
            self.mario['vy'] = self.jump_power
            self.mario['on_ground'] = False

        # 3. Gravity
        self.mario['vy'] += self.gravity
        if self.mario['vy'] > self.max_fall_speed:
            self.mario['vy'] = self.max_fall_speed

        # 4. Resolve X collisions
        self.mario['x'] += self.mario['vx']
        mario_rect = pygame.Rect(self.mario['x'], self.mario['y'], self.mario['w'], self.mario['h'])

        for plat in self.platforms:
            if mario_rect.colliderect(plat):
                if self.mario['vx'] > 0:
                    mario_rect.right = plat.left
                elif self.mario['vx'] < 0:
                    mario_rect.left = plat.right
                self.mario['x'] = mario_rect.x
                self.mario['vx'] = 0

        # 5. Resolve Y collisions
        self.mario['y'] += self.mario['vy']
        mario_rect.y = self.mario['y']
        self.mario['on_ground'] = False

        for plat in self.platforms:
            if mario_rect.colliderect(plat):
                if self.mario['vy'] > 0:   # falling
                    mario_rect.bottom = plat.top
                    self.mario['on_ground'] = True
                elif self.mario['vy'] < 0: # hitting head
                    mario_rect.top = plat.bottom
                self.mario['y'] = mario_rect.y
                self.mario['vy'] = 0

        # 6. Update camera (follows Mario, never scrolls left, clamped to world)
        # Target: keep Mario ~1/3 from the left edge of the viewport
        target_camera_x = self.mario['x'] - self.width // 3
        if target_camera_x > self.camera_x:          # only scroll right
            self.camera_x = target_camera_x
        self.camera_x = max(0.0, min(self.camera_x, self.world_width - self.width))

        # 7. World boundaries
        # Mario cannot move left of the camera's left edge (scrolled-off area is gone)
        if self.mario['x'] < self.camera_x:
            self.mario['x'] = self.camera_x
        # Mario cannot walk past right edge of world
        if self.mario['x'] > self.world_width - self.mario['w']:
            self.mario['x'] = self.world_width - self.mario['w']

        # 8. Fall death
        if self.mario['y'] > self.height:
            done = True
            reward -= 10.0

        # 9. Update enemies (skip dead ones)
        for enemy in self.enemies:
            if enemy.get('dead'):
                continue
            enemy['x'] += enemy['speed'] * enemy['direction']
            if enemy['x'] <= enemy['patrol_min']:
                enemy['x'] = enemy['patrol_min']
                enemy['direction'] = 1
            elif enemy['x'] >= enemy['patrol_max']:
                enemy['x'] = enemy['patrol_max']
                enemy['direction'] = -1

        # Enemy collision — stomp (falling onto top) kills enemy; side/bottom kills Mario
        for enemy in self.enemies:
            if enemy.get('dead'):
                continue
            enemy_rect = pygame.Rect(enemy['x'], enemy['y'], enemy['w'], enemy['h'])
            if not mario_rect.colliderect(enemy_rect):
                continue

            # Stomp: Mario is falling AND his previous bottom was above the enemy's vertical midpoint
            prev_mario_bottom = mario_rect.bottom - self.mario['vy']
            stomped = self.mario['vy'] > 0 and prev_mario_bottom <= enemy_rect.centery

            if stomped:
                enemy['dead'] = True
                reward += 5.0
                self.score += 5
                # Bounce Mario upward (half a normal jump)
                self.mario['vy'] = self.jump_power * 0.55
                self.mario['on_ground'] = False
            else:
                done = True
                reward -= 10.0

        # 10. Coin collection
        for coin in self.coins:
            if not coin['collected'] and mario_rect.colliderect(coin['rect']):
                coin['collected'] = True
                reward += 10.0
                self.score += 10

        # 11. Goal
        if self.goal and mario_rect.colliderect(self.goal):
            done = True
            reward += 50.0

        # 12. Timeout
        if self.steps >= self.max_steps:
            done = True

        # Small per-step penalty to encourage speed
        reward -= 0.01

        obs = self.render()
        return obs, reward, done, info

    # ------------------------------------------------------------------
    # Render
    # ------------------------------------------------------------------
    def render(self):
        """
        Draws the viewport (camera_x … camera_x+width) to an off-screen
        surface and returns a NumPy RGB array of shape (H, W, 3).
        All world-space positions are shifted left by camera_x before drawing.
        """
        cam = int(self.camera_x)

        # Background (sky blue)
        self.screen.fill((107, 140, 255))

        # Platforms (brown)
        for plat in self.platforms:
            screen_rect = pygame.Rect(plat.x - cam, plat.y, plat.w, plat.h)
            pygame.draw.rect(self.screen, (139, 69, 19), screen_rect)

        # Coins (gold)
        for coin in self.coins:
            if not coin['collected']:
                r = coin['rect']
                screen_rect = pygame.Rect(r.x - cam, r.y, r.w, r.h)
                pygame.draw.ellipse(self.screen, (255, 215, 0), screen_rect)

        # Goal (green)
        if self.goal:
            screen_rect = pygame.Rect(self.goal.x - cam, self.goal.y, self.goal.w, self.goal.h)
            pygame.draw.rect(self.screen, (0, 255, 0), screen_rect)

        # Enemies (purple + directional eye; squished flat when dead)
        for enemy in self.enemies:
            sx = int(enemy['x']) - cam
            sy = int(enemy['y'])
            if enemy.get('dead'):
                # Draw a squished flat rectangle at the bottom of where the enemy stood
                squish_rect = pygame.Rect(sx, sy + enemy['h'] - 4, enemy['w'], 4)
                pygame.draw.rect(self.screen, (100, 0, 160), squish_rect)
            else:
                enemy_rect = pygame.Rect(sx, sy, enemy['w'], enemy['h'])
                pygame.draw.rect(self.screen, (160, 32, 240), enemy_rect)
                eye_x = sx + (8 if enemy['direction'] > 0 else 2)
                pygame.draw.circle(self.screen, (255, 255, 255), (eye_x, sy + 4), 2)

        # Mario — red body; yellow when skidding; eye dot shows facing direction
        mario_sx = int(self.mario['x']) - cam
        mario_rect = pygame.Rect(mario_sx, int(self.mario['y']), self.mario['w'], self.mario['h'])
        body_color = (255, 220, 0) if self.mario['skidding'] else (255, 0, 0)
        pygame.draw.rect(self.screen, body_color, mario_rect)
        # Eye: small white dot on the facing side
        eye_x = mario_sx + (10 if self.mario['facing'] > 0 else 2)
        eye_y = int(self.mario['y']) + 4
        pygame.draw.circle(self.screen, (255, 255, 255), (eye_x, eye_y), 2)

        # RGB array (H, W, 3)
        rgb_array = pygame.surfarray.array3d(self.screen)
        return np.transpose(rgb_array, (1, 0, 2))

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def load_scenario_from_json(filepath):
        """Loads a scenario dictionary from a JSON config file."""
        with open(filepath, 'r') as f:
            return json.load(f)


# --- Interactive Demo ---
if __name__ == "__main__":
    env = MarioScenarioEnv()

    # Scrolling parkour level — world is 3× the viewport width
    custom_scenario = {
        'world_width': 768,
        'mario': [20, 180],
        'platforms': [
            # --- Screen 1 (0–256) ---
            [0,   220, 100, 20],   # Start ground
            [130, 180,  50, 10],   # First platform
            [210, 140,  50, 10],   # Rising platform
            # --- Screen 2 (256–512) ---
            [270, 220, 120, 20],   # Ground chunk
            [320, 170,  40, 10],   # Mid platform
            [400, 130,  60, 10],   # High platform
            [460, 190,  40, 10],   # Drop-down
            # --- Screen 3 (512–768) ---
            [520, 220, 248, 20],   # Final ground
            [560, 170,  40, 10],   # Platform before goal
            [640, 130,  40, 10],   # Platform before goal
            [720, 100,  48, 120],  # Goal tower
        ],
        'coins': [
            [145, 150, 10, 10],
            [225, 110, 10, 10],
            [335, 140, 10, 10],
            [415, 100, 10, 10],
            [575, 140, 10, 10],
            [655, 100, 10, 10],
        ],
        # Enemies: [x, y, patrol_min_x, patrol_max_x]
        'enemies': [
            [10,  200,  2,   90],   # Start ground patrol
            [132, 160, 130, 178],   # First platform
            [272, 200, 270, 388],   # Second ground section
            [325, 150, 322, 438],   # Mid platform
            [522, 200, 520, 660],   # Final ground
            [562, 150, 560, 598],   # Platform before goal
        ],
        'goal': [730, 80, 16, 20],
    }

    config_path = "scenarios/level_1.json"
    if os.path.exists(config_path):
        scenario_config = MarioScenarioEnv.load_scenario_from_json(config_path)
    else:
        scenario_config = custom_scenario

    obs = env.reset(scenario=scenario_config)

    display = pygame.display.set_mode((env.width, env.height))
    pygame.display.set_caption("Mario AI Scenario Simulator")
    clock = pygame.time.Clock()

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # Bias toward moving right so the demo scrolls visibly
        action = np.random.choice([0, 1, 1, 1, 2, 2, 5])

        obs, reward, done, info = env.step(action)

        surface = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        display.blit(surface, (0, 0))
        pygame.display.flip()
        clock.tick(30)

        if done:
            print(f"Scenario finished! Score: {env.score}  Camera: {env.camera_x:.0f}")
            env.reset(scenario=custom_scenario)

    pygame.quit()
