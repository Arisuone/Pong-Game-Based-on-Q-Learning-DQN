import numpy as np
import random
import math

def discretize(value, max_value, bins=32):
    """Map continuous values to discrete grid cells"""
    return int(value / max_value * (bins - 1))

class PongEnv:
    def __init__(self, width=400, height=300, bins=32, return_scorer=False):
        self.width, self.height = width, height
        self.bins = bins
        self.return_scorer = return_scorer

        # Reward parameters definition
        self.reward_close = 0.05    # Reward for approaching the ball
        self.reward_hit = 2.0       # Reward for blocking the ball
        self.reward_miss = -1.0     # Missed ball penalty
        self.reward_precise = 0.5   # Precise approach reward

        self.reset()

    def reset(self):
        """Reset ball and paddle positions"""
        # Ensure the ball does not initialize behind the paddle, restrict it to the left half of the court
        self.ball_x = random.randint(50, self.width // 3)
        self.ball_y = random.randint(0, self.height)

        # Randomize ball speed, set within a reasonable range
        speed = random.uniform(3.0, 5.0)

        # Limit the angle range to ±45°, avoiding extreme vertical trajectories
        while True:
            angle = random.uniform(-math.pi/4, math.pi/4)
            vx = speed * math.cos(angle)
            vy = speed * math.sin(angle)
            if abs(vx) >= 1.5:   # Ensure the ball is not approximately vertical
                break

        self.ball_vx = abs(vx)   # Fix the ball's direction to the right side
        self.ball_vy = vy

        # Initialize paddle position in the center
        self.agent_y, self.player_y = self.height // 2, self.height // 2
        return self.get_state()

    def get_state(self):
        """Return the discretized state: ball position + ball velocity + agent paddle position"""
        return np.array([
            discretize(self.ball_x, self.width, self.bins),
            discretize(self.ball_y, self.height, self.bins),
            discretize(self.agent_y, self.height, self.bins),
            discretize(self.ball_vx, 5, self.bins),   # Ball horizontal velocity
            discretize(self.ball_vy, 5, self.bins)    # Ball vertical velocity
        ])

    def step(self, action):
        """Execute one step action: 0 = move up, 1 = move down, 2 = stay still"""
        if action == 0:
            self.agent_y -= 5
        elif action == 1:
            self.agent_y += 5
        self.agent_y = np.clip(self.agent_y, 0, self.height)

        # Update ball position
        self.ball_x += self.ball_vx
        self.ball_y += self.ball_vy

        # Apply continuous slight perturbations to make the velocity more natural
        self.ball_vx += random.choice([-0.1, 0, 0.15])
        self.ball_vy += random.choice([-0.08, 0, 0.08])

        # Boundary handling: prevent the ball from getting stuck and vibrating at the edges
        if self.ball_y <= 0:
            self.ball_y = 1
            self.ball_vy = abs(self.ball_vy) if self.ball_vy != 0 else 1
        elif self.ball_y >= self.height:
            self.ball_y = self.height - 1
            self.ball_vy = -abs(self.ball_vy) if self.ball_vy != 0 else -1

        # Minimum speed constraint to avoid excessively low velocity
        if abs(self.ball_vx) < 3.0:
            self.ball_vx = 3.0 if self.ball_vx >= 0 else -3.0
        if abs(self.ball_vy) < 3.0:
            self.ball_vy = 3.0 if self.ball_vy >= 0 else -3.0

        reward, done, scorer = 0, False, None

        # Reward shaping: encourage the paddle to move closer to the ball
        distance = abs(self.agent_y - self.ball_y)
        reward += max(0, 1 - distance / self.height) * self.reward_close

        # Precise approach reward: if the distance is less than 10, give extra reward
        if distance < 10:
            reward += self.reward_precise

        # Ball reaches the left side (player-controlled)
        if self.ball_x <= 20:
            if abs(self.player_y - self.ball_y) < 30:
                self.ball_vx *= -1
                scorer = "player"   # Player successfully returns the ball, award points
            else:
                reward += self.reward_miss
                scorer = "agent"
                done = True
                if self.return_scorer: 
                    self.reset()

        # # Ball reaches the right side (agent-controlled)
        if self.ball_x >= self.width - 20:
            if abs(self.agent_y - self.ball_y) < 30:
                self.ball_vx *= -1
                reward += self.reward_hit
                scorer = "agent"
            else:
                reward += self.reward_miss
                scorer = "player"
                done = True
                if self.return_scorer: 
                    self.reset()

        if self.return_scorer:
            return self.get_state(), reward, done, scorer
        else:
            return self.get_state(), reward, done