import pygame
import torch
from pong_env import PongEnv
from dqn import DQN
from utils import load_model

# Initialize the Pygame window
pygame.init()
screen = pygame.display.set_mode((400, 300))
clock = pygame.time.Clock()

# Use environment in demo mode (return_scorer=True means returning the scorer)
env = PongEnv(return_scorer=True)
state_dim, action_dim = len(env.get_state()), 3

# Initialize the policy network
policy_net = DQN(state_dim, action_dim)

# Load the trained model
policy_net = load_model(policy_net, path="models/pong.pth")

# Initialize the score
agent_score, player_score = 0, 0
state = env.reset()
done = False

# Game main loop
running = True
while running:
    # Handle quit event
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Player controls the paddle with keyboard up and down keys
    keys = pygame.key.get_pressed()
    if keys[pygame.K_UP]:
        env.player_y -= 5
    if keys[pygame.K_DOWN]:
        env.player_y += 5

    # The agent selects actions using the trained policy
    with torch.no_grad():
        action = policy_net(torch.FloatTensor(state)).argmax().item()

    # Execute action and update environment state
    next_state, reward, done, scorer = env.step(action)
    state = next_state

    # Update the score
    if scorer == "agent":
        agent_score += 1
    if scorer == "player":
        player_score += 1

    # Draw the screen
    screen.fill((0, 0, 0))
    pygame.draw.rect(screen, (255, 255, 255), (10, env.player_y - 30, 10, 60))   # Player paddle
    pygame.draw.rect(screen, (255, 255, 255), (380, env.agent_y - 30, 10, 60))   # Agent paddle
    pygame.draw.circle(screen, (255, 255, 255), (env.ball_x, env.ball_y), 8)     # Ball

    # Display the score
    font = pygame.font.SysFont(None, 24)
    score_text = font.render(f"Player: {player_score}  Agent: {agent_score}", True, (255, 255, 255))
    screen.blit(score_text, (120, 10))

    # Display the immediate reward
    reward_text = font.render(f"Reward: {reward:.2f}", True, (200, 200, 200))
    screen.blit(reward_text, (10, 260))

    # # Display reward parameters (decomposed)
    reward_detail = f"close={env.reward_close}, precise={env.reward_precise}, hit={env.reward_hit}, miss={env.reward_miss}"
    reward_detail_text = font.render(reward_detail, True, (150, 150, 150))
    screen.blit(reward_detail_text, (10, 280))

    # Refresh the screen
    pygame.display.flip()
    clock.tick(60)

pygame.quit()