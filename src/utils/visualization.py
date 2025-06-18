import pygame
from .constants import *

def draw_game_state(game_state):
    """Draw game elements based on current state."""
    screen = pygame.display.get_surface()
    
    # Draw background
    screen.fill((120, 200, 255))
    
    # Draw pipes
    for pipe in game_state['pipes']:
        # Top pipe
        pygame.draw.rect(screen, PIPE_COLOR, (pipe['x'], 0, PIPE_WIDTH, pipe['top']))
        # Bottom pipe
        bottom_y = pipe['top'] + PIPE_GAP
        pygame.draw.rect(screen, PIPE_COLOR, (pipe['x'], bottom_y, PIPE_WIDTH, SCREEN_HEIGHT - bottom_y))
    
    # Draw ground
    pygame.draw.rect(screen, GROUND_COLOR, (0, SCREEN_HEIGHT - GROUND_HEIGHT, SCREEN_WIDTH, GROUND_HEIGHT))
    
    # Draw bird
    bird = game_state['bird']
    pygame.draw.rect(screen, BIRD_COLOR, (bird['x'], bird['y'], BIRD_SIZE, BIRD_SIZE))
    
    # Draw score
    font = pygame.font.Font(None, 36)
    score_text = font.render(f'Score: {game_state["score"]}', True, (0, 0, 0))
    screen.blit(score_text, (10, 10))

def draw_training_info(episode, reward, steps, epsilon):
    """Draw training metrics."""
    screen = pygame.display.get_surface()
    font = pygame.font.Font(None, 24)
    
    # Draw episode info
    episode_text = font.render(f'Episode: {episode}', True, (255, 255, 255))
    reward_text = font.render(f'Reward: {reward:.1f}', True, (255, 255, 255))
    steps_text = font.render(f'Steps: {steps}', True, (255, 255, 255))
    epsilon_text = font.render(f'Epsilon: {epsilon:.2f}', True, (255, 255, 255))
    
    screen.blit(episode_text, (10, 50))
    screen.blit(reward_text, (10, 80))
    screen.blit(steps_text, (10, 110))
    screen.blit(epsilon_text, (10, 140)) 