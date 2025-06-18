import sys
import pygame
import torch
from pathlib import Path
import tkinter as tk
from tkinter import filedialog

# Add project root to Python path
sys.path.append(str(Path(__file__).parent.parent))

from src.rl.agent import DQNAgent
from src.game.core import FlappyBirdCore
from src.utils.constants import *

def select_model():
    """Open file dialog to select model file."""
    root = tk.Tk()
    root.withdraw()
    model_path = filedialog.askopenfilename(
        initialdir=str(Path(__file__).parent.parent / 'models'),
        title="Select Model File",
        filetypes=[("PyTorch Models", "*.pth")]
    )
    return model_path if model_path else None

def draw_vertical_gradient(surface, top_color, bottom_color):
    """Draw vertical gradient background."""
    height = surface.get_height()
    for y in range(height):
        ratio = y / height
        r = int(top_color[0] * (1 - ratio) + bottom_color[0] * ratio)
        g = int(top_color[1] * (1 - ratio) + bottom_color[1] * ratio)
        b = int(top_color[2] * (1 - ratio) + bottom_color[2] * ratio)
        pygame.draw.line(surface, (r, g, b), (0, y), (surface.get_width(), y))

def draw_difficulty_bar(screen, game, font):
    """Draw difficulty progression bar with stage markers."""
    current_speed, current_gap = game._get_current_difficulty()
    
    # Calculate difficulty metrics
    if game.score < 30:
        speed_increase = ((current_speed / PIPE_SPEED) - 1) * 100
        gap_decrease = (1 - (current_gap / PIPE_GAP)) * 100
    elif game.score < 50:
        speed_increase = ((current_speed / PIPE_SPEED) - 1) * 100
        gap_decrease = 10.0
    else:
        speed_increase = 60.0
        gap_decrease = (1 - (current_gap / PIPE_GAP)) * 100
    
    # Draw bar background
    bar_height = 40
    pygame.draw.rect(screen, (0, 0, 0, 128), (0, 0, SCREEN_WIDTH, bar_height))
    
    # Draw metrics
    speed_text = font.render(f'Speed: +{speed_increase:.1f}%', True, (255, 255, 255))
    screen.blit(speed_text, (10, 5))
    gap_text = font.render(f'Gap: -{gap_decrease:.1f}%', True, (255, 255, 255))
    screen.blit(gap_text, (SCREEN_WIDTH - 150, 5))
    
    # Draw progress bar
    progress_width = SCREEN_WIDTH - 20
    progress_height = 10
    progress_x = 10
    progress_y = 25
    
    pygame.draw.rect(screen, (100, 100, 100), 
                    (progress_x, progress_y, progress_width, progress_height))
    
    # Calculate stage boundaries
    stage1_width = int(progress_width * (30/60))
    stage2_width = int(progress_width * (50/60))
    
    # Draw stage progress
    stage1_progress = min(game.score / 30.0, 1.0)
    stage1_filled = int(stage1_width * stage1_progress)
    pygame.draw.rect(screen, (0, 255, 0), 
                    (progress_x, progress_y, stage1_filled, progress_height))
    
    if game.score > 30:
        stage2_progress = min((game.score - 30) / 20.0, 1.0)
        stage2_filled = int((stage2_width - stage1_width) * stage2_progress)
        pygame.draw.rect(screen, (255, 255, 0), 
                        (progress_x + stage1_width, progress_y, stage2_filled, progress_height))
    
    if game.score > 50:
        stage3_progress = min((game.score - 50) / 10.0, 1.0)
        stage3_filled = int((progress_width - stage2_width) * stage3_progress)
        pygame.draw.rect(screen, (255, 0, 0), 
                        (progress_x + stage2_width, progress_y, stage3_filled, progress_height))
    
    # Draw stage markers
    pygame.draw.line(screen, (255, 0, 0),
                    (progress_x + stage2_width, progress_y),
                    (progress_x + stage2_width, progress_y + progress_height),
                    2)
    pygame.draw.line(screen, (255, 255, 0),
                    (progress_x + stage1_width, progress_y),
                    (progress_x + stage1_width, progress_y + progress_height),
                    2)

def play(model_path=None):
    """Run Flappy Bird with trained agent."""
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption('Flappy Bird - AI Agent')
    
    try:
        font = pygame.font.SysFont('comicsansms', 36)
        controls_font = pygame.font.SysFont('comicsansms', 24)
        small_font = pygame.font.SysFont('comicsansms', 18)
    except:
        font = pygame.font.Font(None, 36)
        controls_font = pygame.font.Font(None, 24)
        small_font = pygame.font.Font(None, 18)
    
    clock = pygame.time.Clock()
    
    # Initialize game components
    game = FlappyBirdCore()
    agent = DQNAgent(state_size=7, action_size=2)
    
    if not model_path:
        model_path = str(Path(__file__).parent.parent / 'models' / 'flappy_bird_model_final.pth')
    
    if model_path:
        try:
            checkpoint = torch.load(model_path)
            agent.policy_net.load_state_dict(checkpoint['model_state_dict'])
            agent.policy_net.eval()
            print(f"Loaded model from {model_path}")
        except Exception as e:
            print(f"Error loading model: {e}")
            return
    
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    game.reset()
                if event.key == pygame.K_p:
                    new_model_path = select_model()
                    if new_model_path:
                        try:
                            checkpoint = torch.load(new_model_path)
                            agent.policy_net.load_state_dict(checkpoint['model_state_dict'])
                            agent.policy_net.eval()
                            print(f"Loaded new model from {new_model_path}")
                        except Exception as e:
                            print(f"Error loading model: {e}")
        
        state = game._get_state()
        
        with torch.no_grad():
            state_tensor = torch.from_numpy(state).unsqueeze(0).float()
            q_values = agent.policy_net(state_tensor)
            action = torch.argmax(q_values[0]).item()
        
        game.update(action)
        
        if game.get_game_state()['game_over']:
            game.reset()
        
        # Render game state
        draw_vertical_gradient(screen, (120, 200, 255), (0, 120, 255))
        
        for cx, cy, cr in [(100, 80, 30), (300, 60, 25), (500, 100, 35), (700, 70, 20)]:
            pygame.draw.ellipse(screen, (255,255,255), (cx-cr, cy-cr//2, cr*2, cr))
        
        for pipe in game.get_game_state()['pipes']:
            pygame.draw.rect(screen, PIPE_COLOR, (pipe['x'], 0, PIPE_WIDTH, pipe['top']))
            bottom_y = pipe['top'] + PIPE_GAP
            pygame.draw.rect(screen, PIPE_COLOR, (pipe['x'], bottom_y, PIPE_WIDTH, SCREEN_HEIGHT - bottom_y))
        
        pygame.draw.rect(screen, (60, 200, 60), (0, SCREEN_HEIGHT - GROUND_HEIGHT - 10, SCREEN_WIDTH, 10))
        pygame.draw.rect(screen, GROUND_COLOR, (0, SCREEN_HEIGHT - GROUND_HEIGHT, SCREEN_WIDTH, GROUND_HEIGHT))
        
        bird_state = game.get_game_state()['bird']
        bird_center = (bird_state['x'] + BIRD_SIZE//2, bird_state['y'] + BIRD_SIZE//2)
        pygame.draw.circle(screen, (255, 220, 40), bird_center, BIRD_SIZE//2)
        pygame.draw.circle(screen, (255,255,255), (bird_center[0]+5, bird_center[1]-5), 5)
        pygame.draw.circle(screen, (0,0,0), (bird_center[0]+7, bird_center[1]-5), 2)
        pygame.draw.polygon(screen, (255, 150, 0), [
            (bird_center[0]+BIRD_SIZE//2, bird_center[1]),
            (bird_center[0]+BIRD_SIZE//2+8, bird_center[1]-3),
            (bird_center[0]+BIRD_SIZE//2+8, bird_center[1]+3)
        ])
        
        score_text = font.render(f'Score: {game.get_game_state()["score"]}', True, (0, 0, 0))
        screen.blit(score_text, (10, 50))
        
        controls_text = controls_font.render('Press P to select model, R to reset', True, (255, 255, 255))
        screen.blit(controls_text, (10, SCREEN_HEIGHT - 35))
        
        if game.get_game_state()['game_over']:
            game_over_text = font.render('Game Over! Press R to restart', True, (255, 255, 255))
            text_rect = game_over_text.get_rect(center=(SCREEN_WIDTH/2, SCREEN_HEIGHT/2))
            screen.blit(game_over_text, text_rect)
        
        draw_difficulty_bar(screen, game, small_font)
        
        pygame.display.flip()
        clock.tick(60)
    
    pygame.quit()

if __name__ == '__main__':
    model_path = sys.argv[1] if len(sys.argv) > 1 else None
    play(model_path) 