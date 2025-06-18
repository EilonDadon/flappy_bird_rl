import pygame
import sys
from pathlib import Path

# Add the project root directory to Python path
project_root = str(Path(__file__).parent.parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)

from src.game.core import FlappyBirdCore
from src.utils.constants import *

def draw_vertical_gradient(surface, top_color, bottom_color):
    height = surface.get_height()
    for y in range(height):
        ratio = y / height
        r = int(top_color[0] * (1 - ratio) + bottom_color[0] * ratio)
        g = int(top_color[1] * (1 - ratio) + bottom_color[1] * ratio)
        b = int(top_color[2] * (1 - ratio) + bottom_color[2] * ratio)
        pygame.draw.line(surface, (r, g, b), (0, y), (surface.get_width(), y))

class FlappyBirdGame:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption('Flappy Bird - Manual Play')
        self.clock = pygame.time.Clock()
        self.game = FlappyBirdCore()
        try:
            self.font = pygame.font.SysFont('comicsansms', 36)
            self.controls_font = pygame.font.SysFont('comicsansms', 24)
            self.small_font = pygame.font.SysFont('comicsansms', 18)
        except:
            self.font = pygame.font.Font(None, 36)
            self.controls_font = pygame.font.Font(None, 24)
            self.small_font = pygame.font.Font(None, 18)

    def draw_difficulty_bar(self):
        """Draw the difficulty indicator bar at the top of the screen."""
        # Get current difficulty settings
        current_speed, current_gap = self.game._get_current_difficulty()
        
        # Calculate difficulty percentages
        if self.game.score < 30:
            # Stage 1: Speed increases from 0% to 30%
            speed_increase = ((current_speed / PIPE_SPEED) - 1) * 100
            # Stage 1: Gap decreases from 0% to 10%
            gap_decrease = (1 - (current_gap / PIPE_GAP)) * 100
        elif self.game.score < 50:
            # Stage 2: Speed increases from 30% to 60%
            speed_increase = ((current_speed / PIPE_SPEED) - 1) * 100
            # Stage 2: Gap stays at 10%
            gap_decrease = 10.0
        else:
            # Stage 3: Speed stays at 60%
            speed_increase = 60.0
            # Stage 3: Gap decreases from 10% to 25%
            gap_decrease = (1 - (current_gap / PIPE_GAP)) * 100
        
        # Draw background bar
        bar_height = 40
        pygame.draw.rect(self.screen, (0, 0, 0, 128), (0, 0, SCREEN_WIDTH, bar_height))
        
        # Draw speed indicator
        speed_text = self.small_font.render(f'Speed: +{speed_increase:.1f}%', True, (255, 255, 255))
        self.screen.blit(speed_text, (10, 5))
        
        # Draw gap indicator
        gap_text = self.small_font.render(f'Gap: -{gap_decrease:.1f}%', True, (255, 255, 255))
        self.screen.blit(gap_text, (SCREEN_WIDTH - 150, 5))
        
        # Draw progress bar
        progress_width = SCREEN_WIDTH - 20
        progress_height = 10
        progress_x = 10
        progress_y = 25
        
        # Draw background of progress bar
        pygame.draw.rect(self.screen, (100, 100, 100), 
                        (progress_x, progress_y, progress_width, progress_height))
        
        # Draw stage markers
        stage1_width = int(progress_width * (30/60))  # Stage 1 ends at 30
        stage2_width = int(progress_width * (50/60))  # Stage 2 ends at 50
        
        # Draw stage 1 (green)
        stage1_progress = min(self.game.score / 30.0, 1.0)
        stage1_filled = int(stage1_width * stage1_progress)
        pygame.draw.rect(self.screen, (0, 255, 0), 
                        (progress_x, progress_y, stage1_filled, progress_height))
        
        # Draw stage 2 (yellow) if we're past stage 1
        if self.game.score > 30:
            stage2_progress = min((self.game.score - 30) / 20.0, 1.0)  # Progress in stage 2
            stage2_filled = int((stage2_width - stage1_width) * stage2_progress)
            pygame.draw.rect(self.screen, (255, 255, 0), 
                            (progress_x + stage1_width, progress_y, stage2_filled, progress_height))
        
        # Draw stage 3 (red) if we're past stage 2
        if self.game.score > 50:
            stage3_progress = min((self.game.score - 50) / 10.0, 1.0)  # Progress in stage 3
            stage3_filled = int((progress_width - stage2_width) * stage3_progress)
            pygame.draw.rect(self.screen, (255, 0, 0), 
                            (progress_x + stage2_width, progress_y, stage3_filled, progress_height))
        
        # Draw stage transition markers
        # Red line for stage 2 (50 points)
        pygame.draw.line(self.screen, (255, 0, 0),
                        (progress_x + stage2_width, progress_y),
                        (progress_x + stage2_width, progress_y + progress_height),
                        2)
        
        # Yellow line for stage 1 (30 points)
        pygame.draw.line(self.screen, (255, 255, 0),
                        (progress_x + stage1_width, progress_y),
                        (progress_x + stage1_width, progress_y + progress_height),
                        2)

    def draw(self):
        """Draw the current game state."""
        self.screen.fill(SKY_COLOR)
        
        # Draw vertical gradient
        draw_vertical_gradient(self.screen, (120, 200, 255), (0, 120, 255))
        
        # Draw clouds
        for cx, cy, cr in [(100, 80, 30), (300, 60, 25), (500, 100, 35), (700, 70, 20)]:
            pygame.draw.ellipse(self.screen, (255,255,255), (cx-cr, cy-cr//2, cr*2, cr))
        
        # Draw pipes as rectangles only (no rounded edges)
        for pipe in self.game.get_game_state()['pipes']:
            # Top pipe
            pygame.draw.rect(self.screen, PIPE_COLOR, (pipe['x'], 0, PIPE_WIDTH, pipe['top']))
            # Bottom pipe
            bottom_y = pipe['top'] + PIPE_GAP
            pygame.draw.rect(self.screen, PIPE_COLOR, (pipe['x'], bottom_y, PIPE_WIDTH, SCREEN_HEIGHT - bottom_y))
        
        # Draw grass above ground
        pygame.draw.rect(self.screen, (60, 200, 60), (0, SCREEN_HEIGHT - GROUND_HEIGHT - 10, SCREEN_WIDTH, 10))
        
        # Draw ground
        pygame.draw.rect(self.screen, GROUND_COLOR, (0, SCREEN_HEIGHT - GROUND_HEIGHT, SCREEN_WIDTH, GROUND_HEIGHT))
        
        # Draw bird as a circle
        bird_state = self.game.get_game_state()['bird']
        bird_center = (bird_state['x'] + BIRD_SIZE//2, bird_state['y'] + BIRD_SIZE//2)
        pygame.draw.circle(self.screen, (255, 220, 40), bird_center, BIRD_SIZE//2)
        
        # Draw bird's eye
        pygame.draw.circle(self.screen, (255,255,255), (bird_center[0]+5, bird_center[1]-5), 5)
        pygame.draw.circle(self.screen, (0,0,0), (bird_center[0]+7, bird_center[1]-5), 2)
        
        # Draw bird's beak
        pygame.draw.polygon(self.screen, (255, 150, 0), [
            (bird_center[0]+BIRD_SIZE//2, bird_center[1]),
            (bird_center[0]+BIRD_SIZE//2+8, bird_center[1]-3),
            (bird_center[0]+BIRD_SIZE//2+8, bird_center[1]+3)
        ])
        
        # Draw score
        score_text = self.font.render(f'Score: {self.game.get_game_state()["score"]}', True, (0, 0, 0))
        self.screen.blit(score_text, (10, 50))  # Moved score down to make room for difficulty bar
        
        # Draw controls help
        controls_text = self.controls_font.render('SPACE=Flap  R=Reset', True, (255, 255, 255))
        self.screen.blit(controls_text, (10, SCREEN_HEIGHT - 35))
        
        # Draw game over message
        if self.game.get_game_state()['game_over']:
            game_over_text = self.font.render('Game Over! Press R to restart', True, (255, 255, 255))
            text_rect = game_over_text.get_rect(center=(SCREEN_WIDTH/2, SCREEN_HEIGHT/2))
            self.screen.blit(game_over_text, text_rect)
        
        # Draw difficulty bar
        self.draw_difficulty_bar()
        
        pygame.display.flip()

    def run(self):
        """Main game loop."""
        while True:
            # Handle events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE:
                        if self.game.get_game_state()['game_over']:
                            self.game.reset()
                        else:
                            self.game.update(1)  # Flap
                    if event.key == pygame.K_r:
                        self.game.reset()
            
            # Update game state if not game over
            if not self.game.get_game_state()['game_over']:
                self.game.update(0)  # Gravity
            
            # Draw everything
            self.draw()
            
            # Cap the frame rate
            self.clock.tick(60)

if __name__ == '__main__':
    game = FlappyBirdGame()
    game.run() 