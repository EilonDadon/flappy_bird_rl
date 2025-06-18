import random
import numpy as np
from ..utils.constants import *

class FlappyBirdCore:
    """Core game logic for Flappy Bird, handling physics, collisions, and state management."""
    def __init__(self):
        self.reset()

    def reset(self):
        """Initialize game state for a new round."""
        self.bird_y = (SCREEN_HEIGHT - GROUND_HEIGHT) / 2 - BIRD_SIZE / 2
        self.bird_vel = 0
        self.pipes = []
        self.score = 0
        self.game_over = False
        
        # Initialize first pipe
        initial_pipe_x = SCREEN_WIDTH
        top_height = random.randint(50, GROUND_Y - PIPE_GAP - 50)
        self.pipes.append({"x": initial_pipe_x, "top": top_height, "passed": False})

    def _get_current_difficulty(self):
        """Calculate dynamic difficulty based on score:
        Stage 1 (0-30): Speed 0-30%, Gap 0-10%
        Stage 2 (30-50): Speed 30-60%, Gap 10%
        Stage 3 (50-60): Speed 60%, Gap 10-25%
        """
        # Speed calculation
        if self.score < 30:
            speed_increase = (self.score / 30.0) * 0.3
        elif self.score < 50:
            stage2_progress = (self.score - 30) / 20.0
            speed_increase = 0.3 + (stage2_progress * 0.3)
        else:
            speed_increase = 0.6
        
        # Gap calculation
        if self.score < 30:
            gap_decrease = (self.score / 30.0) * 0.1
        elif self.score < 50:
            gap_decrease = 0.1
        else:
            stage3_progress = (self.score - 50) / 10.0
            gap_decrease = 0.1 + (stage3_progress * 0.15)
        
        return PIPE_SPEED * (1 + speed_increase), PIPE_GAP * (1 - gap_decrease)

    def _get_next_pipe_height(self, current_top, current_gap):
        """Calculate next pipe height with smooth transition."""
        MAX_HEIGHT_CHANGE = 0.15  # Max 15% screen height change
        
        min_height = max(50, current_top - int(SCREEN_HEIGHT * MAX_HEIGHT_CHANGE))
        max_height = min(int(GROUND_Y - current_gap - 50), 
                        current_top + int(SCREEN_HEIGHT * MAX_HEIGHT_CHANGE))
        
        return random.randint(min_height, max_height)

    def update(self, action=None):
        """Update game state by one frame.
        Args:
            action: 1 for jump, 0 for no action, None for manual play
        Returns:
            (state, reward, done) for RL, None for manual play
        """
        current_speed, current_gap = self._get_current_difficulty()

        # Handle jump action
        if action is not None and action == 1:
            self.bird_vel = JUMP_VEL

        # Update bird physics
        self.bird_vel += GRAVITY
        self.bird_vel = min(self.bird_vel, 10)  # Cap velocity
        self.bird_y += self.bird_vel
        self.bird_y = max(0, self.bird_y)  # Prevent going above screen

        # Update pipe positions
        for pipe in self.pipes:
            pipe["x"] -= current_speed
        self.pipes = [pipe for pipe in self.pipes if pipe["x"] > -PIPE_WIDTH]

        # Add new pipe if needed
        if len(self.pipes) == 0 or self.pipes[-1]["x"] < SCREEN_WIDTH - PIPE_DISTANCE:
            last_pipe_top = self.pipes[-1]["top"] if self.pipes else (SCREEN_HEIGHT - GROUND_HEIGHT) // 2
            new_top = self._get_next_pipe_height(last_pipe_top, current_gap)
            self.pipes.append({"x": SCREEN_WIDTH, "top": new_top, "passed": False})

        # Check collisions and scoring
        done = False
        reward = 0

        # Ground/ceiling collision
        if self.bird_y <= 0 or self.bird_y >= GROUND_Y - BIRD_SIZE:
            done = True
            reward = -5 if action is not None else 0

        # Pipe collision and scoring
        for pipe in self.pipes:
            if BIRD_X + BIRD_SIZE > pipe["x"] and BIRD_X < pipe["x"] + PIPE_WIDTH:
                if self.bird_y < pipe["top"] or self.bird_y + BIRD_SIZE > pipe["top"] + current_gap:
                    done = True
                    reward = -5 if action is not None else 0
                elif action is not None:
                    # Reward for being centered in gap
                    gap_center = pipe["top"] + current_gap/2
                    distance_from_center = abs(self.bird_y + BIRD_SIZE/2 - gap_center)
                    reward = 0.1 * (1 - distance_from_center/(current_gap/2))
            elif pipe["x"] + PIPE_WIDTH < BIRD_X and not pipe.get("passed", False):
                pipe["passed"] = True
                self.score += 1
                reward = 1 if action is not None else 0

        # Survival reward
        if not done and action is not None:
            reward += 0.1

        self.game_over = done
        return (self._get_state(), reward, done) if action is not None else None

    def _get_state(self):
        """Get normalized state representation for RL agent."""
        _, current_gap = self._get_current_difficulty()
        
        # Find current and next pipes
        current_pipe = next((p for p in self.pipes if not p["passed"]), self.pipes[-1])
        next_pipe = next((p for p in self.pipes if p != current_pipe and not p["passed"]), self.pipes[-1])

        # Calculate distances
        bird_center_y = self.bird_y + BIRD_SIZE/2
        current_gap_center_y = current_pipe["top"] + current_gap/2
        next_gap_center_y = next_pipe["top"] + current_gap/2

        # Normalize all values
        return np.array([
            (bird_center_y - current_gap_center_y) / SCREEN_HEIGHT,
            (current_pipe["x"] - BIRD_X) / SCREEN_WIDTH,
            (bird_center_y - next_gap_center_y) / SCREEN_HEIGHT,
            (next_pipe["x"] - BIRD_X) / SCREEN_WIDTH,
            self.bird_vel / 15.0,
            (self.bird_y - current_pipe["top"]) / SCREEN_HEIGHT,
            ((current_pipe["top"] + current_gap) - (self.bird_y + BIRD_SIZE)) / SCREEN_HEIGHT
        ], dtype=np.float32)

    def get_game_state(self):
        """Get current game state for rendering."""
        return {
            'bird': {'x': BIRD_X, 'y': self.bird_y},
            'pipes': self.pipes,
            'score': self.score,
            'game_over': self.game_over
        } 