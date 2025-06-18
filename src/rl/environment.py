from ..game.core import FlappyBirdCore

class FlappyBirdEnv:
    """Flappy Bird environment for RL. 
    State: a 7-dimensional vector:
          - vertical_distance_to_current_gap = (bird_y_center - current_gap_y_center) / SCREEN_HEIGHT
          - horizontal_distance_to_current_pipe = (current_pipe_x - bird_x) / SCREEN_WIDTH
          - vertical_distance_to_next_gap = (bird_y_center - next_gap_y_center) / SCREEN_HEIGHT
          - horizontal_distance_to_next_pipe = (next_pipe_x - bird_x) / SCREEN_WIDTH
          - bird_velocity (current vertical velocity) normalized to [-1,1]
          - distance_from_top = (bird_y - current_pipe_top) / SCREEN_HEIGHT
          - distance_from_bottom = (current_pipe_bottom - bird_y) / SCREEN_HEIGHT
    Action space: 2 actions (discrete) – 0: do nothing, 1: flap (apply upward jump to the bird).
    Reward: +1 for each pipe passed, -5 if the bird crashes, +0.1 for staying alive.
    Episode ends when a crash occurs. Game dynamics (gravity, pipe movement) mimic the original Flappy Bird.
    """
    def __init__(self):
        self.game = FlappyBirdCore()
        self.state_size = 7  # Updated to match the new state size
        self.action_size = 2

    def reset(self):
        """Reset the environment to the initial state. Returns the initial state."""
        self.game.reset()
        return self.game._get_state()

    def step(self, action):
        """
        Apply the action (0: no flap, 1: flap) and update the game state by one time-step.
        Returns: (next_state, reward, done)
        """
        return self.game.update(action) 