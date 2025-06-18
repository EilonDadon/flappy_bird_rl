# Flappy Bird RL

A reinforcement learning implementation of Flappy Bird using Deep Q-Learning (DQN). This project demonstrates how to train an AI agent to play Flappy Bird using PyTorch and Pygame.

## Project Structure

```
flappy_bird_rl/
├── src/
│   ├── game/
│   │   ├── __init__.py
│   │   ├── core.py           # Core game logic
│   │   └── manual_game.py    # Manual game implementation
│   ├── rl/
│   │   ├── __init__.py
│   │   ├── agent.py          # DQN agent implementation
│   │   └── environment.py    # RL environment wrapper
│   └── utils/
│       ├── __init__.py
│       └── constants.py      # Shared game constants
├── scripts/
│   ├── train.py             # Training script
│   └── play.py              # Watch trained agent play
├── models/                  # Directory for saved models
├── plots/                   # Directory for training plots
└── requirements.txt
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/flappy-bird-rl.git
cd flappy-bird-rl
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Playing Manually

To play the game manually:
```bash
python -m src.game.manual_game
```

Controls:
- SPACE: Jump
- Close window to quit

### Training the Agent

To train a new agent:
```bash
python scripts/train.py
```

The training script will:
- Train the agent for 2000 episodes by default
- Save model checkpoints every 200 episodes
- Generate a training progress plot
- Save the final model

Training progress will be displayed every 50 episodes, showing:
- Current episode number
- Current score
- Average score over last 50 episodes
- Current exploration rate (epsilon)

### Watching the Trained Agent

To watch a trained agent play:
```bash
python scripts/play.py [model_path]
```

Controls:
- R: Reset game
- P: Select different model file
- Close window to quit

## Implementation Details

### State Space
The agent observes a 7-dimensional state vector:
1. Horizontal distance to next pipe
2. Vertical distance to pipe top
3. Vertical distance to pipe bottom
4. Bird's vertical velocity
5. Bird's vertical position
6. Current pipe speed
7. Current pipe gap size

### Action Space
The agent can take two actions:
- 0: Do nothing
- 1: Jump

### Reward Structure
- +1.0 for passing a pipe
- -5.0 for crashing
- +0.1 for staying alive
- Additional rewards for:
  - Centering in pipe gap
  - Maintaining moderate velocity
  - Staying near screen center

### DQN Architecture
- Input layer: 7 neurons (state size)
- Hidden layer 1: 256 neurons with ReLU and BatchNorm
- Hidden layer 2: 128 neurons with ReLU and BatchNorm
- Hidden layer 3: 64 neurons with ReLU and BatchNorm
- Output layer: 2 neurons (action size)

### Training Parameters
- Learning rate: 0.0005
- Discount factor (gamma): 0.99
- Initial exploration rate (epsilon): 1.0
- Minimum exploration rate: 0.01
- Exploration decay: 0.995
- Replay memory size: 20000
- Batch size: 64
- Target network update frequency: 100 steps

### Difficulty Progression
The game features three difficulty stages:
1. Stage 1 (0-30 points):
   - Speed increases from 0% to 30%
   - Gap decreases from 0% to 10%
2. Stage 2 (30-50 points):
   - Speed increases from 30% to 60%
   - Gap remains at 10%
3. Stage 3 (50-60 points):
   - Speed remains at 60%
   - Gap decreases from 10% to 25%

## Future Improvements

1. Add more sophisticated state representation
2. Implement Double DQN or Dueling DQN
3. Add prioritized experience replay
4. Implement frame stacking for temporal information
5. Add visualizations of the agent's decision-making process
6. Support for custom game parameters
7. Add unit tests
8. Implement parallel training with multiple environments

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details. 