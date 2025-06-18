import sys
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path

# Add project root to Python path
sys.path.append(str(Path(__file__).parent.parent))

from src.rl.agent import DQNAgent
from src.rl.environment import FlappyBirdEnv
from src.utils.constants import *

def train(episodes=2000, save_interval=200):
    """Train DQN agent on Flappy Bird environment."""
    env = FlappyBirdEnv()
    agent = DQNAgent(
        state_size=env.state_size,
        action_size=env.action_size,
        learning_rate=0.001,
        gamma=0.99,
        epsilon=1.0,
        epsilon_min=0.01,
        epsilon_decay=0.995
    )

    # Training metrics
    scores = []
    max_scores = []

    # Initialize directories
    models_dir = Path(__file__).parent.parent / 'models'
    models_dir.mkdir(exist_ok=True)
    plots_dir = Path(__file__).parent.parent / 'plots'
    plots_dir.mkdir(exist_ok=True)

    print("Starting training...")
    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        done = False
        episode_max_score = 0
        last_score = 0

        while not done:
            action = agent.act(state)
            next_state, reward, done = env.step(action)
            
            # Reward shaping
            current_score = env.game.score
            if current_score > last_score:
                reward = 1.0
                last_score = current_score
            elif done:
                reward = -5.0
            else:
                reward = 0.1
                
                # Centering and movement rewards
                if env.game.pipes and env.game.pipes[0]["x"] <= BIRD_X + BIRD_SIZE:
                    pipe = env.game.pipes[0]
                    current_speed, current_gap = env.game._get_current_difficulty()
                    gap_center = pipe["top"] + current_gap/2
                    bird_center = env.game.bird_y + BIRD_SIZE/2
                    distance_from_center = abs(bird_center - gap_center)
                    
                    centering_reward = 0.2 * (1 - distance_from_center/(current_gap/2))
                    reward += centering_reward
                    
                    velocity_reward = 0.1 * (1 - abs(env.game.bird_vel) / 10.0)
                    reward += velocity_reward
                    
                    screen_center = (SCREEN_HEIGHT - GROUND_HEIGHT) / 2
                    height_reward = 0.1 * (1 - abs(bird_center - screen_center) / (SCREEN_HEIGHT/2))
                    reward += height_reward

            agent.memorize(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward

            episode_max_score = max(episode_max_score, env.game.score)
            agent.train_step()

        agent.decay_epsilon()

        scores.append(total_reward)
        max_scores.append(episode_max_score)

        if (episode + 1) % 50 == 0:
            avg_last_50 = np.mean(scores[-50:])
            print(f'Episode {episode + 1}/{episodes}')
            print(f'Score: {total_reward:.2f}')
            print(f'Average Score (last 50): {avg_last_50:.2f}')
            print(f'Epsilon: {agent.epsilon:.3f}')
            print('-' * 50)

        # Save model checkpoint
        if (episode + 1) % save_interval == 0:
            model_path = models_dir / f'flappy_bird_model_episode_{episode + 1}.pth'
            torch.save({
                'episode': episode,
                'model_state_dict': agent.policy_net.state_dict(),
                'optimizer_state_dict': agent.optimizer.state_dict(),
                'epsilon': agent.epsilon,
                'scores': scores,
                'max_scores': max_scores
            }, model_path)
            print(f'Model saved to {model_path}')

    # Plot training metrics
    plt.figure(figsize=(15, 5))
    window = 20
    avg_scores = [np.mean(scores[max(0, i-window+1):i+1]) for i in range(len(scores))]

    plt.plot(scores, label='Score', alpha=0.3)
    plt.plot(avg_scores, label=f'{window}-Episode Moving Average', linewidth=2)
    plt.xlabel('Episode')
    plt.ylabel('Score')
    plt.title('Training Progress - Rewards')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(plots_dir / 'training_progress.png')
    plt.close()

    # Save final model
    final_model_path = models_dir / 'flappy_bird_model_final.pth'
    torch.save({
        'episode': episodes,
        'model_state_dict': agent.policy_net.state_dict(),
        'optimizer_state_dict': agent.optimizer.state_dict(),
        'epsilon': agent.epsilon,
        'scores': scores,
        'max_scores': max_scores
    }, final_model_path)
    print(f'Final model saved to {final_model_path}')

if __name__ == '__main__':
    train()
