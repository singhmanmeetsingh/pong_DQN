import gymnasium as gym
import numpy as np
from models.dqn_agent import DQNAgent
from assignment3_utils import process_frame
import tensorflow as tf
import ale_py

def evaluate_model(model_path, num_episodes=10, render=True):
   
    # Create environment and agent
    env = gym.make('PongDeterministic-v4', 
                  render_mode='human' if render else None)
    agent = DQNAgent()
    agent.load_model(model_path)
    
    # Set epsilon to minimum for exploitation
    agent.epsilon = agent.epsilon_min
    
    # Metrics storage
    episode_data = []
    all_actions = []
    all_q_values = []
    
    print("\nStarting evaluation...")
    
    for episode in range(num_episodes):
        state, _ = env.reset()
        # Process frame and ensure correct shape (84, 80, 1)
        state = process_frame(state, agent.image_shape)
        state = np.squeeze(state, axis=0)  # Remove batch dimension
        
        episode_reward = 0
        episode_actions = []
        episode_q_values = []
        step_count = 0
        done = False
        truncated = False
        
        while not (done or truncated):
            # Get Q-values for logging
            state_input = np.expand_dims(state, axis=0)  # Add batch dimension
            q_values = agent.main_network.predict(state_input, verbose=0)
            episode_q_values.append(q_values[0])
            
            # Choose and take action
            action = agent.choose_action(state)
            episode_actions.append(action)
            
            next_state, reward, done, truncated, _ = env.step(action)
            next_state = process_frame(next_state, agent.image_shape)
            next_state = np.squeeze(next_state, axis=0)
            
            episode_reward += reward
            state = next_state
            step_count += 1
        
        # Store episode data
        episode_data.append({
            'reward': episode_reward,
            'steps': step_count,
            'q_values': episode_q_values,
            'actions': episode_actions
        })
        
        all_actions.extend(episode_actions)
        all_q_values.extend(episode_q_values)
        
        # Log episode metrics to TensorBoard
        with agent.summary_writer.as_default():
            tf.summary.scalar('eval/episode_reward', episode_reward, step=episode)
            tf.summary.scalar('eval/steps', step_count, step=episode)
            tf.summary.scalar('eval/q_value_mean', np.mean(episode_q_values), step=episode)
            tf.summary.scalar('eval/q_value_max', np.max(episode_q_values), step=episode)
            tf.summary.scalar('eval/q_value_min', np.min(episode_q_values), step=episode)
            tf.summary.scalar('eval/q_value_std', np.std(episode_q_values), step=episode)
            
            # Log action distribution
            for action in range(agent.n_actions):
                action_freq = episode_actions.count(action) / len(episode_actions)
                tf.summary.scalar(f'eval/action_{action}_frequency', action_freq, step=episode)
        
        print(f"Episode {episode}: Score = {episode_reward}, Steps = {step_count}")
    
    # Calculate final statistics
    rewards = [ep['reward'] for ep in episode_data]
    steps = [ep['steps'] for ep in episode_data]
    
    final_metrics = {
        'average_reward': np.mean(rewards),
        'reward_std': np.std(rewards),
        'max_reward': max(rewards),
        'min_reward': min(rewards),
        'average_steps': np.mean(steps),
        'q_value_overall_mean': np.mean(all_q_values),
        'q_value_overall_std': np.std(all_q_values)
    }
    
    # Log final evaluation metrics to TensorBoard
    with agent.summary_writer.as_default():
        for metric_name, metric_value in final_metrics.items():
            tf.summary.scalar(f'eval/final_{metric_name}', metric_value, step=0)
        
        # Log final action distribution
        for action in range(agent.n_actions):
            action_freq = all_actions.count(action) / len(all_actions)
            tf.summary.scalar(f'eval/final_action_{action}_frequency', 
                            action_freq, step=0)
    
    # Print final results
    print("\nEvaluation Results:")
    print(f"Average Score: {final_metrics['average_reward']:.2f} ± "
          f"{final_metrics['reward_std']:.2f}")
    print(f"Best Score: {final_metrics['max_reward']}")
    print(f"Worst Score: {final_metrics['min_reward']}")
    print(f"Average Steps per Episode: {final_metrics['average_steps']:.2f}")
    print(f"Average Q-Value: {final_metrics['q_value_overall_mean']:.2f} ± "
          f"{final_metrics['q_value_overall_std']:.2f}")
    
    env.close()
    return final_metrics

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate DQN on Pong')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to the model file')
    parser.add_argument('--episodes', type=int, default=10,
                       help='Number of episodes to evaluate')
    parser.add_argument('--no-render', action='store_true',
                       help='Disable rendering')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    args = parser.parse_args()
    
    # Set random seeds
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)
    
    evaluate_model(args.model_path, args.episodes, not args.no_render)
