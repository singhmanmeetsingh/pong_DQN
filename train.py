import gymnasium as gym
import numpy as np
import tensorflow as tf
import random
from models.dqn_agent import DQNAgent
from assignment3_utils import process_frame, transform_reward
import gc

def train(episodes=1000, render=False):
    env = gym.make('PongDeterministic-v4', 
                  render_mode='human' if render else None)
    agent = DQNAgent()
    
    print("TensorBoard logs will be saved in:", agent.log_dir)
    
    try:
        for episode in range(episodes):
            state, _ = env.reset()
            state = process_frame(state, agent.image_shape)
            state = np.squeeze(state, axis=0)  # Remove batch dimension
            
            total_reward = 0
            done = False
            steps = 0

            #  # Add memory cleanup every 5 episodes
            # if episode % 5 == 0:
            #     # Clear TensorFlow backend
            #     tf.keras.backend.clear_session()
            #     # Force garbage collection
            #     gc.collect()
            
            while not done:
                if render:
                    env.render()
                
                # Choose action
                action = agent.choose_action(state)
                
                # Take action
                next_state, reward, done, truncated, _ = env.step(action)
                next_state = process_frame(next_state, agent.image_shape)
                next_state = np.squeeze(next_state, axis=0)
                reward = transform_reward(reward)
                
                # Store experience
                agent.remember(state, action, reward, next_state, done)
                
                # Train
                loss = agent.train_step()
                
                # Update state and metrics
                state = next_state
                total_reward += reward
                steps += 1
                
                # Update epsilon
                # agent.epsilon = max(agent.epsilon_min, 
                                 # agent.epsilon * agent.epsilon_decay)
                
                if done or truncated:
                    break
            
                # Update epsilon
            agent.epsilon = max(agent.epsilon_min, agent.epsilon * agent.epsilon_decay)
            # Update target network periodically
            if episode % agent.update_target_every == 0:
                agent.update_target_network()
            
            # Log episode metrics
            with agent.summary_writer.as_default():
                tf.summary.scalar('episode_reward', total_reward, step=episode)
                tf.summary.scalar('epsilon', agent.epsilon, step=episode)
                tf.summary.scalar('steps', steps, step=episode)
            
            print(f"\nEpisode: {episode}")
            print(f"Score: {total_reward}")
            print(f"Epsilon: {agent.epsilon:.3f}")
            print(f"Steps: {steps}")
            
            # Save model periodically
            if episode > 0 and episode % 50 == 0:
                agent.save_model(f"episode_{episode}")
                gc.collect()
                tf.keras.backend.clear_session()
        
    except KeyboardInterrupt:
        print("\nTraining interrupted!")
        agent.save_model("interrupted_training")
    
    finally:
        env.close()
    
    return agent

if __name__ == "__main__":
    train(episodes=1000, render=True)
