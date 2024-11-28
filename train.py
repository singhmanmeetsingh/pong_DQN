import gymnasium as gym
import numpy as np
import tensorflow as tf
import random
from models.dqn_agent import DQNAgent
from assignment3_utils import process_frame, transform_reward
import gc
import os

def get_latest_model():
    model_dir = 'saved_models'
    if not os.path.exists(model_dir):
        return None
    models = [f for f in os.listdir(model_dir) if f.endswith('.h5')]
    if not models:
        return None
    latest_model = max(models, key=lambda x: int(x.split('_')[1].split('.')[0]))
    return os.path.join(model_dir, latest_model)

def train(episodes=1000, render=False):
    env = gym.make('PongDeterministic-v4', 
                  render_mode='human' if render else None)
    agent = DQNAgent()
    
    print("TensorBoard logs will be saved in:", agent.log_dir)
    
    try:
        for episode in range(episodes):
            state, _ = env.reset()
            state = process_frame(state, agent.image_shape)
            state = np.squeeze(state, axis=0)
            
            total_reward = 0
            done = False
            steps = 0
            
            # Memory cleanup every 10 episodes
            # if episode % 10 == 0:
            #     gc.collect()
            #     tf.keras.backend.clear_session()
            
            while not done:
                if render:
                    env.render()
                
                action = agent.choose_action(state)
                next_state, reward, done, truncated, _ = env.step(action)
                next_state = process_frame(next_state, agent.image_shape)
                next_state = np.squeeze(next_state, axis=0)
                reward = transform_reward(reward)
                
                agent.remember(state, action, reward, next_state, done)
                loss = agent.train_step()
                
                state = next_state
                total_reward += reward
                steps += 1
                
                if done or truncated:
                    break
            
            # Update epsilon
            agent.epsilon = max(agent.epsilon_min, agent.epsilon * agent.epsilon_decay)
            
            # Update target network periodically
            if episode % agent.update_target_every == 0:
                agent.update_target_network()
            
            # Update episode count
            agent.episode_count = episode
            
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

def resume_training(model_path, episodes=1000, render=False):
    env = gym.make('PongDeterministic-v4', 
                  render_mode='human' if render else None)
    agent = DQNAgent()
    
    # Load the saved model and get starting episode
    print(f"Loading model from: {model_path}")
    agent.load_model(model_path)
    start_episode = agent.episode_count
    
    print(f"Starting resume from episode {start_episode}")
    print("TensorBoard logs will be saved in:", agent.log_dir)
    
    try:
        for episode in range(start_episode, start_episode + episodes):
            state, _ = env.reset()
            state = process_frame(state, agent.image_shape)
            state = np.squeeze(state, axis=0)
            
            total_reward = 0
            done = False
            steps = 0
            
            # Memory cleanup every 10 episodes
            if episode % 10 == 0:
                gc.collect()
                tf.keras.backend.clear_session()
            
            while not done:
                if render:
                    env.render()
                
                action = agent.choose_action(state)
                next_state, reward, done, truncated, _ = env.step(action)
                next_state = process_frame(next_state, agent.image_shape)
                next_state = np.squeeze(next_state, axis=0)
                reward = transform_reward(reward)
                
                agent.remember(state, action, reward, next_state, done)
                loss = agent.train_step()
                
                state = next_state
                total_reward += reward
                steps += 1
                
                if done or truncated:
                    break
            
            # Update epsilon
            agent.epsilon = max(agent.epsilon_min, agent.epsilon * agent.epsilon_decay)
            
            # Update target network periodically
            if episode % agent.update_target_every == 0:
                agent.update_target_network()
            
            # Update episode count
            agent.episode_count = episode
            
            # Log episode metrics
            with agent.summary_writer.as_default():
                tf.summary.scalar('episode_reward', total_reward, step=episode)
                tf.summary.scalar('epsilon', agent.epsilon, step=episode)
                tf.summary.scalar('steps', steps, step=episode)
            
            print(f"\nEpisode: {episode} (Resumed from {start_episode})")
            print(f"Score: {total_reward}")
            print(f"Epsilon: {agent.epsilon:.3f}")
            print(f"Steps: {steps}")
            
            # Save model periodically
            if episode > 0 and episode % 50 == 0:
                agent.save_model(f"resumed_episode_{episode}")
                gc.collect()
                tf.keras.backend.clear_session()
        
    except KeyboardInterrupt:
        print("\nTraining interrupted!")
        agent.save_model("interrupted_resumed_training")
    
    finally:
        env.close()
    
    return agent
