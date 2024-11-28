import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Flatten, Input
from tensorflow.keras.optimizers import Adam
import numpy as np
import os
from datetime import datetime
from utils.replay_buffer import ReplayBuffer
from utils.tensorboard_logger import DQNTensorBoard

class DQNAgent:
    def __init__(self):
        # Hyperparameters
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.995
        # self.batch_size = 8    unccomment it for the testing for 8
        self.batch_size = 16 
        self.update_target_every = 10
        self.learning_rate = 0.00025
        
        # State and action space
        self.image_shape = (84, 80, 1)
        self.n_actions = 6
        
        # Networks
        self.main_network = self._build_network()
        self.target_network = self._build_network()
        self.target_network.set_weights(self.main_network.get_weights())
        
        # Experience replay
        self.memory = ReplayBuffer()
        
        # Training metrics
        self.episode_count = 0
        
        # TensorBoard setup
        current_time = datetime.now().strftime('%Y%m%d-%H%M%S')
        self.log_dir = f'logs/dqn_pong/{current_time}'
        self.summary_writer = tf.summary.create_file_writer(self.log_dir)
        
        # Save directory
        self.save_dir = 'saved_models'
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

    def _build_network(self):
        model = Sequential([
            Input(shape=self.image_shape),
            Conv2D(32, (8, 8), strides=4, activation='relu'),
            Conv2D(64, (4, 4), strides=2, activation='relu'),
            Conv2D(64, (3, 3), strides=1, activation='relu'),
            Flatten(),
            Dense(512, activation='relu'),
            Dense(self.n_actions, activation='linear')
        ])
        
        model.compile(optimizer=Adam(learning_rate=self.learning_rate), 
                     loss='mse')
        return model

    def choose_action(self, state):
        if np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)
            
        state_tensor = np.expand_dims(state, axis=0)
        q_values = self.main_network.predict(state_tensor, verbose=0)
        return np.argmax(q_values[0])

    def train_step(self):
        if len(self.memory) < self.batch_size:
            return 0
        
        # Sample batch
        batch = self.memory.sample(self.batch_size)
        states = np.array([exp[0] for exp in batch])
        actions = np.array([exp[1] for exp in batch])
        rewards = np.array([exp[2] for exp in batch])
        next_states = np.array([exp[3] for exp in batch])
        dones = np.array([exp[4] for exp in batch])
        
        # Get current Q values
        current_q_values = self.main_network.predict(states, verbose=0)
        
        # Get next Q values from target network
        next_q_values = self.target_network.predict(next_states, verbose=0)
        
        # Update Q values
        for i in range(self.batch_size):
            if dones[i]:
                current_q_values[i][actions[i]] = rewards[i]
            else:
                current_q_values[i][actions[i]] = rewards[i] + self.gamma * np.max(next_q_values[i])
        
        # Train the network
        history = self.main_network.fit(states, current_q_values, 
                                      verbose=0, batch_size=self.batch_size)
        loss = history.history['loss'][0]
        
        # Log metrics
        with self.summary_writer.as_default():
            tf.summary.scalar('loss', loss, step=self.episode_count)
            tf.summary.scalar('q_values_mean', np.mean(current_q_values), step=self.episode_count)
            tf.summary.scalar('q_values_max', np.max(current_q_values), step=self.episode_count)
        
        return loss

    def update_target_network(self):
        self.target_network.set_weights(self.main_network.get_weights())

    def remember(self, state, action, reward, next_state, done):
        self.memory.push(state, action, reward, next_state, done)

    def save_model(self, name_prefix):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{name_prefix}_{timestamp}"
        path = os.path.join(self.save_dir, filename)
        
        # Save model weights
        self.main_network.save(f"{path}.h5")
        
        # Save training state
        metrics = {
            'epsilon': self.epsilon,
            'episode': self.episode_count,
            'memory_size': len(self.memory)
        }
        np.save(f"{path}_metrics.npy", metrics)
        print(f"\nModel saved: {path}.h5")
        print(f"Current epsilon: {self.epsilon:.3f}")
        print(f"Episode saved: {self.episode_count}")
        print(f"Memory size: {len(self.memory)}")
    
    def load_model(self, filepath):
        print(f"Loading model from: {filepath}")
        
        # Load model weights
        self.main_network.load_weights(filepath)
        self.target_network.set_weights(self.main_network.get_weights())
        
        # Try to load training state
        try:
            # Look for state file with same name but _metrics.npy instead of .h5
            state_path = filepath.replace('.h5', '_metrics.npy')
            if os.path.exists(state_path):
                state = np.load(state_path, allow_pickle=True).item()
                self.epsilon = state.get('epsilon', self.epsilon)
                self.episode_count = state.get('episode', 0)
                print(f"Restored epsilon: {self.epsilon:.3f}")
                print(f"Restored episode count: {self.episode_count}")
            else:
                print("No state file found, using default values")
                print(f"Using epsilon: {self.epsilon}")
                print(f"Using episode count: {self.episode_count}")
        except Exception as e:
            print(f"Could not load training state: {e}")
            print("Using default values")

    def get_training_state(self):
        """Get current training state"""
        return {
            'epsilon': self.epsilon,
            'episode_count': self.episode_count,
            'memory_size': len(self.memory),
            'learning_rate': self.learning_rate,
            'gamma': self.gamma,
            'batch_size': self.batch_size
        }

    def reset_training_metrics(self):
        """Reset training metrics for new session"""
        current_time = datetime.now().strftime('%Y%m%d-%H%M%S')
        self.log_dir = f'logs/dqn_pong/{current_time}'
        self.summary_writer = tf.summary.create_file_writer(self.log_dir)
        print(f"Reset TensorBoard logs to: {self.log_dir}")
