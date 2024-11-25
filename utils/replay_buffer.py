from collections import deque
import random
import numpy as np

class ReplayBuffer:
    def __init__(self, capacity=15000):  # Increased from 10000 but less than 50000
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        # Convert states to float32 to save memory
        state = np.asarray(state, dtype=np.float32)
        next_state = np.asarray(next_state, dtype=np.float32)
        reward = float(reward)  # Ensure reward is float
         # Remove old items if at capacity
        if len(self.buffer) == self.buffer.maxlen:
            self.buffer.popleft()

        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)
    
    def __len__(self):
        return len(self.buffer)
