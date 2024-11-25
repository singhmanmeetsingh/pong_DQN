import tensorflow as tf
import numpy as np
from datetime import datetime

class DQNTensorBoard:
    def __init__(self, log_dir):
        current_time = datetime.now().strftime('%Y%m%d-%H%M%S')
        self.train_log_dir = f'{log_dir}/train_{current_time}'
        self.eval_log_dir = f'{log_dir}/eval_{current_time}'
        
        self.train_summary_writer = tf.summary.create_file_writer(self.train_log_dir)
        self.eval_summary_writer = tf.summary.create_file_writer(self.eval_log_dir)
    
    def log_training(self, metrics, step):
        with self.train_summary_writer.as_default():
            for name, value in metrics.items():
                tf.summary.scalar(f'train/{name}', value, step=step)
    
    def log_evaluation(self, metrics, step):
        with self.eval_summary_writer.as_default():
            for name, value in metrics.items():
                tf.summary.scalar(f'eval/{name}', value, step=step)
    
    def log_weights(self, model, step):
        with self.train_summary_writer.as_default():
            for layer in model.layers:
                weights = layer.get_weights()
                if weights:
                    tf.summary.histogram(f'weights/{layer.name}', weights[0], step=step)
                    if len(weights) > 1:  # Has biases
                        tf.summary.histogram(f'biases/{layer.name}', weights[1], step=step)
    
    def log_gradients(self, gradients, step):
        with self.train_summary_writer.as_default():
            for grad, var in gradients:
                if grad is not None:
                    tf.summary.histogram(f'gradients/{var.name}', grad, step=step)
    
    def log_action_distribution(self, actions, step):
        with self.train_summary_writer.as_default():
            for action in range(6):  # 6 actions in Pong
                action_freq = np.mean(np.array(actions) == action)
                tf.summary.scalar(f'actions/action_{action}_frequency', action_freq, step=step)
