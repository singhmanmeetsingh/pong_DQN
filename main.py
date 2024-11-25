import argparse
import os
import numpy as np
import tensorflow as tf
import random
from train import train
from evaluate import evaluate_model

def main():
    parser = argparse.ArgumentParser(description='DQN for Pong')
    
    # Mode selection
    parser.add_argument('--mode', type=str, default='train_and_evaluate',
                       choices=['train', 'evaluate', 'train_and_evaluate'],
                       help='Mode to run the agent in')
    
    # Model path for evaluation
    parser.add_argument('--model_path', type=str,
                       help='Path to saved model for evaluation')
    
    # Training/Evaluation settings
    parser.add_argument('--episodes', type=int, default=1000,
                       help='Number of episodes for training or evaluation')
    parser.add_argument('--render', action='store_true',
                       help='Render the environment during training/evaluation')
    
    # Additional settings
    parser.add_argument('--save_freq', type=int, default=50,
                       help='Save model every N episodes')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--gpu', type=int, default=0,
                       help='GPU device to use (default: 0)')
    
    args = parser.parse_args()
    
    # Set GPU device
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    
    # Set random seeds
    np.random.seed(args.seed)
    random.seed(args.seed)
    tf.random.set_seed(args.seed)
    
    if args.mode in ['train', 'train_and_evaluate']:
        print("\n=== Starting Training ===")
        print(f"Episodes: {args.episodes}")
        print(f"Render: {args.render}")
        print(f"Save Frequency: {args.save_freq}")
        print(f"Random Seed: {args.seed}")
        print(f"GPU Device: {args.gpu}")
        
        try:
            agent = train(episodes=args.episodes, render=args.render)
            print("\nTraining completed successfully!")
        except KeyboardInterrupt:
            print("\nTraining interrupted by user!")
            return
        except Exception as e:
            print(f"\nError during training: {e}")
            return
    
    if args.mode in ['evaluate', 'train_and_evaluate']:
        print("\n=== Starting Evaluation ===")
        
        if args.mode == 'evaluate':
            if args.model_path is None:
                print("Error: Must provide model_path for evaluation mode!")
                return
            model_path = args.model_path
        else:
            # Use the latest model from training
            try:
                model_files = [f for f in os.listdir('saved_models') 
                             if f.endswith('.h5')]
                if not model_files:
                    print("Error: No saved models found!")
                    return
                model_path = os.path.join('saved_models', max(model_files))
            except Exception as e:
                print(f"Error finding latest model: {e}")
                return
        
        print(f"Model Path: {model_path}")
        print(f"Evaluation Episodes: {args.episodes}")
        print(f"Render: {args.render}")
        
        try:
            metrics = evaluate_model(
                model_path, 
                num_episodes=min(args.episodes, 100),  # Cap evaluation episodes
                render=args.render
            )
            
            print("\nEvaluation completed successfully!")
            print("\nFinal Metrics:")
            for key, value in metrics.items():
                print(f"{key}: {value:.2f}")
        except Exception as e:
            print(f"\nError during evaluation: {e}")
            return

if __name__ == "__main__":
    print("\nDQN Pong Training/Evaluation")
    print("=" * 30)
    main()
