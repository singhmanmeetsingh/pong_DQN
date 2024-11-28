import argparse
import os
import numpy as np
import tensorflow as tf
import random
from train import train, resume_training, get_latest_model
from evaluate import evaluate_model

def setup_environment(args):
    """Setup GPU and random seeds"""
    print("\n=== Setting Up Environment ===")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    
    # Set random seeds
    np.random.seed(args.seed)
    random.seed(args.seed)
    tf.random.set_seed(args.seed)
    
    print(f"GPU Device: {args.gpu}")
    print(f"Random Seed: {args.seed}")
    
    # Check GPU availability
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"Available GPUs: {len(gpus)}")
        for gpu in gpus:
            print(f"  {gpu}")
    else:
        print("No GPU available, using CPU")

def handle_training(args):
    """Handle training mode"""
    if args.resume:
        print("\n=== Resuming Training ===")
        if args.resume.lower() == 'latest':
            latest_model = get_latest_model()
            if latest_model:
                print(f"Found latest model: {latest_model}")
                args.resume = latest_model
            else:
                print("No saved models found, starting new training")
                args.resume = None
        
        if args.resume:
            print(f"Loading model from: {args.resume}")
            return resume_training(args.resume, episodes=args.episodes, render=args.render)
    
    print("\n=== Starting New Training ===")
    print(f"Episodes: {args.episodes}")
    print(f"Render: {args.render}")
    print(f"Save Frequency: {args.save_freq}")
    
    try:
        agent = train(episodes=args.episodes, render=args.render)
        print("\nTraining completed successfully!")
        return agent
    except KeyboardInterrupt:
        print("\nTraining interrupted by user!")
        return None
    except Exception as e:
        print(f"\nError during training: {e}")
        return None

def handle_evaluation(args, previous_training=False):
    """Handle evaluation mode"""
    print("\n=== Starting Evaluation ===")
    
    # Determine which model to evaluate
    if args.mode == 'evaluate' and args.model_path is None:
        print("Error: Must provide model_path for evaluation mode!")
        return
    
    if previous_training:
        try:
            latest_model = get_latest_model()
            if not latest_model:
                print("Error: No saved models found!")
                return
            model_path = latest_model
        except Exception as e:
            print(f"Error finding latest model: {e}")
            return
    else:
        model_path = args.model_path
    
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

def main():
    parser = argparse.ArgumentParser(description='DQN for Pong')
    
    # Mode selection
    parser.add_argument('--mode', type=str, default='train_and_evaluate',
                       choices=['train', 'evaluate', 'train_and_evaluate'],
                       help='Mode to run the agent in')
    
    # Add resume training option
    parser.add_argument('--resume', type=str,
                       help='Resume training from saved model or "latest"')
    
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
    
    # Setup environment (GPU and seeds)
    setup_environment(args)
    
    # Handle different modes
    if args.mode in ['train', 'train_and_evaluate']:
        agent = handle_training(args)
        if agent is None:
            return
    
    if args.mode in ['evaluate', 'train_and_evaluate']:
        handle_evaluation(args, previous_training=(args.mode == 'train_and_evaluate'))

if __name__ == "__main__":
    print("\nDQN Pong Training/Evaluation")
    print("=" * 30)
    main()
