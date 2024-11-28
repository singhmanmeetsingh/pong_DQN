For logs 
$ tensorboard --logdir=logs/dqn_pong

# Train with default settings
 $ python main.py --mode train

# Train with visualization
$ python main.py --mode train --render

# Train with specific number of episodes
$ python main.py --mode train --episodes 2000

# Train with visualization and specific episodes
$ python main.py --mode train --render --episodes 2000

#resume training on model
$ python main.py --mode train --resume saved_models/your_saved_model.h5
 --episodes 800



# Evaluate a specific model
$ python main.py --mode evaluate --model_path saved_models/your_model.h5


# Evaluate with visualization
python main.py --mode evaluate --model_path saved_models/your_model.h5 --render

# Evaluate with specific number of episodes
python main.py --mode evaluate --model_path saved_models/your_model.h5 --episodes 20

# Evaluate without rendering (faster)
python main.py --mode evaluate --model_path saved_models/your_model.h5 --no-render




