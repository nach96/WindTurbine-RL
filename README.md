# openfast-RL
Reinforcement Learning for for Wind Turbine pitch control.

RL algorithms based on [stable-baselines3](https://stable-baselines.readthedocs.io/en/master/index.html#) 

Trained by interfacing the gym environments:

* [simpleWT-gym](https://github.com/nach96/simpleWT-gym)
* [openfast-gym](https://github.com/nach96/openfast-gym)


## RL_cfg files
The gym environment and the RL model parameters are defined in a json configuration file, placed at /RL_cfg. 

The following parameters are mandatory:

```
{
    "gym_params":{
        "gym_ID": 2,                          # Used for the naming of the Logs
        "gym_package_name": "simpleWT_gym",   # Used to dynamically load the required environment
        "gym_file_name": "simple_wt_gym_2",   # Used to dynamically load the required environment
        "gym_class_name": "SimpleWtGym2",     # Used to dynamically load the required environment
        "Vx": 12.3,                           # Wind speed (if constant)
        "t_max_episode": 100,                 # Maximum episode time [s]
        "wg_nom": 40,                         # Reference rotor speed
        "burn_in_time": 1                     # Time until simulation is reliable
    },
    "RL_model_params":{
        "model_ID": 8,                        # Used for Logs names
        "RL_model": "DDPG",                   # RL Algorithm
        "training_time": 10000,               # Total trining time
        "net_size": 32,                       # Neural network layer size (hardcoded 2 layers MLP policy)
        "noise_std": 0.1,                     # Action noise for exploration in DDPG alg.
        "arguments":{                         # This arguments may vary depending on the RL algorithm
            "learning_starts": 10000,         # Number of steps with random actions before start learning
            "learning_rate": 1e-3,            # Learning rate (gradient descent)
            "gamma": 0.98,                    # Discount factor. How important are future rewards
            "gradient_steps": 1,              
            "train_freq": 1,
            "buffer_size": 1e5,
            "batch_size": 256,
            "verbose":1
        }
    }    
}
```
## Gym arguments
Any gym environment to be called from model_parser shall follow the following arguments:

### SimpleWT
```
def __init__(self,inputFileName="", Vx=18, wg_nom=40, t_max=40, burn_in_time=0, Tem_ini=1.978655e7, Pitch_ini=15.55, pg_nom=1.5e7, logging_level=logging.INFO):

```

### OpenFast
```
def __init__(self, inputFileName="", libraryPath="", max_time=40, Tem_ini=1.978655e7, Pitch_ini=15.55, wg_nom=7.55, pg_nom=1.5e7,enable_myLog=1,myLogName=""):
```

