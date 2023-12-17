from fast_gym_1 import FastGym_1
import os
import numpy as np
#Stable-baselines3
from stable_baselines3 import TD3
from stable_baselines3.common.noise import NormalActionNoise
import beepy
from stable_baselines3.common.logger import configure
import sys
from model_parser import ModelParser
from model_parser import HelperOFRL

#Configuration Files
json_rel_path = "model_hyperparams/params_model_1_5.json"
input_file_rel_path = "..\\FAST_cfg\\IEA-15-240-RWT-Monopile.fst"

mp = ModelParser(json_rel_path,input_file_rel_path)
h = HelperOFRL()

env = FastGym_1(inputFileName=h.get_file_path(input_file_rel_path), **mp.FAST_params, enable_myLog=True, myLogName="td3_1")

#action_noise = NormalActionNoise(mean=np.zeros(1), sigma=0.1 * np.ones(1))
model = TD3("MlpPolicy", env, **mp.model_params, policy_kwargs=mp.net_kwargs)

#Logger
log_Path = h.get_file_path("log_trains/model_1_"+mp.model_ID)
new_logger = configure(log_Path, ["stdout", "csv", "tensorboard"])
model.set_logger(new_logger)

#Learn
try:
    print("LEARNING")
    model.learn(total_timesteps=mp.total_timesteps,log_interval=1,tb_log_name=("1_"+mp.model_ID+"_log"))
    h.log_and_exit(model, env, mp.model_ID)
except KeyboardInterrupt:
    print("EXIT")
    h.log_and_exit(model, env, mp.model_ID)
    sys.exit(1)
    
