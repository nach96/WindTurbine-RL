from model_parser import ModelParser
import logging

logging.basicConfig(level=logging.INFO)

# SimpleWT_gym
#Configuration Files
json_rel_path = "../RL_cfg/params_model_3_9_simpleWT_DDPG.json"
model_rel_path = "../Logs/log_models/model_3_9_11_17_2024_11_36_15"
input_file_rel_path = ""

mp = ModelParser(json_rel_path,input_file_rel_path)
mp.load_model(model_rel_path)
mp.learn()