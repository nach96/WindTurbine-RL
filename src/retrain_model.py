from model_parser import ModelParser
import logging

logging.basicConfig(level=logging.INFO)

# SimpleWT_gym
#Configuration Files
json_rel_path = "../RL_cfg/params_model_8_9_simpleWT_DDPG_retrain.json"
model_rel_path = "../Logs/log_models/model_8_9_12_14_2024_17_30"
input_file_rel_path = ""

mp = ModelParser(json_rel_path,input_file_rel_path)
mp.load_model(model_rel_path)
mp.learn()