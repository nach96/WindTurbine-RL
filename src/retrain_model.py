from model_parser import ModelParser
import logging

logging.basicConfig(level=logging.INFO)

# SimpleWT_gym
#Configuration Files
json_rel_path = "../RL_cfg/params_model_5_10_simpleWT_DDPG.json"
model_rel_path = "../Logs/log_models/model_5_9_12_09_2024_14_08_52"
input_file_rel_path = ""

mp = ModelParser(json_rel_path,input_file_rel_path)
mp.load_model(model_rel_path)
mp.learn()