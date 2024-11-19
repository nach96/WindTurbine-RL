from model_parser import ModelParser
import logging

logging.basicConfig(level=logging.INFO)

# SimpleWT_gym
#Configuration Files
json_rel_path = "../RL_cfg/params_model_3_9_simpleWT_DDPG.json"
#input_file_rel_path = "..\\FAST_cfg\\IEA-15-240-RWT-Monopile.fst"s
input_file_rel_path = ""

mp = ModelParser(json_rel_path,input_file_rel_path)
mp.learn()
