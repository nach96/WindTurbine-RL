from model_parser import ModelParser
import logging

logging.basicConfig(level=logging.INFO)

# SimpleWT_gym
#Configuration Files
json_rel_path = "../RL_cfg/params_model_5_9_simpleWT_DDPG.json"
model_rel_path = "../Logs/log_models/model_3_9_11_25_2024_19_00_41"
input_file_rel_path = ""

mp = ModelParser(json_rel_path,input_file_rel_path)
model = mp.load_model(model_rel_path)

observation = mp.env.reset()
terminated=False
i = 0
while (terminated==False):
    action, _state = mp.model.predict(observation, deterministic=True)
    try:
        observation, reward, terminated, extra = mp.env.step(action)
    except:
        print("Error performing step")
        terminated=True
        break

log_id = "Sim"+mp.gym_ID+"_"+mp.model_ID
mp.h.log_and_exit(mp.model, mp.env, log_id)
