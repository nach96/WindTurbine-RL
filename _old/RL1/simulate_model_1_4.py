from fast_gym_1 import FastGym_1
import os
import numpy as np
from stable_baselines3 import SAC
import pandas as pd


def name_date(name,extension=".csv"):
    from datetime import datetime
    now = datetime.now()
    date_time = now.strftime("%m_%d_%Y_%H_%M_%S")
    file_name = name + "_" + date_time + extension
    return file_name

#Parameters
input_file_rel_path = "..\\FAST_cfg\\IEA-15-240-RWT-Monopile.fst"
input_file_Path = os.path.join(os.path.dirname(__file__), input_file_rel_path)

MAX_T = 50 #seconds
Pg_nom = 15000 #kw
wg_nom = 7.56 #rpm
Tem_ini = 19786767.5 #Nm
Pitch_ini = 15.55 #deg

env = FastGym_1(inputFileName=input_file_Path, max_time=MAX_T, Tem_ini=Tem_ini, Pitch_ini=Pitch_ini, wg_nom=wg_nom, pg_nom=Pg_nom, enable_myLog=True)

file_Path = os.path.join(os.path.dirname(__file__), "log_models/model_1_4_10_22_2023_12_15_34")
print(file_Path)

model = SAC.load(file_Path,env=env)

observations = []
actions = []
rewards = []
observation = env.reset()
terminated=False
i = 0
while (terminated==False):
    action, _state = model.predict(observation, deterministic=True)
    try:
        observation, reward, terminated, extra = env.step(action)
    except:
        print("Error performing step")
        terminated=True
        break

    #Log data every 10 steps
    if i % 10 == 0:
        observations.append({"error": observation[0]})
        observations.append({"Pitch": observation[1]})
        observations.append({"WindSpeed": observation[2]})
        actions.append({"Pitch action": action[0]})
        rewards.append({"reward": reward})
    i+=1
    if terminated:
        print("Terminated")
        break
#Save results
observations_df = pd.DataFrame(observations)
actions_df = pd.DataFrame(actions)
rewards_df = pd.DataFrame(rewards)
frames = [observations_df, actions_df, rewards_df]
result = pd.concat(frames, axis=1)

sim_Path = os.path.join(os.path.dirname(__file__), "log_sims/model_1_4")


result.to_csv(name_date(sim_Path), sep=',', encoding='utf-8')


