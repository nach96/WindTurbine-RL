from ..src.model_parser import ModelParser
from ..src.pid_ref_controller import PIDRefController
import logging
import numpy as np

logging.basicConfig(level=logging.INFO)

# SimpleWT_gym
#Configuration Files
json_rel_path = "../RL_cfg/params_model_8_9_simpleWT_DDPG.json"
#model_rel_path = "../Logs/log_models/model_8_9_12_14_2024_17_30"
input_file_rel_path = ""

def wind_stair(ts):
    if ts<30:
        wind = 12.3
    elif ts<60:
        wind = 12.4
    elif ts<90:
        wind = 12.5
    else:
        wind = 12.35
    return wind  

def sine_wind(env):
        #Freq 0.5Hz; Amplitude = 0.3
        env.Vx = env.Vx_0 + np.sin(0.02*2*np.pi*env.wt_sim.ti)*0.15
        return env.Vx

mp = ModelParser(json_rel_path,input_file_rel_path)
pid = PIDRefController(kp=1, ki=0.1)

observation = mp.env.reset()
terminated=False
i = 0
while (terminated==False):
    #wind = wind_stair(mp.env.wt_sim.ti)
    wind = sine_wind(mp.env)
    observation[2] = wind
    mp.env.Vx=wind
    error = observation[0]
    pitch = pid.step(error)
    try:
        action = [pitch, wind]
        observation, reward, terminated, extra = mp.env.control_step(action)
    except:
        print("Error performing step")
        terminated=True
        break

log_id = "Sim_PID"
mp.h.log_and_exit(mp.model, mp.env, log_id)
