from fast_gym_1 import FastGym_1
import pandas as pd
import os
import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.noise import NormalActionNoise
import beepy
from stable_baselines3.common.logger import configure

def name_date(name,extension=".csv"):
    from datetime import datetime
    now = datetime.now()
    date_time = now.strftime("%m_%d_%Y_%H_%M_%S")
    file_name = name + "_" + date_time + extension
    return file_name

#Parameters
input_file_rel_path = "..\\FAST_cfg\\IEA-15-240-RWT-Monopile.fst"
input_file_Path = os.path.join(os.path.dirname(__file__), input_file_rel_path)

MAX_T = 100 #seconds
Pg_nom = 15000 #kw
wg_nom = 7.56 #rpm
Tem_ini = 19786767.5 #Nm
Pitch_ini = 15.55 #deg

env = FastGym_1(inputFileName=input_file_Path, max_time=MAX_T, Tem_ini=Tem_ini, Pitch_ini=Pitch_ini, wg_nom=wg_nom, pg_nom=Pg_nom, enable_myLog=True, myLogName="td3_1")

#action_noise = NormalActionNoise(mean=np.zeros(1), sigma=0.1 * np.ones(1))
policy_kwargs = dict(net_arch=[32, 32])
model = SAC("MlpPolicy", env, verbose=1,learning_starts=int(1000),gamma=0.98, gradient_steps=-1,
            train_freq=((100, "step")),learning_rate=1e-3,buffer_size=int(1e5), batch_size=256,policy_kwargs=policy_kwargs)

log_Path = os.path.join(os.path.dirname(__file__), "log_trains/model_1_4")
new_logger = configure(log_Path, ["stdout", "csv", "tensorboard"])
model.set_logger(new_logger)

model.learn(total_timesteps=int(1e5),log_interval=1,tb_log_name="1_4_log")

model_Path = os.path.join(os.path.dirname(__file__), "log_models/model_1_4")
model.save(name_date(model_Path,""))
beepy.beep(sound=5)

# Plot Logged variables during training

log_df = pd.DataFrame(env.myLog)
log_df.to_csv(name_date(log_Path), sep=',', encoding='utf-8')


