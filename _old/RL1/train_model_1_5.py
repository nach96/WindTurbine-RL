from fast_gym_1 import FastGym_1
import pandas as pd
import os
import numpy as np
#Stable-baselines3
from stable_baselines3 import TD3
from stable_baselines3.common.noise import NormalActionNoise
import beepy
from stable_baselines3.common.logger import configure
import sys

#ugly trick to make excep keyboardinterrupt work
import os
os.environ['FOR_DISABLE_CONSOLE_CTRL_HANDLER'] = '1'

#KEY PARAMETERS
MODEL_NUMBER = 5
TRAINING_TIME = 5e3 #s
MAX_T = 100 #seconds per epoch
model_number_str = str(MODEL_NUMBER)
total_timesteps = int(TRAINING_TIME/0.01)

def name_date(name,extension=".csv"):
    from datetime import datetime
    now = datetime.now()
    date_time = now.strftime("%m_%d_%Y_%H_%M_%S")
    file_name = name + "_" + date_time + extension
    return file_name

def log_and_exit(model, env):
    model_Path = os.path.join(os.path.dirname(__file__), ("log_models/model_1_"+model_number_str))
    model.save(name_date(model_Path,""))
    beepy.beep(sound=5)

    # Plot Logged variables during training
    log_Path = os.path.join(os.path.dirname(__file__), ("log_trains/model_1_"+model_number_str))
    log_df = pd.DataFrame(env.myLog)
    log_df.to_csv(name_date(log_Path), sep=',', encoding='utf-8')

#FAST Init Parameters
input_file_rel_path = "..\\FAST_cfg\\IEA-15-240-RWT-Monopile.fst"
input_file_Path = os.path.join(os.path.dirname(__file__), input_file_rel_path)
Pg_nom = 15000 #kw
wg_nom = 7.56 #rpm
Tem_ini = 19786767.5 #Nm
Pitch_ini = 15.55 #deg


env = FastGym_1(inputFileName=input_file_Path, max_time=MAX_T, Tem_ini=Tem_ini, Pitch_ini=Pitch_ini, wg_nom=wg_nom, pg_nom=Pg_nom, enable_myLog=True, myLogName="td3_1")

#action_noise = NormalActionNoise(mean=np.zeros(1), sigma=0.1 * np.ones(1))
policy_kwargs = dict(net_arch=[32, 32])
model = TD3("MlpPolicy", env, verbose=1,action_noise=None,learning_starts=int(100),gamma=0.99, gradient_steps=1,
            train_freq=((1, "step")),learning_rate=3e-4,buffer_size=int(1e6), policy_delay=2,batch_size=256,policy_kwargs=policy_kwargs)

log_Path = os.path.join(os.path.dirname(__file__), ("log_trains/model_1_"+model_number_str))
new_logger = configure(log_Path, ["stdout", "csv", "tensorboard"])
model.set_logger(new_logger)

try:
    print("Learning")
    model.learn(total_timesteps=total_timesteps,log_interval=1,tb_log_name=("1_"+model_number_str+"_log"))
    log_and_exit(model, env)
except KeyboardInterrupt:
    print("Hey")
    log_and_exit(model, env)
    sys.exit(1)
    
