import os
import numpy as np
from stable_baselines3 import SAC
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator

from model_parser import ModelParser
import logging

logging.basicConfig(level=logging.INFO)

# SimpleWT_gym
#Configuration Files
json_rel_path = "../RL_cfg/params_model_3_9_simpleWT_DDPG.json"
model_rel_path = "../Logs/log_models/model_3_9_11_25_2024_19_00_41"
input_file_rel_path = ""

mp = ModelParser(json_rel_path,input_file_rel_path)
model = mp.load_model(model_rel_path)


pitch_inc_log = []
error_wg_log = []
action_log = []
Vx_log = []

Vx = 12.3 #m/s
pitch_linespace = np.linspace(0, np.pi/2, 100)
error_wg_linespace = np.linspace(-10, 10, 40)

for pitch in pitch_linespace:
    for error_wg in error_wg_linespace:
    #for Vx in Vx_linespace:
        pitch_ref = pitch
        obs=[error_wg,pitch,Vx,pitch_ref]   
        action, _state = model.predict(obs, deterministic=True)
        pitch_inc_log.append(pitch)
        error_wg_log.append(error_wg)
        action_log.append(action[0])
        #Vx_log.append(Vx)



X = np.array(pitch_inc_log)
Y = np.array(error_wg_log)
#Y = np.array(Vx_log)
Z = np.array(action_log)


# Plot surface
x = np.reshape(X, (100, 40))
y = np.reshape(Y, (100, 40))
z = np.reshape(Z, (100, 40))

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.plot_surface(x, y, z,cmap=cm.coolwarm)

ax.set_xlabel('Pitch')
ax.set_ylabel('error_wg')
#ax.set_ylabel('Vx')
ax.set_zlabel('action')

plt.show()


fig, ax = plt.subplots()

CS=ax.contour(y,x,z)
ax.clabel(CS, inline=True, fontsize=10)
ax.grid(True)
ax.set_xlabel('error_wg')
#ax.set_xlabel('Vx')
ax.set_ylabel('pitch')

plt.show()