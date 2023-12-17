from fast_gym_1 import FastGym_1
import os
import numpy as np
from stable_baselines3 import SAC
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator


def name_date(name,extension=".csv"):
    from datetime import datetime
    now = datetime.now()
    date_time = now.strftime("%m_%d_%Y_%H_%M_%S")
    file_name = name + "_" + date_time + extension
    return file_name

file_Path = os.path.join(os.path.dirname(__file__), "log_models/model_1_4_10_22_2023_12_15_34")
print(file_Path)
model = SAC.load(file_Path)

pitch_log = []
error_wg_log = []
action_log = []
Vx_log = []

Vx = 18 #m/s
pitch_linespace = np.linspace(5, 45, 100)
error_wg_linespace = np.linspace(-10, 10, 40)

for pitch in pitch_linespace:
    for error_wg in error_wg_linespace:
    #for Vx in Vx_linespace:
        obs=[error_wg,pitch,Vx]   
        action, _state = model.predict(obs, deterministic=True)
        pitch_log.append(pitch)
        error_wg_log.append(error_wg)
        action_log.append(action[0])
        #Vx_log.append(Vx)



X = np.array(pitch_log)
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
#ax.set_xlabel('error_wg')
ax.set_xlabel('Vx')
ax.set_ylabel('pitch')

plt.show()