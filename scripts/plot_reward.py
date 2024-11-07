#Script to load a csv file and plot the reward

import pandas as pd
import matplotlib.pyplot as plt
import os

file_path = './WindTurbine-RL/Logs/log_trains/model_2_6_11_07_2024_19_24_22.csv'

os.path.exists(file_path)
df = pd.read_csv(file_path)

#Plot df.w
#plt.plot(df.w)
#plt.show()

#Iterate through df.w and calculate the sum of rewards. Reward is 40-df.w[i]
#Restart the sum each episode. Check the episode with df.ti (when it changes to 0, it is a new episode)
def calculate_episode_reward(df):
    sum = 0
    episode=1
    print("Episode", "Step", "Reward")
    for i in range(len(df.w)):
        if df.time[i] >= 40: #End of episode
            print(episode, i, sum)
            sum = 0
            episode += 1
        sum += -(40 - df.w[i])**2
    return sum
    


calculate_episode_reward(df)
    