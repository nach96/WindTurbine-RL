import pandas as pd
import os
import beepy
import json
from stable_baselines3 import TD3, SAC, PPO
from stable_baselines3.common.logger import configure
import sys
import importlib

#Do this in a function
#from fast_gym_1 import FastGym_1


os.environ['FOR_DISABLE_CONSOLE_CTRL_HANDLER'] = '1'

#Class to parse the JSON file
class ModelParser():
    def __init__(self, json_rel_path, input_file_rel_path):
        self.parse_json(json_rel_path)
        self.h = HelperOFRL()
        self.setup_env(input_file_rel_path)
        self.setup_model()
        self.setup_logger()
        self.learn()

    def setup_env(self,input_file_rel_path):
        #import fast_gym_1 from string "fast_gym_1"
        
        gym_file_name = "fast_gym_"+self.gym_ID
        gym_class_name = "FastGym_"+self.gym_ID

        gym_class = self.import_from(gym_file_name, gym_class_name)

        self.env = gym_class(inputFileName=self.h.get_file_path(input_file_rel_path), **self.FAST_params, enable_myLog=True, myLogName="td3_1")

    def import_from(self, file_name, class_name):
        try:
            module = importlib.import_module(file_name)
        except ImportError:
            print("ERROR: Gym not recognized", file_name)
            sys.exit(1)
        return getattr(module, class_name)
    
    #Setup model from algorighm and hyperparameters
    def setup_model(self):
        if self.RL_model=="TD3":
            self.model = TD3("MlpPolicy", self.env, **self.model_params, policy_kwargs=self.net_kwargs)
        elif self.RL_model=="SAC":
            self.model = SAC("MlpPolicy", self.env, **self.model_params, policy_kwargs=self.net_kwargs)
        elif self.RL_model=="PPO":
            self.model = PPO("MlpPolicy", self.env, **self.model_params, policy_kwargs=self.net_kwargs)
        else:
            print("ERROR: RL model not recognized")
            sys.exit(1)

    def float_to_int(self,in_dict):
        for key, value in in_dict.items():
            if isinstance(value, (int, float)):  # Check if value is a numeric type
                if value % 1 == 0:
                    in_dict[key] = int(value)  # Convert to integer

    def parse_json(self,json_rel_path):
        # Read the JSON file
        # input_file_rel_path = "model_hyperparams/params_model_1_5.json"
        json_file_Path = os.path.join(os.path.dirname(__file__), json_rel_path)

        with open(json_file_Path) as f:
            data = json.load(f)

        self.model_params = data['arguments']
        self.float_to_int(self.model_params)
        
        # Extract the arguments from the JSON data
        self.gym_ID = str(data['gym_ID'])
        self.model_ID = str(data['model_ID'])
        self.RL_model = data['RL_model']
        self.net_size = data["net_size"]
        self.net_kwargs = dict(net_arch=[self.net_size, self.net_size])
        self.FAST_params = data['FAST_params']
        self.training_time = data['training_time']
        self.total_timesteps = int(self.training_time/0.01)

    def setup_logger(self):
        log_Path = self.h.get_file_path("../Logs/log_trains/model_1_"+self.model_ID)
        new_logger = configure(log_Path, ["stdout", "csv", "tensorboard"])
        self.model.set_logger(new_logger)

    def learn(self):
        try:
            print("LEARNING")
            self.model.learn(total_timesteps=self.total_timesteps,log_interval=1,tb_log_name=("1_"+self.model_ID+"_log"))
            self.h.log_and_exit(self.model, self.env, self.model_ID)
        except KeyboardInterrupt:
            print("EXIT")
            self.h.log_and_exit(self.model, self.env, self.model_ID)
            sys.exit(1)
        
class HelperOFRL():
    def name_date(self,name,extension=".csv"):
        from datetime import datetime
        now = datetime.now()
        date_time = now.strftime("%m_%d_%Y_%H_%M_%S")
        file_name = name + "_" + date_time + extension
        return file_name
    
    def get_file_path(self,file_rel_path):
        file_path = os.path.join(os.path.dirname(__file__), file_rel_path)
        return file_path
                                       
    def log_and_exit(self, model, env, id):
        model_Path = self.get_file_path("../Logs/log_models/model_1_"+id)
        model.save(self.name_date(model_Path,""))
        beepy.beep(sound=5)

        # Plot Logged variables during training
        log_Path = self.get_file_path("../Logs/log_trains/model_1_"+id)
        log_df = pd.DataFrame(env.myLog)
        log_df.to_csv(self.name_date(log_Path), sep=',', encoding='utf-8')

