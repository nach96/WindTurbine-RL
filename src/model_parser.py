import pandas as pd
import numpy as np
import os
import beepy
import json
from stable_baselines3 import TD3, SAC, PPO, DDPG
from stable_baselines3.common.logger import configure
from stable_baselines3.common.noise import NormalActionNoise
import sys
import importlib
import logging

os.environ['FOR_DISABLE_CONSOLE_CTRL_HANDLER'] = '1'

#Class to parse the JSON config file and setup the simulation and learning
class ModelParser():
    def __init__(self, json_rel_path, input_file_rel_path):
        self.parse_json(json_rel_path)
        self.h = HelperOFRL()
        self.setup_env(input_file_rel_path)
        self.setup_model()
        self.setup_logger()
        #self.learn()

    def setup_env(self,input_file_rel_path):
        #import gym from gymID        
        gym_class = self.import_from(self.gym_package_name, self.gym_file_name, self.gym_class_name)
        print("Gym imported:", self.gym_file_name, self.gym_class_name)
        #setup environment
        #self.env = gym_class(inputFileName=self.h.get_file_path(input_file_rel_path), **self.FAST_params)
        self.env = gym_class(inputFileName=self.h.get_file_path(input_file_rel_path), Vx=self.gym_vx, t_max=self.gym_t_max_episode, wg_nom=self.gym_wg_nom, burn_in_time=self.burn_in_time)
    def import_from(self, package_name, file_name, class_name):
        try:
            module_name = package_name+"."+file_name
            module = importlib.import_module(module_name)
            env = getattr(module, class_name)
        except ImportError:
            print("ERROR: Gym env file not recognized", file_name)
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
        elif self.RL_model=="DDPG":
            #self.model = DDPG("MlpPolicy", self.env, **self.model_params, policy_kwargs=self.net_kwargs)
            action_noise = NormalActionNoise(mean=np.zeros(1), sigma=self.noise_std * np.ones(1))
            self.model = DDPG("MlpPolicy", self.env, action_noise=action_noise, **self.model_params, policy_kwargs=self.net_kwargs)
        else:
            print("ERROR: RL model not recognized")
            sys.exit(1)

    def load_model(self, model_rel_path):
        model_path = os.path.join(os.path.dirname(__file__), model_rel_path)

        if self.RL_model=="DDPG":
            custom_objects = {'learning_starts': 1}
            #self.model = DDPG.load(model_path, env=self.env, kwargs=dict(learning_starts=1))
            self.model = DDPG.load(model_path, env=self.env, custom_objects=custom_objects)
        else:
            print("ERROR: RL model not ready for loading")
            sys.exit(1)
        return self.model
    def float_to_int(self,in_dict):
        for key, value in in_dict.items():
            if isinstance(value, (int, float)):  # Check if value is a numeric type
                if value % 1 == 0:
                    in_dict[key] = int(value)  # Convert to integer

    #This is 0 robust. If param missing in the cfg file, it breaks. Make it better please
    def parse_json(self,json_rel_path):
        # Read the JSON file
        json_file_Path = os.path.join(os.path.dirname(__file__), json_rel_path)

        with open(json_file_Path) as f:
            data = json.load(f)

        """
        self.model_params = data['arguments']
        self.float_to_int(self.model_params)
        self.gym_ID = str(data['gym_ID'])
        self.model_ID = str(data['model_ID'])
        self.gym_package_name = data['gym_package_name']
        self.gym_file_name = data['gym_file_name']
        self.gym_class_name = data['gym_class_name']
        self.RL_model = data['RL_model']
        self.net_size = data["net_size"]
        self.net_kwargs = dict(net_arch=[self.net_size, self.net_size])
        self.FAST_params = data['FAST_params']
        self.training_time = data['training_time']
        self.total_timesteps = int(self.training_time/0.01)
        self.noise_std = data['noise_std']
        """

        self.set_gym_params(data)
        self.set_model_params(data)      
    
    
    #Function to set a parameter if it exists in data[]
    def set_param(self,param_key,data):
        if param_key in data.keys():
            x = data[param_key]
            return x
        else:
            logging.warning("Parameter "+param_key+" not found in data")
            pass  

    def set_gym_params(self,data):
        gym_params = data['gym_params']
        self.gym_ID = str(self.set_param('gym_ID',gym_params))
        self.gym_package_name = self.set_param('gym_package_name',gym_params)
        self.gym_file_name = self.set_param('gym_file_name',gym_params)
        self.gym_class_name = self.set_param('gym_class_name',gym_params)
        self.gym_vx = self.set_param('Vx',gym_params) 
        self.gym_t_max_episode = self.set_param('t_max_episode',gym_params)
        self.gym_wg_nom = self.set_param('wg_nom',gym_params)
        self.burn_in_time = self.set_param('burn_in_time',gym_params)
        self.control_time_step = self.set_param('control_time_step',gym_params)
        #FAST_params shall contain vx, t_max and wg_nom. Temporary
        self.FAST_params = {'Vx':self.gym_vx,'t_max':self.gym_t_max_episode,'wg_nom':self.gym_wg_nom}

    def set_model_params(self,data):
        RL_model_params = data['RL_model_params']
        self.model_ID = str(self.set_param('model_ID',RL_model_params))
        self.RL_model = self.set_param('RL_model',RL_model_params)
        self.training_time = self.set_param('training_time',RL_model_params)
        self.total_timesteps = int(self.training_time/self.control_time_step)
        self.net_size = self.set_param('net_size',RL_model_params)
        self.net_kwargs = dict(net_arch=self.net_size)
        self.noise_std = self.set_param('noise_std',RL_model_params)
        self.model_params = self.set_param('arguments',RL_model_params)
        self.float_to_int(self.model_params)



    """
    Sets up the logger for the model. The logger is set up to log to stdout, a csv file and
    to a tensorboard log. The log is saved in ../Logs/log_trains/model_<gym_id>_<model_id>
    """
    def setup_logger(self):
        log_id = self.gym_ID+"_"+self.model_ID
        log_Path = self.h.get_file_path("../Logs/log_trains/model_"+log_id)
        new_logger = configure(log_Path, ["stdout", "csv", "tensorboard"])
        self.model.set_logger(new_logger)

    def learn(self,timesteps=None):
        if timesteps is None:
            timesteps = self.total_timesteps

        log_id = self.gym_ID+"_"+self.model_ID
        try:
            logging.info("LEARNING")
            self.model.learn(total_timesteps=timesteps,log_interval=1,tb_log_name=(log_id+"_log"))
            self.h.log_and_exit(self.model, self.env, log_id)
        except KeyboardInterrupt:
            logging.warning("EXIT with KeyboardInterrupt")
            self.h.log_and_exit(self.model, self.env, log_id)
            sys.exit(1)
        except Exception as e:
            logging.warning("EXIT with Exception")
            self.h.log_and_exit(self.model, self.env, log_id)
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
        model_Path = self.get_file_path("../Logs/log_models/model_"+id)
        model.save(self.name_date(model_Path,""))
        beepy.beep(sound=5)

        # Plot Logged variables during training
        log_Path = self.get_file_path("../Logs/log_trains/model_"+id)
        log_df = pd.DataFrame(env.myLog)
        log_df.to_csv(self.name_date(log_Path), sep=',', encoding='utf-8')

