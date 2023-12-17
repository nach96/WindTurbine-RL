from stable_baselines3.common.callbacks import BaseCallback
import numpy as np


class CustomCallback(BaseCallback):
    """
    A custom callback that derives from ``BaseCallback``.

    :param verbose: Verbosity level: 0 for no output, 1 for info messages, 2 for debug messages
    """
    def __init__(self, verbose=0):
        super(CustomCallback, self).__init__(verbose)
        # Those variables will be accessible in the callback
        # (they are defined in the base class)
        # The RL model
        # self.model = None  # type: BaseAlgorithm
        # An alias for self.model.get_env(), the environment used for training
        # self.training_env = None  # type: Union[gym.Env, VecEnv, None]
        # Number of time the callback was called
        # self.n_calls = 0  # type: int
        # self.num_timesteps = 0  # type: int
        # local and global variables
        # self.locals = None  # type: Dict[str, Any]
        # self.globals = None  # type: Dict[str, Any]
        # The logger object, used to report things in the terminal
        # self.logger = None  # stable_baselines3.common.logger
        # # Sometimes, for event callback, it is useful
        # # to have access to the parent object
        # self.parent = None  # type: Optional[BaseCallback]
        


    def _on_rollout_start(self) -> None:
        """
        A rollout is the collection of environment interaction
        using the current policy.
        This event is triggered before collecting new samples.
        """

        
        pass

    def _on_rollout_end(self) -> None:
        """
        This event is triggered before updating the policy.
        """
        pass

    def log_policy_inspection(model):
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

        