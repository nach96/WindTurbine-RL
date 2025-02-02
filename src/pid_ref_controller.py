import numpy as np

class PIDRefController(object):
    def __init__(self, kp=1, ki=0.1, kd=0, dt=0.2, max_pitch=np.pi/2):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.max_pitch = max_pitch
        self.max_pitch_der = np.radians(5)
        self.dt = dt
        self.integral = 0
        self.last_error = 0
        self.last_time = 0
        self.last_action = 0

    def step(self, error):        
        self.integral = self.integral + error * self.dt
        derivative = (error - self.last_error) / self.dt
        self.last_error = error
        action = - (self.kp * error + self.ki * self.integral + self.kd * derivative)
        action = self.rate_limiter(action)
        action = np.clip(action, 0, self.max_pitch)
        return action
    
    def rate_limiter(self, action):
        pre = self.last_action
        if (action - pre) > self.max_pitch_der:
            action = pre + self.max_pitch_der
        elif (action - pre) < -self.max_pitch_der:
            action = pre - self.max_pitch_der
        self.last_action = action
        return action
       
