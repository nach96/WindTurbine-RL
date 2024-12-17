import numpy as np

class PIDRefController(object):
    def __init__(self, kp=1, ki=0.1, kd=0, dt=0.2, max_pitch=np.pi/2):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.max_pitch = max_pitch
        self.dt = dt
        self.integral = 0
        self.last_error = 0
        self.last_time = 0

    def step(self, error):        
        self.integral = self.integral + error * self.dt
        derivative = (error - self.last_error) / self.dt
        self.last_error = error
        action = self.kp * error + self.ki * self.integral + self.kd * derivative
        action = np.clip(action, 0, self.max_pitch)
        return self.kp * error + self.ki * self.integral + self.kd * derivative
       
