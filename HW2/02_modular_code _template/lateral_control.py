import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.interpolate import splprep, splev
from scipy.optimize import minimize
import time


class LateralController:
    '''
    Lateral control using the Stanley controller

    functions:
        stanley 

    init:
        gain_constant (default=5)
        damping_constant (default=0.5)
    '''


    def __init__(self, gain_constant=5, damping_constant=2):

        self.gain_constant = gain_constant
        self.damping_constant = damping_constant
        self.previous_steering_angle = 0


    def stanley(self, waypoints, speed):
        '''
        ##### TODO #####
        one step of the stanley controller with damping
        args:
            waypoints (np.array) [2, num_waypoints]
            speed (float)
        '''

        current_position = np.array([0, 0])

        dx = waypoints[0, 1] - waypoints[0, 0]
        dy = waypoints[1, 1] - waypoints[1, 0]
        desired_heading = np.arctan2(dy, dx)


        current_heading = 0
        #orientation error psi
        orientation_error = desired_heading - current_heading

        orientation_error = (orientation_error + np.pi) % (2 * np.pi) - np.pi

        #cross track error
        closest_waypoint_idx = np.argmin(np.linalg.norm(waypoints - current_position[:, np.newaxis], axis=0))
        cross_track_error = waypoints[1, closest_waypoint_idx] - current_position[1]  # Assuming y-axis is lateral

        #stanley control law
        epsilon = 1e-5
        steering_angle = orientation_error + np.arctan(self.gain_constant * cross_track_error / (speed + epsilon))

        # Damping term
        steering_angle = self.damping_constant * steering_angle + (
                    1 - self.damping_constant) * self.previous_steering_angle

        # Update previous steering angle
        self.previous_steering_angle = steering_angle
        # clip
        return np.clip(steering_angle, -0.4, 0.4) / 0.4






