import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.interpolate import splprep, splev
from scipy.optimize import minimize
import time


def normalize(v):
    norm = np.linalg.norm(v,axis=0) + 0.00001
    return v / norm.reshape(1, v.shape[1])

def curvature(waypoints):
    '''
    ##### TODO #####
    Curvature as the sum of the normalized dot product between the way elements
    Implement second term of the smoothin objective.

    args:
        waypoints [2, num_waypoints] !!!!!
    '''

    # Extract x and y
    x = waypoints[0]
    y = waypoints[1]

    # dx & dy
    dx = np.diff(x)
    dy = np.diff(y)

    #tangents
    tangent_vectors = np.vstack((dx, dy))
    tangent_vectors_normalized = normalize(tangent_vectors)

    #curvature term
    curvature_penalty = 0.0
    for n in range(1, len(x) - 1):
        vec_n = waypoints[:, n + 1] - waypoints[:, n]
        vec_n_minus_1 = waypoints[:, n] - waypoints[:, n - 1]

        norm_n = np.linalg.norm(vec_n) + 1e-10  #division by zero
        norm_n_minus_1 = np.linalg.norm(vec_n_minus_1) + 1e-10  #Avoid division by zero

        if norm_n > 0 and norm_n_minus_1 > 0:
            curvature_penalty += np.dot(vec_n, vec_n_minus_1) / (norm_n * norm_n_minus_1)


    return np.sum(np.abs(np.diff(tangent_vectors_normalized[0]))) + \
        np.sum(np.abs(np.diff(tangent_vectors_normalized[1]))) + curvature_penalty


def smoothing_objective(waypoints, waypoints_center, weight_curvature=40):
    '''
    Objective for path smoothing

    args:
        waypoints [2 * num_waypoints] !!!!!
        waypoints_center [2 * num_waypoints] !!!!!
        weight_curvature (default=40)
    '''
    # mean least square error between waypoint and way point center
    ls_tocenter = np.mean((waypoints_center - waypoints)**2)

    # derive curvature
    curv = curvature(waypoints.reshape(2,-1))

    return -1 * weight_curvature * curv + ls_tocenter


def waypoint_prediction(roadside1_spline, roadside2_spline, num_waypoints=6, way_type = "smooth"):
    '''
    ##### TODO #####
    Predict waypoint via two different methods:
    - center
    - smooth

    args:
        roadside1_spline
        roadside2_spline
        num_waypoints (default=6)
        parameter_bound_waypoints (default=1)
        waytype (default="smoothed")
    '''
    if way_type == "center":
        ##### TODO #####

        # create spline arguments
        spline_params = np.linspace(0, 1, num_waypoints)

        # derive roadside points from spline

        roadside1 = np.array(splev(spline_params, roadside1_spline))
        roadside2 = np.array(splev(spline_params, roadside2_spline))

        # derive center between corresponding roadside points

        center_way_points = (roadside1 + roadside2) / 2

        # output way_points with shape(2 x Num_waypoints)
        return center_way_points

    elif way_type == "smooth":
        ##### TODO #####

        # create spline arguments

        spline_params = np.linspace(0, 1, num_waypoints)

        # derive roadside points from spline

        roadside1 = np.array(splev(spline_params, roadside1_spline))
        roadside2 = np.array(splev(spline_params, roadside2_spline))
        # derive center between corresponding roadside points
        way_points_center = (roadside1 + roadside2) / 2
        way_points_center_flat = way_points_center.flatten()

        # optimization
        way_points = minimize(smoothing_objective,
                      (way_points_center_flat),
                      args=way_points_center_flat)["x"]


        return way_points.reshape(2,-1)


def target_speed_prediction(waypoints, num_waypoints_used=5,
                            max_speed=60, exp_constant=4.5, offset_speed=30):
    '''
    ##### TODO #####
    Predict target speed given waypoints
    Implement the function using curvature()

    args:
        waypoints [2,num_waypoints]
        num_waypoints_used (default=5)
        max_speed (default=60)
        exp_constant (default=4.5)
        offset_speed (default=30)

    output:
        target_speed (float)
    '''

    waypoints_for_curvature = waypoints[:, :num_waypoints_used]

    # implementing our function
    curv = curvature(waypoints_for_curvature)

    # Predict target speed based on curvature
    target_speed = (max_speed - offset_speed) * np.exp(-exp_constant * (np.abs(num_waypoints_used - 2 - curv))) + offset_speed

    return target_speed