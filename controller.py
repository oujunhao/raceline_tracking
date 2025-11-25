import math
import numpy as np
from numpy.typing import ArrayLike

from racetrack import RaceTrack

def get_next_point(
    state : ArrayLike, racetrack : RaceTrack
) -> ArrayLike:
    x, y, steering_angle, lng_velocity, heading = state

    # Find the closest point on the racetrack centerline
    centerline = racetrack.centerline
    diffs = centerline - np.array([x, y])
    dists = np.linalg.norm(diffs, axis=1)
    closest_index = np.argmin(dists)

    # Pure Pursuit Lookahead
    # Lookahead distance Ld
    # Ld = k * v + L0
    k = 0.3
    L0 = 5.0
    Ld = k * lng_velocity + L0
    
    # Find point at distance Ld
    # Simple approximation: accumulate distances along path
    current_dist = 0.0
    next_index = closest_index
    while current_dist < Ld:
        next_index = (next_index + 1) % len(centerline)
        current_dist += np.linalg.norm(centerline[next_index] - centerline[next_index-1])
        if next_index == closest_index: # Looped around
            break
            
    return centerline[next_index]

# self.parameters = np.array([
#             self.wheelbase, # Car Wheelbase
#             -self.max_steering_angle, # x3
#             self.min_velocity, # x4
#             -np.pi, # x5
#             self.max_steering_angle,
#             self.max_velocity,
#             np.pi,
#             -self.max_steering_vel, # u1
#             -self.max_acceleration, # u2
#             self.max_steering_vel,
#             self.max_acceleration
#         ])

class LowerController:
    def __init__(self):
        self.prev_steering_error = 0.0
        self.integral_steering_error = 0.0
        self.prev_velocity_error = 0.0
        self.integral_velocity_error = 0.0

        # PID constants
        self.steering_kp = 15.0
        self.steering_ki = 0.0
        self.steering_kd = 1.0

        self.velocity_kp = 25.0
        self.velocity_ki = 0.0
        self.velocity_kd = 0.1

    def __call__(self, state: ArrayLike, desired: ArrayLike, parameters: ArrayLike) -> ArrayLike:
        # [steer angle, velocity]
        assert(desired.shape == (2,))

        x, y, steering_angle, lng_velocity, heading = state
        desired_angle, desired_velocity = desired

        # Steering PID
        # We want to control the steering angle to match desired_angle
        # The output is steering_velocity (u1)
        
        steering_error = desired_angle - steering_angle
        self.integral_steering_error += steering_error
        steering_derivative = steering_error - self.prev_steering_error
        self.prev_steering_error = steering_error

        steering_control = (self.steering_kp * steering_error + 
                            self.steering_ki * self.integral_steering_error + 
                            self.steering_kd * steering_derivative)

        # Velocity PID
        # We want to control velocity to match desired_velocity
        # The output is acceleration (u2)
        
        velocity_error = desired_velocity - lng_velocity
        self.integral_velocity_error += velocity_error
        velocity_derivative = velocity_error - self.prev_velocity_error
        self.prev_velocity_error = velocity_error

        velocity_control = (self.velocity_kp * velocity_error + 
                            self.velocity_ki * self.integral_velocity_error + 
                            self.velocity_kd * velocity_derivative)

        return np.array([steering_control, velocity_control]).T

lower_controller = LowerController()

def compute_curvature(racetrack: RaceTrack, points=None):
    if points is None:
        points = racetrack.centerline
        attr_name = 'curvature'
    else:
        attr_name = 'path_curvature'

    if hasattr(racetrack, attr_name):
        return

    n_points = len(points)
    curvature = np.zeros(n_points)

    for i in range(n_points):
        p1 = points[i - 1]
        p2 = points[i]
        p3 = points[(i + 1) % n_points]

        # Circumcircle radius
        # a, b, c are side lengths
        a = np.linalg.norm(p1 - p2)
        b = np.linalg.norm(p2 - p3)
        c = np.linalg.norm(p3 - p1)

        # Area of triangle using Heron's formula
        s = (a + b + c) / 2.0
        area = np.sqrt(max(s * (s - a) * (s - b) * (s - c), 0.0))

        if area < 1e-6:
            curvature[i] = 0.0
        else:
            R = (a * b * c) / (4.0 * area)
            curvature[i] = 1.0 / R
            
    setattr(racetrack, attr_name, curvature)

def controller(
    state : ArrayLike, parameters : ArrayLike, racetrack : RaceTrack
) -> ArrayLike:
    x, y, steering_angle, lng_velocity, heading = state

    # Select path
    if racetrack.raceline is not None:
        path_points = racetrack.raceline
        compute_curvature(racetrack, path_points)
        curvature = racetrack.path_curvature
    else:
        path_points = racetrack.centerline
        compute_curvature(racetrack)
        curvature = racetrack.curvature

    # Find the closest point on the path
    diffs = path_points - np.array([x, y])
    dists = np.linalg.norm(diffs, axis=1)
    closest_index = np.argmin(dists)

    # Pure Pursuit Lookahead
    # Lookahead distance Ld
    # Ld = k * v + L0
    k = 0.2
    L0 = 4.0
    Ld = k * lng_velocity + L0
    
    # Find point at distance Ld
    # Simple approximation: accumulate distances along path
    current_dist = 0.0
    next_index = closest_index
    while current_dist < Ld:
        next_index = (next_index + 1) % len(path_points)
        current_dist += np.linalg.norm(path_points[next_index] - path_points[next_index-1])
        if next_index == closest_index: # Looped around
            break
            
    target_point = path_points[next_index]
    
    # Pure Pursuit Steering Control
    # alpha is the angle between the vehicle's heading and the lookahead vector
    lookahead_vector = target_point - np.array([x, y])
    lookahead_angle = np.arctan2(lookahead_vector[1], lookahead_vector[0])
    
    alpha = lookahead_angle - heading
    alpha = (alpha + np.pi) % (2 * np.pi) - np.pi # Normalize
    
    # Ld is the distance to the lookahead point
    Ld = np.linalg.norm(lookahead_vector)
    
    # Steering angle delta = arctan(2L sin(alpha) / Ld)
    # L is wheelbase (parameters[0])
    wheelbase = parameters[0]
    desired_angle = np.arctan2(2 * wheelbase * np.sin(alpha), Ld)
    
    # Clamp desired angle
    max_steer = parameters[4] # max_steering_angle
    desired_angle = np.clip(desired_angle, -max_steer, max_steer)

    # Velocity Control: Curvature-based
    
    # Look ahead for max curvature
    # We need to look ahead based on stopping distance
    # d_stop = v^2 / (2*a_max)
    # a_max braking is ~20 m/s^2
    stopping_dist = (lng_velocity**2) / (2 * 20.0)
    lookahead_dist = max(stopping_dist * 0.4, 10.0) # Min lookahead 10m
    
    max_k = 0.0
    current_dist = 0.0
    idx = closest_index
    
    while current_dist < lookahead_dist:
        idx = (idx + 1) % len(path_points)
        k = curvature[idx]
        if k > max_k:
            max_k = k
        
        # Approx distance
        current_dist += np.linalg.norm(path_points[idx] - path_points[idx-1])
        if idx == closest_index: break
            
    # V = sqrt(a_lat / k)
    # a_lat max ~ 15 m/s^2 (tuned for stability)
    a_lat_max = 80.0 
    if max_k > 1e-3:
        v_limit = np.sqrt(a_lat_max / max_k)
    else:
        v_limit = 120.0 # Max speed
        
    desired_velocity = min(120.0, v_limit)
    desired_velocity = max(desired_velocity, 10.0) # Min speed
    
    return np.array([desired_angle, desired_velocity]).T, target_point