import math
import numpy as np
from numpy.typing import ArrayLike

from simulator import RaceTrack

def lookahead(
    state : ArrayLike, racetrack : RaceTrack, lookahead_distance : float
) -> ArrayLike:
    x, y, heading, lng_velocity, steering_angle = state

    # Find the closest point on the racetrack centerline
    centerline = racetrack.centerline
    diffs = centerline - np.array([x, y])
    dists = np.linalg.norm(diffs, axis=1)
    closest_index = np.argmin(dists)

    # Move along the centerline to find the lookahead point
    total_distance = 0.0
    index = closest_index
    while total_distance < lookahead_distance:
        next_index = (index + 1) % len(centerline)
        segment = centerline[next_index] - centerline[index]
        segment_length = np.linalg.norm(segment)
        total_distance += segment_length
        index = next_index

    return centerline[index]

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
        self.steering_kp = 3.0
        self.steering_ki = 0.0
        self.steering_kd = 0.01

        self.velocity_kp = 1.0
        self.velocity_ki = 0.0
        self.velocity_kd = 0.01

    def __call__(self, state: ArrayLike, desired: ArrayLike, parameters: ArrayLike) -> ArrayLike:
        # [steer angle, velocity]
        assert(desired.shape == (2,))

        x, y, heading, lng_velocity, steering_angle = state
        desired_angle, desired_velocity = desired

        # Steering PID
        steering_error = np.arctan2(
            math.sin(desired_angle - steering_angle),
            math.cos(desired_angle - steering_angle)
        )
        self.integral_steering_error += steering_error
        steering_derivative = steering_error - self.prev_steering_error
        self.prev_steering_error = steering_error

        steering_control = (self.steering_kp * steering_error + 
                            self.steering_ki * self.integral_steering_error + 
                            self.steering_kd * steering_derivative)

        # Velocity PID
        velocity_error = desired_velocity - lng_velocity
        self.integral_velocity_error += velocity_error
        velocity_derivative = velocity_error - self.prev_velocity_error
        self.prev_velocity_error = velocity_error

        velocity_control = (self.velocity_kp * velocity_error + 
                            self.velocity_ki * self.integral_velocity_error + 
                            self.velocity_kd * velocity_derivative)

        return np.array([steering_control, velocity_control]).T

lower_controller = LowerController()

def controller(
    state : ArrayLike, parameters : ArrayLike, racetrack : RaceTrack
) -> ArrayLike:
    x, y, heading, lng_velocity, steering_angle = state

    lookahead_point = lookahead(state, racetrack, 5 + lng_velocity * 0.2)
    
    desired_angle = np.arctan2(
        lookahead_point[1] - y,
        lookahead_point[0] - x
    ) - heading

    # New velocity is proportional to the distance to the lookahead point
    distance_to_lookahead = np.linalg.norm(lookahead_point - np.array([x, y]))
    desired_velocity = min(150.0, distance_to_lookahead * 2.0)
    desired_velocity = 10
    
    
    return np.array([desired_angle, desired_velocity]).T, lookahead_point