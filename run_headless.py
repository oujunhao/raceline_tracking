import numpy as np
from time import time
from racetrack import RaceTrack
from racecar import RaceCar
from controller import lower_controller, controller
import sys

def run_headless(track_file, raceline_file=None, max_steps=20000):
    racetrack = RaceTrack(track_file)
    if raceline_file:
        racetrack.load_raceline(raceline_file)
    car = RaceCar(racetrack.initial_state.T)
    
    # Reset controller state
    lower_controller.prev_steering_error = 0.0
    lower_controller.integral_steering_error = 0.0
    lower_controller.prev_velocity_error = 0.0
    lower_controller.integral_velocity_error = 0.0
    
    dt = car.time_step
    time_elapsed = 0.0
    
    lap_started = False
    lap_finished = False
    start_time = 0.0
    
    violations = 0
    currently_violating = False
    
    for step in range(max_steps):
        # Get control
        desired, lookahead_point = controller(car.state, car.parameters, racetrack)
        cont = lower_controller(car.state, desired, car.parameters)
        car.update(cont)
        
        time_elapsed += dt
        
        # Check progress
        progress = np.linalg.norm(car.state[0:2] - racetrack.centerline[0, 0:2], 2)
        
        if progress > 10.0 and not lap_started:
            lap_started = True
            start_time = time_elapsed
            
        if progress <= 5.0 and lap_started and not lap_finished:
            if time_elapsed - start_time > 10.0:
                lap_finished = True
                lap_time = time_elapsed - start_time
                print(f"Lap finished! Time: {lap_time:.2f}s, Violations: {violations}")
                return lap_time, violations
        
        # Check violations
        car_position = car.state[0:2]
        centerline_distances = np.linalg.norm(racetrack.centerline - car_position, axis=1)
        closest_idx = np.argmin(centerline_distances)
        
        to_right = racetrack.right_boundary[closest_idx] - racetrack.centerline[closest_idx]
        to_left = racetrack.left_boundary[closest_idx] - racetrack.centerline[closest_idx]
        to_car = car_position - racetrack.centerline[closest_idx]
        
        right_dist = np.linalg.norm(to_right)
        left_dist = np.linalg.norm(to_left)
        
        proj_right = np.dot(to_car, to_right) / right_dist if right_dist > 0 else 0
        proj_left = np.dot(to_car, to_left) / left_dist if left_dist > 0 else 0
        
        is_violating = proj_right > right_dist or proj_left > left_dist
        
        if is_violating:
            if not currently_violating:
                violations += 1
                currently_violating = True
                # print(f"Violation at {time_elapsed:.2f}s")
        else:
            currently_violating = False
            
        if step % 500 == 0:
             print(f"T={time_elapsed:.1f}s, V={car.state[3]:.1f}")

    print("Did not finish lap.")
    return None, violations

if __name__ == "__main__":
    track = "./racetracks/Montreal.csv"
    raceline = "./racetracks/Montreal_raceline.csv"
    if len(sys.argv) > 1:
        track = sys.argv[1]
    if len(sys.argv) > 2:
        raceline = sys.argv[2]
        
    run_headless(track, raceline)
