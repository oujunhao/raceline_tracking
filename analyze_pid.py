import numpy as np
import matplotlib.pyplot as plt
from racetrack import RaceTrack
from racecar import RaceCar
from controller import lower_controller, controller

def analyze():
    track_path = "racetracks/Montreal.csv"
    racetrack = RaceTrack(track_path)
    car = RaceCar(racetrack.initial_state.T)
    
    # Reset controller state
    lower_controller.prev_steering_error = 0.0
    lower_controller.integral_steering_error = 0.0
    lower_controller.prev_velocity_error = 0.0
    lower_controller.integral_velocity_error = 0.0
    
    dt = car.time_step
    max_time = 30.0 # Run for 30 seconds
    
    times = []
    steer_desired = []
    steer_actual = []
    vel_desired = []
    vel_actual = []
    
    current_time = 0.0
    
    print("Running simulation for analysis...")
    
    while current_time < max_time:
        desired, lookahead_point = controller(car.state, car.parameters, racetrack)
        cont = lower_controller(car.state, desired, car.parameters)
        
        # Log data
        times.append(current_time)
        
        # Desired angle is relative to heading in controller output, 
        # but let's log the raw controller output vs the car's steering angle state
        # Wait, controller returns [desired_relative_heading, desired_velocity]
        # LowerController takes this.
        # The car state has 'steering_angle' (delta).
        # The LowerController tries to make 'steering_angle' match 'desired_angle'??
        
        # Let's check LowerController logic in controller.py:
        # steering_error = np.arctan2(math.sin(desired_angle - steering_angle), ...)
        # So yes, it treats desired[0] as the target steering angle.
        
        steer_desired.append(desired[0])
        steer_actual.append(car.state[4]) # steering_angle
        
        vel_desired.append(desired[1])
        vel_actual.append(car.state[3]) # lng_velocity
        
        car.update(cont)
        current_time += dt
        
    # Plotting
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    ax1.plot(times, steer_desired, label='Desired Steering', color='blue', alpha=0.7)
    ax1.plot(times, steer_actual, label='Actual Steering', color='red', alpha=0.7)
    ax1.set_title('Steering Response')
    ax1.set_ylabel('Angle (rad)')
    ax1.legend()
    ax1.grid(True)
    
    ax2.plot(times, vel_desired, label='Desired Velocity', color='blue', alpha=0.7)
    ax2.plot(times, vel_actual, label='Actual Velocity', color='red', alpha=0.7)
    ax2.set_title('Velocity Response')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Velocity (m/s)')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig('pid_analysis.png')
    print("Analysis complete. Saved plot to pid_analysis.png")

if __name__ == "__main__":
    analyze()
