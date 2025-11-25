import numpy as np
import matplotlib.pyplot as plt
from racecar import RaceCar
from controller import LowerController

def run_step_test(test_type="velocity"):
    # Initialize car
    initial_state = np.zeros(5)
    car = RaceCar(initial_state)
    
    # Initialize controller with current params
    # We need to instantiate a new one to ensure clean state
    lc = LowerController()
    
    # Simulation parameters
    dt = car.time_step
    max_time = 10.0
    times = []
    actuals = []
    desireds = []
    
    current_time = 0.0
    
    # Step inputs
    target_vel = 50.0
    target_steer = 0.5
    
    while current_time < max_time:
        times.append(current_time)
        
        if test_type == "velocity":
            # Step velocity at t=1.0
            des_v = target_vel if current_time > 1.0 else 0.0
            des_s = 0.0
            
            desired = np.array([des_s, des_v])
            actual = car.state[3] # Velocity
            
            desireds.append(des_v)
            
        elif test_type == "steering":
            # Constant velocity, Step steering at t=1.0
            des_v = 20.0
            des_s = target_steer if current_time > 1.0 else 0.0
            
            desired = np.array([des_s, des_v])
            actual = car.state[2] # Steering angle
            
            desireds.append(des_s)
            
        actuals.append(actual)
        
        # Get control
        cont = lc(car.state, desired, car.parameters)
        car.update(cont)
        
        current_time += dt
        
    return np.array(times), np.array(desireds), np.array(actuals)

def calculate_metrics(times, desireds, actuals):
    # Find step time
    step_idx = np.where(desireds > 0)[0][0]
    step_time = times[step_idx]
    target = desireds[-1]
    
    # Slice data after step
    post_step_actuals = actuals[step_idx:]
    post_step_times = times[step_idx:]
    
    # Rise Time (10% to 90%)
    try:
        idx_10 = np.where(post_step_actuals >= 0.1 * target)[0][0]
        idx_90 = np.where(post_step_actuals >= 0.9 * target)[0][0]
        rise_time = post_step_times[idx_90] - post_step_times[idx_10]
    except:
        rise_time = float('inf')
        
    # Overshoot
    max_val = np.max(post_step_actuals)
    overshoot = (max_val - target) / target * 100.0
    
    # Settling Time (within 2%)
    # Find last time it was outside 2% band
    upper_bound = target * 1.02
    lower_bound = target * 0.98
    
    outside_band = np.where((post_step_actuals > upper_bound) | (post_step_actuals < lower_bound))[0]
    if len(outside_band) > 0:
        settling_time = post_step_times[outside_band[-1]] - step_time
    else:
        settling_time = 0.0
        
    return rise_time, overshoot, settling_time

def tune():
    print("--- Velocity Step Response ---")
    t, d, a = run_step_test("velocity")
    rt, os, st = calculate_metrics(t, d, a)
    print(f"Rise Time: {rt:.4f}s")
    print(f"Overshoot: {os:.2f}%")
    print(f"Settling Time: {st:.4f}s")
    
    plt.figure()
    plt.plot(t, d, label='Desired')
    plt.plot(t, a, label='Actual')
    plt.title(f'Velocity Step Response (OS: {os:.1f}%)')
    plt.legend()
    plt.savefig('velocity_step.png')
    
    print("\n--- Steering Step Response ---")
    t, d, a = run_step_test("steering")
    rt, os, st = calculate_metrics(t, d, a)
    print(f"Rise Time: {rt:.4f}s")
    print(f"Overshoot: {os:.2f}%")
    print(f"Settling Time: {st:.4f}s")
    
    plt.figure()
    plt.plot(t, d, label='Desired')
    plt.plot(t, a, label='Actual')
    plt.title(f'Steering Step Response (OS: {os:.1f}%)')
    plt.legend()
    plt.savefig('steering_step.png')

if __name__ == "__main__":
    tune()
