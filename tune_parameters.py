import os
import glob
import numpy as np
import multiprocessing
import random
import time
import sys
import contextlib
from copy import deepcopy

# Import the controller module to modify globals
# We need to make sure we can import from current directory
sys.path.append(os.getcwd())

import controller
from controller import lower_controller, CONTROLLER_CONFIG
from run_headless import run_headless

# Define parameter ranges
PARAM_RANGES = {
    "steering_kp": (5.0, 25.0),
    "steering_ki": (0.0, 2.0),
    "steering_kd": (0.0, 15.0),
    "velocity_kp": (50.0, 250.0),
    "velocity_ki": (0.0, 2.0),
    "velocity_kd": (0.0, 15.0),
    "lookahead_k": (0.05, 0.3),
    "lookahead_L0": (2.0, 10.0),
    "braking_factor": (0.8, 1.2),
    "steer_limit_factor": (0.5, 1.5),
    "lookahead_brake_scale": (1.0, 5.0)
}

# Tracks to use for tuning
TRACKS_DIR = "racetracks"

def get_tracks():
    track_files = glob.glob(os.path.join(TRACKS_DIR, "*.csv"))
    tracks = []
    for tf in track_files:
        if "_raceline" in tf:
            continue
        
        base_name = os.path.splitext(os.path.basename(tf))[0]
        raceline_file = os.path.join(TRACKS_DIR, f"{base_name}_raceline.csv")
        if not os.path.exists(raceline_file):
            raceline_file = None
        
        tracks.append((tf, raceline_file))
    return sorted(tracks)

def evaluate_params(params):
    # Set parameters in this process
    for k, v in params.items():
        CONTROLLER_CONFIG[k] = v
    
    # Update lower controller instance
    lower_controller.steering_kp = params["steering_kp"]
    lower_controller.steering_ki = params["steering_ki"]
    lower_controller.steering_kd = params["steering_kd"]
    lower_controller.velocity_kp = params["velocity_kp"]
    lower_controller.velocity_ki = params["velocity_ki"]
    lower_controller.velocity_kd = params["velocity_kd"]
    
    total_time = 0.0
    total_violations = 0
    
    tracks = get_tracks()
    
    # Penalty constants
    DNF_PENALTY = 200.0
    VIOLATION_PENALTY = 50.0
    
    for track_file, raceline_file in tracks:
        # Suppress output
        with open(os.devnull, 'w') as fnull:
            with contextlib.redirect_stdout(fnull):
                try:
                    # Reduce max_steps to avoid long hangs on bad params
                    # 1000 steps = 100 seconds. Most laps should be < 30s.
                    lap_time, violations = run_headless(track_file, raceline_file, max_steps=1000)
                except Exception:
                    lap_time = None
                    violations = 100
            
        if lap_time is None:
            lap_time = DNF_PENALTY
            violations += 10 # Extra penalty for DNF
            
        total_time += lap_time
        total_violations += violations
        
    # Fitness: minimize time + penalty for violations
    fitness = total_time + (total_violations * VIOLATION_PENALTY)
    return fitness

def random_individual():
    return {k: random.uniform(v[0], v[1]) for k, v in PARAM_RANGES.items()}

def crossover(p1, p2):
    child = {}
    for k in PARAM_RANGES.keys():
        if random.random() < 0.5:
            child[k] = p1[k]
        else:
            child[k] = p2[k]
    return child

def mutate(individual, mutation_rate=0.3, mutation_scale=0.2):
    mutated = individual.copy()
    for k, v in PARAM_RANGES.items():
        if random.random() < mutation_rate:
            # Gaussian mutation
            change = random.gauss(0, (v[1] - v[0]) * mutation_scale)
            mutated[k] = np.clip(mutated[k] + change, v[0], v[1])
    return mutated

def run_optimization():
    POP_SIZE = 16
    GENERATIONS = 20
    ELITISM = 4
    
    # Initialize population
    population = [random_individual() for _ in range(POP_SIZE)]
    # Include current config as a starting point
    current_config = {k: CONTROLLER_CONFIG[k] for k in PARAM_RANGES.keys()}
    population[0] = current_config
    
    tracks = get_tracks()
    print(f"Starting optimization with {POP_SIZE} individuals for {GENERATIONS} generations.")
    print(f"Tuning on {len(tracks)} tracks.")
    
    # Use spawn to ensure clean state for each process
    ctx = multiprocessing.get_context('spawn')
    
    with ctx.Pool(processes=multiprocessing.cpu_count()) as pool:
        for gen in range(GENERATIONS):
            print(f"\nGeneration {gen+1}/{GENERATIONS}")
            start_time = time.time()
            
            # Evaluate
            fitnesses = pool.map(evaluate_params, population)
            
            # Combine
            pop_fit = list(zip(population, fitnesses))
            pop_fit.sort(key=lambda x: x[1])
            
            best_params, best_fitness = pop_fit[0]
            duration = time.time() - start_time
            print(f"  Best Fitness: {best_fitness:.2f} (Time: {duration:.1f}s)")
            # print(f"  Best Params: {best_params}")
            
            # Selection
            next_pop = [p for p, f in pop_fit[:ELITISM]]
            
            while len(next_pop) < POP_SIZE:
                # Tournament selection
                t_size = 3
                candidates = random.sample(pop_fit, t_size)
                parent1 = min(candidates, key=lambda x: x[1])[0]
                
                candidates = random.sample(pop_fit, t_size)
                parent2 = min(candidates, key=lambda x: x[1])[0]
                
                child = crossover(parent1, parent2)
                child = mutate(child)
                next_pop.append(child)
                
            population = next_pop
            
    print("\nOptimization Complete!")
    print("Best Parameters found:")
    best_params = population[0]
    
    # Format for copy-paste
    print("CONTROLLER_CONFIG = {")
    for k, v in best_params.items():
        print(f'    "{k}": {v:.4f},')
    print("}")

if __name__ == "__main__":
    run_optimization()
