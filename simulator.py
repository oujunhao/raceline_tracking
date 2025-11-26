import numpy as np

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button

from time import time

from racetrack import RaceTrack
from racecar import RaceCar
from controller import lower_controller, controller

class Simulator:

    def __init__(self, rt : RaceTrack):
        matplotlib.rcParams["figure.dpi"] = 100
        matplotlib.rcParams["font.size"] = 8

        self.rt = rt
        self.figure, self.axis = plt.subplots(1, 1, figsize=(12, 8))
        plt.subplots_adjust(bottom=0.35)

        self.axis.set_xlabel("X"); self.axis.set_ylabel("Y")

        # Sliders
        axcolor = 'lightgoldenrodyellow'
        
        # Steer
        ax_kp = plt.axes([0.15, 0.20, 0.75, 0.03], facecolor=axcolor)
        ax_ki = plt.axes([0.15, 0.16, 0.75, 0.03], facecolor=axcolor)
        ax_kd = plt.axes([0.15, 0.12, 0.75, 0.03], facecolor=axcolor)

        # Velocity
        ax_vkp = plt.axes([0.15, 0.08, 0.75, 0.03], facecolor=axcolor)
        ax_vki = plt.axes([0.15, 0.04, 0.75, 0.03], facecolor=axcolor)
        ax_vkd = plt.axes([0.15, 0.00, 0.75, 0.03], facecolor=axcolor)

        self.slider_kp = Slider(ax_kp, 'Steer Kp', 5.0, 15.0, valinit=lower_controller.steering_kp, valstep=0.01)
        self.slider_ki = Slider(ax_ki, 'Steer Ki', 0.0, 1.0, valinit=lower_controller.steering_ki, valstep=0.001)
        self.slider_kd = Slider(ax_kd, 'Steer Kd', 3.0, 9.0, valinit=lower_controller.steering_kd, valstep=0.01)

        self.slider_vkp = Slider(ax_vkp, 'Vel Kp', 100.0, 200.0, valinit=lower_controller.velocity_kp, valstep=0.1)
        self.slider_vki = Slider(ax_vki, 'Vel Ki', 0.0, 1.0, valinit=lower_controller.velocity_ki, valstep=0.001)
        self.slider_vkd = Slider(ax_vkd, 'Vel Kd', 4.0, 12.0, valinit=lower_controller.velocity_kd, valstep=0.01)

        self.slider_kp.on_changed(self.update_sliders)
        self.slider_ki.on_changed(self.update_sliders)
        self.slider_kd.on_changed(self.update_sliders)
        self.slider_vkp.on_changed(self.update_sliders)
        self.slider_vki.on_changed(self.update_sliders)
        self.slider_vkd.on_changed(self.update_sliders)
        
        # Initial run
        self.update_sliders(None)

    def update_sliders(self, val):
        lower_controller.steering_kp = self.slider_kp.val
        lower_controller.steering_ki = self.slider_ki.val
        lower_controller.steering_kd = self.slider_kd.val
        lower_controller.velocity_kp = self.slider_vkp.val
        lower_controller.velocity_ki = self.slider_vki.val
        lower_controller.velocity_kd = self.slider_vkd.val
        
        self.simulate_lap()
        self.plot_results()

    def simulate_lap(self):
        self.car = RaceCar(self.rt.initial_state.T)
        lower_controller.prev_steering_error = 0.0
        lower_controller.integral_steering_error = 0.0
        lower_controller.prev_velocity_error = 0.0
        lower_controller.integral_velocity_error = 0.0
        
        dt = self.car.time_step
        max_steps = 2000 
        
        self.trajectory = []
        self.violation_points = []
        self.lap_finished = False
        self.lap_time = 0.0
        self.violations = 0
        
        lap_started = False
        time_elapsed = 0.0
        currently_violating = False
        
        total_points = len(self.rt.centerline)
        last_closest_idx = 0
        
        for _ in range(max_steps):
            self.trajectory.append(self.car.state[0:2].copy())
            
            desired, _ = controller(self.car.state, self.car.parameters, self.rt)
            cont = lower_controller(self.car.state, desired, self.car.parameters)
            self.car.update(cont)
            
            time_elapsed += dt
            
            # Progress
            car_position = self.car.state[0:2]
            dists = np.linalg.norm(self.rt.centerline - car_position, axis=1)
            closest_idx = np.argmin(dists)
            
            if not lap_started:
                if closest_idx > total_points * 0.05 and closest_idx < total_points * 0.5:
                    lap_started = True
                    
            if lap_started and not self.lap_finished:
                if last_closest_idx > total_points * 0.9 and closest_idx < total_points * 0.1:
                    self.lap_finished = True
                    self.lap_time = time_elapsed
                    break
            
            last_closest_idx = closest_idx
            
            # Violations
            to_right = self.rt.right_boundary[closest_idx] - self.rt.centerline[closest_idx]
            to_left = self.rt.left_boundary[closest_idx] - self.rt.centerline[closest_idx]
            to_car = car_position - self.rt.centerline[closest_idx]
            
            right_dist = np.linalg.norm(to_right)
            left_dist = np.linalg.norm(to_left)
            
            proj_right = np.dot(to_car, to_right) / right_dist if right_dist > 0 else 0
            proj_left = np.dot(to_car, to_left) / left_dist if left_dist > 0 else 0
            
            is_violating = proj_right > right_dist or proj_left > left_dist
            
            if is_violating:
                self.violation_points.append(car_position.copy())
                if not currently_violating:
                    self.violations += 1
                    currently_violating = True
            else:
                currently_violating = False

    def plot_results(self):
        self.axis.cla()
        self.rt.plot_track(self.axis)
        
        traj = np.array(self.trajectory)
        if len(traj) > 0:
            self.axis.plot(traj[:, 0], traj[:, 1], 'b-', linewidth=1)
            
        if len(self.violation_points) > 0:
            v_pts = np.array(self.violation_points)
            self.axis.plot(v_pts[:, 0], v_pts[:, 1], 'r.', markersize=3)
            
        info_text = f"Lap Time: {self.lap_time:.2f}s\nViolations: {self.violations}"
        if not self.lap_finished:
            info_text = "Lap DNF\nViolations: " + str(self.violations)
            
        self.axis.text(0.02, 0.98, info_text, transform=self.axis.transAxes, 
                       verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        self.figure.canvas.draw_idle()

    def start(self):
        pass