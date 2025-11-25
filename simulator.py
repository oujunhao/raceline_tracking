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
        matplotlib.rcParams["figure.dpi"] = 300
        matplotlib.rcParams["font.size"] = 8

        self.rt = rt
        self.figure, self.axis = plt.subplots(1, 1, figsize=(16, 10))
        plt.subplots_adjust(bottom=0.25)

        self.axis.set_xlabel("X"); self.axis.set_ylabel("Y")

        self.car = RaceCar(self.rt.initial_state.T)

        self.lap_time_elapsed = 0
        self.lap_start_time = None
        self.lap_finished = False
        self.lap_started = False
        self.track_limit_violations = 0
        self.currently_violating = False

        # Sliders
        axcolor = 'lightgoldenrodyellow'
        ax_kp = plt.axes([0.25, 0.1, 0.65, 0.03], facecolor=axcolor)
        ax_ki = plt.axes([0.25, 0.06, 0.65, 0.03], facecolor=axcolor)
        ax_kd = plt.axes([0.25, 0.02, 0.65, 0.03], facecolor=axcolor)

        self.slider_kp = Slider(ax_kp, 'Steer Kp', 0.0, 20.0, valinit=lower_controller.steering_kp)
        self.slider_ki = Slider(ax_ki, 'Steer Ki', 0.0, 1.0, valinit=lower_controller.steering_ki)
        self.slider_kd = Slider(ax_kd, 'Steer Kd', 0.0, 5.0, valinit=lower_controller.steering_kd)

        self.slider_kp.on_changed(self.update_sliders)
        self.slider_ki.on_changed(self.update_sliders)
        self.slider_kd.on_changed(self.update_sliders)

        # Reset Button
        resetax = plt.axes([0.025, 0.05, 0.1, 0.04])
        self.button = Button(resetax, 'Reset', color=axcolor, hovercolor='0.975')
        self.button.on_clicked(self.reset_simulation)

    def update_sliders(self, val):
        lower_controller.steering_kp = self.slider_kp.val
        lower_controller.steering_ki = self.slider_ki.val
        lower_controller.steering_kd = self.slider_kd.val

    def reset_simulation(self, event):
        self.car = RaceCar(self.rt.initial_state.T)
        self.lap_time_elapsed = 0
        self.lap_start_time = time()
        self.lap_finished = False
        self.lap_started = False
        self.track_limit_violations = 0
        self.currently_violating = False
        
        lower_controller.prev_steering_error = 0.0
        lower_controller.integral_steering_error = 0.0
        lower_controller.prev_velocity_error = 0.0
        lower_controller.integral_velocity_error = 0.0

    def check_track_limits(self):
        car_position = self.car.state[0:2]
        
        min_dist_right = float('inf')
        min_dist_left = float('inf')
        
        for i in range(len(self.rt.right_boundary)):
            dist_right = np.linalg.norm(car_position - self.rt.right_boundary[i])
            dist_left = np.linalg.norm(car_position - self.rt.left_boundary[i])
            
            if dist_right < min_dist_right:
                min_dist_right = dist_right
            if dist_left < min_dist_left:
                min_dist_left = dist_left
        
        centerline_distances = np.linalg.norm(self.rt.centerline - car_position, axis=1)
        closest_idx = np.argmin(centerline_distances)
        
        to_right = self.rt.right_boundary[closest_idx] - self.rt.centerline[closest_idx]
        to_left = self.rt.left_boundary[closest_idx] - self.rt.centerline[closest_idx]
        to_car = car_position - self.rt.centerline[closest_idx]
        
        right_dist = np.linalg.norm(to_right)
        left_dist = np.linalg.norm(to_left)
        
        proj_right = np.dot(to_car, to_right) / right_dist if right_dist > 0 else 0
        proj_left = np.dot(to_car, to_left) / left_dist if left_dist > 0 else 0
        
        is_violating = proj_right > right_dist or proj_left > left_dist
        
        if is_violating and not self.currently_violating:
            self.track_limit_violations += 1
            self.currently_violating = True
        elif not is_violating:
            self.currently_violating = False

    def run(self):
        try:
            if self.lap_finished:
                exit()

            self.figure.canvas.flush_events()
            self.axis.cla()

            self.rt.plot_track(self.axis)

            self.axis.set_xlim(self.car.state[0] - 200, self.car.state[0] + 200)
            self.axis.set_ylim(self.car.state[1] - 200, self.car.state[1] + 200)

            desired, lookahead_point = controller(self.car.state, self.car.parameters, self.rt)
            
            self.axis.plot(
                [self.car.state[0], lookahead_point[0]], 
                [self.car.state[1], lookahead_point[1]], 
                "-"
            )

            cont = lower_controller(self.car.state, desired, self.car.parameters)
            self.car.update(cont)
            self.update_status()
            self.check_track_limits()

            self.axis.arrow(
                self.car.state[0], self.car.state[1], \
                self.car.wheelbase*np.cos(self.car.state[4]), \
                self.car.wheelbase*np.sin(self.car.state[4])
            )

            self.axis.text(
                self.car.state[0] + 195, self.car.state[1] + 195, "Lap completed: " + str(self.lap_finished),
                horizontalalignment="right", verticalalignment="top",
                fontsize=8, color="Red"
            )

            self.axis.text(
                self.car.state[0] + 195, self.car.state[1] + 170, "Lap time: " + f"{self.lap_time_elapsed:.2f}",
                horizontalalignment="right", verticalalignment="top",
                fontsize=8, color="Red"
            )

            self.axis.text(
                self.car.state[0] + 195, self.car.state[1] + 155, "Track violations: " + str(self.track_limit_violations),
                horizontalalignment="right", verticalalignment="top",
                fontsize=8, color="Red"
            )

            self.figure.canvas.draw()
            return True

        except KeyboardInterrupt:
            exit()

    def update_status(self):
        progress = np.linalg.norm(self.car.state[0:2] - self.rt.centerline[0, 0:2], 2)

        if progress > 10.0 and not self.lap_started:
            self.lap_started = True
    
        if progress <= 1.0 and self.lap_started and not self.lap_finished:
            self.lap_finished = True
            self.lap_time_elapsed = time() - self.lap_start_time

        if not self.lap_finished and self.lap_start_time is not None:
            self.lap_time_elapsed = time() - self.lap_start_time

    def start(self):
        # Run the simulation loop every 1 millisecond.
        self.timer = self.figure.canvas.new_timer(interval=1)
        self.timer.add_callback(self.run)
        self.lap_start_time = time()
        self.timer.start()