import numpy as np

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize

from time import time

from racetrack import RaceTrack
from racecar import RaceCar
from controller import lower_controller, controller, CONTROLLER_CONFIG

class Simulator:

    def __init__(self, rt : RaceTrack):
        matplotlib.rcParams["figure.dpi"] = 100
        matplotlib.rcParams["font.size"] = 8

        self.rt = rt
        
        # Auto-rotate if track is tall
        w = np.max(self.rt.centerline[:, 0]) - np.min(self.rt.centerline[:, 0])
        h = np.max(self.rt.centerline[:, 1]) - np.min(self.rt.centerline[:, 1])
        
        if h > w * 1.2:
            print("Auto-rotating track 90 degrees CW for better fit")
            # Rotate points: x' = y, y' = -x
            def rot(pts):
                return np.column_stack((pts[:, 1], -pts[:, 0]))
                
            self.rt.centerline = rot(self.rt.centerline)
            self.rt.right_boundary = rot(self.rt.right_boundary)
            self.rt.left_boundary = rot(self.rt.left_boundary)
            if self.rt.raceline is not None:
                self.rt.raceline = rot(self.rt.raceline)
                
            # Rotate initial state: [x, y, steer, vel, heading]
            x, y, s, v, head = self.rt.initial_state
            self.rt.initial_state = np.array([y, -x, s, v, head - np.pi/2])
            
            # Update MPL paths
            self.rt.mpl_centerline.vertices = self.rt.centerline
            self.rt.mpl_right_track_limit.vertices = self.rt.right_boundary
            self.rt.mpl_left_track_limit.vertices = self.rt.left_boundary
            if self.rt.raceline is not None:
                self.rt.mpl_raceline.vertices = self.rt.raceline

        self.figure, self.axis = plt.subplots(1, 1)
        try:
            mng = plt.get_current_fig_manager()
            mng.full_screen_toggle()
        except:
            self.figure.set_size_inches(16, 9)
            
        plt.subplots_adjust(left=0.0, right=0.82, bottom=0.0, top=1.0)

        self.axis.set_axis_off()

        # Track limits
        buff = 50
        self.min_x = np.min(self.rt.centerline[:, 0]) - buff
        self.max_x = np.max(self.rt.centerline[:, 0]) + buff
        self.min_y = np.min(self.rt.centerline[:, 1]) - buff
        self.max_y = np.max(self.rt.centerline[:, 1]) + buff

        # Dynamic layout
        track_ratio = (self.max_x - self.min_x) / (self.max_y - self.min_y)
        screen_ratio = 16/9
        needed_width = (track_ratio / screen_ratio) + 0.15
        plot_right = np.clip(needed_width, 0.4, 0.80)
        
        plt.subplots_adjust(left=0.0, right=plot_right, bottom=0.0, top=1.0)
        self.axis.set_axis_off()

        # Sliders
        axcolor = 'lightgoldenrodyellow'
        
        self.sx = plot_right + 0.05
        self.sw = 0.95 - self.sx
        
        # Slider height and spacing
        sh = 0.03
        sp = 0.04
        
        # Starting Y positions
        y_steer = 0.85
        y_vel = y_steer - (3 * sp) - 0.02
        y_look = y_vel - (3 * sp) - 0.02
        y_dyn = y_look - (2 * sp) - 0.02
        
        # Steer PID
        ax_kp = plt.axes([self.sx, y_steer, self.sw, sh], facecolor=axcolor)
        ax_ki = plt.axes([self.sx, y_steer - sp, self.sw, sh], facecolor=axcolor)
        ax_kd = plt.axes([self.sx, y_steer - 2*sp, self.sw, sh], facecolor=axcolor)

        # Velocity PID
        ax_vkp = plt.axes([self.sx, y_vel, self.sw, sh], facecolor=axcolor)
        ax_vki = plt.axes([self.sx, y_vel - sp, self.sw, sh], facecolor=axcolor)
        ax_vkd = plt.axes([self.sx, y_vel - 2*sp, self.sw, sh], facecolor=axcolor)
        
        # Lookahead
        ax_lk = plt.axes([self.sx, y_look, self.sw, sh], facecolor=axcolor)
        ax_l0 = plt.axes([self.sx, y_look - sp, self.sw, sh], facecolor=axcolor)
        
        # Dynamics
        ax_bf = plt.axes([self.sx, y_dyn, self.sw, sh], facecolor=axcolor)
        ax_slf = plt.axes([self.sx, y_dyn - sp, self.sw, sh], facecolor=axcolor)
        ax_lbs = plt.axes([self.sx, y_dyn - 2*sp, self.sw, sh], facecolor=axcolor)

        # Sliders
        self.slider_kp = Slider(ax_kp, 'Steer Kp', 0.0, 30.0, valinit=CONTROLLER_CONFIG["steering_kp"], valstep=0.1)
        self.slider_ki = Slider(ax_ki, 'Steer Ki', 0.0, 5.0, valinit=CONTROLLER_CONFIG["steering_ki"], valstep=0.01)
        self.slider_kd = Slider(ax_kd, 'Steer Kd', 0.0, 20.0, valinit=CONTROLLER_CONFIG["steering_kd"], valstep=0.1)

        self.slider_vkp = Slider(ax_vkp, 'Vel Kp', 0.0, 300.0, valinit=CONTROLLER_CONFIG["velocity_kp"], valstep=1.0)
        self.slider_vki = Slider(ax_vki, 'Vel Ki', 0.0, 5.0, valinit=CONTROLLER_CONFIG["velocity_ki"], valstep=0.01)
        self.slider_vkd = Slider(ax_vkd, 'Vel Kd', 0.0, 20.0, valinit=CONTROLLER_CONFIG["velocity_kd"], valstep=0.1)
        
        self.slider_lk = Slider(ax_lk, 'Lookahead K', 0.0, 0.5, valinit=CONTROLLER_CONFIG["lookahead_k"], valstep=0.01)
        self.slider_l0 = Slider(ax_l0, 'Lookahead L0', 0.0, 15.0, valinit=CONTROLLER_CONFIG["lookahead_L0"], valstep=0.1)
        
        self.slider_bf = Slider(ax_bf, 'Brake Factor', 0.5, 1.5, valinit=CONTROLLER_CONFIG["braking_factor"], valstep=0.01)
        self.slider_slf = Slider(ax_slf, 'Steer Lim Fac', 0.1, 2.0, valinit=CONTROLLER_CONFIG["steer_limit_factor"], valstep=0.01)
        self.slider_lbs = Slider(ax_lbs, 'Look Brake Sc', 1.0, 10.0, valinit=CONTROLLER_CONFIG["lookahead_brake_scale"], valstep=0.1)

        # Callbacks
        sliders = [self.slider_kp, self.slider_ki, self.slider_kd,
                   self.slider_vkp, self.slider_vki, self.slider_vkd,
                   self.slider_lk, self.slider_l0,
                   self.slider_bf, self.slider_slf, self.slider_lbs]
                   
        for s in sliders:
            s.on_changed(self.update_sliders)
        
        # Reset Button
        resetax = plt.axes([self.sx, 0.25, self.sw*0.6, 0.04])
        self.button = Button(resetax, 'Reset', color=axcolor, hovercolor='0.975')
        self.button.on_clicked(self.reset_simulation)
        
        # Initial run
        self.update_sliders(None)

    def update_sliders(self, val):
        # Update Lower Controller
        lower_controller.steering_kp = self.slider_kp.val
        lower_controller.steering_ki = self.slider_ki.val
        lower_controller.steering_kd = self.slider_kd.val
        lower_controller.velocity_kp = self.slider_vkp.val
        lower_controller.velocity_ki = self.slider_vki.val
        lower_controller.velocity_kd = self.slider_vkd.val
        
        # Update Global Config
        CONTROLLER_CONFIG["steering_kp"] = self.slider_kp.val
        CONTROLLER_CONFIG["steering_ki"] = self.slider_ki.val
        CONTROLLER_CONFIG["steering_kd"] = self.slider_kd.val
        CONTROLLER_CONFIG["velocity_kp"] = self.slider_vkp.val
        CONTROLLER_CONFIG["velocity_ki"] = self.slider_vki.val
        CONTROLLER_CONFIG["velocity_kd"] = self.slider_vkd.val
        
        CONTROLLER_CONFIG["lookahead_k"] = self.slider_lk.val
        CONTROLLER_CONFIG["lookahead_L0"] = self.slider_l0.val
        CONTROLLER_CONFIG["braking_factor"] = self.slider_bf.val
        CONTROLLER_CONFIG["steer_limit_factor"] = self.slider_slf.val
        CONTROLLER_CONFIG["lookahead_brake_scale"] = self.slider_lbs.val
        
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
        self.velocities = []
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
            self.velocities.append(self.car.state[3])
            
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
        self.axis.set_xlim(self.min_x, self.max_x)
        self.axis.set_ylim(self.min_y, self.max_y)
        self.axis.set_aspect('equal')
        self.rt.plot_track(self.axis)
        
        traj = np.array(self.trajectory)
        vels = np.array(self.velocities)
        
        if len(traj) > 1:
            # Create a set of line segments so that we can color them individually
            # This creates the points as a N x 1 x 2 array so that we can stack points
            # together easily to get the segments. The segments array for line collection
            # needs to be (num_lines) x (points_per_line) x 2 (for x and y)
            points = traj.reshape(-1, 1, 2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)

            # Create a continuous norm to map from data points to colors
            norm = Normalize(vmin=0, vmax=100) # Max velocity is 100
            lc = LineCollection(segments, cmap='plasma', norm=norm)
            
            # Set the values used for colormapping
            lc.set_array(vels[:-1])
            lc.set_linewidth(2)
            line = self.axis.add_collection(lc)
            
            # Add colorbar if it doesn't exist
            if not hasattr(self, 'cbar'):
                self.cbar = self.figure.colorbar(line, ax=self.axis, label='Velocity (m/s)')
            else:
                self.cbar.update_normal(line)
            
        if len(self.violation_points) > 0:
            v_pts = np.array(self.violation_points)
            self.axis.plot(v_pts[:, 0], v_pts[:, 1], 'r.', markersize=3)
            
        info_text = f"Lap Time: {self.lap_time:.2f}s\nViolations: {self.violations}"
        if not self.lap_finished:
            info_text = "Lap DNF\nViolations: " + str(self.violations)
            
        self.axis.text(0.02, 0.98, info_text, transform=self.axis.transAxes, 
                       verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        self.figure.canvas.draw_idle()

    def reset_simulation(self, event):
        self.car = None
        lower_controller.reset()
        self.trajectory = []
        self.violation_points = []
        self.lap_finished = False
        self.lap_time = 0.0
        self.violations = 0

        self.axis.cla()
        self.rt.plot_track(self.axis)
        self.figure.canvas.draw_idle()

    def start(self):
        pass