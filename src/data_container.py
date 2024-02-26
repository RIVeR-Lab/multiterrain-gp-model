"""Container for commanded velocitiesand ground truth positions and velocities"""

import numpy as np
import math
import pandas as pd

from configurations import ModelingParameters as JP
from filter import first_order_lpf

class DataContainer:
    def __init__(self, cmd_file_name, loc_file_name):
        
        """
        Main interface container for all data. Resamples commands and ground truth positions.
        Computes velocities using backward differencing and filtering

        Args:
            cmd_file_name : Commanded velocities file name
            loc_file_name : Localization ground truth data file name
        """

        # Load the data for commanded velocities and ground truth positions
        self.cmd_df = pd.read_csv(cmd_file_name, names=JP.cmd_names)
        self.loc_df = pd.read_csv(loc_file_name, names=JP.loc_names)

        # Desired resampling rate and resampled time delta
        self.resampled_freq_hz = JP.resampled_freq_hz
        self.resampled_dt_s = JP.resampled_dt_s

        # Raw cmd data and ground truth localization positions and headings
        # These will be resampled. Positions and headings are further low pass filtered
        self.raw_cmd_time = self.cmd_df.loc[ : ,'time (seconds)'].to_numpy()
        self.raw_cmd_lin_vel = self.cmd_df.loc[ : ,'V_linear_x (m/s)'].to_numpy()
        self.raw_cmd_ang_vel = self.cmd_df.loc[ : ,'V_ang_z (rad/s)'].to_numpy()

        self.raw_loc_time    = self.loc_df.loc[ : ,"time (seconds)"].to_numpy()
        self.raw_loc_x       = self.loc_df.loc[ : ,"position_x (meters)"].to_numpy()
        self.raw_loc_y       = self.loc_df.loc[ : ,"position_y (meters)"].to_numpy()
        self.raw_loc_heading = self.unwrap_heading()

        # Initialize filtered x,y,heading values
        self.loc_x = None; self.loc_y = None; self.loc_heading = None; self.loc_time = None

        # Initialize linear and angular velocities
        self.loc_lin_vel = None; self.loc_ang_vel = None

        # Initialize the x,y,yaw velocities in the local frame
        # These are used for benchmarking
        self.loc_x_vel = None; self.loc_y_vel = None

        # Initialize the right and left wheel speeds measurement
        self.loc_right_wheel_speed = None;self.loc_left_wheel_speed = None

        # Initialize the right and left wheel speeds commands
        self.cmd_right_wheel_speed = None; self.cmd_left_wheel_speed = None

        # Initialize the commanded linear and angular speeds and command times
        self.cmd_time=None; self.cmd_lin_vel=None;self.cmd_ang_vel=None

        # Commanded velocities and ground truth localization to be resampled 
        # to the same frequency and same time stamps
        self.resampled_cmd()
        self.resampled_loc()

        # Compute the localization based ground truth velocities
        self.compute_velocities()

    def resampled_cmd(self):
        
        """
        Resample the commanded velocities to the desired sampling rate
        """
        
        # Ensuring commands and localization have same start and end time
        self.start_time = max(self.raw_cmd_time[0]  , self.raw_loc_time[0])
        self.end_time   = min(self.raw_cmd_time[-1] , self.raw_loc_time[-1])
        
        sample_cmd_time = self.start_time
        
        resampled_cmd_time = [];resampled_cmd_lin_vel = [] ;resampled_cmd_ang_vel = []
        
        search_idx = 0 
        num_cmds = len(self.raw_cmd_time)
        
        while sample_cmd_time <= self.end_time and search_idx < num_cmds-1:
            if sample_cmd_time >= self.raw_cmd_time[search_idx] and sample_cmd_time <= self.raw_cmd_time[search_idx+1]:
                resampled_cmd_time.append(sample_cmd_time)
                resampled_cmd_lin_vel.append(self.raw_cmd_lin_vel[search_idx])
                resampled_cmd_ang_vel.append(self.raw_cmd_ang_vel[search_idx])
                
                sample_cmd_time += self.resampled_dt_s
            else:
                search_idx+=1
        
        # Store the time synced and resampled commanded velocities
        self.cmd_time    = np.asarray(resampled_cmd_time)
        self.cmd_lin_vel = np.asarray(resampled_cmd_lin_vel)
        self.cmd_ang_vel = np.asarray(resampled_cmd_ang_vel)


    def resampled_loc(self):
        
        """
        Match the time stamps to the resampled commanded velocities 
        """
        
        resampled_loc_time = [];resampled_loc_x = [];resampled_loc_y = [];resampled_loc_heading = []
        
        search_idx = 0
        num_locs = len(self.raw_loc_time)
        
        # Find the localization ground truths at the command times
        for time_val in self.cmd_time:
            while search_idx < num_locs-1:
                if time_val >= self.raw_loc_time[search_idx] and time_val <= self.raw_loc_time[search_idx+1]:
                    dt = time_val - self.raw_loc_time[search_idx]
                    vx = (self.raw_loc_x[search_idx+1] - self.raw_loc_x[search_idx]) / ( self.raw_loc_time[search_idx+1] - self.raw_loc_time[search_idx]  )
                    vy = (self.raw_loc_y[search_idx+1] - self.raw_loc_y[search_idx]) / ( self.raw_loc_time[search_idx+1] - self.raw_loc_time[search_idx]  )
                    vheading = (self.raw_loc_heading[search_idx+1] - self.raw_loc_heading[search_idx]) / ( self.raw_loc_time[search_idx+1] - self.raw_loc_time[search_idx]  )
                    
                    resampled_loc_time.append(time_val)
                    resampled_loc_x.append(self.raw_loc_x[search_idx] + vx*dt)
                    resampled_loc_y.append(self.raw_loc_y[search_idx] + vy*dt)
                    resampled_loc_heading.append(self.raw_loc_heading[search_idx] + vheading*dt)
                    
                    break
                else:
                    search_idx+=1
        
        # Store the time synced and resampled ground truths
        self.loc_time    = self.cmd_time      
        self.loc_x       = np.asarray(resampled_loc_x)
        self.loc_y       = np.asarray(resampled_loc_y)
        self.loc_heading = np.asarray(resampled_loc_heading)
        
    def unwrap_heading(self):
        """
        Since we primarily deal with velocities 
        found by backward differencing, avoid jumps
        in heading from -pi to +pi, by starting angle computations from 2pi
        
        Returns:
            unwrapped_gt_heading (numpy array) : Angle unwrapped heading
        """

        unwrapped_gt_heading = [2*math.pi]
        raw_gt_heading = self.loc_df.loc[ : ,"heading (rad)"].to_numpy()
        num_heading_values = len(raw_gt_heading)

        for heading_idx in range(num_heading_values-1):
            diff_heading = raw_gt_heading[heading_idx+1] - raw_gt_heading[heading_idx]

            if abs(diff_heading) > math.pi: #corner case of pi/-pi
                diff_heading -= np.sign(diff_heading)*2*math.pi
            
            unwrapped_gt_heading.append(unwrapped_gt_heading[-1] + diff_heading)

        return np.asarray(unwrapped_gt_heading)
    
    def compute_velocities(self):
        
        """
        Interface to filter and compute commanded and ground truth velocities in 
        different frames of reference. 
        """

        # First Order LPF on raw position values
        positions = [self.loc_x,self.loc_y,self.loc_heading]

        [self.loc_x,self.loc_y,self.loc_heading] = first_order_lpf(positions)
        
        # Backward finite differencing to compute velocities
        output_vel = self.backward_diff_vel()
        
        filtered_lin_vel = output_vel[0]
        filtered_ang_vel = output_vel[1]

        # Velocity Category1 : Ground truth local frame linear in x direction and angular velocities
        self.loc_lin_vel = filtered_lin_vel; self.loc_ang_vel = filtered_ang_vel

        # Velocity Category2 : Ground truth right and left wheel speeds
        self.loc_right_wheel_speed, self.loc_left_wheel_speed  = \
            self.compute_wheel_speeds(filtered_lin_vel,filtered_ang_vel)
        
        # Velocity Category3 : Commanded left and right wheel speeds
        self.cmd_right_wheel_speed, self.cmd_left_wheel_speed  = \
            self.compute_wheel_speeds(self.cmd_lin_vel,self.cmd_ang_vel)
        
        # Velocity Category4 : Ground truth local frame linear velocities x AND y
        self.loc_x_vel,self.loc_y_vel = self.compute_x_y_velocities()

    def backward_diff_vel(self):
        """
        Use finite differencing of position data to for velocity generation
        """
        loc_lin_vel = [0.0]
        loc_ang_vel = [0.0]

        # Compute velocities using backward differencing
        for idx in range(1,len(self.loc_time)):
            dheading_rad = self.loc_heading[idx] - self.loc_heading[idx-1]
            dt_s = self.resampled_dt_s

            w_radps = dheading_rad / self.resampled_dt_s
            dx_m = self.loc_x[idx] - self.loc_x[idx-1]
            dy_m = self.loc_y[idx] - self.loc_y[idx-1]

            dx_dt = dx_m / dt_s
            dy_dt = dy_m / dt_s

            vx_mps = dx_dt + JP.a_m * w_radps * math.sin(self.loc_heading[idx])
            vy_mps = dy_dt - JP.a_m * w_radps * math.cos(self.loc_heading[idx])

            u_mps = abs(math.sqrt(vx_mps**2 + vy_mps**2))

            loc_lin_vel.append(u_mps)
            loc_ang_vel.append(w_radps)
        
        return np.asarray(loc_lin_vel) , np.asarray(loc_ang_vel)

    def compute_wheel_speeds(self,lin_vel,ang_vel):
        
        """
        Convert local frame linear and angular speeds
        to left and right wheel speeds

        Args:
            lin_vel (numpy array): Linear Velocity in the local frame
            ang_vel (numpy array): Angular Velocity in the local frame

        Returns:
            tuple of numpy array: Right and Left wheel speeds respectively
        """
        
        right_wheel_speed = (lin_vel + 0.5 * ang_vel * JP.b_m) / JP.r_m
        left_wheel_speed  = (lin_vel - 0.5 * ang_vel * JP.b_m) / JP.r_m
    
        return right_wheel_speed,left_wheel_speed
    
    def compute_x_y_velocities(self):
    
        """
        Compute local frame linear velocity (x,y) and angular velocity

        Args:
            data_container (class DataContainer): DataContainer object 

        Returns:
            tuple of numpy array: Tuple of numpy array (xvel,yvel)
        """
        
        x_vel = [0.]; y_vel = [0.]
        
        for idx in range(1,len(self.loc_time)):
            dt_hat   = JP.resampled_dt_s
            dx_hat   = self.loc_x[idx] - self.loc_x[idx-1]
            dy_hat   = self.loc_y[idx] - self.loc_y[idx-1]
            
            xdot = dx_hat / dt_hat; ydot = dy_hat / dt_hat
            
            omega   = self.loc_ang_vel[idx]
            heading = self.loc_heading[idx]
            
            a_eqn = np.array( [  [ np.cos(heading) , -np.sin(heading)   ]   , [ np.sin(heading) , np.cos(heading)   ]  ]  ) 
            b_eqn = np.array( [   xdot + JP.a_m*omega*np.sin(heading)  , ydot - JP.a_m*omega*np.cos(heading) ]    )
            
            sol = np.linalg.solve(a_eqn,b_eqn)
            
            x_vel.append(sol[0]); y_vel.append(sol[1])
            
        return (np.asarray(x_vel) , np.asarray(y_vel))

