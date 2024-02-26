"""Implementation of the Extended Differential Drive with 5 parameters
kinematic model (EDD5) for Skid-Steer Wheeled Mobile Robot. Based 
on the paper https://ieeexplore.ieee.org/abstract/document/9108696"""

import numpy as np
from scipy.optimize import least_squares
import os

from configurations import ModelingParameters as JP

r = JP.r_m; b = JP.b_m

def res_extended_diff_drive_five_params(x, cmd_w_l, cmd_w_r, vx_hat, vy_hat, vheading_hat):
    """
    Residual function for EDD5 Model. Based on equation 7 in the paper

    Args:
        x: [alphar, alphal, xv, yr, yl]
        cmd_w_l : Commanded left wheel velocity
        cmd_w_r : Commanded right wheel velocity
        vx_hat  : Local frame longitudinal velocity
        vy_hat  : Local frame lateral velocity
        vheading_hat : Local frame heading velocity    
    """
    alphar = x[0]; alphal = x[1]; xv = x[2]; yr = x[3]; yl = x[4]
    
    pre_multiplier = r / (yl - yr)

    elem_x          = pre_multiplier * ( -yr * alphal* cmd_w_l + yl * alphar * cmd_w_r ) - vx_hat
    elem_y          = pre_multiplier * (  xv * alphal* cmd_w_l - xv * alphar * cmd_w_r ) - vy_hat
    elem_heading    = pre_multiplier * ( -1. * alphal* cmd_w_l + 1. * alphar * cmd_w_r ) - vheading_hat
    
    return np.array( [ elem_x,elem_y,elem_heading ] ).flatten()

class KinematicModel:

    def __init__(self, data_container,terrain_type):

        # Make a local copy of the DataContainer object
        self.data_container = data_container

        # Name of terrain for storing parameters
        self.terrain_type = terrain_type
    
    def compute_paramters(self):
        
        # Warm start the optimization problem with these weights
        x0 = np.array( [0.9063,0.9042,0.1777,-0.2002,0.2528] )

        # Commanded left and right wheel speeds
        cmd_w_l = self.data_container.cmd_left_wheel_speed; cmd_w_r = self.data_container.cmd_right_wheel_speed

        # Local frame x,y,heading velocities
        vx_hat = self.data_container.loc_x_vel; vy_hat = self.data_container.loc_y_vel;  vheading_hat = self.data_container.loc_ang_vel

        # Inputs for least squares
        args=(cmd_w_l,cmd_w_r,vx_hat,vy_hat,vheading_hat)

        # Solve the least squares problem
        least_squares_soln = least_squares(res_extended_diff_drive_five_params,x0=x0,loss='soft_l1',f_scale=0.8,args=args)

        # Extract the solution of the least squares problem
        kinematic_model_parameters = least_squares_soln.x

        # Save the parameters
        parent_dir = os.path.join(os.getcwd(),"models")

        os.makedirs(parent_dir,exist_ok=True)

        file_name = os.path.join(parent_dir,"edd_5_" + self.terrain_type + ".npy")

        np.save(file_name, kinematic_model_parameters)

        return kinematic_model_parameters
        