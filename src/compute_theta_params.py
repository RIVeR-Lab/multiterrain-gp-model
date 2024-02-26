"""Compute θ1 to θ6 using linear regression and low pass filtering. Equations are based on
https://ieeexplore.ieee.org/abstract/document/574537. In essence after re arranging the equations
of motion, we remove the dependency on the noisy acceleration values via a low pass filtering 
both sides of equations. Some useful resources to understand this more are
1) https://underactuated.csail.mit.edu/sysid.html#Gautier97
2) https://www.youtube.com/watch?v=phCpKYUPsXs

In the ICRA paper, we eliminated a section on this due to space constraints
"""

import numpy as np
from sklearn.linear_model import LinearRegression
import os

from configurations import ModelingParameters as JP
from filter import first_order_lpf

def compute_theta(data_container):
    
    """
    Find parameters θ1 to θ6 from the dynamic model

    Args:
        data_container: DataContainer class having processed veloctiies

    Returns:
        numpy array of floats representing the theta parameters.
    """

    # Cutoff frequency in hertz
    cutoff_hz = JP.theta_lpf_cutoff_hz
    
    # Measured velocities in local frame
    loc_lin_vel   = data_container.loc_lin_vel
    loc_ang_vel   = data_container.loc_ang_vel
    
    # Commanded velocities in local frame
    cmd_lin_vel = data_container.cmd_lin_vel
    cmd_ang_vel = data_container.cmd_ang_vel


    lam = 2*np.pi*cutoff_hz
    
    alpha1  = lam * loc_lin_vel - lam * ( first_order_lpf ([loc_lin_vel],cutoff_hz=cutoff_hz) [0] )
    beta1   = -first_order_lpf([loc_ang_vel*loc_ang_vel],cutoff_hz=cutoff_hz) [0]
    gamma1  = first_order_lpf([loc_lin_vel],cutoff_hz=cutoff_hz) [0]
    delta1  = first_order_lpf([cmd_lin_vel],cutoff_hz=cutoff_hz) [0]
    
    alpha2  =  lam*loc_ang_vel - lam* ( first_order_lpf( [loc_ang_vel],cutoff_hz=cutoff_hz)[0] )
    beta2   =  first_order_lpf([loc_lin_vel*loc_ang_vel],cutoff_hz=cutoff_hz)[0]
    gamma2  =  first_order_lpf([loc_ang_vel],cutoff_hz=cutoff_hz)[0]
    delta2  =  first_order_lpf([cmd_ang_vel],cutoff_hz=cutoff_hz)[0]

    X = np.transpose(np.vstack( (alpha1,alpha2,beta1,gamma1,beta2,gamma2) ))
    Y = delta1 + delta2
    
    theta = LinearRegression(fit_intercept=False).fit(X,Y).coef_

    parent_dir = os.path.join(os.getcwd(),"models")

    os.makedirs(parent_dir,exist_ok=True)

    theta_file_name = os.path.join(parent_dir,"theta.npy")

    np.save(theta_file_name,theta)

    return theta