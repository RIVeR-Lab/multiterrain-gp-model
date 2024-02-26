"""
Integration functions for forward simulation
"""

def nominal_dynamics_func(t, vel, *params):
    """
    Dynamics equations considering theta 1 to theta 6.
    Does not consider GP learnt errors.

    Args:
        t (float): Time of evaluation of dynamics
        vel (array): Current linear and angular velocity of the robot
        params (tuple of list) -- params[0] - theta, params[1] - commanded velocities

    Returns:
        veldot (array) : current derivative of velocities based on nominal dynamics
    """

    # Extract linear and angular velocities
    u = vel[0]; omega = vel[1]

    # Extract Parameters θ1 to θ6
    theta1 = params[0][0];theta2=params[0][1];theta3=params[0][2];theta4=params[0][3];theta5=params[0][4];theta6=params[0][5]

    # Extract commanded  velocities
    u_ref = params[1][0]; omega_ref = params[1][1]

    # Nominal velocity dynamics
    vdot = (theta3/theta1)*omega**2 - (theta4/theta1)*u + (1./theta1)*u_ref, \
            (-theta5/theta2)*u*omega - (theta6/theta2)*omega + (1./theta2)*omega_ref
    
    return vdot