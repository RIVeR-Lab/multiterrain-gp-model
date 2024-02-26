"""
Base Class for propagating the mean and variance of the dynamics
"""

import os
import torch
import casadi as ca
from configurations import ModelingParameters as JP
import numpy as np



# Precomputed value of theta
theta = np.load( os.path.join(os.getcwd(),"models/theta.npy") )

theta1 = theta[0]; theta2 = theta[1]; theta3 = theta[2]
theta4 = theta[3]; theta5 = theta[4]; theta6 = theta[5]

# Delta t to propagate the dynamics by
dt = JP.resampled_dt_s

# Distance between robot center of mass and center of the rear axle
a_m = JP.a_m

def nominal_dynamics_ode(x,u):
    
    """
    Summary:
        ODE representation of the nominal robot dynamics without GP compensation
    Args:
        x(ca.MX) -- Current robot state -- [X,Y,psi,v,omega]
        u(ca.MX) -- Currnet robot controls -- [v_ref,omega_ref]
    Returns:
        nom_dyn (ca.MX) -- Nominal dynamics in continuous form
    """
    
    # Robot state space
    X = x[0]; Y = x[1] ; psi = x[2] ; v = x[3] ; omega = x[4]
    
    # Robot control actions
    v_ref = u[0] ; omega_ref = u[1]

    # Equations of motion
    Xdot = v * ca.cos(psi) - a_m * omega * ca.sin(psi)
    Ydot = v * ca.sin(psi) + a_m * omega * ca.cos(psi)
    psidot = omega
    vdot = (theta3/theta1) * omega**2 - (theta4/theta1) * v + (1./theta1) * v_ref
    omegadot = (-theta5/theta2) * v * omega - (theta6/theta2) * omega + (1./theta2) * omega_ref
    
    # Collate the vectors
    nom_dyn = [Xdot,
            Ydot,
            psidot,
            vdot,
            omegadot]
    
    return ca.vertcat(*nom_dyn)

class UncertaintyPropagation:
    def __init__(self,blend_gp):
        self.device = torch.device("cuda:0")
        
        self.blend_gp = blend_gp
        self.data_container = blend_gp.data_container

        self.n = JP.num_robot_states

        self.create_nominal_dynamics_casadi()

    def gp_mean_func(self,model,likelihood,x):
        """
        Summary:
            Generic mean function for a GP linear/angular for any terrain
            Used for computing the jacobian of GP dynamics wrt robot state
        Args:
            model (gpytorch.model) -- GP model type
            likelihood (gpytorch.likelihood) -- Gaussian likelihood function
            x (torch.tensor) -- Query point x (lin vel, ang vel, lin cmd vel, ang cmd vel)
        Returns:
            mean (torch.tensor) -- Mean value for the GP at the query point x
        """

        model.eval(); likelihood.eval()
        predictions = likelihood(model(x))
        mean = predictions.mean
        
        return mean        

    def gp_variance_func(self,model,likelihood,x):

        """
        Summary:
            Generic variance function for a GP linear/angular for any terrain
            Used for computing the contribution of GP uncertainity to overall state uncertainty
        Args:
            model (gpytorch.model) -- GP model type
            likelihood (gpytorch.likelihood) -- Gaussian likelihood function
            x (torch.tensor) -- Query point x (lin vel, ang vel, lin cmd vel, ang cmd vel)
        Returns:
            mean (torch.tensor) -- Variance value for the GP at the query point x
        """

        model.eval(); likelihood.eval()
        predictions = likelihood(model(x))
        var = predictions.variance
        
        return var

    def create_nominal_dynamics_casadi(self):
        # Create CasADi function to compute the derivative of the nominal dynamics
        # wrt robot state for linearizatio based uncertainty propagation
        x = ca.MX.sym('x',5)
        u = ca.MX.sym('u',2)

        # Dictionary to set up the nominal dynamics integrator
        ode_dict = {'x':x, 'p':u  , 'ode': nominal_dynamics_ode(x,u)}

        # Create the discrete time representation of the nominal robot dynamics
        # Last two arguments represent the time duration for which to integrate
        self.nominal_dynamics_integrator = ca.integrator('nominal_dynamics_integrator','cvodes',ode_dict,0.0,JP.resampled_dt_s)

        # CasADi function to compute nominal dynamics based next robot state
        self.nominal_dynamics_func = ca.Function("nominal_dynamics_func",[x,u],[self.nominal_dynamics_integrator(x0=x,p=u)['xf']],\
                                                    ["state","control"] , ["next_state"]  )
        
        # Derivative of nominal robot dynamics wrt robot state, dimension (5X5)
        self.nominal_dynamics_casadi_derivative = ca.Function('nominal_dynamics_casadi_derivative',[x,u],[ca.jacobian( self.nominal_dynamics_integrator(x0=x,p=u)['xf'],x ) ], \
                                                       ["state","control"] , ["jac_x"])


    def get_mean_dict(self):
        return  NotImplementedError

    def get_variance_list(self):
        return  NotImplementedError
    
    def propagate_dynamics(self):
        return  NotImplementedError
    
    