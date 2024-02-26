"""
Propagation of robot states using Taylor series expansion
"""

import numpy as np
import torch

from uncertainty_propagation import UncertaintyPropagation
from configurations import ModelingParameters as JP


class LinearUncertaintyPropagation(UncertaintyPropagation):
    def __init__(self,blend_gp):
        super(LinearUncertaintyPropagation, self).__init__(blend_gp)

        self.propagate_dynamics()

    def propagate_dynamics(self):
        
        """
        Based on the SSMR's initial state and a sequence of commanded velocities, predict the mean and variance for
        all the states in response to the commands based on Taylor series expansion
        """

        # Initialize with the measured initial state of the robot and the commanded values
        curr_x = self.data_container.loc_x[0]
        curr_y = self.data_container.loc_y[0]
        curr_heading = self.data_container.loc_heading[0]
        curr_lin_vel = self.data_container.loc_lin_vel[0]
        curr_ang_vel = self.data_container.loc_ang_vel[0]

        # Output dictionary, save the results of the predicted states mean
        self.ekf_mean_dict = {"x":[curr_x],
                       "y":[curr_y],
                       "heading":[curr_heading],
                       "lin_vel":[curr_lin_vel],
                       "ang_vel":[curr_ang_vel]}

        # State Covariance matrix for each time step
        self.ekf_variance_list = [np.zeros( (5,5) )] 

        # Loop through the commands time series
        for idx in range(len(self.data_container.cmd_time)-1):
            # Extract the current commanded linear and angular velocities
            curr_cmd_lin_vel = self.data_container.cmd_lin_vel[idx]
            curr_cmd_ang_vel = self.data_container.cmd_ang_vel[idx]

            ############ Compute the contribution of nominal dynamics to the next robot state mean ############
            # Initial robot state
            state   = [curr_x,curr_y,curr_heading,curr_lin_vel,curr_ang_vel]
            # Applied control action
            control = [curr_cmd_lin_vel,curr_cmd_ang_vel]
            nominal_next_state = np.array(self.nominal_dynamics_func(state=state , control = control)["next_state"]).reshape(5,)   

            ############### Compute the contribution of nominal dynamics to the next robot state variance ##############
            # Derivative of the nominal casadi dynamics
            casadi_derivative = self.nominal_dynamics_casadi_derivative(state=state,control=control)["jac_x"]
            # Nominal dynamics derivative
            nominal_dyn_derivative = np.array(casadi_derivative)

            #########################################################################################################

            ############### Compute the contribution of GP dynamics to the next robot state mean ##############

            gp_next_state_mean = np.zeros( (5) )
            gp_dyn_derivative = np.zeros ( ( 5,5 ) )

            gp_input = torch.tensor([curr_cmd_lin_vel,curr_cmd_ang_vel,curr_lin_vel,curr_ang_vel]).reshape(1,4).to(self.device).float()

            gp_dyn_variance = np.zeros( ( 5,5) )

            # Loop through each terrain, calculate the linear and angular means and variances for those terrains 
            # Using weights from the history of observations, compute  a weighted sum of the means and variances
            for terrain in JP.terrain_types:

                # Extract the linear and angular velocity models and likelihood
                lin_model = self.blend_gp.linear_gp_models_dict[terrain]
                ang_model = self.blend_gp.angular_gp_models_dict[terrain]
                lin_likelihood = self.blend_gp.linear_gp_likelihood_dict[terrain]
                ang_likelihood = self.blend_gp.angular_gp_likelihood_dict[terrain]

                # Weight for this time instant for this terrain based on history of observations
                weight_val = self.blend_gp.weights[terrain][idx]

                ############## Mean Propagation ##############
                # Contribution of linear and angular means                
                lin_gp_output = self.gp_mean_func(lin_model,lin_likelihood,gp_input)
                ang_gp_output = self.gp_mean_func(ang_model,ang_likelihood,gp_input)

                # The first three states are kinematic and thus only integrated from the dynamics
                # The last two dynamic states have GP compensation
                gp_next_state_mean[3] += weight_val * lin_gp_output
                gp_next_state_mean[4] += weight_val * ang_gp_output

                ############## Setup for Taylor Series Component of Variance Propagation ##############

                # Derivative of the GP dynamics for this terrain
                terrain_gp_dyn_derivative = np.zeros( (5,5) )

                # Compute the derivative of the linear mean dynamics at the query point
                lin_gp_der = torch.autograd.functional.jacobian(lambda x : self.gp_mean_func(lin_model,lin_likelihood,x),gp_input).squeeze()
                ang_gp_der = torch.autograd.functional.jacobian(lambda x : self.gp_mean_func(ang_model,ang_likelihood,x),gp_input).squeeze()
                
                # Order of GP inputs -- cmd_lin, cmd_ang, curr_lin, curr_ang, so we extract the derivatives using following indices
                terrain_gp_dyn_derivative[3,3] = lin_gp_der[2]; terrain_gp_dyn_derivative[3,4] = lin_gp_der[3]
                terrain_gp_dyn_derivative[4,3] = ang_gp_der[2]; terrain_gp_dyn_derivative[4,4] = ang_gp_der[3]
                
                # Weighted sum of GP dynamics derivative
                gp_dyn_derivative +=  weight_val * terrain_gp_dyn_derivative
                
                ############# GP Dynamics Contribution to next state variance(weight squares here) ###############
                gp_dyn_variance[3][3] = weight_val**2 * self.gp_variance_func(lin_model,lin_likelihood,gp_input)
                gp_dyn_variance[4][4] = weight_val**2 * self.gp_variance_func(ang_model,ang_likelihood,gp_input)
                
            #####################################################################################
                
            #########Output for this iteration###########
                                                         
            # Effective next state prediction is the sum total of the contributions
            final_next_state = nominal_next_state + gp_next_state_mean
            
            # Update the current state for the next prediction step
            curr_x = final_next_state[0]
            curr_y = final_next_state[1]
            curr_heading = final_next_state[2]
            curr_lin_vel = final_next_state[3]
            curr_ang_vel = final_next_state[4]
            
            # Store prediction results back into the dictionary of lists
            self.ekf_mean_dict["x"].append(curr_x)
            self.ekf_mean_dict["y"].append(curr_y)
            self.ekf_mean_dict["heading"].append(curr_heading)
            self.ekf_mean_dict["lin_vel"].append(curr_lin_vel)
            self.ekf_mean_dict["ang_vel"].append(curr_ang_vel)
            
            # Computing effective next state variance
            prev_state_variance = self.ekf_variance_list[-1]
            
            # Gradient of the total dynamics 
            total_dyn_der = nominal_dyn_derivative + gp_dyn_derivative  ; total_dyn_der_transpose = total_dyn_der.transpose()          
            
            # Propagation from previous state uncertainty via linearization
            
            taylor_variance = np.matmul( np.matmul(total_dyn_der,prev_state_variance) , total_dyn_der_transpose )    
            
            final_state_variance = gp_dyn_variance + taylor_variance

            # Add to final measurement list
            self.ekf_variance_list.append(final_state_variance)

    def get_mean_dict(self):
        return self.ekf_mean_dict
    
    def get_variance_list(self):
        return self.ekf_variance_list