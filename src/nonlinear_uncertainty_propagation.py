"""
Propagation of robot states using Unscented Transforms
"""

import numpy as np
import math
from scipy.linalg import sqrtm
import torch

from uncertainty_propagation import UncertaintyPropagation
from configurations import ModelingParameters as JP


class NonLinearUncertaintyPropagation(UncertaintyPropagation):
    def __init__(self,blend_gp):
        super(NonLinearUncertaintyPropagation, self).__init__(blend_gp)
        
        self.propagate_dynamics()

    def propagate_dynamics(self):
        """
        Based on the SSMR's initial state and a sequence of commanded velocities, predict the mean and variance for
        all the states in response to the commands based on sigma point transform
        """
        # Compute weights for the sigma points
        self.compute_sigma_point_weights()

        # Initialize with the measured initial state of the robot 
        curr_x = self.data_container.loc_x[0]
        curr_y = self.data_container.loc_y[0]
        curr_heading = self.data_container.loc_heading[0]
        curr_lin_vel = self.data_container.loc_lin_vel[0]
        curr_ang_vel = self.data_container.loc_ang_vel[0]

        # Output dictionary, save the results of the predicted states mean
        self.ukf_mean_dict = {"x":[curr_x],
                       "y":[curr_y],
                       "heading":[curr_heading],
                       "lin_vel":[curr_lin_vel],
                       "ang_vel":[curr_ang_vel]}
        
        # State Covariance matrix for each time step
        self.ukf_variance_list = [ np.zeros( (5,5) )]

        # Loop through the command time series
        for idx in range(len(self.data_container.cmd_time)-1):
            
            # Initialize with the initial commanded velocities
            curr_cmd_lin_vel = self.data_container.cmd_lin_vel[idx]
            curr_cmd_ang_vel = self.data_container.cmd_ang_vel[idx]
            
            ############ Compute the contribution of nominal dynamics to the next robot state mean ############
            
            nominal_state = np.array( [curr_x,curr_y,curr_heading,curr_lin_vel,curr_ang_vel] )
            
            gp_next_state_mean = np.zeros( (5) )
            
            gp_input = torch.tensor([curr_cmd_lin_vel,curr_cmd_ang_vel,curr_lin_vel,curr_ang_vel]).reshape(1,4).to(self.device).float()
            
            nominal_variance = self.ukf_variance_list[-1]
            
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
                # The last two states have GP compensation
                gp_next_state_mean[3] += weight_val * lin_gp_output
                gp_next_state_mean[4] += weight_val * ang_gp_output
                
                
                ############# GP Dynamics Contribution to next state variance ###############
                gp_dyn_variance[3][3] = weight_val**2 * self.gp_variance_func(lin_model,lin_likelihood,gp_input)
                gp_dyn_variance[4][4] = weight_val**2 * self.gp_variance_func(ang_model,ang_likelihood,gp_input)
                
            
            #####################################################################################

            # Create an augmented state vector from the nominal dynamics and the GP mean                    
            z = np.concatenate( (nominal_state,gp_next_state_mean ) )
            
            # Create an augmented diagonal matrix from the diagonal elements of the nominal variance matrix and the gp variance matrix
            P = np.block([ [nominal_variance , np.zeros((self.n,self.n))] , [np.zeros((self.n,self.n)),gp_dyn_variance] ])  

            # Square root of augmented_variance matrix
            S = sqrtm(P)
        
            # Compute sigma points for this iteration
            self.compute_sigma_points(z,S)
            
            # Pass sigma points through the nominal + GP dynamics function
            self.process_sigma_points(curr_cmd_lin_vel,curr_cmd_ang_vel)
            
            # Compute the next state mean and variance as a weighted sum of the processed sigma points
            final_next_state = self.weighted_sigma_points_mean()
            final_state_variance = self.weighted_sigma_points_variance(final_next_state)

            # Update the current state for the next prediction step
            curr_x = final_next_state[0]
            curr_y = final_next_state[1]
            curr_heading = final_next_state[2]
            curr_lin_vel = final_next_state[3]
            curr_ang_vel = final_next_state[4]
            
            # Store prediction results back into the dictionary of lists
            self.ukf_mean_dict["x"].append(curr_x)
            self.ukf_mean_dict["y"].append(curr_y)
            self.ukf_mean_dict["heading"].append(curr_heading)
            self.ukf_mean_dict["lin_vel"].append(curr_lin_vel)
            self.ukf_mean_dict["ang_vel"].append(curr_ang_vel)
            
            self.ukf_variance_list.append(final_state_variance)

    def compute_sigma_point_weights(self):
        """
        Summary:
            Compute the weights for all the (4n+1) sigma points
        """
        
        # The weight for the mean sigma point transform is JP.mean_sigma_point_weight
        # Thus, (kappa)/(2n+kappa) = JP.mean_sigma_point_weight
        self.kappa = (2.0 * JP.mean_sigma_point_weight * self.n) / (1.0 - JP.mean_sigma_point_weight)
        
        # Weights for the non-mean (4n) sigma points are thus,
        other_sigma_point_weight = (1.0) / ( (2.0) * (2.0*self.n + self.kappa) ) 
        
        # Compute the weights for the 4n+1 sigma points
        self.sigma_point_weights_np = np.array( (4*self.n+1) * [other_sigma_point_weight] )
        
        # Assign weight to the first sigma point
        self.sigma_point_weights_np[0] = self.kappa / (2*self.n + self.kappa)

    def compute_sigma_points(self,z,S):
        
        """
        Summary:
            Compute the 4n+1 sigma points based on the nominal state dynamics and the GP estimates
        Args:
            z (np.array) -- Augmented robot state [nominaldynamics,GP estimate] shape (10,)
            S (np.array) -- Square root matrix of the augmented robot variance (diag(nominal variance,gp variance))
        """
        
        # Augmented sigma points - nominal dynamics and GP dynamics
        self.augmented_sigma_points = [z]
        
        # Nominal Dynamics sigma point
        self.nominal_sigma_points = [ z[0:self.n] ]
        
        # GP dynamics sigma point
        self.gp_sigma_points = [z[self.n:]]

        # Multiplier to each column of S
        col_multiplier = math.sqrt(2*self.n + self.kappa)

        # Loop through all the columns of S and use that to compute the 4*n sigma points
        for col_idx in range(S.shape[1]):

            # Extract out the given column of S
            S_col = S[:,col_idx]
            
            # Compute the sigma point in one direction of the mean
            augmented_sigma_point_dir_one = z + col_multiplier * S_col
            nominal_sigma_point_dir_one = augmented_sigma_point_dir_one[0:self.n]
            gp_sigma_point_dir_one = augmented_sigma_point_dir_one[self.n:]
            
            self.augmented_sigma_points.append(augmented_sigma_point_dir_one)
            self.nominal_sigma_points.append(nominal_sigma_point_dir_one)
            self.gp_sigma_points.append(gp_sigma_point_dir_one)

            
            # Compute the sigma point in the opposite direction of the mean
            augmented_sigma_point_dir_two = z - col_multiplier * S_col
            nominal_sigma_point_dir_two = augmented_sigma_point_dir_two[0:self.n]
            gp_sigma_point_dir_two = augmented_sigma_point_dir_two[self.n:]
            
            self.augmented_sigma_points.append(augmented_sigma_point_dir_two)
            self.nominal_sigma_points.append(nominal_sigma_point_dir_two)
            self.gp_sigma_points.append(gp_sigma_point_dir_two)
            
        # Log the number of sigma points
        self.num_sigma_pts = len(self.augmented_sigma_points)

    def weighted_sigma_points_mean(self):
        
        """
        Summary:
            Compute weighted sum of sigma points as the final next state prediction
        Returns:
            final_next_state (np.array) -- Final next state prediction
        """
        
        final_next_state = np.zeros(self.n)
        
        for sigma_pt_idx in range(self.num_sigma_pts):
            final_next_state +=  self.sigma_point_weights_np[sigma_pt_idx] * self.propagated_sigma_points[sigma_pt_idx] 
        
        return final_next_state
    
    def weighted_sigma_points_variance(self,final_next_state):
        
        """
        Summary:
            Compute a weighted sum of the variances of all the sigma points from the final state prediction
        Args:
            final_next_state (np.array) -- Final next state prediction
        Return:
            final_state_variance (np.array) -- Final next state variance
        """
        
        final_state_variance = np.zeros( ( self.n,self.n) )
        
        mean_val = final_next_state.reshape(-1,1)
        
        for sigma_pt_idx in range(self.num_sigma_pts):
            current_sigma_point = self.propagated_sigma_points[sigma_pt_idx].reshape(-1,1)

            deviation_vec = current_sigma_point - mean_val
            deviation_vec_transpose = deviation_vec.transpose()
            weight_val = self.sigma_point_weights_np[sigma_pt_idx]

            final_state_variance += weight_val * np.matmul(deviation_vec,deviation_vec_transpose)
        
        return final_state_variance
                
    def process_sigma_points(self,curr_cmd_lin_vel,curr_cmd_ang_vel):
        
        """
        Summary:
            Pass the sigma points through the nominal + GP dynamics and compute 
            values of propagated sigma points. These values are then weighted later
            into overall next state mean and variance
        Args:
            curr_cmd_lin_vel (numpy array) -- Commanded linear velocity
            curr_cmd_ang_vel (numpy array) -- Commanded angular velocity
        """
        
        # 4n+1 sigma points
        total_sigma_pts = len(self.augmented_sigma_points)

        # List of dynamics propagated sigma points
        self.propagated_sigma_points = []
        
        for sigma_pt_idx in range(total_sigma_pts):
            
            state = self.nominal_sigma_points[sigma_pt_idx].tolist()
            control = [curr_cmd_lin_vel,curr_cmd_ang_vel]
            
            nominal_next_state = np.array(self.nominal_dynamics_func(state=state,control=control)["next_state"]).reshape(self.n,)
            gp_next_state = self.gp_sigma_points[sigma_pt_idx]
            
            self.propagated_sigma_points.append(nominal_next_state + gp_next_state)

    def get_mean_dict(self):
        return self.ukf_mean_dict
    
    def get_variance_list(self):
        return self.ukf_variance_list