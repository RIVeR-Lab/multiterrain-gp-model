import os
from scipy.integrate import solve_ivp
import numpy as np
import torch
import gpytorch
import cvxpy
import pandas as pd
from sklearn import preprocessing

from data_container import DataContainer
from integration_functions import nominal_dynamics_func
from configurations import ModelingParameters as JP
from train_gp import ExactGPModel

"""
The final mean and covariance predictions of the unmodeled dynamics of the SSMR 
at any given time instant are a weighted sum of the predictions from all the GP models. 
Using a look-back history of the robot motion, here we compute these weights using convex optimization

Input  -- time series (cmd, loc),
Output -- (1) all GP means/covs for each terrain (2) weights for each time step
"""

class BlendGP:

    def __init__(self,cmd_file_name, loc_file_name,theta):
        
        self.device = torch.device('cuda:0')

        self.cmd_file_name = cmd_file_name
        self.loc_file_name = loc_file_name
        self.theta = theta

        # Number of terrains
        self.num_terrains = len(JP.terrain_types)

        # Moving horizon look back window
        self.K = JP.look_back_horizon

        # Dictionary of linear and angular velocity models for each terrain and their corresponding likelihoods
        self.linear_gp_models_dict     = {}; self.angular_gp_models_dict = {}
        self.linear_gp_likelihood_dict = {}; self.angular_gp_likelihood_dict = {}

        # Dictionary of mean and variances for the given time series for all the terrain
        self.linear_mean_dict  = {}; self.linear_var_dict  = {}
        self.angular_mean_dict = {}; self.angular_var_dict = {}

        # Weight values at each time instance of the given time series
        # Saved as a dictionary, each terrain gets a value 0.<=w_terrain<=1. 
        self.weights = {}

        # Fill out the models dictionary
        self.load_final_models()

        # Load and process the commanded velocities and ground truth localization values
        self.load_time_series()

        # Predict the errors b/w measured values and nominal physics model using GPs
        # Do this for all the terrains for the entire time series first
        # For the given time series, save the mean and variance for all the terrains
        self.predict_and_store_errors()

        # Then compute the weights for the entire time series in a moving window fashion
        self.compute_weights()

    def predict_and_store_errors(self):
        
        """
        Predict the linear and angular velocity deviations between the measured next state and the nominal physics based model
        """
        
        for terrain_type in JP.terrain_types:
            
            lin_model = self.linear_gp_models_dict[terrain_type]
            lin_likelihood = self.linear_gp_likelihood_dict[terrain_type]
            
            ang_model = self.angular_gp_models_dict[terrain_type]
            ang_likelihood = self.angular_gp_likelihood_dict[terrain_type]
            
            # Assume both the models have the same training inputs
            # If loaded from final_models folder, this is a valid assumption
            
            train_inputs_numpy = np.asarray(lin_model.train_inputs[0].tolist())
            scaler = preprocessing.StandardScaler().fit(train_inputs_numpy)
            train_inputs_numpy = scaler.transform(train_inputs_numpy)
            
            linear_mean_val_list = []; angular_mean_val_list = []
            linear_var_val_list  = []; angular_var_val_list  = []
            
            for idx in range(len(self.ground_truth_ang_error)):
                standardized_input_tensor =  torch.from_numpy(scaler.transform(self.gp_inputs[idx])).to(self.device).float()
                
                # Get ready to make predictions by putting models and likelihoods  in eval mode
                lin_model.eval();lin_likelihood.eval()
                ang_model.eval();ang_likelihood.eval()
                
                with torch.no_grad(), gpytorch.settings.fast_pred_var():
                    lin_predictions_dist = lin_likelihood(lin_model(standardized_input_tensor))
                    ang_predictions_dist = ang_likelihood(ang_model(standardized_input_tensor))
                    
                    linear_mean_val = lin_predictions_dist.mean.cpu().numpy();     angular_mean_val = ang_predictions_dist.mean.cpu().numpy()
                    linear_var_val  = lin_predictions_dist.variance.cpu().numpy(); angular_var_val  = ang_predictions_dist.variance.cpu().numpy()
                    
                    linear_mean_val_list.append(linear_mean_val); angular_mean_val_list.append(angular_mean_val)
                    linear_var_val_list.append(linear_var_val); angular_var_val_list.append(angular_var_val)
            
            self.linear_mean_dict[terrain_type]  = linear_mean_val_list 
            self.angular_mean_dict[terrain_type] = angular_mean_val_list
            self.linear_var_dict[terrain_type]   = linear_var_val_list
            self.angular_var_dict[terrain_type]  = angular_var_val_list

    def compute_weights(self):
        
        """
        Based on the predictions from the GP and the ground truth, compute weights for the entire time series 
        """
        
        #################################### Create the problem #######################################
        
        # Previous Weight
        self.prev_w_var = cvxpy.Parameter((self.num_terrains,1))
        
        # Decision variable -- Weights for each terrain
        self.w_var = cvxpy.Variable((self.num_terrains,1))
        
        # Ground truth for each terrain  for both the GPs.
        # Row vector -- Stacked vertically as:
        # gt_lin_gp_t0,gt_ang_gp_t0,...gt_lin_gp_tK,gt_ang_gp_tK
        self.Y_par = cvxpy.Parameter((2*self.K,1))        
        
        # Pooled mean estimates for all the terrains and all their GP types
        # Each row corresponds to one terrain. Each row tacked vertically as 
        # terrain_1_lin_pred_t0,terrain_1_ang_pred_t0,..,terrain_1_lin_pred_tK,terrain_1_ang_pred_tK
        self.F_par = cvxpy.Parameter((2*self.K,self.num_terrains))

        # Sum of squares error
        objective = cvxpy.sum_squares(self.Y_par - self.F_par @ self.w_var)
            
        # Penalize deviation between previous and current w
        objective += JP.regularization_eps * cvxpy.norm( self.w_var - self.prev_w_var ,1 )
        
        constraints = [self.w_var >=0.0, self.w_var <=1.0 , cvxpy.sum(self.w_var) == 1.0]
        
        self.w_prob = cvxpy.Problem(cvxpy.Minimize(objective) , constraints )

        ############################################################################################



        # Initialize weights for each terrain as having uniform probability for all terrains
        default_weight = 1./len(JP.terrain_types)

        for terrain in JP.terrain_types:
            self.weights[terrain] = self.no_data_pts*[default_weight]
        
        prev_w = np.asarray(self.num_terrains*[default_weight]).reshape(-1,1)
        
        for idx in range(self.no_data_pts-1):
            
            # Let a horizon build up
            if idx < JP.look_back_horizon:
                continue
            
            self.Y_par.value, self.F_par.value = self.compute_Y_and_F(idx)
            
            self.prev_w_var.value = prev_w
            
            self.w_prob.solve(solver=cvxpy.OSQP, polish=True,eps_abs=0.001, adaptive_rho=True, eps_rel=0.001, verbose=False, warm_start=True)
            
            self.w = self.w_var.value
            
            for terrain_idx in range(len(self.w_var.value)):
                self.weights[JP.terrain_types[terrain_idx]][idx] = self.w_var.value[terrain_idx][0]

            prev_w = self.w


    def compute_Y_and_F(self,idx):
        
        """
        Cost function for weights computation is 2-norm squared of ||Y-Fw||
        """
        
        curr_idx = idx; end_idx = curr_idx - self.K
        
        Y = np.zeros( ( 2*self.K , 1 ) )
        F = np.zeros( ( 2*self.K,self.num_terrains ) )
        
        while curr_idx >= end_idx:
            
            for horizon_idx in range(2*self.K):
                
                if horizon_idx % 2 == 0: #linear
                    
                    Y[horizon_idx] = self.ground_truth_lin_error[curr_idx]
                    
                    for terrain_idx in range(len(JP.terrain_types)):
                        F[horizon_idx][terrain_idx] = self.linear_mean_dict[JP.terrain_types[terrain_idx]][curr_idx]
                    
                else: #angular
                    Y[horizon_idx] = self.ground_truth_ang_error[curr_idx]
                    
                    for terrain_idx in range(len(JP.terrain_types)):
                        F[horizon_idx][terrain_idx] = self.angular_mean_dict[JP.terrain_types[terrain_idx]][curr_idx]
                        
            curr_idx -= 1
            
        return (Y,F)
    
    def load_time_series(self):
        
        """
        Load up the commanded velocities and ground truth localization values
        Time sync them, compute ground truth velocities and ground truth deviations between
        measured next state and next state predicted by the nominal physics based model
        """

        # Find the time synced commanded and ground truth velocities
        self.data_container = DataContainer(cmd_file_name=self.cmd_file_name,loc_file_name=self.loc_file_name)
        
        # Number of data points
        self.no_data_pts = len(self.data_container.cmd_time)
        
        params = (self.theta,[0.,0.]) #theta, commanded velocities
        
        # Inputs to the GPs 
        self.gp_inputs = []

        # Outputs of the GPs -- Ground Truths
        self.ground_truth_lin_error = []
        self.ground_truth_ang_error = []
        
        for idx in range(self.no_data_pts - 1):
            params[1][0] = self.data_container.cmd_lin_vel[idx]
            params[1][1] = self.data_container.cmd_ang_vel[idx]
            
            v0 = (self.data_container.loc_lin_vel[idx],self.data_container.loc_ang_vel[idx])
            
            sol = solve_ivp(nominal_dynamics_func,(0,JP.resampled_dt_s),v0,args=params)
            
            if sol.success == True:
                nominal_next_lin_vel = sol.y[0][-1]
                nominal_next_ang_vel = sol.y[1][-1]
            else:
                raise RuntimeError ("Integration unsuccessful, unable to apply nominal unicycle dynamic model")
            
            gp_lin_error = self.data_container.loc_lin_vel[idx+1] - nominal_next_lin_vel
            gp_ang_error = self.data_container.loc_ang_vel[idx+1] - nominal_next_ang_vel
            
            # Plug data into the lists
            inputs = [self.data_container.cmd_lin_vel[idx],self.data_container.cmd_ang_vel[idx], \
                self.data_container.loc_lin_vel[idx],self.data_container.loc_ang_vel[idx]]
            
            inputs_np = np.asarray(inputs).reshape(-1,4)
            
            self.gp_inputs.append(np.asarray(inputs_np)) #1X4
            
            self.ground_truth_lin_error.append(gp_lin_error)
            self.ground_truth_ang_error.append(gp_ang_error)


    def load_final_models(self):

        """
        Loop through all the terrains and the trained GPs for linear and angular velocities
        and store the training datsets and the model into a dictionary
        """

        # Find the best models for all the different terrains
        for terrain_type in JP.terrain_types: # terrain_types = ["Asphalt", "Grass", "Tile"]
            gp_data_base_path = os.path.join(os.getcwd(),"data/{}/GP_Datasets/GP_Training".format(terrain_type))

            for trained_gp_type in JP.trained_gp_types_list: #["Linear","Angular"]
                likelihood = gpytorch.likelihoods.GaussianLikelihood(noise_constraint=gpytorch.constraints.GreaterThan(1e-6),)

                train_inputs_tensor,train_targets_tensor = \
                    self.load_data_from_csv( os.path.join(gp_data_base_path,"Train.csv") , trained_gp_type )
                
                model = ExactGPModel(train_inputs_tensor,train_targets_tensor,likelihood)

                # hyperparam_storage_file = os.path.join(os.getcwd(),trained_gp_type,"model.pth")

                hyperparam_storage_file = \
                    os.path.join(os.getcwd(), "models/{}_{}.pth".format(trained_gp_type,terrain_type))

                state_dict = torch.load(hyperparam_storage_file)

                model.load_state_dict(state_dict)

                if trained_gp_type == "Linear":
                    self.linear_gp_models_dict[terrain_type] = model.to(self.device).float()
                    self.linear_gp_likelihood_dict[terrain_type] = likelihood.to(self.device).float()
                else: #"Angular"
                    self.angular_gp_models_dict[terrain_type] = model.to(self.device).float()
                    self.angular_gp_likelihood_dict[terrain_type] = likelihood.to(self.device).float()

    def load_data_from_csv(self,csv_file,trained_gp_type):
        """
        Summary:
            Load training and testing data from csv files for GP model creation
        Args:
            csv_file : csv file location for the GP training/testing dataset
            trained_gp_type  : "Linear" or "Angular", to decide what the targets will be 
        """
        data_frame = pd.read_csv(csv_file)
        
        inputs_tensor = self.load_gp_inputs(data_frame).contiguous().to(self.device).float()
        
        gp_error = None
        
        if trained_gp_type == "Linear":
            gp_error = data_frame.loc[ : ,"gp_op_lin_error"].to_numpy()
        else:
            gp_error = data_frame.loc[ : ,"gp_op_ang_error"].to_numpy()
            
        outputs_tensor = torch.tensor(gp_error).contiguous().to(self.device).float()
        
        return inputs_tensor,outputs_tensor
    
    def load_gp_inputs(self,data_frame):
        
        # Load the GP inputs from the pandas dataframe
        
        cmd_lin  =  data_frame.loc[ : ,"gp_ip_cmd_lin"].to_numpy().reshape(-1,1)
        cmd_ang  =  data_frame.loc[ : ,"gp_ip_cmd_ang"].to_numpy().reshape(-1,1)   
        curr_lin =  data_frame.loc[ : ,"gp_ip_curr_lin"].to_numpy().reshape(-1,1)   
        curr_ang =  data_frame.loc[ : ,"gp_ip_curr_ang"].to_numpy().reshape(-1,1)
        
        inputs_tensor = torch.tensor(  np.hstack( (cmd_lin,cmd_ang,curr_lin,curr_ang)   )  )
        
        return inputs_tensor
    
