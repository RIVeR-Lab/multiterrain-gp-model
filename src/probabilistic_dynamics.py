"""
Example script to:
1) Train the GP models based on processed raw data
2) Train benchmark kinematic model
3) Using a test time series, use the trained models and 
    ensemble Gaussian process and compare prediction accuracy

The plots and results in the paper were generated based on running the 
component scripts in this example
"""

import os
import numpy as np
import torch

from compute_theta_params import compute_theta
from data_container import DataContainer
from all_data import AllData
from all_data_unique_cmd import AllDataUniqueCmd
from configurations import ModelingParameters as JP
from compute_edd5_params import KinematicModel,\
    res_extended_diff_drive_five_params as res_fn
from create_gp_train_test import CreateGPTrainTest
from train_gp import GaussianProcess
from blend_gp import BlendGP
from linear_uncertainty_propagation import LinearUncertaintyPropagation
from nonlinear_uncertainty_propagation import NonLinearUncertaintyPropagation

class ProbabilisticDynamics:

    def __init__(self):

        # Ensure proper set-up
        print("Performing sanity checks..")
        self._sanity_checks()

        # Compute θ1 to θ6
        print("Computing θ1 to θ6..")
        self._compute_theta()

        # Data preprocessing
        print("Generating GP data for all terrains..")
        AllData(self.theta)

        print("Generating GP data based on unique command values..")
        AllDataUniqueCmd()

        print("Computing kinematic parameters..")
        self._compute_edd5()
        
        print("Creating GP training/testing data..(This may take a while..)")
        CreateGPTrainTest()

        print("Training GP Models for all terrains..")
        self._train_gp_models()

        # With the GP and benchmark models trained,
        # benchmark these models for a random time-series
        print("Benchmarking kinematic and probabilistic motion models..")
        self._benchmark()

    def _compute_model_errors(self):
        """
        Based on predictions from both the modeling procedure compute cumulative
        absolute errors in linear and angular velocities
        """

        gp_lin_error = 0.; gp_ang_error = 0.
        kinematic_lin_error = 0.; kinematic_ang_error = 0.

        for idx in range(self.benchmark_num_pts):
            # Ground truth
            gt_lin_vel = self.ground_truth_values_dict["lin_vel"][idx]
            gt_ang_vel = self.ground_truth_values_dict["ang_vel"][idx]

            # Probabilistic Motion Model
            gp_lin_vel = self.gp_mean_dict["lin_vel"][idx]
            gp_ang_vel = self.gp_mean_dict["ang_vel"][idx]

            # Kinematic Motion Model
            kinematic_lin_vel = self.kinematic_predictions_dict["lin_vel"][idx]
            kinematic_ang_vel = self.kinematic_predictions_dict["ang_vel"][idx]

            gp_lin_error += np.abs(gt_lin_vel - gp_lin_vel)
            gp_ang_error += np.abs(gt_ang_vel - gp_ang_vel)

            kinematic_lin_error += np.abs(gt_lin_vel - kinematic_lin_vel)
            kinematic_ang_error += np.abs(gt_ang_vel - kinematic_ang_vel)
        
        print("GP Linear Error: {}, Angular Error: {}".format(gp_lin_error,gp_ang_error))
        print("EDD5 Linear Error: {}, Angular Error: {}".format(kinematic_lin_error,kinematic_ang_error))

    def _benchmark(self):
        # Randomly chosing a high twist, high linear velocity
        # time-series for validating modeling methods accuracy

        terrain_type = "Tile"
        file_number = "164"

        base_folder_name = os.path.join(os.getcwd(),"data/Tile/ConstantVel")
        benchmark_cmd_file_name = os.path.join(base_folder_name,"cmd/{}.txt".format(file_number))
        benchmark_loc_file_name = os.path.join(base_folder_name,"loc/{}.txt".format(file_number))

        # Loop through the time series and compute the weights for each terrain
        blend_gp = BlendGP(cmd_file_name=benchmark_cmd_file_name,loc_file_name=benchmark_loc_file_name,theta=self.theta)
        _data_container = blend_gp.data_container

        # Propagate mean and variance of the dynamics
        # We will compare the mean with the corresponding
        # predictions from the kinematic model
        
        probabilistic_dynamics = None

        if JP.uncertainty_propagation_method == "Linear":
            probabilistic_dynamics = LinearUncertaintyPropagation(blend_gp)
        elif JP.uncertainty_propagation_method == "NonLinear":
            probabilistic_dynamics = NonLinearUncertaintyPropagation(blend_gp)
        else:
            raise NotImplementedError("Wrong uncertainty propagation config value")
        
        # Extract the mean and variance of the predictions
        self.gp_mean_dict = probabilistic_dynamics.get_mean_dict()
        self.variance_list = probabilistic_dynamics.get_variance_list()

        ####### Compute predictions from the kinematic model #######
        edd5_params = np.load(os.path.join(os.getcwd(), "models/edd_5_{}.npy".format(terrain_type)))  

        # Compute predicted velocities from kinematic model
        self.compute_kinematic_predictions(edd5_params,_data_container)

        self._compute_model_errors()

    def compute_kinematic_predictions(self, edd5_params, _data_container):
        """
        Compute the one-step linear and angular velocity predictions based on kinematic model
        Also extract ground truth velocities for benchmarking later

        Args:
            edd5_params : Parameters of the kinematic model
            _data_container : DataContainer object with ground truths and commands
        """

        # Dictionaries for kinematic predictions, ground truth and commanded velocities
        self.kinematic_predictions_dict = {"lin_vel":[_data_container.loc_lin_vel[0]],"ang_vel":[_data_container.loc_ang_vel[0]]}
        self.ground_truth_values_dict   = {"lin_vel":[_data_container.loc_lin_vel[0]],"ang_vel":[_data_container.loc_ang_vel[0]]}
        self.commanded_values_dict = {"lin_vel":[_data_container.cmd_lin_vel[0]],"ang_vel":[_data_container.cmd_ang_vel[0]]}

        self.benchmark_num_pts = len(_data_container.cmd_time)

        for idx in range(len(_data_container.cmd_time)-1):
            # Extract the current commanded linear and angular velocities
            curr_cmd_left_wheel_speed = _data_container.cmd_left_wheel_speed[idx]
            curr_cmd_right_wheel_speed = _data_container.cmd_right_wheel_speed[idx]

            # Actual final next state
            self.ground_truth_values_dict["lin_vel"].append(_data_container.loc_lin_vel[idx+1])  
            self.ground_truth_values_dict["ang_vel"].append(_data_container.loc_ang_vel[idx+1])

            # Commanded next state
            self.commanded_values_dict["lin_vel"].append(_data_container.cmd_lin_vel[idx+1])
            self.commanded_values_dict["ang_vel"].append(_data_container.cmd_ang_vel[idx+1])
            
            # Setup for sending to kinematic model
            x = edd5_params
            cmd_w_l = curr_cmd_left_wheel_speed
            cmd_w_r = curr_cmd_right_wheel_speed

            # Since we are using the residual function, 
            # just set what is being subtracted from model output to zero
            vx_hat = np.array([0.]); vy_hat = np.array([0.]); vheading_hat = np.array([0.])

            # Output of the robot model
            benchmark_vel = res_fn(x,cmd_w_l,cmd_w_r,vx_hat,vy_hat,vheading_hat)

            # Plug back into the dictionary
            self.kinematic_predictions_dict["lin_vel"].append(benchmark_vel[0])
            self.kinematic_predictions_dict["ang_vel"].append(benchmark_vel[2]) 

    def _train_gp_models(self):
        """
        Train the GP models for all the terrains 
        and the linear and angular velocity errors
        """
        for terrain_type in JP.terrain_types:
            for gp_type in JP.trained_gp_types_list:
                # Locate the GP train and test datasets for the particular terrain
                training_dataset = os.path.join(os.getcwd(),"data",terrain_type,"GP_Datasets","GP_Training","Train.csv")
                testing_dataset = os.path.join(os.getcwd(),"data",terrain_type,"GP_Datasets","GP_Training","Test.csv")
                model_file_name = os.path.join(os.getcwd(),"models/{}_{}.pth".format(gp_type,terrain_type))

                GaussianProcess(gp_type,terrain_type,training_dataset,testing_dataset,model_file_name)

    def _compute_edd5(self):
        """Compute the kinematic model parameters for each terrain"""
        # Build a dictionary of terrains and their kinematic parameters
        self.kinematic_models_per_terrain_dict = {}

        for terrain_type in JP.terrain_types:
            # Locate the command and localization files corresponding to long distance
            connecting_str = "data/" + terrain_type + "/LongDistance"
            base_path_name = os.path.join(os.getcwd(),connecting_str)
            edd5_cmd_file_name = os.path.join(base_path_name,"cmd/1.txt")
            edd5_loc_file_name = os.path.join(base_path_name,"loc/1.txt")

            kinematic_model = KinematicModel(DataContainer(edd5_cmd_file_name,edd5_loc_file_name), terrain_type)

            kinematic_model_parameters = kinematic_model.compute_paramters()

            self.kinematic_models_per_terrain_dict[terrain_type] = kinematic_model_parameters

    def _compute_theta(self):
        # Find parameters θ1 to θ6 from the dynamic model based on asphalt 
        # Long Distance time series. Since these parameters do not depend
        # on the terrain, any long distance dataset file would work
        base_path_name = os.path.join(os.getcwd(),"data/Asphalt/LongDistance")
        theta_cmd_file_name = os.path.join(base_path_name,"cmd/1.txt")
        theta_loc_file_name = os.path.join(base_path_name,"loc/1.txt")
        self.theta = compute_theta(DataContainer(theta_cmd_file_name, theta_loc_file_name))

    def _sanity_checks(self):
        # Ensure GPU
        if torch.cuda.is_available() == False:
            raise RuntimeError("GPU needed for training Gaussian Process Regression Model")

        # Check the data folder has been populated from google drive
        if os.path.exists(os.path.join(os.getcwd(),"data")) == False:
            raise RuntimeError("Please download dataset as in the README")

if __name__ == "__main__":
    print("-------")
    ProbabilisticDynamics()
    print("-------")