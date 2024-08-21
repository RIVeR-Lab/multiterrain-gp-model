#!/usr/bin/env python3

import os
import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture as GMM
from sklearn.metrics import pairwise_distances_argmin_min

from configurations import ModelingParameters as JP

class CreateGPTrainTest:
    
    def __init__(self, common_csv_file, num_train_pts=None, num_test_pts=None):
        """
        Initialize with the path to the combined CSV file and parameters for training/testing points.
        """
        self.common_csv_file = common_csv_file
        self.num_train_pts = num_train_pts
        self.num_test_pts = num_test_pts
        
        # Load data from the combined CSV file
        self.load_from_csv()

        # Create GP dataset by clustering points and selecting training/testing points
        self.create_gp_dataset()
        
    def load_from_csv(self):
        """
        Extract data from the combined CSV file into members of this class.
        """
        data_frames = pd.read_csv(self.common_csv_file)

        self.ip_cmd_lin    = data_frames.loc[:, "ip_cmd_lin"].to_numpy()
        self.ip_cmd_ang    = data_frames.loc[:, "ip_cmd_ang"].to_numpy()
        self.ip_curr_lin   = data_frames.loc[:, "ip_curr_lin"].to_numpy()
        self.ip_curr_ang   = data_frames.loc[:, "ip_curr_ang"].to_numpy()
        
        self.asphalt_op_lin_error  = data_frames.loc[:, "Asphalt_Lin_Error"].to_numpy()
        self.asphalt_op_ang_error  = data_frames.loc[:, "Asphalt_Ang_Error"].to_numpy()

        self.grass_op_lin_error    = data_frames.loc[:, "Grass_Lin_Error"].to_numpy()
        self.grass_op_ang_error    = data_frames.loc[:, "Grass_Ang_Error"].to_numpy()

        self.tile_op_lin_error     = data_frames.loc[:, "Tile_Lin_Error"].to_numpy()
        self.tile_op_ang_error     = data_frames.loc[:, "Tile_Ang_Error"].to_numpy()
    
    def create_gp_dataset(self):
        """
        Cluster the points and then select training and testing points.
        The point in each cluster that is closest to the centroid is chosen as
        the training point. Testing point is any other point in the cluster at random.
        """
        
        # Total number of points 
        total_num_pts = len(self.ip_cmd_lin)
        
        # Input to the clustering
        X = np.vstack((self.ip_cmd_lin, self.ip_cmd_ang, self.ip_curr_lin, self.ip_curr_ang)).T
        
        # Compute the centroids based on the GMM clustering algorithm
        gmm = GMM(n_components=self.num_train_pts, n_init=3, random_state=0)
        gmm.fit(X)
        centroids = gmm.means_

        # Find training points as the points in the cluster closest to the centroid
        self.clustering_training_pts_idx, _ = pairwise_distances_argmin_min(centroids, X)
        
        # Select training points
        self.clustering_training_pt_ip = X[self.clustering_training_pts_idx]

        # Select testing points
        indices_array = np.arange(total_num_pts)
        indices_not_used_in_training = np.delete(indices_array, self.clustering_training_pts_idx)
        self.clustering_testing_pts_idx = np.random.choice(indices_not_used_in_training, self.num_test_pts, replace=False)
        self.clustering_testing_pt_ip = X[self.clustering_testing_pts_idx]

        # Save the outputs of the clustering min dist as training points
        training_dict = {
            'gp_ip_cmd_lin': self.clustering_training_pt_ip[:, 0],
            'gp_ip_cmd_ang': self.clustering_training_pt_ip[:, 1],
            'gp_ip_curr_lin': self.clustering_training_pt_ip[:, 2],
            'gp_ip_curr_ang': self.clustering_training_pt_ip[:, 3],
            'Asphalt_Lin_Error': self.asphalt_op_lin_error[self.clustering_training_pts_idx],
            'Asphalt_Ang_Error': self.asphalt_op_ang_error[self.clustering_training_pts_idx],
            'Grass_Lin_Error': self.grass_op_lin_error[self.clustering_training_pts_idx],
            'Grass_Ang_Error': self.grass_op_ang_error[self.clustering_training_pts_idx],
            'Tile_Lin_Error': self.tile_op_lin_error[self.clustering_training_pts_idx],
            'Tile_Ang_Error': self.tile_op_ang_error[self.clustering_training_pts_idx],
        }

        testing_dict = {
            'gp_ip_cmd_lin': self.clustering_testing_pt_ip[:, 0],
            'gp_ip_cmd_ang': self.clustering_testing_pt_ip[:, 1],
            'gp_ip_curr_lin': self.clustering_testing_pt_ip[:, 2],
            'gp_ip_curr_ang': self.clustering_testing_pt_ip[:, 3],
            'Asphalt_Lin_Error': self.asphalt_op_lin_error[self.clustering_testing_pts_idx],
            'Asphalt_Ang_Error': self.asphalt_op_ang_error[self.clustering_testing_pts_idx],
            'Grass_Lin_Error': self.grass_op_lin_error[self.clustering_testing_pts_idx],
            'Grass_Ang_Error': self.grass_op_ang_error[self.clustering_testing_pts_idx],
            'Tile_Lin_Error': self.tile_op_lin_error[self.clustering_testing_pts_idx],
            'Tile_Ang_Error': self.tile_op_ang_error[self.clustering_testing_pts_idx],
        }
        
        # Save as CSV
        base_save_path = os.path.dirname(self.common_csv_file)
        train_path = os.path.join(base_save_path,"Train.csv")
        test_path  = os.path.join(base_save_path,"Test.csv")
        
        os.makedirs(os.path.dirname(train_path), exist_ok=True)
        pd.DataFrame(training_dict).to_csv(train_path, index=False)
        pd.DataFrame(testing_dict).to_csv(test_path, index=False)
        
        print(f"Training data saved to: {train_path}")
        print(f"Testing data saved to: {test_path}")


# Set the path to your combined CSV file
common_csv_file = "/home/ananya/Code/multiterrain-gp-model/data/combined_terrain_errors/combined_terrain_errors.csv"

# Run the process to create the GP datasets
gp_creator = CreateGPTrainTest(common_csv_file, JP.gp_clustering_num_train_pts,JP.gp_clustering_num_test_pts)
