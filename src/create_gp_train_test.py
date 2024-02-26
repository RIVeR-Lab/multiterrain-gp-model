#! /usr/bin/python3

'''
Data processing step: From the all_gp_data and unique_gp_data csv files create two types of 
GP training and testing datasets
1) Split the data sets into train and test based on a configured ratio
2) Reduce the number of training points to a configured value 
   by using clustering via GMMs
'''

import os
import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture as GMM
from sklearn.metrics import pairwise_distances_argmin_min

from configurations import ModelingParameters as JP

class CreateGPTrainTest:
    
    def __init__(self):
        
        # Loop through all the terrains
        for terrain_type in JP.terrain_types:
            
            data_location = os.path.join(os.getcwd(),"data",terrain_type,"GP_Datasets","unique_gp_cmds.csv")
            
            data_frames = pd.read_csv(data_location)

            self.load_from_csv(data_frames)

            if os.path.isfile(data_location) == False:
                raise RuntimeError("Need to create GP datasets before running this script")
            
            base_save_path = os.path.join(os.getcwd(),"data",terrain_type,"GP_Datasets","GP_Training")
            
            os.makedirs(base_save_path,exist_ok=True)

            self.create_gp_dataset(base_save_path)
        
    def load_from_csv(self,data_frames):
        
        '''
        Extract data from the csv file into members of this class
        '''

        self.ip_cmd_lin    = data_frames.loc[ : , "ip_cmd_lin"].to_numpy()
        self.ip_cmd_ang    = data_frames.loc[ : , "ip_cmd_ang"].to_numpy()
        self.ip_curr_lin   = data_frames.loc[ : , "ip_curr_lin"].to_numpy()
        self.ip_curr_ang   = data_frames.loc[ : , "ip_curr_ang"].to_numpy()
        
        self.op_lin_error  = data_frames.loc[ : , "op_lin_error"].to_numpy()
        self.op_ang_error  = data_frames.loc[ : , "op_ang_error"].to_numpy()
    
    def create_gp_dataset(self,base_save_path):
        """
        Cluster the points and then select training and testing points
        The point in each cluster that is closest to the centroid is chosen as
        the training point. Testing point is any other point in the cluster at random
        """
        
        # Total number of points 
        total_num_pts = len(self.ip_cmd_lin)
        
        # Input to the clustering
        X = []
        
        #Corresponding output
        Y = []
        
        for point_idx in range(total_num_pts):
            # Input ordering : cmd lin, cmd ang, curr lin, curr ang
            new_pt_ip = [self.ip_cmd_lin[point_idx],self.ip_cmd_ang[point_idx],self.ip_curr_lin[point_idx],self.ip_curr_ang[point_idx]]
            
            # Output ordering : op lin error, op ang error
            new_pt_op = [self.op_lin_error[point_idx],self.op_ang_error[point_idx]]
            
            X.append(new_pt_ip); Y.append(new_pt_op)
        
        # Convert back to array
        X = np.asarray(X); Y = np.asarray(Y)            
        contiguous_X = np.ascontiguousarray(X)
        centroids = None
        
        # Compute the centroids based on the GMM clustering algorithm
        centroids = GMM(n_components=JP.gp_clustering_num_train_pts,n_init=3,random_state=0).fit(X).means_

        # Find training pts as the points in the cluster closest to the centroid
        self.clustering_training_pts_idx,_ = pairwise_distances_argmin_min(centroids,contiguous_X)
        
        # Save the outputs of the clustering min dist as training points
        self.clustering_training_pt_ip = X[self.clustering_training_pts_idx]
        self.clustering_training_pt_op = Y[self.clustering_training_pts_idx]

        # To generate the test, remove the elements from the training pt indices
        indices_array = np.asarray(list(range(total_num_pts)))
        
        indices_not_used_in_training = np.delete(indices_array,self.clustering_training_pts_idx) 

        self.clustering_testing_pts_idx = np.random.choice(indices_not_used_in_training,JP.gp_clustering_num_test_pts,replace=False)
        
        self.clustering_testing_pt_ip = X[self.clustering_testing_pts_idx]
        self.clustering_testing_pt_op = Y[self.clustering_testing_pts_idx]

        # Generate Training and Testing dictionaries
        training_dict = {}
        testing_dict = {}
        
        # Load elements into the dictionary
        
        # Training dictionary
        training_dict['gp_ip_cmd_lin'] = self.clustering_training_pt_ip[:,0].tolist()
        training_dict['gp_ip_cmd_ang'] = self.clustering_training_pt_ip[:,1].tolist()
        training_dict['gp_ip_curr_lin']= self.clustering_training_pt_ip[:,2].tolist()
        training_dict['gp_ip_curr_ang']= self.clustering_training_pt_ip[:,3].tolist()
        
        training_dict['gp_op_lin_error']=self.clustering_training_pt_op[:,0].tolist()
        training_dict['gp_op_ang_error']=self.clustering_training_pt_op[:,1].tolist()
        
        # Testing Dictionary
        testing_dict['gp_ip_cmd_lin'] = self.clustering_testing_pt_ip[:,0].tolist()
        testing_dict['gp_ip_cmd_ang'] = self.clustering_testing_pt_ip[:,1].tolist()
        testing_dict['gp_ip_curr_lin']= self.clustering_testing_pt_ip[:,2].tolist()
        testing_dict['gp_ip_curr_ang']= self.clustering_testing_pt_ip[:,3].tolist()
        
        testing_dict['gp_op_lin_error']=self.clustering_testing_pt_op[:,0].tolist()
        testing_dict['gp_op_ang_error']=self.clustering_testing_pt_op[:,1].tolist()
        
        training_dataframe = pd.DataFrame(training_dict)
        testing_dataframe  = pd.DataFrame(testing_dict)
        
        # Save as csv
        train_path = os.path.join(base_save_path,"Train.csv")
        test_path  = os.path.join(base_save_path,"Test.csv")
        
        training_dataframe.to_csv(train_path)
        testing_dataframe.to_csv(test_path)