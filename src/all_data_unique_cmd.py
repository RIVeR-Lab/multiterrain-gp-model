#! /usr/bin/python3

'''
Data processing step : From the bulk GP dataset, single out the values which are unique based on the first decimal 
point of the GP inputs. This way we can create a smaller dataset which is the average value of all the unique datasets
A crude way to filter
'''

import os
import pandas as pd
import numpy as np

from configurations import ModelingParameters as JP

class AllDataUniqueCmd:
    
    def __init__(self):
        
        # Loop through all the terrains
        for terrain_type in JP.terrain_types:
            
            # Location of the csv file that contains all the GP data
            all_data_file_path = os.path.join(os.getcwd(),"data",terrain_type,"GP_Datasets","all_gp_data.csv")
            
            # Check to see the file with all the data points exists
            if os.path.isfile(all_data_file_path) == False:
                raise RuntimeError ("Need to ensure the file containing all the GP data points is created first")
            
            # Base directory to store the bulk gp unique dataset
            self.all_data_unique_cmd_file_path = os.path.join(os.getcwd(),"data",terrain_type,"GP_Datasets","unique_gp_cmds.csv")
            
            # Read csv file containing all the data points
            self.data_frames = pd.read_csv(all_data_file_path)
        
            # Load the data from the csv file into this class object
            self.load_from_csv()
        
            # Find all the unique GP input values
            self.unique_gp_data_pts_dict = {}
        
            # This would be same as the dictionary above, just the values will be averged
            # Thus we will no longer have a list of tuples but just one tuple
            # for the error in linear and angular dimensions. We will also drop the counter
            self.unique_gp_data_pts_dict_averaged = {}
        
            self.find_unique_pts()
       
            self.create_unique_gp_dataset()
        
            # Dump data into a csv file
            print("Generating Unique GP Data for terrain: {}..".format(terrain_type))
            
            self.save_dataset()
            
            # Reset the dictionaries to empty for the next terrain
            self.unique_gp_data_pts_dict = {}
            self.unique_gp_data_pts_dict_averaged = {}
        
        
    def save_dataset(self):
        
        key_list = list(self.unique_gp_data_pts_dict.keys())
        
        dataset_dict = {}
        dataset_dict["ip_cmd_lin"]  = []
        dataset_dict["ip_cmd_ang"]  = []
        dataset_dict["ip_curr_lin"] = []
        dataset_dict["ip_curr_ang"] = []
        
        dataset_dict["op_lin_error"] = []
        dataset_dict["op_ang_error"] = []
        
        
        for unique_gp_ip_key in key_list:
            
            gp_ip_element  = unique_gp_ip_key
            gp_op_element  = self.unique_gp_data_pts_dict_averaged[unique_gp_ip_key]
            
            dataset_dict["ip_cmd_lin"].append(gp_ip_element[0])
            dataset_dict["ip_cmd_ang"].append(gp_ip_element[1])
            dataset_dict["ip_curr_lin"].append(gp_ip_element[2])
            dataset_dict["ip_curr_ang"].append(gp_ip_element[3])
            
            dataset_dict["op_lin_error"].append(gp_op_element[0])
            dataset_dict["op_ang_error"].append(gp_op_element[1])
        
        dataset_dict["counter"] = list(range(0,len(dataset_dict["ip_cmd_lin"])))
        data_frame = pd.DataFrame(dataset_dict)
        data_frame.to_csv(self.all_data_unique_cmd_file_path)
        
    
    def create_unique_gp_dataset(self):
        # Create a new bulk dataset which is the average of all the errors
        # both linear and angular for all the unique GP input points
        # We can then make grids for this dataset

        # All the key values based on the unique GP inputs
        key_list = list(self.unique_gp_data_pts_dict.keys())
        
        for unique_gp_ip_key in key_list:
            
            gp_op_tuple_list = self.unique_gp_data_pts_dict[unique_gp_ip_key]

            lin_error = []; ang_error = []

            for gp_op_tuple_idx in range(len(gp_op_tuple_list)):
                lin_error.append(gp_op_tuple_list[gp_op_tuple_idx][0])
                ang_error.append(gp_op_tuple_list[gp_op_tuple_idx][1])

            final_lin_error = np.mean(lin_error); final_ang_error = np.mean(ang_error)
        
            self.unique_gp_data_pts_dict_averaged[unique_gp_ip_key] = (final_lin_error,final_ang_error)
            
        
    def find_unique_pts(self):
        
        '''
        Assigns data points to unique commanded values.We can later average them to denoise
        '''
        
        for idx in range(len(self.counter)):
            gp_ip = (self.gp_ip_cmd_lin[idx],self.gp_ip_cmd_ang[idx],self.gp_ip_curr_lin[idx],self.gp_ip_curr_ang[idx])
            
            current_gp_op_tuple =  (self.gp_op_lin_error[idx] , self.gp_op_ang_error[idx],self.counter[idx]) #lin error, ang error, counter value in original dataset 
            
            if gp_ip not in list(self.unique_gp_data_pts_dict.keys()):
                self.unique_gp_data_pts_dict[gp_ip] = [current_gp_op_tuple] 
            else:
                self.unique_gp_data_pts_dict[gp_ip].append(current_gp_op_tuple)
        
    def load_from_csv(self):
        
        # Load all the relevant data from the all GP dataset specified to the desired number of decimal places
        
        # Commanded linear velocity rounded to desired number of decimal points(defaults to 1)
        self.gp_ip_cmd_lin = np.around(self.data_frames.loc[ : ,"ip_cmd_lin"].to_numpy() , decimals=1)
        
        # Indices list for all the commanded linear velocities equal to zero
        zero_cmd_vel_indices = []
        
        for idx in range(len(self.gp_ip_cmd_lin)):
            if self.gp_ip_cmd_lin[idx] > -0.01 and self.gp_ip_cmd_lin[idx] < 0.01:
                zero_cmd_vel_indices.append(idx)
        
        # Load GP data points from the all_gp_data files. 
        # Remove the points where the commanded linear velocities are zero
        # Truncate the commands to single decimal point
        
        self.file_name = np.delete(self.data_frames.loc[ : ,"file_name"].to_numpy(),zero_cmd_vel_indices)
        
        self.counter = np.asarray(range(len(self.file_name)))
        
        self.gp_ip_cmd_lin = np.delete(np.around(self.data_frames.loc[ : ,"ip_cmd_lin"].to_numpy() , decimals=1),zero_cmd_vel_indices)  
        
        self.gp_ip_curr_lin = np.delete(np.around(self.data_frames.loc[ : ,"ip_curr_lin"].to_numpy() , decimals=1),zero_cmd_vel_indices)
        
        self.gp_ip_cmd_ang = np.delete(np.around(self.data_frames.loc[ : ,"ip_cmd_ang"].to_numpy() , decimals=1),zero_cmd_vel_indices)     
        
        self.gp_ip_curr_ang = np.delete(np.around(self.data_frames.loc[ : ,"ip_curr_ang"].to_numpy(),decimals=1),zero_cmd_vel_indices)
        
        self.gp_op_lin_error = np.delete(self.data_frames.loc[ : ,"op_lin_error"].to_numpy(),zero_cmd_vel_indices)
        
        self.gp_op_ang_error = np.delete(self.data_frames.loc[ : ,"op_ang_error"].to_numpy(),zero_cmd_vel_indices)