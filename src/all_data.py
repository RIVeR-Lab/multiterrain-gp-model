'''
Data processing step. Based on user set parameters for computing velocities, loops through all the available terrains
and the corresponding dataset folder types and extracts out the outputs for Gaussian Process Regression (GPR)
Outputs saved as a csv file to be later for GPR training.
'''

import os
from scipy.integrate import solve_ivp
import pandas as pd
import numpy as np

from data_container import DataContainer
from configurations import ModelingParameters as JP
from integration_functions import nominal_dynamics_func

class AllData:

    def __init__(self,theta):
        
        """
        Takes in data from all the terrains, converts them into GP inputs
        and outputs. Stores the results for model training
        
        Args:
            theta : Pre-computed θ1 to θ6 for the dynamic unicycle model
        """

        # θ1 to θ6 in the dynamic unicycle model
        self.theta = theta

        # Counter to keep track of the datapoint number
        self.counter = []

        # Source file name from which the particular datapoint originated
        self.file_name = []

        ######### GP Inputs: Saved as a list #########
        # current commanded linear velocity, current commanded angular velocity, 
        # current linear velocity, current angular velocity in order
        self.gp_ip_cmd_lin  =[]; self.gp_ip_cmd_ang=[]
        self.gp_ip_curr_lin =[];self.gp_ip_curr_ang=[]

        ######### GP Outputs #########
        # The deviation/error between measured next state and
        # the one estimated using dynamics model containing effects of θ1-θ6
        self.gp_op_lin_error=[];self.gp_op_ang_error=[]

        # Fill out the GP input/output lists
        # Remove zero commanded linear and angular velocities
        # Save the resultant dataset as a csv file
        self.create_and_save_dataset()

    def create_and_save_dataset(self):
        
        """
        Compute differences in measured next velocities and those suggested by the nominal model
        Remove zero commanded linear and angular velocities. Save the dataset as a csv file
        """

        # Loop through all the terrains
        for terrain_type in JP.terrain_types:
            
            # Reset member variables containing data set from previous terrain
            # before starting data collect for this terrain
            self.reset()

            # Loop over the dataset types
            for dataset_type in JP.dataset_types:
                
                # Get to the commanded velocity and ground truth localization folders
                base_path_name = os.path.join(os.getcwd(),"data",terrain_type,dataset_type)
                cmd_base_path_name = os.path.join(base_path_name,"cmd")
                loc_base_path_name = os.path.join(base_path_name,"loc")

                # Number of files in the directory
                no_files = len(os.listdir(cmd_base_path_name))

                # Get to the cmd vel and localization files
                for file_number_idx in range(1,no_files+1):
                    file_number = str(file_number_idx) + ".txt"
                    cmd_file_name = os.path.join(cmd_base_path_name,file_number)
                    loc_file_name = os.path.join(loc_base_path_name,file_number)

                    # Process commands and localization
                    data_container = DataContainer(cmd_file_name=cmd_file_name,loc_file_name=loc_file_name)

                    # No of data points contained in the resampled commands and ground truth velocities
                    no_data_pts = len(data_container.cmd_time)

                    # For use of integration function dynamics_func
                    params = (self.theta,[0.,0.])  #theta,commanded velocities lin and ang

                    # Loop through all the data points and compute individual one step
                    # GP inputs and outputs
                    for idx in range(no_data_pts-1):
                        
                        # Data point number
                        self.counter.append(len(self.counter))

                        # Current file name
                        curr_file_name = terrain_type + "_" + dataset_type + "_" +str(file_number_idx)
                        self.file_name.append(curr_file_name)

                        # Set the params to commanded linear and angular velocities
                        params[1][0] = data_container.cmd_lin_vel[idx]
                        params[1][1] = data_container.cmd_ang_vel[idx]

                        # Initial condition of the linear and angular velocities
                        v0 = (data_container.loc_lin_vel[idx],data_container.loc_ang_vel[idx])

                        # Solve the initial value problem to predict the linear and angular
                        # velocities at the next time step in absence of GP correction 
                        sol = solve_ivp(nominal_dynamics_func,(0,JP.resampled_dt_s),v0,args=params)

                        if sol.success == True: #Check for success of integration
                            
                            # Extract out solution of integration as  predicted next linear and angular velocities
                            predicted_next_lin_vel = sol.y[0][-1]
                            predicted_next_ang_vel = sol.y[1][-1]
                            
                            # Difference between measured next state velocity and predicted next state velocity
                            # This is the output of the GP.
                            gp_lin_error = data_container.loc_lin_vel[idx+1] - predicted_next_lin_vel
                            gp_ang_error = data_container.loc_ang_vel[idx+1] - predicted_next_ang_vel
                            
                            self.gp_op_lin_error.append(gp_lin_error)
                            self.gp_op_ang_error.append(gp_ang_error)
                            
                            self.gp_ip_cmd_lin.append(data_container.cmd_lin_vel[idx])
                            self.gp_ip_cmd_ang.append(data_container.cmd_ang_vel[idx])
                            self.gp_ip_curr_lin.append(data_container.loc_lin_vel[idx])
                            self.gp_ip_curr_ang.append(data_container.loc_ang_vel[idx])
                        
                        else: #integration not successful
                            raise RuntimeError ("Could not integrate the dynamics function")

            # Remove the points which have zeros as commanded linear velocities
            self.remove_zero_cmds()

            # Save the dataset into a csv file
            self.save_dataset(terrain_type)

    def reset(self):
        
        """
        Reset all the data lists before starting on
        the next terrain
        """
        
        self.counter = []
        self.file_name = []
        self.gp_ip_cmd_lin  =[]; self.gp_ip_cmd_ang=[];self.gp_ip_curr_lin=[];self.gp_ip_curr_ang=[]
        self.gp_op_lin_error=[];self.gp_op_ang_error=[]

    def remove_zero_cmds(self):
        '''
        Points with zero commanded linear and angular velocities form ~10 % of the dataset
        These points dont add much value since the robot is being asked to be stationary and we assume
        the skid can be ignored for this corner case.
        '''
        indices_to_delete = []
        
        # Find the indices where the robot is asked to come to a stop
        # that is commanded velocity is zero. These points constitute around
        # 11 % of the total dataset but should not lead to any noticeable skid 
        # When infering on the robot, we can just use the physics based model 
        
        for idx in range(len(self.gp_ip_cmd_lin)):
            if self.gp_ip_cmd_lin[idx] > -0.01 and self.gp_ip_cmd_lin[idx] < 0.01:
                indices_to_delete.append(idx)
        
        # Remove the elements from the dataset one-by-one
        self.gp_ip_cmd_lin = np.delete(self.gp_ip_cmd_lin,indices_to_delete)
        self.gp_ip_cmd_ang = np.delete(self.gp_ip_cmd_ang,indices_to_delete)
        self.gp_ip_curr_lin = np.delete(self.gp_ip_curr_lin,indices_to_delete)
        self.gp_ip_curr_ang = np.delete(self.gp_ip_curr_ang,indices_to_delete)
        
        self.gp_op_lin_error = np.delete(self.gp_op_lin_error,indices_to_delete)
        self.gp_op_ang_error = np.delete(self.gp_op_ang_error,indices_to_delete)
        
        self.file_name = np.delete(self.file_name,indices_to_delete)
        
        # Now redo the counter
        self.counter = list(range(len(self.gp_ip_cmd_lin)))

    def save_dataset(self,terrain_type):
    
        """
        Store the GP dataset in the form of a csv file
        """
        
        # Names of GP dataset fields
        # Setting as dictionary to allow not saving certain fields in the dataset
        # Default to saving all
        
        # Set the following to False if not intending to log in the csv file
        self.gp_dataset_names = {"counter":(True,self.counter),\
                                "file_name":(True,self.file_name),\
                                    
                                "ip_cmd_lin":(True,self.gp_ip_cmd_lin),\
                                "ip_cmd_ang":(True,self.gp_ip_cmd_ang),\
                                        
                                "ip_curr_lin":(True,self.gp_ip_curr_lin),\
                                "ip_curr_ang":(True,self.gp_ip_curr_ang),\
                                    
                                "op_lin_error":(True,self.gp_op_lin_error),\
                                "op_ang_error":(True,self.gp_op_ang_error),}

        # Name of the file to save to 
        file_dir  = os.path.join(os.getcwd(),"data",terrain_type,"GP_Datasets")
        os.makedirs(file_dir,exist_ok=True)
        file_name = os.path.join(file_dir,"all_gp_data.csv")

        gp_dataset_name_keys = list(self.gp_dataset_names.keys())
        
        dataset_dict = {}
        
        # Check if we want to save the list
        for name_key in gp_dataset_name_keys:
            if self.gp_dataset_names[name_key][0] == True:
                dataset_dict[name_key] =  self.gp_dataset_names[name_key][1]
                
        data_frame = pd.DataFrame(dataset_dict)
        
        os.makedirs(file_dir,exist_ok=True)
        
        print("Generating All GP Data for terrain: {}..".format(terrain_type))
        
        data_frame.to_csv(file_name)


if __name__ == "__main__":
    all_data = AllData()