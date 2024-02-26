"""Train a Gaussian Process Regression Model to predict the mean and covariance estimates of the linear 
and angular velocity errors using the GPyTorch library"""

import numpy as np
import torch
import gpytorch
import pandas as pd
from sklearn import preprocessing

from configurations import ModelingParameters as JP

class ExactGPModel(gpytorch.models.ExactGP):
    
    def __init__(self,train_inputs,train_targets,likelihood):
        """Override class of the ExactGP Model.

        Args:
            train_inputs (torch.Tensor): N X input dimension
            train_targets (torch.Tensor): N X output dimension
            likelihood (torch.Tensor): Gaussian Likelihood
        """

        # Instantiate the base class
        super(ExactGPModel, self).__init__(train_inputs, train_targets, likelihood)
        
        # Zero Mean Prior Function
        self.mean_module = gpytorch.means.ZeroMean()
        
        # Radial basis function Kernel
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(ard_num_dims=train_inputs.shape[1]))
    
    def forward(self,x):
        """Compute multivariate normal distrubution evaluated at x

        Args:
            x (torch.Tensor): Dimension N X input dimension

        Returns:
            gpytorch.distributions : Multivariate normal distribution evaluated at x 
            from the prior mean and covar function
        """
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x,covar_x)

class GaussianProcess:
    """
    Main interface class for a GP Model
    1) Trains Model 2) Loads Model
    """

    def __init__(self, gp_type, terrain_type, training_dataset, testing_dataset, model_file_name):
        
        # Whether we are training a linear or angular velocity GP
        self.gp_type = gp_type

        # Terrain type 
        self.terrain_type = terrain_type

        # Create file name for the model
        self.model_file_name = model_file_name

        # Load the datasets and extract GP inputs and outputs
        self.training_dataset = training_dataset

        self.testing_dataset = testing_dataset

        self.create_gp_input_outputs()

        # Gaussian Likelihood for the ExactGP Model
        self.likelihood = gpytorch.likelihoods.GaussianLikelihood(noise_constraint=gpytorch.constraints.GreaterThan(1e-6),).cuda().float()

        # Create the model
        self.model = ExactGPModel(self.training_inputs_tensor,self.training_outputs_tensor,self.likelihood).cuda().float()

        # Train and store the GP Model
        self.train_gp()

    def create_gp_input_outputs(self):
        # Load the training and testing pandas dataframes.
        training_data_frame = pd.read_csv(self.training_dataset)
        testing_data_frame  = pd.read_csv(self.testing_dataset)

        # Collate inputs into training and testing tensors
        self.training_inputs_tensor = torch.tensor(np.hstack(self.load_gp_inputs(training_data_frame)))
        self.testing_inputs_tensor  = torch.tensor(np.hstack(self.load_gp_inputs(testing_data_frame)))
        
        # Normalize the training and testing inputs
        # Converts each dimension of the training inputs to a standard normal with 0 mean and unit covariance
        self.scaler = preprocessing.StandardScaler().fit(self.training_inputs_tensor.numpy())
        self.training_inputs_tensor = torch.from_numpy(self.scaler.transform(self.training_inputs_tensor.numpy()))
        self.testing_inputs_tensor  = torch.from_numpy(self.scaler.transform(self.testing_inputs_tensor.numpy()))
    
        # Load the GP outputs
        train_gp_error = None; test_gp_error = None

        if self.gp_type == "Linear":
            train_gp_error  = training_data_frame.loc[ : ,"gp_op_lin_error"].to_numpy()
            test_gp_error   = testing_data_frame.loc [ : ,"gp_op_lin_error"].to_numpy()
        elif self.gp_type == "Angular":
            train_gp_error = training_data_frame.loc[ : ,"gp_op_ang_error"].to_numpy()
            test_gp_error  = testing_data_frame.loc [ : ,"gp_op_ang_error"].to_numpy()
        else:
            raise RuntimeError("Only Linear and Angular velocity GPs can be trained")

        # Collate outputs into training and testing tensors
        self.training_outputs_tensor  = torch.tensor(train_gp_error)
        self.testing_outputs_tensor   = torch.tensor(test_gp_error)
        
        # Dimension Book-keeping
        self.input_dimension    = self.training_inputs_tensor.shape[1]
        self.output_dimension   = self.training_outputs_tensor.shape
        self.n_training_samples = self.training_inputs_tensor[0]

        # GPU as the device
        output_device = torch.device('cuda:0')

        # Transfer to GPU
        self.training_inputs_tensor = self.training_inputs_tensor.contiguous().to(output_device).float()
        self.training_outputs_tensor = self.training_outputs_tensor.contiguous().to(output_device).float()
        self.testing_inputs_tensor = self.testing_inputs_tensor.contiguous().to(output_device).float()
        self.testing_outputs_tensor = self.testing_outputs_tensor.contiguous().to(output_device).float()


    def load_gp_inputs(self,data_frame):
        
        # Load the GP inputs from the pandas dataframe
        
        cmd_lin  = data_frame.loc[ : ,"gp_ip_cmd_lin"].to_numpy().reshape(-1,1)
        cmd_ang  = data_frame.loc[ : ,"gp_ip_cmd_ang"].to_numpy().reshape(-1,1)
        curr_lin = data_frame.loc[ : ,"gp_ip_curr_lin"].to_numpy().reshape(-1,1)
        curr_ang = data_frame.loc[ : ,"gp_ip_curr_ang"].to_numpy().reshape(-1,1)
        
        return cmd_lin, cmd_ang, curr_lin, curr_ang
    

    def train_gp(self):
        """"Train the GP Models and store results"""

        # ADAM Optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=JP.lr)

        # Loss Function for the GP training process -- the marginal log likelihood
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood,self.model)

        # Initialize loss to apply backprop on
        train_loss = torch.tensor(0.)

        # Lowest instantaneous training loss
        best_train_loss = torch.tensor(9999999.)

        # Train the model and save at the least encountered loss value
        print("-----")
        print("Training {} GP-model for terrain {}".format(self.gp_type, self.terrain_type))

        opt_iter = 0
        test_loss = torch.tensor(0.)
        train_loss = torch.tensor(0.)

        while opt_iter < JP.max_opt_iter:

            with torch.no_grad(),gpytorch.settings.fast_pred_var():
                # Document testing loss to see trends
                self.model.eval(); self.likelihood.eval()
                test_output = self.model(self.testing_inputs_tensor.contiguous())
                test_loss = -mll(test_output,self.testing_outputs_tensor)
            
            # Put the model and likelihood back in training mode to resume training
            self.model.train(); self.likelihood.train()

            # Zero gradients from the previous iteration
            self.optimizer.zero_grad()

            # Output from the model, compute the loss and backprop gradients
            train_output = self.model(self.training_inputs_tensor)
            train_loss = -mll(train_output,self.training_outputs_tensor)

            # Backprop gradients
            train_loss.backward()

            # Update parameters, increment step
            self.optimizer.step(); opt_iter+=1

            # Display training status
            if opt_iter % 100 == 0:
                print("Training iter: {}/{}, Training Loss: {}, Testing Loss: {}".format(opt_iter,JP.max_opt_iter,train_loss.item(),test_loss.item()))
            
            # Save Model at lowest loss
            if train_loss < best_train_loss:
                best_train_loss = train_loss
                state_dict = self.model.state_dict()
                torch.save(state_dict,self.model_file_name)

        print("--------")
        print("Training Complete")
        print("--------")
