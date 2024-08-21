#!/usr/bin/env python3
'''
Load a pre-trained batch GP model for three terrains and linear/angular velocity residuals
and perform inference and plotting.
'''

# Requisite imports
import time
import torch
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from gpytorch.models import ExactGP
from gpytorch.means import ZeroMean
from gpytorch.kernels import ScaleKernel, RBFKernel
from gpytorch.distributions import MultitaskMultivariateNormal, MultivariateNormal
from gpytorch.likelihoods import MultitaskGaussianLikelihood
from gpytorch.settings import fast_pred_var

# Set the device to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Parameters
num_tasks = 6  # Number of independent tasks (GPs) - 3 terrains, 2 GPs/terrain
input_dimension = 4  # 4D input
profiling_iterations = 1500

# Define a Batch Independent Multitask GP Model
class BatchIndependentMultitaskGPModel(ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super().__init__(train_x, train_y, likelihood)
        
        # Define the mean module with batch shape for multitasking
        self.mean_module = ZeroMean(batch_shape=torch.Size([num_tasks]))
        
        # Define the RBF kernel with batch shape for multitasking and ARD for each input dimension
        self.covar_module =\
            ScaleKernel(RBFKernel(ard_num_dims=input_dimension, batch_shape=torch.Size([num_tasks])), batch_shape=torch.Size([num_tasks]))
        
    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultitaskMultivariateNormal.from_batch_mvn(MultivariateNormal(mean_x, covar_x))

# Load the training data (needed to initialize the model, even though we won't retrain it)
train_data_path = '/home/ananya/Code/multiterrain-gp-model/data/combined_terrain_errors/Train.csv'
train_df = pd.read_csv(train_data_path)

# Extract inputs (same for all GPs) and outputs (different for each GP)
train_x = train_df[['gp_ip_cmd_lin', 'gp_ip_cmd_ang', 'gp_ip_curr_lin', 'gp_ip_curr_ang']].values
train_x = torch.tensor(train_x,dtype=torch.float32).to(device)

train_y = train_df[['Asphalt_Lin_Error', 'Asphalt_Ang_Error',
                    'Grass_Lin_Error', 'Grass_Ang_Error',
                    'Tile_Lin_Error', 'Tile_Ang_Error']].values

train_y = torch.tensor(train_y, dtype=torch.float32).to(device)

# Initialize the model and likelihood
likelihood = MultitaskGaussianLikelihood(num_tasks=num_tasks).to(device)
model = BatchIndependentMultitaskGPModel(train_x, train_y, likelihood).to(device)

# Load the model's saved state
model.load_state_dict(torch.load('batch_gp_model_hyperparameters.pth'))

# Set the model and likelihood to evaluation mode
model.eval()
likelihood.eval()

# Load the testing data
test_data_path = '/home/ananya/Code/multiterrain-gp-model/data/combined_terrain_errors/Test.csv'
test_df = pd.read_csv(test_data_path)

test_x = torch.tensor(test_df[['gp_ip_cmd_lin', 'gp_ip_cmd_ang', 'gp_ip_curr_lin', 'gp_ip_curr_ang']].values, dtype=torch.float32).to(device)

test_y = train_df[['Asphalt_Lin_Error', 'Asphalt_Ang_Error',
                    'Grass_Lin_Error', 'Grass_Ang_Error',
                    'Tile_Lin_Error', 'Tile_Ang_Error']].values

# Inference
with torch.no_grad(), fast_pred_var():
    observed_pred = likelihood(model(test_x))
    
# Extract predictions for all tasks
predicted_mean = observed_pred.mean.cpu().numpy()
lower, upper = observed_pred.confidence_region()
lower = lower.cpu().numpy()
upper = upper.cpu().numpy()

# Plot the results
plt.figure(figsize=(10, 8))
for task in range(num_tasks):
    plt.subplot(3, 2, task + 1)
    plt.plot(range(len(predicted_mean[:, task])), test_y[:, task], 'r*', label=f'GT Mean - Task {task + 1}')
    plt.plot(range(len(predicted_mean[:, task])), predicted_mean[:, task], 'k*', label=f'Predicted Mean - Task {task + 1}')
    plt.fill_between(range(len(predicted_mean[:, task])), lower[:, task], upper[:, task], color='lightblue', alpha=0.5, label='Confidence Interval')
    plt.legend()
    plt.title(f'GP Model: Predictions for Task {task + 1}')
    plt.xlabel('Data Point Index')
    plt.ylabel('Output')

plt.tight_layout()
plt.show()

# Profiling loop
timings = np.empty(profiling_iterations)

with torch.no_grad(), fast_pred_var():
    for idx in range(profiling_iterations):
        start = time.perf_counter_ns()
        observed_pred = likelihood(model(test_x))
        end = time.perf_counter_ns()
        timings[idx] = end - start
    
print(f"\nTime to predict: {timings.mean()*1e-6:.4f} ms ± {timings.std()*1e-6:.4f}")
