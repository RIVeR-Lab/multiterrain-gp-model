#!/usr/bin/env python3#!/usr/bin/env python3
'''
Train 6 GPs independently and compare with Batch GP model
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
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.distributions import MultivariateNormal
from gpytorch.settings import fast_pred_var

# Set the device to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Parameters
num_tasks = 6  # Number of independent tasks (GPs) - 3 terrains, 2 GPs/terrain
input_dimension = 4  # 4D input
training_iterations = 2500
learning_rate = 0.01
profiling_iterations = 1500

# Set the random seed for reproducibility
torch.manual_seed(42)

# Define a single GP model class for training independently
class IndependentGPModel(ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = ZeroMean()
        self.covar_module = ScaleKernel(RBFKernel(ard_num_dims=input_dimension))
        
    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)

# Load the training data
train_data_path = '/home/ananya/Code/multiterrain-gp-model/data/combined_terrain_errors/Train.csv'
train_df = pd.read_csv(train_data_path)

# Extract inputs (same for all GPs) and outputs (different for each GP)
train_x = train_df[['gp_ip_cmd_lin', 'gp_ip_cmd_ang', 'gp_ip_curr_lin', 'gp_ip_curr_ang']].values
train_x = torch.tensor(train_x, dtype=torch.float32).to(device)

# Extract outputs (with correct column names for each terrain)
train_y = train_df[['Asphalt_Lin_Error', 'Asphalt_Ang_Error',
                    'Grass_Lin_Error', 'Grass_Ang_Error',
                    'Tile_Lin_Error', 'Tile_Ang_Error']].values

# Convert train_y to tensors for each GP independently
train_y_tensors = [torch.tensor(train_y[:, i], dtype=torch.float32).to(device) for i in range(num_tasks)]

# Train independent GPs
independent_models = []
likelihoods = []
for i in range(num_tasks):
    likelihood = GaussianLikelihood().to(device)
    model = IndependentGPModel(train_x, train_y_tensors[i], likelihood).to(device)
    
    model.train()
    likelihood.train()
    
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    mll = ExactMarginalLogLikelihood(likelihood, model)
    
    for j in range(training_iterations):
        optimizer.zero_grad()
        output = model(train_x)
        loss = -mll(output, train_y_tensors[i])
        print('Iter %d/%d - Loss: %.3f' % (j + 1, training_iterations, loss.item()))
        loss.backward()
        optimizer.step()
    
    independent_models.append(model)
    likelihoods.append(likelihood)

# Load the testing data
test_data_path = '/home/ananya/Code/multiterrain-gp-model/data/combined_terrain_errors/Test.csv'
test_df = pd.read_csv(test_data_path)

test_x = torch.tensor(test_df[['gp_ip_cmd_lin', 'gp_ip_cmd_ang', 'gp_ip_curr_lin', 'gp_ip_curr_ang']].values, dtype=torch.float32).to(device)

test_y = train_df[['Asphalt_Lin_Error', 'Asphalt_Ang_Error',
                   'Grass_Lin_Error', 'Grass_Ang_Error',
                   'Tile_Lin_Error', 'Tile_Ang_Error']].values

# Inference and plotting for independent GPs
plt.figure(figsize=(10, 8))
for i in range(num_tasks):
    model = independent_models[i]
    likelihood = likelihoods[i]
    
    model.eval()
    likelihood.eval()
    
    with torch.no_grad(), fast_pred_var():
        observed_pred = likelihood(model(test_x))
        
    predicted_mean = observed_pred.mean.cpu().numpy()
    lower, upper = observed_pred.confidence_region()
    lower = lower.cpu().numpy()
    upper = upper.cpu().numpy()
    
    plt.subplot(3, 2, i + 1)
    plt.plot(range(len(predicted_mean)), test_y[:, i], 'r*', label=f'GT Mean - Task {i + 1}')
    plt.plot(range(len(predicted_mean)), predicted_mean, 'k*', label=f'Predicted Mean - Task {i + 1}')
    plt.fill_between(range(len(predicted_mean)), lower, upper, color='lightblue', alpha=0.5, label='Confidence Interval')
    plt.legend()
    plt.title(f'Independent GP: Predictions for Task {i + 1}')
    plt.xlabel('Data Point Index')
    plt.ylabel('Output')

plt.tight_layout()
plt.show()

# Save the independent models' hyperparameters (as an example for the first model)
torch.save(independent_models[0].state_dict(), 'independent_gp_model_hyperparameters.pth')

# Profiling loop for independent GPs
timings = np.empty(profiling_iterations)

with torch.no_grad(), fast_pred_var():
    for idx in range(profiling_iterations):
        start = time.perf_counter_ns()
        observed_pred = likelihoods[0](independent_models[0](test_x))  # Profile the first GP as an example
        end = time.perf_counter_ns()
        timings[idx] = end - start
    
print(f"\nTime to predict with independent GPs: {timings.mean()*1e-6:.4f} ms ± {timings.std()*1e-6:.4f}")
