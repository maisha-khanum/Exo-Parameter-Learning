import numpy as np
import random
import time
import matplotlib.pyplot as plt
from optimization.cma_helper import init_constants, upd_plot_data, sample_param, f_multi, f_all, rank_params, cma_update, init_opt_vars, init_xmean, constrain_params, plot_single_mode
import os

# CMA parameters (tweak these to change the outcome of the simulation)
num_gens_cma = 20 # number of generations before simulation terminates (adjust for longer or shorter optimizations)
meas_noise = 0.001 # noise in CMA estimates
offset_std_in_params = 0.1 # standard deviation of the "true" parameters, increase to make the optimal parameters spread over a larger range
scalar_std_in_params = 0.1 # underlying scalar offset of "true" parameters, increase to make the optimal parameters spread over a larger range
init_sigma_val = 0.2 # initial sigma value, controlling the covariance (similar to the range) of the first CMA generation.
speed_type = 'normal' # 'uniform' # type of distribution to sample walking speed from
speed_mean = 1.35 # mean value for the walking speed distribution m/s for the normal distribution
speed_std = 0.15 # standard deviation in walking speed distribution (m/s)
meta_plot = [] # storing data to plot
start_time = time.time() # saving initial timestamp

# Optimization parameter definitions (not recommended to change these)
weight = 68 # Participant weight (kg) used to normalize the peak torque magnitude
seed = 3 # randomization seed
random.seed(seed) # initializing randomization for consitent results between runs
np.random.seed(seed) # initializing randomization for consitent results between runs
N = 2 # Number of optimization dimensions (2 torque parameters for peak torque magnitude and rise time)
f_params = np.array([[0.64051748, 0.58666667],[0.72715882,0.70916667],[0.80887392,0.85]]) # initial values of the torque parameters (peak torque, rise time) for the three optimization bins based on the ranges of walking speed
m = 'Normal' #'Re-initialization' # 
bins = np.array([0.9, 1.24, 1.42, 1.75]) # defining speed ranges of the bins
bin = 0 # initalize bin parameter to 0
spd_bins = len(bins)-1 # number of optimization bins based on the defined ranges of walking speed
sigma = init_sigma_val*np.ones(spd_bins) # initalize the CMA optimization sigma, controlling the covariance
λ, constants = init_constants(N, num_gens_cma) # initialized optimization constant values
param_bounds = np.zeros((N,2)) # defining an array to store the bounds of the torque parameters
param_bounds[0,:] = np.array([0,1.]) # peak torque, max 1 = Nm/kg 
param_bounds[1,:] = np.array([0.25,1.]) # normalized rise time ((% gait cycle)/40%), max = 40%, min = 10%
rise_scalar = 40.0 # constant value used to normalize the rise time (40% gait cycle) 
new_params = np.array([0.701, 54.59, 27.8/rise_scalar, 9.98]) # initialize the four torque parameter values to the generic assistance
upd_flag = False # flag to track cma updates
last_cond_ind = 0 # intialize counter variable


# Initialization Phase
# Define an initial set of parameters (e.g., torque, angles, velocities).
initial_parameters = initialize_parameters()
reward_model = initialize_reward_model()  # Based on past optimization data

# Outer Loop (RL Agent)

for iteration in range(total_iterations):
    # Step 1: RL Agent selects a parameterization
    selected_param_set = select_parameterization(reward_model)  # e.g., {torque, angle difference}

    # Step 2: Pass the selected parameterization to the inner loop
    performance_metrics = inner_loop(selected_param_set)
    
    # Step 3: Adaptation Phase - Update the reward model based on performance
    reward_model = update_reward_model(reward_model, performance_metrics)

# Inner Loop (Optimization)
def inner_loop(selected_param_set):
    # Step 1: Optimize the selected parameters using CMA-ES
    optimized_params = cma_optimization(selected_param_set)

    # Step 2: Evaluate the result (e.g., metabolic cost reduction, stability metrics)
    performance_metrics = evaluate_performance(optimized_params)

    # Step 3: Return the performance metrics to the outer loop
    return performance_metrics

# Optimization using CMA-ES
def cma_optimization(selected_param_set):
    # Initialize CMA parameters (sigma, mean, constants, etc.)
    init_sigma, init_mean, constants = initialize_cma(selected_param_set)

    # Main loop for CMA optimization
    for generation in range(num_generations):
        # Generate new candidate solutions
        new_candidates = generate_candidates(init_mean, init_sigma)
        
        # Evaluate candidate solutions
        candidate_performance = evaluate_candidates(new_candidates)

        # Rank candidates and update CMA parameters
        ranked_candidates = rank_candidates(candidate_performance)
        init_mean, init_sigma = update_cma_parameters(ranked_candidates)

    # Return the best parameters after optimization
    return init_mean  # Best parameters after CMA-ES

# Evaluate performance of parameters (e.g., metabolic cost reduction)
def evaluate_performance(optimized_params):
    # Evaluate the performance for metabolic cost reduction or stability
    performance = calculate_performance(optimized_params)
    return performance

# Helper functions for selecting and updating the reward model
def initialize_parameters():
    # Define initial parameters based on walking conditions, torque, etc.
    return initial_parameters

def initialize_reward_model():
    # Initialize the reward model using past optimization data
    return reward_model

def select_parameterization(reward_model):
    # Use the reward model to select a parameterization (e.g., torque, angle)
    selected_param_set = select_best_param_set_based_on_reward(reward_model)
    return selected_param_set

def update_reward_model(reward_model, performance_metrics):
    # Update the reward model using the performance metrics of the selected parameters
    updated_reward_model = adjust_reward_model_based_on_performance(reward_model, performance_metrics)
    return updated_reward_model

# Helper functions for CMA-ES initialization and updates
def initialize_cma(selected_param_set):
    # Initialize CMA-ES with parameters (mean, covariance, etc.)
    return init_mean, init_sigma, constants

def generate_candidates(init_mean, init_sigma):
    # Generate new candidates based on the current CMA-ES parameters
    return candidates

def evaluate_candidates(candidates):
    # Evaluate the candidates using the objective function
    return candidate_performance

def rank_candidates(candidate_performance):
    # Rank the candidates based on their performance (fitness)
    return ranked_candidates

def update_cma_parameters(ranked_candidates):
    # Update the CMA parameters (mean, covariance, etc.) based on ranked candidates
    return updated_mean, updated_sigma

# Performance evaluation function
def calculate_performance(params):
    # Calculate the metabolic cost or other metrics based on the parameters
    return performance_metric
