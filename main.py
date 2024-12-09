import numpy as np
import random
import time
import matplotlib.pyplot as plt
from optimization.cma_helper import init_constants, upd_plot_data, sample_param, f_multi, f_all, rank_params, cma_update, init_opt_vars, init_xmean, constrain_params, plot_single_mode
import os

# MAB Parameters
epsilon = 0.1  # Exploration rate (probability of exploring a random parameter set)
num_arms = 3  # Number of parameter sets to choose from (based on speed bins)
num_trials = 100  # Number of trials to run
num_gens_cma = 20 # number of generations before simulation terminates (adjust for longer or shorter optimizations)

# CMA parameters (not changed from your original code)
speed_type = 'normal'  # 'uniform' or 'normal' for sampling walking speed distribution
speed_mean = 1.35  # mean value for the walking speed distribution m/s for the normal distribution
speed_std = 0.15  # standard deviation in walking speed distribution (m/s)
seed = 3  # randomization seed
random.seed(seed)
np.random.seed(seed)

# Load training data
training_data = np.load("training_data.npy")  # 5180x64
train_labels = np.load("train_labels.npy")  # 5180x1

# MAB setup
Q_values = np.zeros(num_arms)  # Average reward for each arm (parameter set)
N_arms = np.zeros(num_arms)  # Count of how many times each arm was pulled (parameter set)
arms = np.array([0, 1, 2])  # The arms are the indices of the bins, e.g., 3 bins for different walking speeds



# Simulation loop using epsilon-greedy MAB
for trial in range(num_trials):
    if random.random() < epsilon:
        # Exploration: Randomly choose an arm (random parameter set)
        chosen_arm = random.choice(arms)
    else:
        # Exploitation: Choose the arm with the best average reward so far
        chosen_arm = np.argmax(Q_values)

    # Get the selected parameter set based on chosen arm
    bin_params = f_params[chosen_arm]  # Assuming f_params stores the parameter sets for different speed bins
    
    # Sample walking speed
    spd = np.random.normal(loc=speed_mean, scale=speed_std)  # or use uniform as needed
    
    # Generate parameters and evaluate metabolic cost
    bin_gen_data = sample_param(bin_params, param_bounds)  # Generate parameter sample based on arm choice
    metabolic_cost = evaluate_metabolic_cost(bin_gen_data, training_data, train_labels)  # This is the reward (lower cost is better)
    
    # Update MAB (average reward for chosen arm)
    N_arms[chosen_arm] += 1
    Q_values[chosen_arm] += (metabolic_cost - Q_values[chosen_arm]) / N_arms[chosen_arm]  # Incremental mean update
    
    print(f"Trial {trial+1}/{num_trials}, Arm {chosen_arm}, Metabolic Cost: {metabolic_cost}")

    # Optionally store data for plotting
    plot_sig_data, plot_rew_data, plot_mean_data = upd_plot_data(bin_opt_vars, gen_counter, bin, plot_sig_data, plot_rew_data, plot_mean_data, len(bins)-1, constants)

# Plot results after all trials
print("Final Q-values (average reward):", Q_values)
print("Final N_arms (arm pull counts):", N_arms)
plot_single_mode(plot_sig_data, plot_rew_data, plot_mean_data, goal, num_gens_cma, spd_bins, bins, gen_counter)
