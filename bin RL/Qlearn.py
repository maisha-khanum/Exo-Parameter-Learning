import numpy as np
import random
from collections import defaultdict
import time
import numpy.matlib
from scipy.sparse.linalg import svds
import matplotlib.pyplot as plt
from helper import init_constants, reinit_bin_upd, upd_plot_data, sample_param, f_multi, f_all, rank_params, cma_multi, cma_update, init_opt_vars, init_xmean, constrain_params, plot_single_mode
from bins_helper import initialize_params, calculate_constrained_bin_edges, calculate_bin_edges_from_sizes, find_max_below_threshold
from viz_helper import plot_density_with_colored_bins
import os
from scipy.stats import norm
from tqdm import tqdm

# Environment parameters
speed_mean = 1.35
speed_std = 0.15
min_val = 0.9
max_val = 1.75
max_bins = 10
min_bins = 1
min_bin_size = 10
# num_gens_cma = 40
lambda_cma = 10  # Number of samples per generation
convergence_threshold = 0.05  # Threshold for average sigma
c_penalty = 0.0001  # Cost factor for the number of bins
meas_noise = 0.001
init_sigma_val = 0.2

# CMA parameters (tweak these to change the outcome of the simulation)
num_gens_cma = 9 # number of generations before simulation terminates (adjust for longer or shorter optimizations)
meas_noise = 0.001 # noise in CMA estimates
offset_std_in_params = 0.1 # standard deviation of the "true" parameters, increase to make the optimal parameters spread over a larger range
scalar_std_in_params = 0.1 # underlying scalar offset of "true" parameters, increase to make the optimal parameters spread over a larger range
init_sigma_val = 0.2 # initial sigma value, controlling the covariance (similar to the range) of the first CMA generation.
speed_type = 'normal' #'uniform' # 'normal' # type of distribution to sample walking speed from
meta_plot = [] # storing data to plot
start_time = time.time() # saving initial timestamp

torque_range = (0.6, 0.85)  # Example range for peak torque
rise_time_range = (0.55, 0.9)  # Example range for rise time

# Optimization parameter definitions
weight = 68 # Participant weight (kg) used to normalize the peak torque magnitude
seed = 3 # randomization seed
random.seed(seed) # initializing randomization for consitent results between runs
np.random.seed(seed) # initializing randomization for consitent results between runs
N = 2 # Number of optimization dimensions (2 torque parameters for peak torque magnitude and rise time)
# m = 'Normal' #'Re-initialization' #
m = 'Re-initialization'
max_num_bins = 10

# f_params = initialize_params(num_bins, torque_range, rise_time_range) # initial values of the torque parameters (peak torque, rise time) for the three optimization bins based on the ranges of walking speed
# sigma = init_sigma_val*np.ones(num_bins) # initalize the CMA optimization sigma, controlling the covariance
λ, constants = init_constants(N, num_gens_cma) # initialized optimization constant values
param_bounds = np.zeros((N,2)) # defining an array to store the bounds of the torque parameters
param_bounds[0,:] = np.array([0,1.]) # peak torque, max 1 = Nm/kg 
param_bounds[1,:] = np.array([0.25,1.]) # normalized rise time ((% gait cycle)/40%), max = 40%, min = 10%
rise_scalar = 40.0 # constant value used to normalize the rise time (40% gait cycle) 
new_params = np.array([0.701, 54.59, 27.8/rise_scalar, 9.98]) # initialize the four torque parameter values to the generic assistance
upd_flag = False # flag to track cma updates
last_cond_ind = 0 # intialize counter variable

# randomly sample the location of the optimal torque parameters for the simulation
f_offset = np.random.uniform(low=-offset_std_in_params, high=offset_std_in_params, size=(max_num_bins, 1))
f_mult = np.random.uniform(low=1-scalar_std_in_params, high=1+scalar_std_in_params, size=1)

# randomly sample the walking speeds of all conditions in the simulated optimization
if speed_type == 'normal':
    cond_speeds = np.random.normal(loc=speed_mean, scale=speed_std, size=num_gens_cma*λ)
elif speed_type == 'uniform':
    cond_speeds = np.random.uniform(low=speed_mean-speed_std, high=speed_mean+speed_std, size=num_gens_cma*λ)

# Q-Learning parameters
alpha = 0.1  # Learning rate
gamma = 0.9  # Discount factor
epsilon = 0.5  # Exploration rate
num_episodes = 500

# Helper function to calculate bin edges
def calculate_bin_edges(num_bins, min_val, max_val):
    return np.linspace(min_val, max_val, num_bins + 1)

def cma_simulation(state):    
    """
    Simulates the CMA-ES optimization process across speed bins for walking conditions.
    
    Parameters:
        state.
    Returns:
        dict: Results including optimization parameters, plotting data, and CMA data.
    """
    bin_sizes = state
    num_bins = len(state)
    bins = calculate_bin_edges_from_sizes(speed_mean, speed_std, bin_sizes, min_val, max_val)

    # print("num_bins", num_bins)

    f_params = initialize_params(num_bins, torque_range, rise_time_range) # initial values of the torque parameters (peak torque, rise time) for the three optimization bins based on the ranges of walking speed
    sigma = init_sigma_val*np.ones(num_bins) # initalize the CMA optimization sigma, controlling the covariance
    λ, constants = init_constants(N, num_gens_cma) # initialized optimization constant values
    param_bounds = np.zeros((N,2)) # defining an array to store the bounds of the torque parameters
    param_bounds[0,:] = np.array([0,1.]) # peak torque, max 1 = Nm/kg 
    param_bounds[1,:] = np.array([0.25,1.]) # normalized rise time ((% gait cycle)/40%), max = 40%, min = 10%
    rise_scalar = 40.0 # constant value used to normalize the rise time (40% gait cycle) 
    new_params = np.array([0.701, 54.59, 27.8/rise_scalar, 9.98]) # initialize the four torque parameter values to the generic assistance
    upd_flag = False # flag to track cma updates
    last_cond_ind = 0 # intialize counter variable
    
    # Define matrices to store the optimization parameters and relevant plotting data
    sigma = init_sigma_val * np.ones(num_bins)
    x_mean = init_xmean(bins, N, f_params, np.linspace(0.8, 1.7, num_bins))
    plot_sig_data = np.zeros((num_bins, num_gens_cma + 1))
    plot_sig_data[:, 0] = sigma
    plot_rew_data = np.zeros((num_bins, num_gens_cma + 1))
    goal = f_params * f_mult + f_offset[:num_bins]
    plot_rew_data[:, 0] = f_all(goal, meas_noise, x_mean)
    plot_mean_data = np.zeros((num_bins, num_gens_cma + 1, N))
    plot_mean_data[:, 0, :] = x_mean

    # Initialize optimization variables for each bin
    bin_opt_vars = [init_opt_vars(x_mean[i, :], sigma[i], N) for i in range(num_bins)]
    cond_counter = np.zeros(num_bins, dtype=int)
    gen_counter = np.zeros(num_bins, dtype=int)
    bin_gen_data = np.zeros((num_bins, λ, N))
    bin_gen_params = np.zeros((num_bins, λ, N))
    bin_gen_params[:, 0, :] = x_mean
    constants.append(param_bounds)
    constants.append(meas_noise)
    constants.append(goal)

    # Main CMA simulation loop
    for i, spd in enumerate(cond_speeds):
        bin_idx = np.where(spd > bins)[0][-1]
        bin_gen_data[bin_idx, cond_counter[bin_idx], :] = bin_gen_params[bin_idx, cond_counter[bin_idx], :]
        cond_counter[bin_idx] += 1
        
        if cond_counter[bin_idx] % λ == 0:
            arindex, arx = rank_params(constants, bin_idx, bin_gen_data[bin_idx, :, :])
            bin_opt_vars[bin_idx] = cma_multi([constants, bin_opt_vars[bin_idx]], arindex, bin_gen_params[bin_idx, :, :])
            
            # opt_vars = [xmean, xold, sigma, pc, ps, B, D, C, invsqrtC, eigeneval, local_cnt] # what cma output is

            if m == 'Re-initialization':
                bin_opt_vars, upd_flag = reinit_bin_upd(bin_opt_vars, bin_idx, len(bins) - 1, bins, 2, sigma[0], upd_flag, m)
            
            cond_counter[bin_idx] = 0
            gen_counter[bin_idx] += 1
            plot_sig_data, plot_rew_data, plot_mean_data = upd_plot_data(
                bin_opt_vars, gen_counter, bin_idx, plot_sig_data, plot_rew_data, plot_mean_data, len(bins) - 1, constants
            )
        
        bin_gen_params[bin_idx, cond_counter[bin_idx], :] = sample_param(bin_opt_vars[bin_idx], param_bounds)
        # new_params = np.zeros(3)
        # new_params[0] = bin_gen_params[bin_idx, cond_counter[bin_idx], 0] * weight
        # new_params[2] = bin_gen_params[bin_idx, cond_counter[bin_idx], 0] * rise_scalar
        # print("New params:", new_params)
    
    g = find_max_below_threshold(plot_sig_data, convergence_threshold) # for minimizing g
    sigma = np.max(plot_sig_data[:, -3])

    return g, sigma
    # return {
    #     "opt_results": bin_opt_vars,
    #     "plot_sig_data": plot_sig_data,
    #     "plot_rew_data": plot_rew_data,
    #     "plot_mean_data": plot_mean_data,
    #     "cma_data": {
    #         "gen_data": bin_gen_data,
    #         "gen_params": bin_gen_params,
    #         "gen_counter": gen_counter,
    #         "cond_counter": cond_counter,
    #     }
    # }

def apply_action(state, action): 
    """
    Apply an action to modify the bin configuration.
    
    Parameters:
        state (list): Current bin configuration.
        action (tuple): Action to modify bins, e.g., ('merge', idx) or ('split', idx, num_split).
    
    Returns:
        list: New bin configuration after applying the action.
    """
    new_state = state.copy()
    action_type = action[0]

    if action_type == 'merge':
        # Merge two adjacent bins at indices i and i+1
        idx = action[1]
        if idx < len(new_state) - 1:  # Ensure there is a next bin to merge with
            new_state[idx] += new_state.pop(idx + 1)

    elif action_type == 'split':
        # Split the bin at index `idx` into two bins of divisible sizes
        idx, split_sizes = action[1], action[2]
        if sum(split_sizes) == new_state[idx] and all(size % int(100 / max_num_bins) == 0 for size in split_sizes):
            new_state = new_state[:idx] + list(split_sizes) + new_state[idx + 1:]

    return new_state

# Reward function
def reward_function(g, sigma, num_bins, c_penalty): # TODO
    # return -(g + c_penalty * num_bins)
    return -(sigma - c_penalty * num_bins)
    
# State 0
def initialize_bins(): # TODO
    """Initialize a random bin configuration."""
    return [20, 20, 60]  # Example: 3 bins (0-20%, 20-40%, 40-100%)

# Initialize Q-table
# Q = defaultdict(lambda: np.zeros(max_bins - min_bins + 1))  # Actions: possible bin splits
Q = {}

# Q-learning loop
for episode in tqdm(range(num_episodes)):
    # Initialize environment
    state = initialize_bins()
    epsilon = 0.1  # Exploration rate
    total_reward = 0
    done = False

    while not done:
        # Choose action using epsilon-greedy policy, TODO DECIDE HOW MUCH TO SPLIT BY
        actions = []
        if len(state) > 1:
            adjacent_pairs = [(i, i+1) for i in range(len(state) - 1)]
            random.shuffle(adjacent_pairs)
            for i, j in adjacent_pairs:
                actions.append(('merge', i, j))

        # Split action: Split bins into two, ensuring one size is divisible by 20
        if len(state) < max_num_bins:
            for i, bin_size in enumerate(state):
                if bin_size > min_bin_size:
                    # Find all valid splits where both resulting bins are divisible by 20
                    valid_splits = [
                        (x, bin_size - x)
                        for x in range(min_bin_size, bin_size, min_bin_size)
                        if (bin_size - x) % min_bin_size == 0
                    ]
                    for split in valid_splits:
                        actions.append(('split', i, split))

        actions += [('no_op',)]  # Include no-op action
                
        if np.random.uniform(0, 1) < epsilon:
            print("EXPLORE")
            action = random.choice(actions)  # Explore
        else:
            action_values = [Q.get((tuple(state), a), 0) for a in actions]
            action = actions[np.argmax(action_values)]  # Exploit

        # Apply action to update state
        new_state = apply_action(state, action)

        # Simulate CMA-ES optimization
        g, sigma = cma_simulation(new_state)

        # Compute reward
        r = reward_function(g, sigma, len(new_state), c_penalty)

        # Update Q-value
        if (tuple(state), action) not in Q:
            Q[(tuple(state), action)] = 0
        Q[(tuple(state), action)] += 0.1 * (r + 0.9 * max(Q.get((tuple(new_state), a), 0) for a in actions) - Q[(tuple(state), action)])

        # Update state and total reward
        state = new_state
        total_reward += r

        print(np.mean(sigma))
        done = True
        # Termination condition (e.g., convergence met)
        if np.mean(sigma) < convergence_threshold:
            done = True

    # Reduce epsilon over time to favor exploitation
    epsilon = max(0.01, epsilon * 0.99)

    # Log progress
    if (episode + 1) % 50 == 0:
        print(f"Episode {episode + 1}/{num_episodes}, Total Reward: {total_reward}")

print(Q)

# Extract optimal policy
optimal_policy = {state: np.argmax(actions) + min_bins for state, actions in Q.items()}
print("Optimal policy (num_bins per state):", optimal_policy)

# Visualizations
# plot_density_with_colored_bins(cond_speeds, bin_edges)
