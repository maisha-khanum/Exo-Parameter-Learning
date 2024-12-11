import numpy as np
import random
from collections import defaultdict
import time
import numpy.matlib
from scipy.sparse.linalg import svds
import matplotlib.pyplot as plt
from helper import init_constants, reinit_bin_upd, upd_plot_data, sample_param, f_multi, f_all, rank_params, cma_multi, cma_update, init_opt_vars, init_xmean, constrain_params, plot_single_mode
from bins_helper import initialize_params, calculate_constrained_bin_edges, calculate_bin_edges_from_sizes, find_max_below_threshold
from viz_helper import *
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
c_penalty = 0.001  # Cost factor for the number of bins
meas_noise = 0.001
init_sigma_val = 0.2

# CMA parameters (tweak these to change the outcome of the simulation)
num_gens_cma = 20 # number of generations before simulation terminates (adjust for longer or shorter optimizations)
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
    cond_speeds = np.random.uniform(low=speed_mean-2*speed_std, high=speed_mean+2*speed_std, size=num_gens_cma*λ)

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
        new_params = []
        new_params.append(bin_gen_params[bin_idx, cond_counter[bin_idx], 0] * weight)
        new_params.append(bin_gen_params[bin_idx, cond_counter[bin_idx], 0] * rise_scalar)
        # print("New params:", new_params)
    
    g = find_max_below_threshold(plot_sig_data, convergence_threshold) # for minimizing g
    sigma = np.max(plot_sig_data[:, -10])

    return g, sigma, new_params
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
    return [30,40,30]  # Example: 3 bins (0-20%, 20-40%, 40-100%)

# Initialize Q-table
# Q = defaultdict(lambda: np.zeros(max_bins - min_bins + 1))  # Actions: possible bin splits
Q = {}

# Track best state over episodes
best_states_per_episode = []
best_sigma_per_episode = []
best_param_per_episode = []
# Q-learning loop
max_no_improvement_steps = 10  # Maximum steps without improvement before terminating the episode
no_improvement_count = 0  # Counter for steps without improvement

for episode in tqdm(range(num_episodes)):
    # Initialize environment
    state = initialize_bins()
    epsilon = 0.1  # Exploration rate
    total_reward = 0
    done = False
    best_reward = float('-inf')
    last_state = None  # Variable to store the last state (to prevent undo action)

    while not done:
        best_state = state

        # Generate actions
        actions = []
        if len(state) > 1:
            adjacent_pairs = [(i, i + 1) for i in range(len(state) - 1)]
            random.shuffle(adjacent_pairs)
            for i, j in adjacent_pairs:
                actions.append(('merge', i, j))

        if len(state) < max_num_bins:
            for i, bin_size in enumerate(state):
                if bin_size > min_bin_size:
                    valid_splits = [
                        (x, bin_size - x)
                        for x in range(min_bin_size, bin_size, min_bin_size)
                        if (bin_size - x) % min_bin_size == 0
                    ]
                    for split in valid_splits:
                        actions.append(('split', i, split))

        actions += [('no_op',)]  # Include no-op action

        # Remove actions that would undo the previous one (prevent returning to the last state)
        if last_state is not None:
            if state == last_state:  # If we are already in the last state, don't split or merge to it
                actions = [a for a in actions if a[0] != 'merge' and a[0] != 'split']
        
        # Choose action using epsilon-greedy policy
        if np.random.uniform(0, 1) < epsilon:
            action = random.choice(actions)  # Explore
        else:
            action_values = [Q.get((tuple(state), a), 0) for a in actions]
            action = actions[np.argmax(action_values)]  # Exploit

        # Apply action
        new_state = apply_action(state, action)

        # Simulate CMA-ES optimization
        g, sigma, new_params = cma_simulation(new_state)

        # Compute reward
        r = reward_function(g, sigma, len(new_state), c_penalty)

        # Update Q-value
        Q[(tuple(state), action)] = Q.get((tuple(state), action), 0) + \
            0.1 * (r + 0.9 * max(Q.get((tuple(new_state), a), 0) for a in actions) - Q.get((tuple(state), action), 0))

        # Update best state if reward improves
        if r > best_reward:
            best_state = new_state
            best_reward = r
            no_improvement_count = 0  # Reset counter as the reward improved
        else:
            no_improvement_count += 1  # Increment counter if reward didn't improve
        
        # print(no_improvement_count)

        # Update state and track last state
        last_state = state  # Store current state as the last state
        state = new_state

        # Check if no improvement has occurred for too many steps
        if no_improvement_count >= max_no_improvement_steps:
            done = True  # End the episode

    best_sigma_per_episode.append(sigma)
    best_states_per_episode.append(len(best_state))
    best_param_per_episode.append(new_params)

    # Update epsilon for exploration
    epsilon = max(0.01, epsilon * 0.99)

    # Log progress
    if (episode + 1) % 10 == 0:
        print(f"Episode {episode + 1}/{num_episodes}, Best Reward: {best_reward}")

    # Log progress
    if (episode + 1) % 10 == 0:
        print(f"Episode {episode + 1}/{num_episodes}, Best Reward: {best_reward}")



# print(Q)

# Extract optimal policy
# optimal_policy = {state: np.argmax(actions) + min_bins for state, actions in Q.items()}
# print("Optimal policy (num_bins per state):", optimal_policy)


# Extract optimal policy and best state
optimal_policy = {}
best_state = None
max_q_value = float('-inf')

for state_action, value in Q.items():
    state, action = state_action
    
    # Update optimal policy
    if state not in optimal_policy or Q[(state, optimal_policy[state])] < value:
        optimal_policy[state] = action
    
    # Track the best state
    if value > max_q_value:
        max_q_value = value
        best_state = state

# Print results
print("Optimal policy (state -> action):", optimal_policy)
print("Best state:", best_state)
print("Maximum Q-value:", max_q_value)

print(min(best_sigma_per_episode))

# Visualizations
# Get the directory of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Define the path for the visualization folder relative to the script directory
visualization_folder = os.path.join(script_dir, 'visualizations')

# Create the visualization folder if it doesn't exist
os.makedirs(visualization_folder, exist_ok=True)

# Save the density plot with colored bins
best_bins = calculate_bin_edges_from_sizes(speed_mean, speed_std, list(best_state), min_val, max_val)
plot_density_with_colored_bins(cond_speeds, best_bins)
# plt.savefig(os.path.join(visualization_folder, 'density_with_colored_bins.png'), dpi=300)
# plt.close()

# Save the plot for the number of bins in the best state per episode
plot_bestsigma_per_episode_comp(num_episodes, best_sigma_per_episode, 0.07390039505777192)

# Save the histogram of bin sizes across all episodes
plot_bestbin_hist(best_states_per_episode)