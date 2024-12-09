# This script runs a simulated real-world optimization using covariance matrix adaptation.
# For simulation, a set of "optimal" parameters are provided based on random sampling.
# Running cma_test.py will perform an optimization that converges towards these optimal parameters.
# To emulate opportunistic optimization, this code updates three sets of optimization parameters based on three ranges of walking speeds.
# Thus, separate paramters are computed for slow, normal, and fast ranges of walking speeds.
# Here the walking speed is sampled from a normal distribution following real-world data.

import numpy as np
import random
import time
import numpy.matlib
from scipy.sparse.linalg import svds
import matplotlib.pyplot as plt
from cma_helper import init_constants, reinit_bin_upd, upd_plot_data, sample_param, f_multi, f_all, rank_params, cma_multi, cma_update, init_opt_vars, init_xmean, constrain_params, plot_single_mode
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
# f_params = np.array([[0.64051748, 0.58666667],[0.72715882,0.70916667],[0.80887392,0.85],[0.81887392,0.857]]) # initial values of the torque parameters (peak torque, rise time) for the three optimization bins based on the ranges of walking speed
m = 'Normal' #'Re-initialization' # 
bins = np.array([0.9, 1.24, 1.42, 1.75]) # defining speed ranges of the bins
# bins = np.array([0.9, 1.24, 1.42, 1.75, 1.8]) # defining speed ranges of the bins

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

# randomly sample the location of the optimal torque parameters for the simulation
f_offset = np.random.uniform(low=-offset_std_in_params, high=offset_std_in_params, size=(spd_bins, 1))
f_mult = np.random.uniform(low=1-scalar_std_in_params, high=1+scalar_std_in_params, size=1)

# randomly sample the walking speeds of all conditions in the simulated optimization
if speed_type == 'normal':
    cond_speeds = np.random.normal(loc=speed_mean, scale=speed_std, size=num_gens_cma*λ)
elif speed_type == 'uniform':
    cond_speeds = np.random.uniform(low=speed_mean-speed_std, high=speed_mean+speed_std, size=num_gens_cma*λ)

# defining matrices to store the optimization parameters and relevant plotting data
# x_mean = init_xmean(bins, N, f_params, np.array([0.8, 1.25, 1.7])) # starting param values
x_mean = init_xmean(bins, N, f_params, np.array([0.8, 1.25, 1.7, 1.78])) # starting param values

plot_sig_data = np.zeros((spd_bins,num_gens_cma+1))
plot_sig_data[:,0] = sigma
plot_rew_data = np.zeros((spd_bins,num_gens_cma+1))
goal = f_params*f_mult + f_offset # computing simulated "optimal" parameters
plot_rew_data[:,0] = f_all(goal, meas_noise, x_mean)
plot_mean_data = np.zeros((spd_bins,num_gens_cma+1, N))
plot_mean_data[:,0,:] = x_mean
bin_opt_vars = []
for i in range(spd_bins):
    bin_opt_vars.append(init_opt_vars(x_mean[i,:], sigma[i], N))
cond_counter = np.zeros((spd_bins), dtype=int)
gen_counter = np.zeros((spd_bins), dtype=int)
bin_gen_data = np.zeros((spd_bins, λ, N)) # stores the data for the conditions, or in sim just function evals
bin_gen_params = np.zeros((spd_bins, λ, N)) # stores the params for the conditions
bin_gen_params[:,0,:] = x_mean # initialize the params
constants.append(param_bounds)
constants.append(meas_noise)
constants.append(goal)

# main CMA simulation loop
for i, spd in enumerate(cond_speeds): # handle each simulated walking bout sequentially
    bin = np.where(spd > bins)[0][-1] # select the appropriate optimization bin based on the condition walking speed
    bin_gen_data[bin, cond_counter[bin], :] = bin_gen_params[bin, cond_counter[bin], :] # generate torque parameters from the appropriate optimization bin of parameters to evaluate for this condition
    cond_counter[bin] += 1 # increment the number of conditions completed for this optimization bin
    if cond_counter[bin] % λ == 0: # determine if generation is finished
        arindex, arx = rank_params(constants, bin, bin_gen_data[bin,:,:]) # rank the conditions from this generation from best to worst
        bin_opt_vars[bin] = cma_multi([constants, bin_opt_vars[bin]], arindex, bin_gen_params[bin,:,:]) # passing bin specific parameters to cma
        if m == 'Re-initialization': # determine if other bins should be updated as well
            bin_opt_vars, upd_flag = reinit_bin_upd(bin_opt_vars, bin, len(bins)-1, bins, 2, sigma[0], upd_flag, m) # update other optimization bins based on this completed generation
        cond_counter[bin] = 0 # reset condition count
        gen_counter[bin] += 1 # increment the generation count for this optimization bin
        plot_sig_data, plot_rew_data, plot_mean_data = upd_plot_data(bin_opt_vars, gen_counter, bin, plot_sig_data, plot_rew_data, plot_mean_data, len(bins)-1, constants) # store data for plotting
    bin_gen_params[bin, cond_counter[bin], :] = sample_param(bin_opt_vars[bin], param_bounds) # sample new param from the mean/sigma for that bin
    new_params[0] = bin_gen_params[bin, cond_counter[bin], 0]*weight # normalize the sampled peak torque magnitudes by subject weight
    new_params[2] = bin_gen_params[bin, cond_counter[bin], 0]*rise_scalar # normalize the sampled rise time values
    print("New params:", new_params) # print the new parameters to evaluate
    cma_data = [i, bin_gen_data, bin_gen_params, plot_sig_data, plot_rew_data, plot_mean_data, bin_opt_vars, cond_counter, gen_counter] # save cma values
opt_results =  bin_opt_vars # save optimization parameters

# plotting code
print("Mode", m)
plot_single_mode(plot_sig_data, plot_rew_data, plot_mean_data, goal, num_gens_cma, spd_bins, bins, gen_counter)
meta_plot.append([plot_sig_data, plot_rew_data, plot_mean_data])
print("Gen counts:", gen_counter)
print("Run time (s):", time.time() - start_time)
