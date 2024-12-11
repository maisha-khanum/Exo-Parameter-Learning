# functions to assist with the CMA simulation
import numpy as np
import random
import time
import numpy.matlib
from scipy.sparse.linalg import svds
import matplotlib.pyplot as plt

# initialize the constant values for CMA optimization
def init_constants(N, num_gens_cma):
    λ = 4+int(3*np.log(N))
    stopeval = λ*(num_gens_cma) # samples * generations
    mu = λ // 2
    weights = np.log(mu + 1 / 2) - np.log(np.asarray(range(1, mu + 1))).astype(np.float32)
    weights = weights / np.sum(weights)
    mueff = (np.sum(weights) ** 2) / np.sum(weights ** 2)
    cc = (4 + mueff / N) / (N + 4 + 2 * mueff / N)
    cs = (mueff + 2) / (N + mueff + 5)
    c1 = 2 / ((N + 1.3) ** 2 + mueff)
    cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((N + 2) ** 2 + mueff))
    damps = 1 + 2 * max(0, ((mueff - 1) / (N + 1)) ** 0.5 - 1) + cs
    chiN = N ** 0.5 * (1 - 1 / (4 * N) + 1 / (21 * N ** 2))
    return λ, [N, λ, stopeval, mu, weights, mueff, cc, cs, c1, cmu, damps, chiN]

# when generation is complete, update optimization parameters of the other optimization parameters in the other bins of walking speed
def reinit_bin_upd(bin_opt_vars, bin, spd_bins, bins, N, init_sigma, upd_flag, mode, upd_thresh = 0.5):
    change_vec = bin_opt_vars[bin][0] - bin_opt_vars[bin][1] # just apply same update to all bins
    change_sigma = np.clip(bin_opt_vars[bin][2]/init_sigma, 0., 1.)
    if change_sigma < upd_thresh and not upd_flag:
        upd_flag = True
        temp_flag = True
    else:
        temp_flag = False
    for i in range(spd_bins):
        if i != bin:
            sigma_scalar = np.clip(bin_opt_vars[i][2]/init_sigma, 0., 1.)
            if change_sigma < sigma_scalar:
                if mode != 'Re-init sigma':
                    bin_opt_vars[i][0] = bin_opt_vars[i][0] + change_vec*sigma_scalar # correct the xmean
                for j in range(2,8):
                    bin_opt_vars[i][j] = bin_opt_vars[bin][j]*sigma_scalar + bin_opt_vars[i][j]*(1-sigma_scalar)

    return bin_opt_vars, upd_flag

# extract optimization data to plot
def upd_plot_data(bin_opt_vars, gen_counter, bin, plot_sig_data, plot_rew_data, plot_mean_data, spd_bins, constants):
    N, λ, stopeval, mu, weights, mueff, cc, cs, c1, cmu, damps, chiN, param_bounds, meas_noise, f_params = constants
    tot_gens = np.sum(gen_counter)
    for i in range(spd_bins):
        plot_mean_data[i,tot_gens,:] = bin_opt_vars[i][0]
        plot_sig_data[i,tot_gens] = bin_opt_vars[i][2]
        plot_rew_data[i,tot_gens] = f_multi(f_params[i,:], meas_noise, bin_opt_vars[i][0])
    return plot_sig_data, plot_rew_data, plot_mean_data

# sample torque parameters for one condition given the CMA parameters and constrain the sample within the defined range (param_bounds)
def sample_param(opt_vars, param_bounds):
    xmean, xold, sigma, pc, ps, B, D, C, invsqrtC, eigeneval, local_cnt = opt_vars
    sample = constrain_params(xmean + sigma * B.dot(D * np.random.randn(len(D))), param_bounds)
    return sample

# evaluate the parameters for a given condition (x_eval) by computing the distance between the optimal parameters (f_params) with added noise (meas_noise) 
def f_multi(f_params, meas_noise, x_eval): # simulated reward function
    noise = np.array([1+random.gauss(0, meas_noise), 1+random.gauss(0, meas_noise)])
    val = np.sum(np.abs((x_eval - f_params*noise))) # make negative?
    return val

# evaluate all conditions for a given generation for each of the three optimization bins based on ranges of walking speed
def f_all(f_vec, meas_noise, x_vec):
    spd_bins = x_vec.shape[0]
    tot_val = np.zeros(spd_bins)
    for i in range(spd_bins):
        tot_val[i] = f_multi(f_vec[i,:], meas_noise, x_vec[i,:])
    return tot_val

# order conditions from one generation from best to worst
def rank_params(constants, bin, arx):
    N, λ, stopeval, mu, weights, mueff, cc, cs, c1, cmu, damps, chiN, param_bounds, meas_noise, f_params = constants
    arfitness = np.zeros(λ)
    for k in range(λ):
        arfitness[k] = f_multi(f_params[bin], meas_noise, arx[k])
    arindex = np.argsort(arfitness)
    return arindex, arx#, arfitness

# load the optimization parameters from the proper optimization bin and perform a CMA update to those paramters
def cma_multi(opt_inputs, arindex, arx):
    constants, opt_vars = opt_inputs
    mean_list = []
    # constants
    N, λ, stopeval, mu, weights, mueff, cc, cs, c1, cmu, damps, chiN, param_bounds, meas_noise, f_params = constants
    xmean, xold, sigma, pc, ps, B, D, C, invsqrtC, eigeneval, local_cnt = opt_vars
    local_cnt += λ
    xold = xmean
    xmean = weights.dot(arx[arindex[0:mu]])
    # update CMA parameters
    opt_vars = [xmean, xold, sigma, pc, ps, B, D, C, invsqrtC, eigeneval, local_cnt]
    new_opt_vars = cma_update(constants, opt_vars, arx, arindex)
    return new_opt_vars#, [new_opt_vars[2]]

# update the CMA paramters based on the ordered conditions from one generation of optimization
def cma_update(constants, opt_vars, arx, arindex):
    N, λ, stopeval, mu, weights, mueff, cc, cs, c1, cmu, damps, chiN, _, _, _ = constants
    xmean, xold, sigma, pc, ps, B, D, C, invsqrtC, eigeneval, local_cnt = opt_vars

    ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * invsqrtC.dot((xmean - xold) / sigma)
    hsig = np.linalg.norm(ps) / np.sqrt(1 - (1 - cs) ** (2 * local_cnt / λ)) / chiN < 1.4 + 2 / (N + 1)
    pc = (1 - cc) * pc + hsig * np.sqrt(cc * (2 - cc) * mueff) * ((xmean - xold) / sigma)
    artmp = (1 / sigma) * (arx[arindex[0:mu]] - xold)
    C = (1 - c1 - cmu) * C + c1 * (pc.dot(pc.T) + (1 - hsig) * cc * (2 - cc) * C) + cmu * artmp.T.dot(np.diag(weights)).dot(artmp)
    sigma = sigma * np.exp((cs / damps) * (np.linalg.norm(ps) / chiN - 1))
    # print(sigma)
    if local_cnt - eigeneval > λ / (c1 + cmu) / N / 10:
        eigeneval = local_cnt
        C = np.triu(C) + np.triu(C, 1).T
        D, B = np.linalg.eig(C)
        D = np.sqrt(D)
        invsqrtC = B.dot(np.diag(D ** -1).dot(B.T))

    opt_vars = [xmean, xold, sigma, pc, ps, B, D, C, invsqrtC, eigeneval, local_cnt]
    return opt_vars

# setting initial values for the optimization parameters
def init_opt_vars(x_mean, sigma, N):
    pc = np.zeros(N).astype(np.float32)
    ps = np.zeros(N).astype(np.float32)
    B = np.eye(N, N).astype(np.float32)
    D = np.ones(N).astype(np.float32)
    C = B * np.diag(D ** 2) * B.T
    invsqrtC = B * np.diag(D ** -1) * B.T
    eigeneval = 0
    local_cnt = 0
    opt_vars = [x_mean, x_mean, sigma, pc, ps, B, D, C, invsqrtC, eigeneval, local_cnt]
    # print(opt_vars)
    # print(np.array(opt_vars).shape)
    # print("HELLO")
    return opt_vars
    # return np.copy(opt_vars)

# set the mean value of the optimization parameters based on the defined ranges of walking speed (def_speeds) and defined torque parameters (def_params)
def init_xmean(bins, N, def_params, def_speeds):
    new_params = np.zeros((len(bins)-1, N))
    new_speeds = np.zeros(len(bins)-1)
    for i in range(len(bins)-1):
        avg_spd = (bins[i] + bins[i+1])/2
        if abs(avg_spd) > 100:
            if avg_spd > 100:
                avg_spd = bins[i]
            else:
                avg_spd = bins[i+1]
        new_speeds[i] = avg_spd
    for j, spd in enumerate(new_speeds):
        if len(def_speeds) == 1:
            new_params[j, :] = def_params[0, :]
        elif spd in def_speeds:
            spd_ind = np.where(spd == def_speeds)[0][0]
            new_params[j,:] = def_params[spd_ind,:]
        else: # interpolate
            spd_ind = np.where(spd > def_speeds)[0][-1]
            spd_diff = (spd - def_speeds[spd_ind])/(def_speeds[spd_ind+1]-def_speeds[spd_ind])
            new_params[j,:] = def_params[spd_ind,:]*(1-spd_diff) + spd_diff*def_params[spd_ind+1,:]
    return new_params

# constrain torque parameters within the defined range of values in param_bounds
def constrain_params(sampled_param, param_bounds):
    param_min = param_bounds[:,0]
    param_max = param_bounds[:,1]
    constrain_sample = np.minimum(np.maximum(sampled_param, param_min), param_max)
    return constrain_sample

# extract the mean values of the torque parameters from the saved data to create a plot
def grabMeans(save_mean_data, gen_counter):
    spd_bins = save_mean_data.shape[0]
    save = []
    for i in range(spd_bins):
        save.append(np.expand_dims(save_mean_data[i,gen_counter[i],:],axis=0))
    savecat = np.concatenate(save,axis=0)
    return savecat

# plot the parameter values for each of the three bins of walking speed (slow, normal, fast)
def plot_single_mode(plot_sig_data, plot_rew_data, plot_mean_data, goal_all, num_gens_cma, spd_bins, bins, gen_counter):
    """
    Generalized function to handle any number of bins and plot optimization results.
    """
    all_mean_data = np.copy(plot_mean_data)
    min_gen_plots = num_gens_cma + 1

    # Plot sigma data
    plt.figure()
    labels = [f"{bins[i]} to {bins[i+1]}" for i in range(spd_bins)]
    plt.plot(plot_sig_data[:, :min_gen_plots - 1].T)
    plt.xlabel('Number of Generations')
    plt.ylabel('Optimization Covariance Indicating Convergence (Sigma)')
    plt.legend(labels, title="Speed Bins")
    plt.show()

    # Calculate errors
    err = plot_mean_data - np.expand_dims(goal_all, axis=1)  # Difference from goal
    abs_err = np.abs(err[:, :, 0]) + np.abs(err[:, :, 1])
    abs_err = abs_err.T
    mean_abs_err = np.mean(abs_err[:min_gen_plots, :], axis=-1)
    std_abs_err = np.std(abs_err[:min_gen_plots, :], axis=-1)

    # Create subplots for each bin
    fig, ax = plt.subplots(spd_bins, 1, figsize=(6.4, 2 * spd_bins))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    plot_indiv_run = False

    for i in range(spd_bins):
        if plot_indiv_run:
            # Individual run plotting (if enabled)
            r = 0  # Placeholder for run index, adjust if needed
            ax[i].plot(all_mean_data[i, :num_gens_cma, 1, r], all_mean_data[i, :num_gens_cma, 0, r], color=colors[i % len(colors)])
            ax[i].scatter(all_mean_data[i, 0, 1, r], all_mean_data[i, 0, 0, r], marker='o', color=colors[i % len(colors)])
            ax[i].scatter(all_mean_data[i, num_gens_cma, 1, r], all_mean_data[i, num_gens_cma, 0, r], marker='x', color=colors[i % len(colors)])
        else:
            # Plot aggregated mean data for the bin
            ax[i].plot(plot_mean_data[i, :num_gens_cma, 1], plot_mean_data[i, :num_gens_cma, 0], color=colors[i % len(colors)])
            ax[i].scatter(plot_mean_data[i, 0, 1], plot_mean_data[i, 0, 0], marker='o', color=colors[i % len(colors)], label='Starting parameters')
            ax[i].scatter(plot_mean_data[i, num_gens_cma - 1, 1], plot_mean_data[i, num_gens_cma - 1, 0], marker='x', color=colors[i % len(colors)], label='Final parameters')
        
        # Add goal parameters as red stars
        ax[i].scatter(goal_all[i, 1], goal_all[i, 0], color='red', marker='*', s=150, label='Goal parameters')

        # Set labels and titles dynamically
        ax[i].set_xlabel('Rise Time')
        ax[i].set_ylabel('Peak Torque')
        ax[i].set_title(f'Bin {i+1}: {labels[i]}', loc='left')
        ax[i].set_xlim(0.4, 1.1)
        ax[i].set_ylim(0.5, 0.9)

    # Add a single legend at the bottom of the figure
    fig.legend(
        loc='lower center',
        bbox_to_anchor=(0.5, -0.05),
        ncol=spd_bins,
        title="Legend",
        frameon=False
    )
    
    # Adjust layout for better visibility
    fig.subplots_adjust(hspace=0.5)
    plt.show()

