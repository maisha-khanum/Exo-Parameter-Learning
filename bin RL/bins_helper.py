
import numpy as np
import random
import time
import numpy.matlib
from scipy.sparse.linalg import svds
import matplotlib.pyplot as plt
import os
from scipy.stats import norm

# Example usage
mean = 1.328  # Approximate mean
std = 0.212   # Approximate standard deviation
num_bins = 5
min_val = 0.9
max_val = 1.75

def initialize_params(num_bins, torque_range, rise_time_range):
    """
    Initializes the parameters for optimization bins based on the given number of bins and ranges.

    Parameters:
        num_bins (int): The number of bins to split the parameters into.
        torque_range (tuple): The range (min, max) for peak torque.
        rise_time_range (tuple): The range (min, max) for rise time.

    Returns:
        f_params (ndarray): Initialized parameters (peak torque, rise time) for each bin.
    """
    # Create linearly spaced values for each parameter
    peak_torque_values = np.linspace(torque_range[0], torque_range[1], num_bins)
    rise_time_values = np.linspace(rise_time_range[0], rise_time_range[1], num_bins)
    
    # Combine the values into a 2D array
    f_params = np.column_stack((peak_torque_values, rise_time_values))
    return f_params

def find_max_below_threshold(arrays, threshold):
    """
    Find the first index where the maximum value across arrays falls below a given threshold.

    Parameters:
        arrays (np.ndarray): 2D array where each row represents an array.
        threshold (float): The threshold value.

    Returns:
        int: The index where the maximum value across arrays is below the threshold, or -1 if not found.
    """
    arrays = np.array(arrays)  # Ensure input is a numpy array
    max_values = np.max(arrays, axis=0)  # Compute maximum value across arrays at each index
    
    for i, max_value in enumerate(max_values):
        if max_value < threshold:
            return i  # Return the first index where the condition is met
    
    return -1  # Return -1 if no such index is found

def calculate_constrained_bin_edges(mean, std, num_bins, min_val, max_val):
    """
    Calculate bin edges spaced equally in percentiles for a normal distribution, 
    constrained by min and max values.

    Parameters:
        mean (float): The mean of the distribution.
        std (float): The standard deviation of the distribution.
        num_bins (int): The number of bins.
        min_val (float): The minimum value of the bins.
        max_val (float): The maximum value of the bins.

    Returns:
        np.ndarray: Array of bin edges.
    """
    # Calculate the cumulative distribution function (CDF) for min and max
    min_cdf = norm.cdf(min_val, loc=mean, scale=std)
    max_cdf = norm.cdf(max_val, loc=mean, scale=std)
    
    # Generate equally spaced percentiles within the CDF range
    percentiles = np.linspace(min_cdf, max_cdf, num_bins + 1)
    
    # Convert percentiles back to values using the inverse CDF (ppf)
    bin_edges = norm.ppf(percentiles, loc=mean, scale=std)
    
    return np.round(bin_edges, 2)

def calculate_bin_edges_from_sizes(mean, std, bin_sizes, min_val, max_val):
    """
    Calculate bin edges from an array of bin sizes, constrained by min and max values.
    
    Parameters:
        mean (float): The mean of the distribution.
        std (float): The standard deviation of the distribution.
        bin_sizes (list of int): The sizes (percentages) of each bin. Should sum to 100.
        min_val (float): The minimum value of the bins.
        max_val (float): The maximum value of the bins.

    Returns:
        np.ndarray: Array of bin edges.
    """
    if sum(bin_sizes) != 100:
        raise ValueError("Bin sizes must sum to 100.")

    # Calculate the cumulative distribution function (CDF) for min and max
    min_cdf = norm.cdf(min_val, loc=mean, scale=std)
    max_cdf = norm.cdf(max_val, loc=mean, scale=std)
    
    # Map bin sizes to the cumulative CDF range
    cdf_range = max_cdf - min_cdf
    cumulative_sizes = np.cumsum([0] + bin_sizes) / 100.0  # Convert percentages to fractions
    cumulative_cdfs = min_cdf + cdf_range * cumulative_sizes

    # Convert cumulative CDF values to bin edges using the inverse CDF (ppf)
    bin_edges = norm.ppf(cumulative_cdfs, loc=mean, scale=std)
    
    return np.round(bin_edges, 2)