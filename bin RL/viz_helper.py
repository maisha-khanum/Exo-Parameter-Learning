import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import matplotlib.cm as cm
import os

# Get the directory of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Define the path for the visualization folder relative to the script directory
visualization_folder = os.path.join(script_dir, 'visualizations')

# Create the visualization folder if it doesn't exist
os.makedirs(visualization_folder, exist_ok=True)

def plot_density_with_colored_bins(cond_speeds, bin_edges):
    """
    Plots a histogram of the walking speeds, a KDE for probability density,
    and highlights each bin with a different color.

    Parameters:
    - cond_speeds: array-like, sampled walking speeds
    - bin_edges: array-like, edges of bins to highlight
    """
    # Generate KDE for smoothed probability density
    kde = gaussian_kde(cond_speeds)
    x_vals = np.linspace(min(cond_speeds), max(cond_speeds), 1000)
    kde_vals = kde(x_vals)

    # Plot the histogram
    plt.figure(figsize=(10, 6))
    plt.hist(cond_speeds, bins=20, density=True, alpha=0.5, color='gray', edgecolor='black', label='Walking Speed Histogram')

    # Plot the KDE
    plt.plot(x_vals, kde_vals, label='Probability Density (KDE)', color='blue', linewidth=2)

    # Highlight bins with different colors
    num_bins = len(bin_edges) - 1
    color_map = cm.get_cmap('viridis', num_bins)  # Use a colormap to assign unique colors
    for i in range(num_bins):
        plt.axvspan(bin_edges[i], bin_edges[i + 1], color=color_map(i), alpha=0.3)

    # Customize the plot
    plt.title('Probability Density vs. Walking Speed with Colored Bins', fontsize=14)
    plt.xlabel('Walking Speed (m/s)', fontsize=12)
    plt.ylabel('Probability Density', fontsize=12)
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(visualization_folder, 'density_with_colored_bins.png'), dpi=300)
    plt.close()
    # plt.show()


def plot_bestsigma_per_episode(num_episodes, best_sigma_per_episode):
    plt.figure(figsize=(10, 6))
    plt.plot(range(num_episodes), best_sigma_per_episode, label='Number of Bins')
    plt.title('Number of Bins in Best Sigma per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Number of Bins')
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(visualization_folder, 'sigma_per_episode.png'), dpi=300)
    plt.close()

def plot_bestsigma_per_episode_comp(num_episodes, best_sigma_per_episode, comp):
    plt.figure(figsize=(10, 6))
    plt.plot(range(num_episodes), best_sigma_per_episode, label='Sigma for Best Bin')
    
    # Add a horizontal line at y = 0.07390039505777192
    plt.axhline(y=comp, color='r', linestyle='--', label='Sigma for Generic Bin')

    plt.title('Sigma of Best Bin per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Sigma')
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(visualization_folder, 'sigma_per_episode.png'), dpi=300)
    plt.close()


def plot_bestbin_hist(best_states_per_episode):
    # Save the histogram of bin sizes across all episodes
    plt.figure(figsize=(10, 6))
    plt.hist(best_states_per_episode, bins=range(min(best_states_per_episode), max(best_states_per_episode) + 1), edgecolor='black', alpha=0.7)
    plt.title('Distribution of Bin Sizes in Best States Across Episodes')
    plt.xlabel('Number of Bins')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.savefig(os.path.join(visualization_folder, 'bin_sizes_distribution.png'), dpi=300)
    plt.close()

# # Parameters for the simulation
# speed_mean = 1.35
# speed_std = 0.15
# num_gens_cma = 9
# λ = 10  # Number of samples per generation
# # speed_type = 'uniform'  # Type of distribution ('normal' or 'uniform')
# speed_type = 'normal'

# # Randomly sample the walking speeds
# if speed_type == 'normal':
#     cond_speeds = np.random.normal(loc=speed_mean, scale=speed_std, size=num_gens_cma*λ)
# elif speed_type == 'uniform':
#     cond_speeds = np.random.uniform(low=speed_mean-2*speed_std, high=speed_mean+2*speed_std, size=num_gens_cma*λ)

# # Example bin edges for highlighting
# bin_edges = [0.9, 1.24, 1.42, 1.75]

# # Call the function to plot
# plot_density_with_colored_bins(cond_speeds, bin_edges)
