import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from cma import CMAEvolutionStrategy

# Steeper 2D bowl-shaped objective function
def objective_function(x):
    return (x[0]**2 + x[1]**2) + 10*np.sin(x[0]) + 2*x[1]  # Steeper quadratic bowl

# Function to plot covariance ellipses
def plot_ellipse(ax, mean, cov, color):
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]
    theta = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
    width, height = 2 * np.sqrt(vals)
    ellipse = Ellipse(xy=mean, width=width, height=height, angle=theta, edgecolor=color, fc='None', lw=1)  # Thinner lines
    ax.add_patch(ellipse)

# Color gradient for generations
num_generations = 12  # Change this value to adjust the number of generations
colors = plt.cm.get_cmap('autumn', num_generations)  # Use the number of generations

# Scaling and unscaling functions
def scale_to_bounds(x, lower_bound=-4, upper_bound=4):
    scaled = lower_bound + x * (upper_bound - lower_bound)
    scaled = np.minimum(np.maximum(scaled, lower_bound), upper_bound)
    return scaled

def scale_to_01(x, lower_bound=-4, upper_bound=4):
    return (x - lower_bound) / (upper_bound - lower_bound)

# Initialize CMA-ES with values scaled between 0 and 1
initial_mean = np.array([0.8, 0.6])  # Initial mean in 0-1 space
sigma = 0.3  # Initial step size
cmaes = CMAEvolutionStrategy(initial_mean, sigma)
for i in range(3,6):
    cmaes.sp.weights[i] = 0

# Generate the 2D landscape (for filled contour plot)
x = np.linspace(-4, 4, 100)
y = np.linspace(-4, 4, 100)
X, Y = np.meshgrid(x, y)
Z = objective_function([X, Y])

# Initialize data storage for plotting
candidates_all = []
covariances_all = []
means_all = []
arrows = []

# Track the previous mean for arrows
prev_mean = scale_to_bounds(initial_mean)  # Scale initial mean to landscape bounds

# Set noise level for sampled costs
noise_level = 0.0

# To track true costs and values over generations
mean_costs = []
mean_values = []
sigmas = []

# Perform optimization and store data
for iteration in range(num_generations):  # Run for specified generations
    candidates_scaled = cmaes.ask()  # Ask for candidates in 0-1 space
    candidates_scaled[-1] = cmaes.mean
    
    # Scale candidates from 0-1 to actual landscape bounds [-4, 4]
    candidates = [scale_to_bounds(cand) for cand in candidates_scaled]
    
    color = colors(iteration)  # Color for this generation
    
    # Store candidates and means
    candidates_all.append(candidates)
    means_all.append(scale_to_bounds(cmaes.mean))

    # Extract the covariance matrix (sigma^2 * C) and scale to landscape
    cov_scaled = cmaes.sigma**2 * cmaes.C
    cov = np.diag([8, 8]) @ cov_scaled @ np.diag([8, 8])  # Scale covariance to landscape bounds
    covariances_all.append(cov)
    
    # Sample the costs on the landscape with added noise
    costs = [
        objective_function(cand) * np.random.normal(1, noise_level)  # Apply noise
        for cand in candidates
    ]
    
    # Store the true cost and mean value of the mean
    mean_costs.append(objective_function(scale_to_bounds(cmaes.mean)))
    mean_values.append(scale_to_bounds(cmaes.mean))
    sigmas.append(cmaes.sigma)

    # Return scaled candidates and costs to CMA-ES
    cmaes.tell(candidates_scaled, costs)
    
    # Store the arrow information for plotting later
    arrows.append((prev_mean, scale_to_bounds(cmaes.mean)))
    prev_mean = scale_to_bounds(cmaes.mean)


# Plot stored candidates and means
for iteration in range(num_generations):
    # Create plot with GridSpec
    fig = plt.figure(figsize=(10, 6))  # Wider aspect ratio
    gs = fig.add_gridspec(3, 6)  # 3 rows, 6 columns

    # First subplot for optimization process (large and square)
    ax1 = fig.add_subplot(gs[0:3, 0:4])  # Take the top three rows and first four columns
    ax1.contourf(X, Y, Z, levels=50, cmap='Greens')  # Lighter color map for better visibility
    
    # Plot sampled points (smaller marker size)
    for i in range(iteration+1):
        candidates = candidates_all[i]
        color = colors(i) if i==iteration else [.5, .5, .5] # Color for this generation
        size = 10 if i==iteration else 5
        ax1.scatter(*zip(*candidates), color=color, s=size, label=f'Generation {iteration}' if iteration == 0 else "")  # Smaller markers
    
        # Plot covariance ellipse (thinner lines)
        plot_ellipse(ax1, means_all[i], covariances_all[i], color=color)

        ax1.scatter(means_all[i][0],means_all[i][1], color=color, marker='x')

    # Plot arrows showing movement from previous mean to current mean (black arrows)
    for prev_mean, current_mean in arrows[:iteration]:
        ax1.annotate('', xy=current_mean, xytext=prev_mean,
                    arrowprops=dict(facecolor='black', edgecolor='black', shrink=0.05, width=0.5, headwidth=5, linestyle='-', linewidth=1, alpha=0.8))

    # Final plot settings: bound xlim and ylim to [-4, 4]
    ax1.set_xlim([-4, 4])
    ax1.set_ylim([-4, 4])
    ax1.set_title('Cost landscape, means(x\'s) sampled points (dots), \n covariance matrices (ellipses)')
    ax1.set_xlabel('x1')
    ax1.set_ylabel('x2')

    # Second subplot for sigma over generations (smaller)
    ax2 = fig.add_subplot(gs[0, 4:6])  # Take the first row and last two columns
    ax2.plot(range(1, num_generations + 1)[:iteration+1], sigmas[:iteration+1], marker='o', color='blue')
    ax2.set_title('Sigma')
    ax2.set_xlabel('Generation')
    ax2.set_ylabel('Sigma')
    ax2.grid()

    # Third subplot for mean cost over generations (smaller)
    ax3 = fig.add_subplot(gs[1, 4:6])  # Take the second row and last two columns
    ax3.plot(range(1, num_generations + 1)[:iteration+1], mean_costs[:iteration+1], marker='o', color='orange')
    ax3.set_title('True Cost of Mean')
    ax3.set_xlabel('Generation')
    ax3.set_ylabel('True Cost')
    ax3.grid()

    # Fourth subplot for mean values over generations (smaller)
    ax4 = fig.add_subplot(gs[2, 4:6])  # Take the third row and last two columns
    mean_x, mean_y = zip(*mean_values)
    ax4.plot(range(1, num_generations + 1)[:iteration+1], mean_x[:iteration+1], marker='o', label='Mean x1', color='purple')
    ax4.plot(range(1, num_generations + 1)[:iteration+1], mean_y[:iteration+1], marker='o', label='Mean x2', color='green')
    ax4.set_title('Mean Values')
    ax4.set_xlabel('Generation')
    ax4.set_ylabel('Mean Values')
    ax4.legend()
    ax4.grid()

    plt.tight_layout()
    # plt.show()

    output_dir = './generation_figs'
    os.makedirs(output_dir, exist_ok=True)

    fig.savefig(f'./generation_figs/gen{iteration}.png')
print('done')
    

