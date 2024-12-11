import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# Parameters
mean = 1.35
std = 0.15
x = np.linspace(mean - 4 * std, mean + 4 * std, 1000)

# Normal distribution curve
pdf = norm.pdf(x, mean, std)

# Percentile thresholds
percentiles = [0, 33, 66, 100]
z_values = [norm.ppf(p / 100, mean, std) for p in percentiles]

# Plot the curve
plt.figure(figsize=(8, 5))
plt.plot(x, pdf, label='Normal Distribution', color='black')

# Shaded regions
colors = ['#a6cee3', '#1f78b4', '#b2df8a']
labels = ['0-33%', '33-66%', '66-100%']
for i in range(len(percentiles) - 1):
    plt.fill_between(x, pdf, where=(x >= z_values[i]) & (x < z_values[i + 1]),
                     color=colors[i], alpha=0.6, label=labels[i])

# Annotations and aesthetics
plt.axvline(mean, color='red', linestyle='--', label=f'Mean = {mean}')
plt.title('Normal Distribution with Percentile Shading')
plt.xlabel('Value')
plt.ylabel('Probability Density')
plt.legend()
plt.grid(alpha=0.3)
plt.show()
