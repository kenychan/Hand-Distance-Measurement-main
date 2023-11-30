import numpy as np
from sklearn.neighbors import KernelDensity
from sklearn.preprocessing import MinMaxScaler

# Sample 2D points (replace this with your own dataset)
points = np.random.rand(100, 2)  # Generating random points for demonstration
print(points)
# Calculate kernel density estimation
kde = KernelDensity(bandwidth=0.2, kernel='gaussian')
kde.fit(points)

# Evaluate the KDE on a grid
x_min, x_max = points[:, 0].min() - 0.1, points[:, 0].max() + 0.1
y_min, y_max = points[:, 1].min() - 0.1, points[:, 1].max() + 0.1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
grid_points = np.c_[xx.ravel(), yy.ravel()]
densities = np.exp(kde.score_samples(grid_points))
densities = densities.reshape(xx.shape)

# Normalize the densities to [0, 1]
scaler = MinMaxScaler()
normalized_densities = scaler.fit_transform(densities)

# Calculate a single value indicating tightness of the cluster
mean_density = np.mean(normalized_densities)

print("Mean density:", mean_density)
