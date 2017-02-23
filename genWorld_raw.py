import numpy as np
import scipy

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

from generateWorld4 import *

# vel = load_velocity_fields('floor0','floor1')
vel = load_velocity_fields('frame9900','frame9903')

world, shape = genWorld(vel)

crop = 32
X = world[0,:].reshape(shape)[crop:-crop:,crop:-crop:]
Y = world[1,:].reshape(shape)[crop:-crop:,crop:-crop:]
Z = world[2,:].reshape(shape)[crop:-crop:,crop:-crop:]

fig = plt.figure()
ax = fig.gca(projection='3d')

# Plot the surface.
surf = ax.plot_surface(X, Y, Z, cmap=cm.hot,
                       linewidth=0, antialiased=False)

ax.set_aspect('equal')
# Customize the z axis.
# ax.set_zlim(-1.01, 1.01)
# ax.zaxis.set_major_locator(LinearLocator(10))
# ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

# Add a color bar which maps values to colors.
# fig.colorbar(surf, shrink=0.5, aspect=5)
fig.colorbar(surf)



plt.show()
