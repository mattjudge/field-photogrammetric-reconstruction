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

worldx = world[0,:]
worldy = world[1,:]
worldz = world[2,:]

# regular grid covering the domain of the data
X, Y = np.meshgrid(np.arange(-0.5, 0.5, 0.05), np.arange(-0.5, 0.5, 0.05))
XX = X.flatten()
YY = Y.flatten()
data = world.transpose()
# best-fit linear plane
A = np.c_[data[:, 0], data[:, 1], np.ones(data.shape[0])]
C, _, _, _ = np.linalg.lstsq(A, data[:, 2])  # coefficients

# evaluate it on grid
Z = C[0] * X + C[1] * Y + C[2]

# or expressed using matrix/vector product
# Z = np.dot(np.c_[XX, YY, np.ones(XX.shape)], C).reshape(X.shape)

centroid = np.mean(world, axis=1, keepdims=True)
print("centroid", centroid)
world -= centroid

cos_t = 1/np.sqrt(C[0]**2 + C[1]**2 + 1)
sin_t = np.sin(np.arccos(cos_t))
ux = cos_t * -C[1]
uy = cos_t * C[0]
n = np.sqrt(ux**2 + uy**2)
ux /= n
uy /= n

R = np.array([
    [cos_t + ux**2 * (1-cos_t), ux*uy*(1-cos_t),            uy*sin_t],
    [ux*uy*(1-cos_t),           cos_t + uy**2 * (1-cos_t),  -ux*sin_t],
    [-uy*sin_t,                 ux*sin_t,                   cos_t]
])

# world[2,:] -= -C[2]
world = R.dot(world)

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


max_range = np.array([X.max()-X.min(), Y.max()-Y.min(), Z.max()-Z.min()]).max() / 2.0

mid_x = (X.max()+X.min()) * 0.5
mid_y = (Y.max()+Y.min()) * 0.5
mid_z = (Z.max()+Z.min()) * 0.5
ax.set_xlim(mid_x - max_range, mid_x + max_range)
ax.set_ylim(mid_y - max_range, mid_y + max_range)
ax.set_zlim(mid_z - max_range, mid_z + max_range)
# ax.set_zlim(mid_z - max_range/50, mid_z + max_range/50)

plt.show()
