import numpy as np
# import visvis as vv
import scipy

from generateWorld2 import *

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

print("ux", ux)
print("uy", uy)
print("ux**2 + uy**2", ux**2 + uy**2)

R = np.array([
    [cos_t + ux**2 * (1-cos_t), ux*uy*(1-cos_t),            uy*sin_t],
    [ux*uy*(1-cos_t),           cos_t + uy**2 * (1-cos_t),  -ux*sin_t],
    [-uy*sin_t,                 ux*sin_t,                   cos_t]
])

# world[2,:] -= -C[2]
world = R.dot(world)

worldx = world[0,:]
worldy = world[1,:]
worldz = world[2,:] * 10


# import pylab
# pylab.figure()
# pylab.imshow(worldx.reshape(vel[0].shape), interpolation='none', cmap=pylab.cm.hot)  # , vmin=-4, vmax=4)
# pylab.colorbar()
# pylab.title('x per pixel')
# pylab.savefig('__pp_x.png')
#
# pylab.figure()
# pylab.imshow(worldy.reshape(vel[0].shape), interpolation='none', cmap=pylab.cm.hot)  # , vmin=-4, vmax=4)
# pylab.colorbar()
# pylab.title('y per pixel')
# pylab.savefig('__pp_y.png')
#
# pylab.figure()
# pylab.imshow(worldz.reshape(vel[0].shape), interpolation='none', cmap=pylab.cm.hot)  # , vmin=-4, vmax=4)
# pylab.colorbar()
# pylab.title('z per pixel')
# pylab.savefig('__pp_z.png')

# app = vv.use()
#
# # prepare axes
# a = vv.gca()
# a.cameraType = '3d'
# a.daspectAuto = False
# print("view", a.camera.GetViewParams())
# # a.SetView(loc=(-1000,0,0))
# # a.camera.SetView(None, loc=(-1000,0,0))
# # a.SetLimits(rangeX=(-0.2, 0.2), rangeY=(-0.5, 0.5), rangeZ=(-0.5, 0), margin=0.02)
#
# # draw points
# # pp = vv.Pointset(world.transpose())
# # l = vv.plot(pp, ms='.', mc='r', mw='1', ls='', mew=0)
# # l.alpha = 0.2
# #
# # pb = vv.Pointset(np.vstack([XX, YY, Z.flatten()]).transpose())
# # lb = vv.plot(pb, ms='.', mc='b', mw='1', ls='', mew=0)
# # lb.alpha = 0.2
#
#
# # draw mesh
# # width = vel[0].shape[1]//1
# # height = int(np.sqrt(worldx.shape[0] * 9/16))
# # height = 1080
# width = 1920 // 30
# l = vv.surf(worldx.reshape(shape), worldy.reshape(shape), worldz.reshape(shape))
# l.colormap = vv.CM_HOT
#
# a.axis.xLabel = 'x'
# a.axis.yLabel = 'y'
# a.axis.zLabel = 'z'
#
# app.Run()



from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter


fig = plt.figure()
ax = fig.gca(projection='3d')

# Make data.
# X = np.arange(-5, 5, 0.25)
# Y = np.arange(-5, 5, 0.25)
# X, Y = np.meshgrid(X, Y)
# R = np.sqrt(X**2 + Y**2)
# Z = np.sin(R)

# Plot the surface.
surf = ax.plot_surface(worldx.reshape(shape), worldy.reshape(shape), worldz.reshape(shape), cmap=cm.hot,
                       linewidth=0, antialiased=False)

# Customize the z axis.
# ax.set_zlim(-1.01, 1.01)
# ax.zaxis.set_major_locator(LinearLocator(10))
# ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

# Add a color bar which maps values to colors.
fig.colorbar(surf, shrink=0.5, aspect=5)

plt.show()
