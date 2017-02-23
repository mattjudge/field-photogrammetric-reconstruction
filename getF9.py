from generate_registrations import *

import numpy as np

# vel = load_velocity_fields('floor0','floor1')
vel = load_velocity_fields('frame9900','frame9903')
shapex = vel.shape[2]
shapey = vel.shape[1]

detl = 4
X, Y = np.meshgrid(np.arange(shapex), np.arange(shapey))
vel = vel[:,::detl,::detl]
X = X[::detl,::detl]
Y = Y[::detl,::detl]

# u  = np.array([230, 878, 1663, 543, 1333, 374, 960, 1477]).reshape((1,-1))
# v  = np.array([319, 133,  273, 427,  422, 906, 742,  847]).reshape((1,-1))
# u2 = np.array([220, 875, 1670, 535, 1336, 360, 957, 1484]).reshape((1,-1))
# v2 = np.array([331, 140,  284, 442,  437, 936, 768,  875]).reshape((1,-1))

u2 = X - vel[0]*shapex
v2 = Y - vel[1]*shapey
u, v = X, Y

w1 = np.concatenate([u.reshape((1,-1)), v.reshape((1,-1))])
w2 = np.concatenate([u2.reshape((1,-1)), v2.reshape((1,-1))])

F, mask = cv2.findFundamentalMat(w1.transpose(), w2.transpose())
print('F', F)
print(np.linalg.det(F))


# fku = 2.56183420e+03
fku = 2200
fkv = fku#*9//16
K = np.array([[fku, 0, shapex//2],
              [0, fkv, shapey//2],
              [0, 0, 1]])

# E, mask = cv2.findEssentialMat(w1.transpose(), w2.transpose(), 2000, (shapey//2, shapex//2))
E, mask = cv2.findEssentialMat(w1.transpose(), w2.transpose(), K)
print("E ", E)
R1, R2, t = cv2.decomposeEssentialMat(E)
print("R1", R1)
print("R2", R2)
print("t", t)
P = K.dot(np.concatenate([R1, np.zeros((3, 1))], axis=1))
P2 = K.dot(np.concatenate([R2, t], axis=1))

print("w1", w1)

def null(A, eps=1):
    u, s, vh = np.linalg.svd(A)
    # return vh[-1:, :].T
    # print("s", s, np.argmin(s))
    null_mask = (s <= eps)
    # print(null_mask)
    null_space = np.compress(null_mask, vh, axis=0)
    return null_space.T
    # print("n ", n)
    # print("n2", vh[-1:,:].T)

world = np.zeros((4, w1.shape[1]))
Pstack = np.vstack([P[2,:], P[2,:]]) - P[0:2,:]
P2stack = np.vstack([P2[2,:], P2[2,:]]) - P2[0:2,:]
PP2stack = np.vstack([Pstack, P2stack])
w1w2stack = np.vstack([w1, w2])
for i in range(w1.shape[1]):
    A1 = w1[:,i:i+1] * np.vstack([P[2,:], P[2,:]]) - P[0:2,:]
    A2 = w2[:,i:i+1] * np.vstack([P2[2,:], P2[2,:]]) - P2[0:2,:]
    A = np.vstack([A1, A2])
    # n = null(A)
    # A = w1w2stack[:, i:i+1] * PP2stack
    u, s, vh = np.linalg.svd(A)
    # print(n)
    n = vh[-1:, :].T
    world[:,i:i+1] = n

world /= world[-1,:]
print(world)

check1 =P.dot(world[:,3])
check1 /= check1[-1]
print(check1, w1[:,3])

check2 =P2.dot(world[:,3])
check2 /= check2[-1]
print(check2, w2[:,3])

print("world shape", world.shape)

thx = np.deg2rad(30)
Rx = np.array([[1, 0, 0],
               [0, np.cos(thx), np.sin(thx)],
               [0, -np.sin(thx), np.cos(thx)]])
# world = Rx.dot(world[0:3,:])

thy = np.deg2rad(0)
Ry = np.array([[np.cos(thx), 0, -np.sin(thx)],
               [0, 1, 0],
               [np.sin(thx), 0, np.cos(thx)]])
# world = Ry.dot(Rx).dot(world[0:3,:])
# world[2,:] *= 30
# world[2,:] -= min(world[2,:])
# world = np.linalg.inv(R2).dot(world[0:3,:])

worldx = world[0,:]
worldy = world[1,:]
worldz = world[2,:]

print("x shape", world.shape)
print("worldx shape", worldx.shape)

import pylab
pylab.figure()
pylab.imshow(worldx.reshape(vel[0].shape), interpolation='none', cmap=pylab.cm.hot)  # , vmin=-4, vmax=4)
pylab.colorbar()
pylab.title('x per pixel')
pylab.savefig('__pp_x.png')

pylab.figure()
pylab.imshow(worldy.reshape(vel[0].shape), interpolation='none', cmap=pylab.cm.hot)  # , vmin=-4, vmax=4)
pylab.colorbar()
pylab.title('y per pixel')
pylab.savefig('__pp_y.png')

pylab.figure()
pylab.imshow(worldz.reshape(vel[0].shape), interpolation='none', cmap=pylab.cm.hot)  # , vmin=-4, vmax=4)
pylab.colorbar()
pylab.title('z per pixel')
pylab.savefig('__pp_z.png')


import visvis as vv

app = vv.use()

detail = 1
# m2 = vv.surf(worldx[::detail], worldy[::detail], worldz[::detail])
pp = vv.Pointset(np.concatenate([worldx[::detail], worldy[::detail], worldz[::detail]], axis=0).reshape((-1, 3)))

# prepare axes
a = vv.gca()
a.cameraType = '3d'
a.daspectAuto = False
print("view", a.camera.GetViewParams())
# a.SetView(loc=(-1000,0,0))
# a.camera.SetView(None, loc=(-1000,0,0))

# draw points
# l = vv.plot(pp, ms='.', mc='r', mw='5', ls='', mew=0)
# l.alpha = 0.2
print(worldx.shape)
print(vel.shape)
l = vv.surf(worldx.reshape(vel[0].shape), worldy.reshape(vel[0].shape), worldz.reshape(vel[0].shape))
a.SetLimits(rangeX=(-0.2, 0.2), rangeY=(-0.5, 0.5), rangeZ=(-0.5, 0), margin=0.02)
app.Run()