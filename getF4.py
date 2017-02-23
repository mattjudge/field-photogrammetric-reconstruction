from generate_registrations import *

import numpy as np
from numpy.linalg import lstsq


vel = load_velocity_fields('frame9900','frame9903')
shapex = vel.shape[2]
shapey = vel.shape[1]

X, Y = np.meshgrid(np.arange(shapex), np.arange(shapey))

u  = np.array([230, 878, 1663, 543, 1333, 374, 960, 1477]).reshape((1,-1))
v  = np.array([319, 133,  273, 427,  422, 906, 742,  847]).reshape((1,-1))
u2 = np.array([220, 875, 1670, 535, 1336, 360, 957, 1484]).reshape((1,-1))
v2 = np.array([331, 140,  284, 442,  437, 936, 768,  875]).reshape((1,-1))

w1 = np.concatenate([u, v], axis=0)
w2 = np.concatenate([u2, v2], axis=0)
# print(w1, w2)
# print(np.checkVector(2, -1, False))
F, mask  = cv2.findFundamentalMat(w1.transpose(), w2.transpose())
print('F', F)
print(np.linalg.det(F))


fku = 2500
fkv = fku#*9//16
K = np.array([[fku, 0, shapex//2],
              [0, fkv, shapey//2],
              [0, 0, 1]])

""" findEssentialMat(points1, points2, cameraMatrix[, method[, prob[, threshold[, mask]]]]) -> retval, mask
or  findEssentialMat(points1, points2[, focal[, pp[, method[, prob[, threshold[, mask]]]]]]) -> retval, mask """
# E, mask = cv2.findEssentialMat(w1.transpose(), w2.transpose(), 2000, (shapey//2, shapex//2))
E, mask = cv2.findEssentialMat(w1.transpose(), w2.transpose(), K)

""" decomposeEssentialMat(E[, R1[, R2[, t]]]) -> R1, R2, t """
R1, R2, t = cv2.decomposeEssentialMat(E)
print("R1", R1)
print("R2", R2)
print("t", t)
P = K.dot(np.concatenate([R1, np.zeros((3,1))], axis=1))
P2 = K.dot(np.concatenate([R2, t], axis=1))


# newu = X
# newvv = Y
uu2 = X - vel[0]*shapex
vv2 = Y - vel[1]*shapey
print("u2", u2)
print("coords", (u[0,2], v[0,2]), (u2[0,2], v2[0,2]), (uu2[v[0,2], u[0,2]], vv2[v[0,2], u[0,2]]))
print("coords", (u[0,3], v[0,3]), (u2[0,3], v2[0,3]), (uu2[v[0,3], u[0,3]], vv2[v[0,3], u[0,3]]))
print("coords", (u[0,6], v[0,6]), (u2[0,6], v2[0,6]), (uu2[v[0,6], u[0,6]], vv2[v[0,6], u[0,6]]))
u, v = X, Y
u2, v2 = uu2, vv2


# d = 50000
d = 1
w1 = np.concatenate([u[::d].reshape((1,-1)), v[::d].reshape((1,-1)), np.ones_like(u[::d].reshape((1,-1)))])
w2 = np.concatenate([u2[::d].reshape((1,-1)), v2[::d].reshape((1,-1)), np.ones_like(u2[::d].reshape((1,-1)))])

print("shape", w1.shape)
print("computing triangulation")
""" triangulatePoints(projMatr1, projMatr2, projPoints1, projPoints2[, points4D]) -> points4D """
# x = cv2.triangulatePoints(P, P2, w1, w2)
print("P P2 shape", P.shape, P2.shape)
A = np.concatenate([P, P2], axis=0)
print("A", A)
# b = np.concatenate(cv2.convertPointsToHomogeneous(w1), cv2.convertPointsToHomogeneous(w2))
b = np.concatenate([w1, w2])
print("b", b)
x, residuals, rank, s = np.linalg.lstsq(A, b)
print("computed")

# print(w1)
# x = w1
print("x shape", x.shape)
worldx = (x[0,:] / x[-1,:])#.reshape((shapey, shapex))
worldy = (x[1,:] / x[-1,:])#.reshape((shapey, shapex))
worldz = (x[2,:] / x[-1,:])#.reshape((shapey, shapex))
print("worldx shape", worldx.shape)
# print(x)
# world = cv2.convertPointsFromHomogeneous(x)
print(worldx.shape)
# world.append(x/x[-1])
# world = x / x[-1]
import visvis as vv

app = vv.use()

detail = 1
# m2 = vv.surf(worldx[::detail], worldy[::detail], worldz[::detail])
pp = vv.Pointset(np.concatenate([worldx[::detail], worldy[::detail], worldz[::detail]], axis=0).reshape((-1, 3)))

# prepare axes
a = vv.gca()
a.cameraType = '3d'
a.daspectAuto = False

# draw points
l = vv.plot(pp, ms='.', mc='r', mw='9', ls='', mew=0 )
l.alpha = 0.1

app.Run()