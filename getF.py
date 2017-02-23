from generate_registrations import *

import numpy as np
from numpy.linalg import lstsq


vel = load_velocity_fields('frame9900','frame9903')
shapex = vel.shape[2]
shapey = vel.shape[1]

u = [320, 960, 1600, 640, 1280, 320, 960, 1600]
v = [270, 270, 270, 540, 540, 810, 810, 810]

X, Y = np.meshgrid(np.arange(shapex), np.arange(shapey))

print(u,v)

print(vel.shape)
u2 = vel[0, v,u]*shapex
v2 = -vel[1, v,u]*shapey  # minus?

# a = np.array([u2*u, u2*v, u2, v2*u, v2*v, v2, u, v])#.transpose()
# print(a.shape)
# print(a)
#
# b = -np.ones((8,1))
# # b = np.array([1,2,3,4,5,6,7,8,9])
# Fl,residu,rank,sing = lstsq(a, b)
# print(Fl)
# F = np.reshape(np.append(Fl,1), (3,3))
# # np.reshape()
# # F = np.array([Fl[0:3],Fl[3:6],Fl[6:9]])
# print(F)
# print(np.linalg.det(F))

u  = np.array([230, 878, 1663, 543, 1333, 374, 960, 1477])
v  = np.array([319, 133,  273, 427,  422, 906, 742,  847])
u2 = np.array([220, 875, 1670, 535, 1336, 360, 957, 1484])
v2 = np.array([331, 140,  284, 442,  437, 936, 768,  875])

A = np.array([u2*u, u2*v, u2, v2*u, v2*v, v2, u, v, [1]*8]).transpose()
A2 = np.append(A, [[0]*9], axis=0)
print(A2)

def null(A, eps=1e-15):
    u, s, vh = np.linalg.svd(A)
    null_mask = (s <= eps)
    null_space = np.compress(null_mask, vh, axis=0)
    return null_space.T

print(null(A2))
print(A.dot(null(A2)))

F = np.reshape(null(A2), (3,3))
# print(F)
print(np.linalg.det(F))

K = np.array([[0.00012, 0, 1920//2], [0, 0.00012, 1080//2], [0, 0, 1]])

E = K.transpose().dot(F).dot(K)

U, S, V = np.linalg.svd(E)
Tx = U.dot(np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 0]])).dot(U.transpose())
R = U.dot(np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])).dot(V)

T = np.array([Tx[2,1], Tx[0,2], Tx[1,0]]).reshape((-1,1))
# print("E", E)
# print('Tx', Tx)
print('T', T)
print('R', R)

P = K.dot(np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]]))
P2 = K.dot(np.concatenate([R, T], axis=1))
print(P, P2)



u = X
v = Y
u2 = u + vel[0]*shapex
v2 = Y - vel[1]*shapey

print(u2)
w1 = np.concatenate([u.reshape((1,-1)), v.reshape((1,-1)), np.ones_like(u.reshape((1,-1)))])
w2 = np.concatenate([u2.reshape((1,-1)), v2.reshape((1,-1)), np.ones_like(u2.reshape((1,-1)))])

x, resid, rank, s = np.linalg.lstsq(np.concatenate([P, P2]), np.concatenate([w1, w2]))
# world.append(x/x[-1])
world = x / x[-1]

print("world", world)

worldx = world[0].reshape((shapey, shapex))
worldy = world[1].reshape((shapey, shapex))
worldz = world[2].reshape((shapey, shapex))

# coords = x, y = np.meshgrid(range(shapey), range(shapex))
# print(coords)
# world = []
#
# for i, _ in enumerate(u):
#     w1 = np.array([u[i], v[i], 1]).reshape((-1,1))
#     w2 = np.array([u2[i], v2[i], 1]).reshape((-1,1))
#
#     x, resid, rank, s = np.linalg.lstsq(np.concatenate([P, P2]), np.concatenate([w1, w2]))
#     # world.append(x/x[-1])
#     world.append(x/x[-1])
#
# print('world', np.array(world))

import visvis

f = visvis.gca()
# f.daspect = 1, -1, 30  # z x 30
# m = visvis.surf(X[50:-50, 50:-50], Y[50:-50, 50:-50], Z[50:-50, 50:-50], cv2.cvtColor(img, cv2.COLOR_BGR2RGB)[50:-50, 50:-50, :])
m2 = visvis.surf(worldx, , 50:-50], Z[50:-50, 50:-50])
m2.colormap = visvis.CM_HOT
plt.waitforbuttonpress()