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


u = X
v = Y
u2 = u + vel[0]*shapex
v2 = Y - vel[1]*shapey

# print(u2)
w1 = np.concatenate([u.reshape((1,-1)), v.reshape((1,-1))])
w2 = np.concatenate([u2.reshape((1,-1)), v2.reshape((1,-1))])

K = np.array([[0.00012, 0, 1920//2], [0, 0.00012, 1080//2], [0, 0, 1]])


# retval, H1, H2 = cv2.stereoRectifyUncalibrated(w1, w2, F, (shapey, shapex))
# print(H1)
K = np.array([[  2.56183420e+03,   0.00000000e+00,   1.07756895e+03],
 [  0.00000000e+00,   2.55767107e+03 ,  7.55023347e+02],
 [  0.00000000e+00  , 0.00000000e+00 ,  1.00000000e+00]])
dist = np.array([[  7.93167424e-02,  -1.32774998e+00 , -2.29753304e-03 ,  1.57676713e-03,
    2.62586571e+01]] )
R =  np.array([[-0.12271646],
       [ 0.01283177],
       [ 3.13452517]])
T =  np.array([[  3.06213849],
       [  2.35298446],
       [ 18.69406276]])
ret = 0.19691314823331685
R1, R2, P1, P2, Q, validPixROI1, validPixROI2 = cv2.stereoRectify(K, dist, K, dist, (shapey, shapex), R, T)
world = cv2.reprojectImageTo3D((vel[0]*16*shapex).astype(int), Q)
print(world.shape)
# world.append(x/x[-1])
# world = x / x[-1]

print("world", world)

worldx = world[:,:,0]
worldy = world[:,:,1]
worldz = world[:,:,2]


import visvis as vv

app = vv.use()
# f = visvis.gca()
# # f.daspect = 1, -1, 30  # z x 30
# # m = visvis.surf(X[50:-50, 50:-50], Y[50:-50, 50:-50], Z[50:-50, 50:-50], cv2.cvtColor(img, cv2.COLOR_BGR2RGB)[50:-50, 50:-50, :])
detail = 50
m2 = vv.surf(worldx[::detail], worldy[::detail], worldz[::detail])
app.Run()
# m2.colormap = visvis.CM_HOT
# plt.waitforbuttonpress()