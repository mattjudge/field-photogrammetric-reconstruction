from generate_registrations import *

import numpy as np

def genWorld(vel):
    shapex = vel.shape[2]
    shapey = vel.shape[1]

    detl = 1
    X, Y = np.meshgrid(np.arange(shapex), np.arange(shapey))
    vel = vel[:,::detl,::detl]
    X = X[::detl,::detl]
    Y = Y[::detl,::detl]

    # u  = np.array([230, 878, 1663, 543, 1333, 374, 960, 1477]).reshape((1,-1))
    # v  = np.array([319, 133,  273, 427,  422, 906, 742,  847]).reshape((1,-1))
    # u2 = np.array([220, 875, 1670, 535, 1336, 360, 957, 1484]).reshape((1,-1))
    # v2 = np.array([331, 140,  284, 442,  437, 936, 768,  875]).reshape((1,-1))


    # u2 = X - vel[0]*shapex
    # v2 = Y - vel[1]*shapey
    # u, v = X, Y

    # velocity vectors map pixels in f2 to their locations in f1
    u = X + vel[0]*shapex
    v = Y + vel[1]*shapey
    u2, v2 = X, Y

    w1 = np.concatenate([u.reshape((1,-1)), v.reshape((1,-1))])
    w2 = np.concatenate([u2.reshape((1,-1)), v2.reshape((1,-1))])

    F, mask = cv2.findFundamentalMat(w1.transpose(), w2.transpose())
    print('F', F)
    print(np.linalg.det(F))


    fku = 1883  # estimated
    # fku = 8000

    K = np.array([[fku,     0, shapex//2],
                  [0,     fku, shapey//2],
                  [0,       0,         1]])

    # E, mask = cv2.findEssentialMat(w1.transpose(), w2.transpose(), 2000, (shapey//2, shapex//2))
    E, mask = cv2.findEssentialMat(w1.transpose(), w2.transpose(), K)
    print("E ", E)
    # E = K.T.dot(F.dot(K))
    R2, R1, t = cv2.decomposeEssentialMat(E)
    print("R1", R1)
    print("R2", R2)
    print("t", t)
    P = K.dot(np.concatenate([R1, np.zeros((3, 1))], axis=1))
    P2 = K.dot(np.concatenate([R2, t], axis=1))

    print("w1", w1)
    print("w2", w2.astype(int))

    # def null(A, eps=1):
    #     u, s, vh = np.linalg.svd(A)
    #     return vh[-1:, :].T
    #     # null_mask = (s <= eps)
    #     # null_space = np.compress(null_mask, vh, axis=0)
    #     # return null_space.T

    # world = np.ones((4, w1.shape[1]))
    print("triangulating vertices")
    finalshape = (shapey, shapex)
    # detl = 1
    #
    # finalshape = w1[0,:].reshape((shapey, shapex))[::detl,::detl].shape
    # print("triangulated shape", finalshape)
    #
    # w1 = np.vstack([
    #                    w1[0,:].reshape((shapey, shapex))[::detl,::detl].flatten(),
    #                    w1[1,:].reshape((shapey, shapex))[::detl,::detl].flatten()
    #                ])
    # w2 = np.vstack([
    #                    w2[0,:].reshape((shapey, shapex))[::detl,::detl].flatten(),
    #                    w2[1,:].reshape((shapey, shapex))[::detl,::detl].flatten()
    #                ])

    # manually triangulate
    # world = np.zeros((4, finalshape[0]*finalshape[1]))
    # for i in range(0, w1.shape[1], 1):
    #     # A1 = w1[:,i:i+1] * np.vstack([P[2,:], P[2,:]]) - P[0:2,:]
    #     # A2 = w2[:,i:i+1] * np.vstack([P2[2,:], P2[2,:]]) - P2[0:2,:]
    #     # A = np.vstack([A1, A2])
    #     # # n = null(A)
    #     # # A = w1w2stack[:, i:i+1] * PP2stack
    #     # u, s, vh = np.linalg.svd(A)
    #     # # print(n)
    #     # n = vh[-1:, :].T
    #     n = cv2.triangulatePoints(P.astype(float), P2.astype(float), w1[:,i:i+1].astype(float), w2[:,i:i+1].astype(float))
    #     world[:,i:i+1] = n
    #R1, R2, P1, P2, Q, validPixROI1, validPixROI2 = cv2.stereoRectify(K, np.zeros((1, 4)), K, np.zeros((1, 4)), (shapey, shapex), R1, t)
    #""" stereoRectify(cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2, imageSize, R, T[, R1[, R2[, P1[, P2[, Q[, flags[, alpha[, newImageSize]]]]]]]]) -> R1, R2, P1, P2, Q, validPixROI1, validPixROI2 """

    world = cv2.triangulatePoints(P2.astype(float), P.astype(float), w2.astype(float), w1.astype(float))
    # world = cv2.triangulatePoints(P, P2, w1, w2)
    print("triangulated.")

    world /= world[-1,:]
    # print(world)

    check1 =P.dot(world[:,3])
    check1 /= check1[-1]
    print("check", check1, w1[:,3])

    check2 =P2.dot(world[:,3])
    check2 /= check2[-1]
    print("check", check2, w2[:,3])

    print("world shape", world.shape)

    return world[:-1,:], finalshape
