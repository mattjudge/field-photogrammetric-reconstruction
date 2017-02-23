from generate_registrations import *

import numpy as np

def getFundamental(u1, v1, u2, v2):
    u1 = u1.reshape((-1, 1))
    v1 = v1.reshape((-1, 1))
    u2 = u2.reshape((-1, 1))
    v2 = v2.reshape((-1, 1))
    LHS = np.hstack([u2*u1, u2*v1, u2, v2*u1, v2*v1, v2, u1, v1, np.ones_like(v1)])
    U, s, VT = np.linalg.svd(LHS)
    # print(np.min(s), s.flatten()[-1])  # verify last column of V is nullspace
    # NS = VT[:, -1:].transpose()
    NS = VT[-1:, :].transpose()
    print(NS.shape)
    return NS.reshape((3, 3)) / NS[-1]

def getRandT(E):
    U, s, VT = np.linalg.svd(E)
    Tx = U.dot(
        np.array([
            [0, 1, 0],
            [-1, 0, 0],
            [0, 0, 0]
        ])
    ).dot(U.transpose())

    R = U.dot(
        np.array([
            [0, -1, 0],
            [1, 0, 0],
            [0, 0, 1]
        ])
    ).dot(VT)

    t = np.array([
        [Tx[2][1]],
        [Tx[0][2]],
        [Tx[1][0]],
    ])

    return R, t

def getProjections(K, R, t):
    P1 = K.dot(
        np.hstack([np.eye(3), np.zeros((3, 1))])
    )
    P2 = K.dot(
        np.hstack([R, t])
    )
    return P1, P2

def genWorld(vel):
    shapex = vel.shape[2]
    shapey = vel.shape[1]
    imagesize = vel.shape

    X, Y = np.meshgrid(np.arange(shapex), np.arange(shapey))

    # velocity vectors map pixels in f2 to their locations in f1
    u1shaped = X + vel[0]*shapex
    v1shaped = Y + vel[1]*shapey
    u2shaped, v2shaped = X, Y

    u1 = u1shaped.flatten()
    v1 = v1shaped.flatten()
    u2 = u2shaped.flatten()
    v2 = v2shaped.flatten()

    d = 100
    Fu1, Fv1, Fu2, Fv2 = u1[::d], v1[::d], u2[::d], v2[::d]

    # Fu1  = np.array([230, 878, 1663, 543, 1333, 374, 960, 1477]).reshape((1,-1))
    # Fv1  = np.array([319, 133,  273, 427,  422, 906, 742,  847]).reshape((1,-1))
    # Fu2 = np.array([220, 875, 1670, 535, 1336, 360, 957, 1484]).reshape((1,-1))
    # Fv2 = np.array([331, 140,  284, 442,  437, 936, 768,  875]).reshape((1,-1))

    # print("computing F with", len(Fu1), "data points")
    # F = getFundamental(Fu1, Fv1, Fu2, Fv2)
    # print("F", F)
    # print(np.linalg.det(F))
    #
    w1 = np.concatenate([u1.reshape((1,-1)), v1.reshape((1,-1))])
    w2 = np.concatenate([u2.reshape((1,-1)), v2.reshape((1,-1))])
    # F2, mask = cv2.findFundamentalMat(w1.transpose(), w2.transpose())
    # print('F2', F2)
    # print(np.linalg.det(F2))


    fku = 1883  # estimated
    # fku = 8000

    K = np.array([[fku,     0, shapex//2],
                  [0,     fku, shapey//2],
                  [0,       0,         1]])

    # E = K.transpose().dot(F).dot(K)
    # print("E", E)

    # E, mask = cv2.findEssentialMat(w1.transpose(), w2.transpose(), 2000, (shapey//2, shapex//2))
    E2, mask = cv2.findEssentialMat(w1.transpose(), w2.transpose(), K)
    print("E2", E2)

    R, t = getRandT(E2)
    P1, P2 = getProjections(K, R, t)
    print("P1", P1)
    print("P2", P2)

    # E = K.T.dot(F.dot(K))
    R2b, R1b, tb = cv2.decomposeEssentialMat(E2)
    P1b = K.dot(np.concatenate([R1b, np.zeros((3, 1))], axis=1))
    P2b = K.dot(np.concatenate([R2b, t], axis=1))
    print("P1b", P1b)
    print("P2b", P2b)

    print("w1 as int", w1.astype(int))
    print("w2 as int", w2.astype(int))

    # def null(A, eps=1):
    #     u, s, vh = np.linalg.svd(A)
    #     return vh[-1:, :].T
    #     # null_mask = (s <= eps)
    #     # null_space = np.compress(null_mask, vh, axis=0)
    #     # return null_space.T

    # world = np.ones((4, w1.shape[1]))
    print("triangulating vertices")
    finalshape = (shapey, shapex)
    # detl = 5
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
    #
    # # manually triangulate
    # world = np.zeros((4, finalshape[0]*finalshape[1]))
    # for i in range(0, w1.shape[1], 1):
    #     A1 = w1[:,i:i+1] * np.vstack([P1[2,:], P1[2,:]]) - P1[0:2,:]
    #     A2 = w2[:,i:i+1] * np.vstack([P2[2,:], P2[2,:]]) - P2[0:2,:]
    #     A = np.vstack([A1, A2])
    #     # n = null(A)
    #     # A = w1w2stack[:, i:i+1] * PP2stack
    #     u, s, vh = np.linalg.svd(A)
    #     # print(n)
    #     n = vh[-1:, :].T
    #     world[:,i:i+1] = n

    world = cv2.triangulatePoints(P2.astype(float), P1.astype(float), w2.astype(float), w1.astype(float))
    # world = cv2.triangulatePoints(P, P2, w1, w2)  # crashes, bug in opencv
    print("triangulated.")

    world /= world[-1,:]
    # print(world)

    check1 =P1.dot(world[:,3])
    check1 /= check1[-1]
    print("check", check1, w1[:,3])

    check2 =P2.dot(world[:,3])
    check2 /= check2[-1]
    print("check", check2, w2[:,3])

    print("world shape", world.shape)

    return world[:-1,:], finalshape
