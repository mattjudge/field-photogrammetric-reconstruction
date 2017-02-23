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

    # incorrect mapping:
    # u2 = X - vel[0]*shapex
    # v2 = Y - vel[1]*shapey
    # u, v = X, Y

    # velocity vectors map pixels in f2 to their locations in f1
    u = X + vel[0]*shapex
    v = Y + vel[1]*shapey
    u2, v2 = X, Y

    w1 = np.concatenate([u.reshape((1,-1)), v.reshape((1,-1))])
    w2 = np.concatenate([u2.reshape((1,-1)), v2.reshape((1,-1))])

    objectPoints = np.array([
        [-3, 2, 0],
        [0, 2, 0],
        [3, 2, 0],
        [-2, 1, 0],
        [0, 1, 0],
        [2, 1, 0],
        [-1, 0, 0],
        [0, 0, 0],
        [1, 0, 0]
    ])
    imagePoints1 = np.array([
        [u[50, 50], v[50, 50]],
        [u[50, shapex//2], v[50, shapex//2]],
        [u[50, -50], v[50, -50]],
        [u[shapey//2, 50], v[shapey//50, 50]],
        [u[shapey//2, shapex//2], v[shapey//50, shapex//2]],
        [u[shapey//2, -50], v[shapey//50, -50]],
        [u[-50, 50], v[-50, 50]],
        [u[-50, shapex//2], v[-50, shapex//2]],
        [u[-50, -50], v[-50, -50]],
    ])
    imagePoints2 = np.array([
        [u2[50, 50], v2[50, 50]],
        [u2[50, shapex//2], v2[50, shapex//2]],
        [u2[50, -50], v2[50, -50]],
        [u2[shapey//2, 50], v2[shapey//50, 50]],
        [u2[shapey//2, shapex//2], v2[shapey//50, shapex//2]],
        [u2[shapey//2, -50], v2[shapey//50, -50]],
        [u2[-50, 50], v2[-50, 50]],
        [u2[-50, shapex//2], v2[-50, shapex//2]],
        [u2[-50, -50], v2[-50, -50]],
    ])

    cameraMatrix1 = None
    distCoeffs1 = None
    cameraMatrix2 = None
    distCoeffs2 = None
    imageSize = (shapey, shapex)


    retval, cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2, R, T, E, F = cv2.stereoCalibrate(
        [objectPoints.astype('float32')], [imagePoints1.astype('float32')], [imagePoints2.astype('float32')], cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2, imageSize
    )

    print(cameraMatrix1)
    print(cameraMatrix2)
    print("R", R)
    print("T", T)
    print("E", E)
    print("F", F)

    R1, R2, P1, P2, Q, validPixROI1, validPixROI2 = cv2.stereoRectify(cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2, imageSize, R, T)
    print("R1", R1)
    print("R2", R2)
    print("P1", P1)
    print("P2", P2)
    print("Q", Q)
    print("validPixROI1", validPixROI1)
    print("validPixROI2", validPixROI2)

    map1, map2 = cv2.initUndistortRectifyMap(cameraMatrix1, distCoeffs1, R1, P1, imageSize, cv2.CV_32FC1)
    ub = cv2.remap(u.astype('float32'), map1, map2, cv2.INTER_LINEAR)
    vb = cv2.remap(v.astype('float32'), map1, map2, cv2.INTER_LINEAR)

    map1, map2 = cv2.initUndistortRectifyMap(cameraMatrix2, distCoeffs2, R2, P2, imageSize, cv2.CV_32FC1)
    u2b = cv2.remap(u2.astype('float32'), map1, map2, cv2.INTER_LINEAR)
    v2b = cv2.remap(v2.astype('float32'), map1, map2, cv2.INTER_LINEAR)


    u, v, u2, v2 = ub, vb, u2b, v2b
    w1 = np.concatenate([u.reshape((1,-1)), v.reshape((1,-1))])
    w2 = np.concatenate([u2.reshape((1,-1)), v2.reshape((1,-1))])

    world = cv2.triangulatePoints(P1, P2, w1, w2)



    world /= world[-1,:]
    print(world)

    check1 = P1.dot(world[:,3])
    check1 /= check1[-1]
    print("check", check1, w1[:,3])

    check2 =P2.dot(world[:,3])
    check2 /= check2[-1]
    print("check", check2, w2[:,3])

    print("world shape", world.shape)

    return world[:-1,:], imageSize
