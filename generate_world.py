
import numpy as np
import cv2

import generate_registrations
import pointcloud


def get_fundamental(u1, v1, u2, v2):
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


def get_rt(E):
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


def get_projections_from_rt(K, R, t):
    P1 = K.dot(
        np.hstack([np.eye(3), np.zeros((3, 1))])
    )
    P2 = K.dot(
        np.hstack([R, t])
    )
    return P1, P2


def estimate_world_projections(vel, P1=None, P2=None):
    # vel should be pre-cropped to ensure good data points (otherwise E becomes unstable)
    imshape = vel[0].shape
    shapey, shapex = imshape
    vel[1] = vel[1].clip(-1, 0)

    X, Y = np.meshgrid(np.arange(shapex), np.arange(shapey))

    # velocity vectors map pixels in f2 to their locations in f1
    u1shaped = X + vel[0]*shapex
    v1shaped = Y + vel[1]*shapey
    u2shaped, v2shaped = X, Y

    w1 = np.vstack([u1shaped.flatten(), v1shaped.flatten()])
    w2 = np.vstack([u2shaped.flatten(), v2shaped.flatten()])
    print("w1 as int", w1.astype(int))
    print("w2 as int", w2.astype(int))


    fku = 1883  # estimated
    # fku = 1700

    K = np.array([[fku,     0, shapex//2],
                  [0,     fku, shapey//2],
                  [0,       0,             1]])

    # E, mask = cv2.findEssentialMat(w1.transpose(), w2.transpose(), 2000, (shapey//2, shapex//2))
    E, mask = cv2.findEssentialMat(w1.transpose(), w2.transpose(), K)
    print("E", E)
    # print("mask", mask.T)
    # print(np.sum(mask), ":", mask.size)
    # print(mask.size - np.sum(mask))
    # print((mask.size - np.sum(mask)) / mask.size)

    print("recovering pose")
    if P1 is None or P2 is None:
        # retval, R, t, mask = cv2.recoverPose(E, w1.transpose(), w2.transpose(), K, mask=mask)
        retval, R, t, mask = cv2.recoverPose(E, w1.transpose(), w2.transpose(), K, mask=None)
        P1, P2 = get_projections_from_rt(K, R, t)
    else:
        # closest to last one
        R1, R2, t = cv2.decomposeEssentialMat(E)
        Rtcombs = (
            (R1, t),
            (R1, -t),
            (R2, t),
            (R2, -t),
        )
        projs = list(map(lambda rt: get_projections_from_rt(K, *rt), Rtcombs))
        diffs = list(map(
            lambda x: np.linalg.norm(P1-x[0]) + np.linalg.norm(P2-x[1]), projs
        ))
        print(diffs)
        P1, P2 = projs[np.argmin(diffs)]
        R, t = Rtcombs[np.argmin(diffs)]

    # print("mask", mask.T)
    # print(np.sum(mask), ":", mask.size)
    # print(mask.size - np.sum(mask))
    # print((mask.size - np.sum(mask)) / mask.size)

    print("triangulating vertices")
    points = cv2.triangulatePoints(P1.astype(float), P2.astype(float), w1.astype(float), w2.astype(float))
    # world = cv2.triangulatePoints(P, P2, w1, w2)  # crashes, bug in opencv
    print("triangulated.")
    # world[:, mask.transpose()] = float('nan')

    points /= points[-1,:]

    check1 = P1.dot(points[:, points.shape[1]//4])
    check1 /= check1[-1]
    print("check", check1, w1[:, points.shape[1]//4])

    check2 = P2.dot(points[:, points.shape[1]//4])
    check2 /= check2[-1]
    print("check", check2, w2[:, points.shape[1]//4])

    print("world shape", points.shape)

    # return points[:-1,:], imshape, P1, P2
    return pointcloud.PointCloud(points[:-1,:], imshape, P1, P2, R, t)


def generate_world(fname1, fname2, P1=None, P2=None, visual=True):
    vel = generate_registrations.load_velocity_fields(fname1, fname2)
    # crop to ensure good E fit
    crop = 50
    vel = vel[:,crop:-crop, crop:-crop]

    cloud = estimate_world_projections(vel, P1, P2)

    # cloud.points = pointcloud.align_points_with_xy(cloud.points)

    if visual:
        # visualise_world_visvis(X, Y, Z)
        pointcloud.visualise_worlds_mplotlib(cloud)

    return cloud


def generate_world_average(fnames):
    fnamepairs = [(fnames[i], fnames[i+1]) for i in range(len(fnames)-1)]
    vels = np.array(list(map(lambda fnm: generate_registrations.load_velocity_fields(*fnm), fnamepairs)))
    print(vels.shape)

    # crop vels
    crop = 50
    vels = vels[:, :, crop:-crop, crop:-crop]
    print(vels.shape)

    nregs = vels.shape[0]
    imagesize = vels[0][0].shape
    shapex = imagesize[1]
    shapey = imagesize[0]

    X, Y = np.meshgrid(np.arange(shapex), np.arange(shapey))

    worldavg = np.zeros((3, shapey, shapex))
    cumreg = np.array([X, Y]).astype('float32')
    P1 = P2 = None

    for vel in vels:
        # print(vel.shape)
        world, shape, P1, P2 = estimate_world_projections(vel, P1, P2)

        # clip world
        world[0] = world[0].clip(-60, 60)
        world[1] = world[1].clip(-40, 20)
        world[2] = world[2].clip(30, 150)

        # # velocity vectors map pixels in f2 to their locations in f1
        # u1 = X + vel[0]*shapex
        # v1 = Y + vel[1]*shapey
        # u2, v2 = X, Y

        X = world[0, :].reshape(imagesize)
        Y = world[1, :].reshape(imagesize)
        Z = world[2, :].reshape(imagesize)

        # dst(x, y) = src(mapx(x, y), mapy(x, y))
        worldavg[0] += cv2.remap(X, cumreg[0].astype('float32'), cumreg[1].astype('float32'), cv2.INTER_LINEAR)
        worldavg[1] += cv2.remap(Y, cumreg[0].astype('float32'), cumreg[1].astype('float32'), cv2.INTER_LINEAR)
        worldavg[2] += cv2.remap(Z, cumreg[0].astype('float32'), cumreg[1].astype('float32'), cv2.INTER_LINEAR)

        cumreg += vel

    worldavg /= nregs

    flattenedworld = np.vstack([worldavg[0].flatten(), worldavg[1].flatten(), worldavg[2].flatten()])
    worldavg = pointcloud.align_points_with_xy(flattenedworld)

    crop = 100
    X = worldavg[0, :].reshape(imagesize)[crop:-crop, crop:-crop]
    Y = worldavg[1, :].reshape(imagesize)[crop:-crop, crop:-crop]
    Z = worldavg[2, :].reshape(imagesize)[crop:-crop, crop:-crop]

    pointcloud.visualise_world_mplotlib(X, Y, Z)

    return worldavg


if __name__ == "__main__":
    cloud1 = generate_world('frame9900', 'frame9903', visual=False)
    cloud2 = generate_world('frame9903', 'frame9906', cloud1.P1, cloud1.P2, visual=False)
    cloud3 = generate_world('frame9906', 'frame9909', cloud2.P1, cloud2.P2, visual=False)
    cloud4 = generate_world('frame9909', 'frame9912', cloud3.P1, cloud3.P2, visual=False)

    cloud2.points = cloud1.R.T.dot(cloud2.points - cloud1.t)

    cloud3.points = cloud2.R.T.dot(cloud3.points - cloud2.t)
    cloud3.points = cloud1.R.T.dot(cloud3.points - cloud1.t)

    cloud4.points = cloud3.R.T.dot(cloud4.points - cloud3.t)
    cloud4.points = cloud2.R.T.dot(cloud4.points - cloud2.t)
    cloud4.points = cloud1.R.T.dot(cloud4.points - cloud1.t)

    pointcloud.visualise_worlds_mplotlib(cloud1, cloud2, cloud3, cloud4)
    # world, P1, P2 = generate_world('frame9912', 'frame9915', P1, P2)
    # world, P1, P2 = generate_world('frame9915', 'frame9918', P1, P2)
    # world, P1, P2 = generate_world('frame9918', 'frame9921', P1, P2)
    # world, P1, P2 = generate_world('frame9921', 'frame9924', P1, P2)  # bad
    # world, P1, P2 = generate_world('frame9924', 'frame9927', P1, P2)
    # world, P1, P2 = generate_world('frame9927', 'frame9930', P1, P2)

    # generate_world_average(('frame9900', 'frame9903', 'frame9906', 'frame9909'))
    # generate_world_average(('frame9900', 'frame9903', 'frame9906', 'frame9909', 'frame9912', 'frame9915'))
    # generate_world_average(('frame9900', 'frame9903', 'frame9906', 'frame9909', 'frame9912', 'frame9915', 'frame9918'))
    # generate_world_average((
    #     'frame9900', 'frame9903', 'frame9906', 'frame9909', 'frame9912', 'frame9915',
    #     'frame9918', 'frame9921', 'frame9924', 'frame9927', 'frame9930'
    # ))
