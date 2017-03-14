
import numpy as np
import cv2
from scipy import interpolate
import dtcwt.registration

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


def create_pixel_correspondences(vel):
    velx, vely = vel
    imshape = velx.shape
    shapey, shapex = imshape
    vely = vely.clip(-1, 0)

    X, Y = np.meshgrid(np.arange(shapex), np.arange(shapey))

    # velocity vectors map pixels in f2 to their locations in f1
    u1shaped = X + velx*shapex
    v1shaped = Y + vely*shapey
    u2shaped, v2shaped = X, Y

    return np.dstack((u1shaped, v1shaped)), np.dstack((u2shaped, v2shaped))


def estimate_projections(correspondences, P1=None, P2=None):
    # points should be pre-cropped to ensure good data points (otherwise E becomes unstable)
    corr1, corr2 = correspondences
    imshape = corr1[:, :, 0].shape
    shapey, shapex = imshape

    w1 = np.vstack((corr1[:, :, 0].flat, corr1[:, :, 1].flat))
    w2 = np.vstack((corr2[:, :, 0].flat, corr2[:, :, 1].flat))
    print("w1 as int", w1.astype(int))
    print("w2 as int", w2.astype(int))

    # TODO: ensure shapex, shapey are correct estimations given cropping of vels
    fku = 1883  # estimated
    K = np.array([[fku,     0, shapex//2],
                  [0,     fku, shapey//2],
                  [0,       0,         1]])

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
        # TODO: refine sampling method
        retval, R, t, mask = cv2.recoverPose(E, w1[:,::100].transpose(), w2[:,::100].transpose(), K, mask=None)
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

    print("R", R)
    print("t", t)
    # print("P1", P1)
    # print("P2", P2)

    # print("mask", mask.T)
    # print(np.sum(mask), ":", mask.size)
    # print(mask.size - np.sum(mask))
    # print((mask.size - np.sum(mask)) / mask.size)
    return P1, P2, R, t


def generate_cloud(correspondences, P1, P2, R, t):
    corr1, corr2 = correspondences
    imshape = corr1[:, :, 0].shape
    shapey, shapex = imshape

    w1 = np.vstack((corr1[:, :, 0].flat, corr1[:, :, 1].flat))
    w2 = np.vstack((corr2[:, :, 0].flat, corr2[:, :, 1].flat))
    print("w1 as int", w1.astype(int))
    print("w2 as int", w2.astype(int))

    print("triangulating vertices")
    points = cv2.triangulatePoints(P1.astype(float), P2.astype(float), w1.astype(float), w2.astype(float))
    # world = cv2.triangulatePoints(P, P2, w1, w2)  # crashes, bug in opencv
    print("triangulated.")
    # world[:, mask.transpose()] = float('nan')

    points /= points[-1, :]

    check1 = P1.dot(points[:, points.shape[1] // 4])
    check1 /= check1[-1]
    print("check", check1, w1[:, points.shape[1] // 4])

    check2 = P2.dot(points[:, points.shape[1] // 4])
    check2 /= check2[-1]
    print("check", check2, w2[:, points.shape[1] // 4])

    print("world shape", points.shape)

    return pointcloud.PointCloud(points[:-1, :], imshape, P1, P2, R, t)


def generate_world(fnum1, fnum2, P1=None, P2=None, visual=True):
    vel = generate_registrations.load_velocity_fields(fnum1, fnum2)
    # crop to ensure good E fit
    crop = 50
    vel = vel[:, crop:-crop, crop:-crop]

    cloud = estimate_projections(vel, P1, P2)

    cloud.points = pointcloud.align_points_with_xy(cloud.points)

    if visual:
        # visualise_world_visvis(X, Y, Z)
        pointcloud.visualise_worlds_mplotlib(cloud)

    return cloud


def generate_world_average_old(fnums):
    fnumpairs = [(fnums[i], fnums[i+1]) for i in range(len(fnums)-1)]
    vels = np.array(list(map(lambda fnm: generate_registrations.load_velocity_fields(*fnm), fnumpairs)))
    print(vels.shape)
    # print("vels", vels)

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

    cloud = None

    for vel in vels:
        # print("x vel", vel[0]*shapex)
        # print("y vel", vel[1]*shapey)
        # print("min y vel", np.min(vel[1]*shapey))
        # print(vel.shape)
        if cloud is None:
            cloud = estimate_projections(vel, None, None)
        else:
            cloud = estimate_projections(vel, cloud.P1, cloud.P2)
        world = cloud.points

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

        vel[0,:,:] *= shapex
        vel[1,:,:] *= shapey
        cumreg += vel

    worldavg /= nregs

    flattenedworld = np.vstack([worldavg[0].flatten(), worldavg[1].flatten(), worldavg[2].flatten()])
    worldavg = pointcloud.align_points_with_xy(flattenedworld)

    crop = 100
    X = worldavg[0, :].reshape(imagesize)[crop:-crop, crop:-crop]
    Y = worldavg[1, :].reshape(imagesize)[crop:-crop, crop:-crop]
    Z = worldavg[2, :].reshape(imagesize)[crop:-crop, crop:-crop]

    avgcloud = pointcloud.PointCloud(np.vstack([X.flatten(), Y.flatten(), Z.flatten()]), X.shape, None, None, None, None)
    # avgcloud.points = pointcloud.align_points_with_xy(avgcloud.points)
    pointcloud.visualise_worlds_mplotlib(avgcloud)


def gen_running_avg(fnums, initcloud=None):
    # generates a "moving average"
    # beware that initial frames become increasingly "blurred" over a greater number of input frames
    # should only be used for small numbers of frames

    fnumpairs = [(fnums[i], fnums[i+1]) for i in range(len(fnums)-1)]
    vels = np.array(list(map(lambda fnm: generate_registrations.load_velocity_fields(*fnm), fnumpairs)))
    print(vels.shape)

    # crop vels
    crop = 50
    vels = vels[:, :, crop:-crop, crop:-crop]
    print(vels.shape)

    nregs = vels.shape[0]
    imgshape = vels[0][0].shape
    shapey, shapex = imgshape

    X, Y = np.meshgrid(np.arange(shapex, dtype=np.float32),
                       np.arange(shapey, dtype=np.float32))

    # worldavg = np.zeros((shapey*3, shapex, 3))
    # avgX, avgY = np.meshgrid(np.arange(worldavg.shape[1], dtype=np.float32),
    #                          np.arange(worldavg.shape[0], dtype=np.float32))
    # worldcnt = np.zeros((shapey*3, shapex))
    worldavg = np.zeros((shapey, shapex, 3))

    cloud = initcloud

    for vel in vels:
        # print(vel.shape)
        if cloud is None:
            cloud = estimate_projections(vel, None, None)
        else:
            cloud = estimate_projections(vel, cloud.P1, cloud.P2)
        world = cloud.points

        # clip world
        # world[0] = world[0].clip(-60, 60)
        # world[1] = world[1].clip(-40, 20)
        # world[2] = world[2].clip(30, 150)

        # # velocity vectors map pixels in f2 to their locations in f1
        # u1 = X + vel[0]*shapex
        # v1 = Y + vel[1]*shapey
        # u2, v2 = X, Y

        wX = world[0, :].reshape(imgshape)
        wY = world[1, :].reshape(imgshape)
        wZ = world[2, :].reshape(imgshape)

        # worldavg = dtcwt.registration.normsample(worldavg, X/shapex + vel[0], Y/shapey + vel[1]) + np.dstack((wX, wY, wZ))

        mapX = (X + vel[0]*shapex).astype('float32')
        mapY = (Y + vel[1]*shapey).astype('float32')
        worldavg = cv2.remap(worldavg, mapX, mapY,
                             interpolation=cv2.INTER_LINEAR,  # INTER_LANCZOS4,
                             borderMode=cv2.BORDER_TRANSPARENT) + np.dstack((wX, wY, wZ))
        # worldavg[:, :, 0] = cv2.remap(worldavg[:, :, 0], mapX, mapY, cv2.INTER_LANCZOS4) + wX
        # worldavg[:, :, 1] = cv2.remap(worldavg[:, :, 1], mapX, mapY, cv2.INTER_LANCZOS4) + wY
        # worldavg[:, :, 2] = cv2.remap(worldavg[:, :, 2], mapX, mapY, cv2.INTER_LANCZOS4) + wZ

    worldavg /= nregs
    # print("max count", np.max(worldcnt))

    # display world average contribution counts
    # cntX, cntY = np.meshgrid(np.arange(worldcnt.shape[1]), np.arange(worldcnt.shape[0]))
    # cntcloud = pointcloud.PointCloud(np.vstack([cntX.flatten(), cntY.flatten(), worldcnt.flatten()]), worldcnt.shape,
    #                                  None, None, None, None)
    # pointcloud.visualise_worlds_mplotlib(cntcloud)

    # display world average
    # mask = worldcnt > 2
    # worldavg = worldavg[:, mask] / worldcnt[mask]  # just points enabled in mask
    flattenedworld = np.vstack([worldavg[:,:,0].flatten(), worldavg[:,:,1].flatten(), worldavg[:,:,2].flatten()])
    worldavg = pointcloud.align_points_with_xy(flattenedworld)


    crop = 100
    X = worldavg[0, :].reshape(imgshape)[crop:-crop, crop:-crop]
    Y = worldavg[1, :].reshape(imgshape)[crop:-crop, crop:-crop]
    Z = worldavg[2, :].reshape(imgshape)[crop:-crop, crop:-crop]

    avgcloud = pointcloud.PointCloud(np.vstack([X.flatten(), Y.flatten(), Z.flatten()]), X.shape, None, None, None, None)
    # avgcloud.points = pointcloud.align_points_with_xy(avgcloud.points)
    pointcloud.visualise_worlds_mplotlib(avgcloud)

    # from scipy.interpolate import griddata
    # # Z = griddata(
    # #     np.vstack([avgX[mask].flatten(), avgY[mask].flatten()]).T,
    # #     worldavg[2].flatten(),
    # #     np.vstack([avgX.flatten(), avgY.flatten()]).T,
    # #     method="nearest"
    # # )
    #
    # globX, globY = np.meshgrid(np.arange(np.min(worldavg[0]), np.max(worldavg[0])),
    #                            np.arange(np.min(worldavg[1]), np.max(worldavg[1])))
    #
    # Z = griddata(
    #     np.vstack([worldavg[0].flatten(), worldavg[1].flatten()]).T,
    #     worldavg[2].flatten(),
    #     np.vstack([globX.flatten(), globY.flatten()]).T,
    #     method="linear"
    # )
    #
    # avgcloud = pointcloud.PointCloud(np.vstack([globX.flatten(), globY.flatten(), Z.flatten()]), globX.shape, None, None, None, None)
    # pointcloud.visualise_worlds_mplotlib(avgcloud)


def average_clouds(clouds):
    detail = 5  # bins per unit
    data = np.stack([c.points for c in clouds], axis=0)
    xmin, ymin, zmin = np.min(data, (0, 2))
    xmax, ymax, zmax = np.max(data, (0, 2))
    print("data shape", data.shape)
    print("data min", np.min(data, (0, 2)))
    print("data max", np.max(data, (0, 2)))

    data = np.hstack([c.points for c in clouds])
    print("data shape", data.shape)

    # print("flattening to xy plane")
    # data = pointcloud.align_points_with_xy(data)

    # rounddata = np.round(data / (1/detail)) * (1/detail)  # round to nearest (1/detail)
    # intdata[:,-1,:] = data[]# round x and y for all sets (not z)
    # rounddata[2,:] = data[2,:]

    xmin = int(np.floor(xmin))
    ymin = int(np.floor(ymin))
    xmax = int(np.ceil(xmax))
    ymax = int(np.ceil(ymax))

    # xshape = (xmax - xmin + 1)*detail
    # yshape = (ymax - ymin + 1)*detail

    xarr, yarr = np.arange(xmin, xmax+1, 1/detail), np.arange(ymin, ymax+1, 1/detail)
    X, Y = np.meshgrid(xarr, yarr)
    yshape, xshape = X.shape
    print("X shape", X.shape)
    print("Y shape", Y.shape)

    # index data
    indexdata = np.vstack([
        np.rint((data[0, :] - xmin) * detail),
        np.rint((data[1, :] - ymin) * detail),
        data[2, :]
    ])
    avgz = np.zeros((yshape, xshape))
    pcount = np.zeros_like(avgz)
    print("indexdata min", np.min(indexdata, 1))
    print("indexdata max", np.max(indexdata, 1))

    print("binning data")
    for x, y, z in indexdata.T:
        i, j = int(x), int(y)
        avgz[j, i] += z
        pcount[j, i] += 1

    pvals = pcount > 3
    avgz[pvals] /= pcount[pvals]
    print("tot pcount", np.sum(pcount))
    print("accepted bins", np.sum(pcount[pvals]), pcount[pvals].shape)

    print("interpolating data")
    # f = interpolate.interp2d(X[pvals], Y[pvals], avgz[pvals], kind='cubic')
    # Z = f(xarr, yarr)

    # Z = interpolate.griddata(data[:-1,:].T, data[-1,:].T, np.vstack([X.flatten(), Y.flatten()]).T, method='linear')
    Z = interpolate.griddata(np.vstack([X[pvals], Y[pvals]]).T, avgz[pvals].T, np.vstack([X.flatten(), Y.flatten()]).T, method='linear')

    print("Z shape", Z.shape)

    avgcloud = pointcloud.PointCloud(np.vstack([X.flatten(), Y.flatten(), Z.flatten()]), X.shape, None, None, None, None)
    avgcloud.points = pointcloud.align_points_with_xy(avgcloud.points)
    pointcloud.visualise_worlds_mplotlib(avgcloud)


def gen_world_avg_pairs_gc(fnums, initcloud=None):
    # generate frame number pairs
    fnumpairs = [(fnums[i], fnums[i+1]) for i in range(len(fnums)-1)]

    # generate pair velocity fields
    vels = np.array(list(map(lambda fnm: generate_registrations.load_velocity_fields(*fnm), fnumpairs)))
    print(vels.shape)
    # crop vels
    vels = vels[:, :, 50:-50, 50:-50]
    print(vels.shape)

    nregs = vels.shape[0]
    imgshape = vels[0][0].shape
    shapey, shapex = imgshape

    # generate pair clouds
    clouds = []
    cloud = initcloud
    for vel in vels:
        cloud = None
        corr = create_pixel_correspondences(vel)
        # print(vel.shape)
        if cloud is None:
            P1, P2, R, t = estimate_projections(corr, None, None)
        else:
            P1, P2, R, t = estimate_projections(corr, cloud.P1, cloud.P2)

        cropcorr1 = corr[0][400:-50, 50:-50, :]
        cropcorr2 = corr[1][400:-50, 50:-50, :]
        cloud = generate_cloud((cropcorr1, cropcorr2), P1, P2, R, t)
        clouds.append(cloud)

    vels = vels[:, :, 400:-50, 50:-50]

    velshape = vels[0][0].shape
    X, Y = np.meshgrid(np.arange(velshape[1], dtype=np.float32),
                       np.arange(velshape[0], dtype=np.float32))

    print("cloudshape", clouds[0].imageshape)
    print("velshape", vels[0, 0, :, :].shape)

    # generate moving average of clouds
    avgperiod = 5
    avgclouds = []
    for i in range(len(clouds) - (avgperiod - 1)):
        avg = np.zeros_like(clouds[i].get_shaped())
        # print("avg shape", avg.shape)
        for j in range(avgperiod):
            index = i + j
            p = clouds[index].get_shaped()
            mapX = (X + vels[index, 0, :, :] * shapex).astype('float32')
            mapY = (Y + vels[index, 1, :, :] * shapey).astype('float32')
            avg = p + cv2.remap(avg, mapX, mapY,
                            interpolation=cv2.INTER_LINEAR,  # INTER_LANCZOS4,
                            borderMode=cv2.BORDER_TRANSPARENT)
        avg /= avgperiod

        crop = 50
        avgcloudX = avg[:, :, 0][100:-crop, crop:-crop]
        avgcloudY = avg[:, :, 1][100:-crop, crop:-crop]
        avgcloudZ = avg[:, :, 2][100:-crop, crop:-crop]

        avgcloud = pointcloud.PointCloud(np.vstack([avgcloudX.flatten(), avgcloudY.flatten(), avgcloudZ.flatten()]),
                                        avgcloudX.shape, None, None, None, None)
        avgclouds.append(avgcloud)
        # avgcloud.points = pointcloud.align_points_with_xy(avgcloud.points)
        # pointcloud.visualise_worlds_mplotlib(avgcloud)

    del vels
    # shift everything into global coordinates, trusting R and t from the projection matrices to be correct
    for pos in range(avgperiod, nregs):
        for i in range(pos - avgperiod + 1):
            print("shifting", i, "into global coords")
            avgclouds[i].points = clouds[pos].R.dot(avgclouds[i].points) + clouds[pos].t
    pointcloud.visualise_worlds_mplotlib(*avgclouds)

    del clouds

    # # merge all average sets into one cloud
    # globalavgp = np.hstack([c.points for c in avgclouds])
    #
    # # flatten to the x, y plane
    # globalavgp = pointcloud.align_points_with_xy(globalavgp)
    #
    # # bin in x, y grid
    # globgrid =

    average_clouds(avgclouds)


if __name__ == "__main__":
    # cloud1 = generate_world('frame9900', 'frame9903', visual=False)
    # cloud2 = generate_world('frame9903', 'frame9906', cloud1.P1, cloud1.P2, visual=False)
    # cloud3 = generate_world('frame9906', 'frame9909', cloud2.P1, cloud2.P2, visual=False)
    # cloud4 = generate_world('frame9909', 'frame9912', cloud3.P1, cloud3.P2, visual=False)
    #
    # cloud2.points = cloud1.R.T.dot(cloud2.points - cloud1.t)
    #
    # cloud3.points = cloud2.R.T.dot(cloud3.points - cloud2.t)
    # cloud3.points = cloud1.R.T.dot(cloud3.points - cloud1.t)
    #
    # cloud4.points = cloud3.R.T.dot(cloud4.points - cloud3.t)
    # cloud4.points = cloud2.R.T.dot(cloud4.points - cloud2.t)
    # cloud4.points = cloud1.R.T.dot(cloud4.points - cloud1.t)
    #
    # print(min(cloud1.points[0,:]))
    # print(max(cloud1.points[0,:]))
    # print(min(cloud1.points[1,:]))
    # print(max(cloud1.points[1,:]))
    #
    # # pointcloud.visualise_worlds_mplotlib(cloud1)#, cloud2, cloud3, cloud4)
    # average_clouds((cloud1, cloud2, cloud3, cloud4))

    # verify sensible results
    # cloud0 = generate_world(9900, 9903, visual=True)
    # cloud1 = generate_world(9903, 9906, cloud0.P1, cloud0.P2, visual=True)
    # cloud2 = generate_world(9906, 9909, cloud1.P1, cloud1.P2, visual=True)
    # cloud3 = generate_world(9909, 9912, cloud2.P1, cloud2.P2, visual=True)
    # cloud4 = generate_world(9912, 9915, cloud3.P1, cloud3.P2, visual=True)
    # cloud5 = generate_world(9915, 9918, cloud4.P1, cloud4.P2, visual=True)
    # cloud6 = generate_world(9918, 9921, cloud5.P1, cloud5.P2, visual=True)
    # cloud7 = generate_world(9921, 9924, cloud6.P1, cloud6.P2, visual=True)  # bad
    # cloud8 = generate_world(9924, 9927, cloud7.P1, cloud7.P2, visual=True)
    # cloud9 = generate_world(9927, 9930, cloud8.P1, cloud8.P2, visual=True)

    # generate_world_average_new(list(range(9900, 9931, 3)))  # 9900 to 9930 inclusive (10 pairs)
    # generate_world_average_old([9900, 9903])


    initP1 = np.array([[1.88300000e+03, 0.00000000e+00, 9.10000000e+02, 0.00000000e+00],
                       [0.00000000e+00, 1.88300000e+03, 4.90000000e+02, 0.00000000e+00],
                       [0.00000000e+00, 0.00000000e+00, 1.00000000e+00, 0.00000000e+00]])
    initP2 = np.array([[1.88348095e+03, -1.18549676e-01, 9.09004116e+02, -8.07573272e+02],
                       [1.72582273e+00, 1.88372392e+03, 4.87206551e+02, 6.61298027e+02],
                       [5.29438024e-04, 1.48277499e-03, 9.99998761e-01, -8.24083973e-01]])
    initcloud = pointcloud.PointCloud(None, None, initP1, initP2, None, None)

    # gen_world_avg_pairs_gc([9900, 9903, 9906, 9909, 9912, 9915])#, initcloud=initcloud)
    # gen_world_avg_pairs_gc(list(range(9900, 9930, 3)))#, initcloud=initcloud)
    gen_world_avg_pairs_gc(list(range(9900, 10000, 3)))#, initcloud=initcloud)
    # gen_world_avg_pairs_gc(list(range(9900, 10100, 3)))#, initcloud=initcloud)
    # gen_running_avg([9900, 9903, 9906, 9909], initcloud=initcloud)
    # gen_running_avg(list(range(9900, 9904, 3)))
    # generate_world_average_old(list(range(9900, 9910, 3)))
