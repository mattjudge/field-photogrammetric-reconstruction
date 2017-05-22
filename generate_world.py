
import numpy as np
import cv2
from scipy import interpolate, linalg
import dtcwt.registration

import generate_registrations
import pointcloud, video

# from joblib import Memory
# mem = Memory(cachedir='./data/')


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
    vely = vely.clip(-1, 0)  # todo: check expected limits

    X, Y = np.meshgrid(np.arange(shapex), np.arange(shapey))

    # velocity vectors map pixels in f2 to their locations in f1
    u1shaped = X + velx*shapex
    v1shaped = Y + vely*shapey
    u2shaped, v2shaped = X, Y

    return np.dstack((u1shaped, v1shaped)), np.dstack((u2shaped, v2shaped))


def estimate_projections(correspondences):
    # points should be pre-cropped to ensure good data points (otherwise E becomes unstable)
    corr1, corr2 = correspondences
    imshape = corr1[:, :, 0].shape
    shapey, shapex = imshape

    w1 = np.vstack((corr1[:, :, 0].flat, corr1[:, :, 1].flat))
    w2 = np.vstack((corr2[:, :, 0].flat, corr2[:, :, 1].flat))
    # print("w1 as int", w1.astype(int))
    # print("w2 as int", w2.astype(int))

    # TODO: ensure shapex, shapey are correct estimations given cropping of vels
    fku = 1883  # estimated
    K = np.array([[fku,     0, shapex//2],
                  [0,     fku, shapey//2],
                  [0,       0,         1]])

    # E, mask = cv2.findEssentialMat(w1.transpose(), w2.transpose(), 2000, (shapey//2, shapex//2))
    E, mask = cv2.findEssentialMat(w1.transpose(), w2.transpose(), K)
    # print("E", E)
    # print("mask", mask.T)
    # print(np.sum(mask), ":", mask.size)
    # print(mask.size - np.sum(mask))
    # print((mask.size - np.sum(mask)) / mask.size)

    print("recovering pose")
    # retval, R, t, mask = cv2.recoverPose(E, w1.transpose(), w2.transpose(), K, mask=mask)
    # TODO: refine sampling method
    retval, R, t, mask = cv2.recoverPose(E, w1[:,::100].transpose(), w2[:,::100].transpose(), K, mask=None)
    P1, P2 = get_projections_from_rt(K, R, t)
    # print("R", R)
    # print("t", t)
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
    # print("w1 as int", w1.astype(int))
    # print("w2 as int", w2.astype(int))

    print("triangulating vertices")
    points = cv2.triangulatePoints(P1.astype(float), P2.astype(float), w1.astype(float), w2.astype(float))
    # world = cv2.triangulatePoints(P, P2, w1, w2)  # crashes, bug in opencv
    print("triangulated.")
    # world[:, mask.transpose()] = float('nan')

    points /= points[-1, :]

    # check1 = P1.dot(points[:, points.shape[1] // 4])
    # check1 /= check1[-1]
    # print("check", check1, w1[:, points.shape[1] // 4])
    #
    # check2 = P2.dot(points[:, points.shape[1] // 4])
    # check2 /= check2[-1]
    # print("check", check2, w2[:, points.shape[1] // 4])
    #
    # print("world shape", points.shape)

    return pointcloud.PointCloud(points[:-1, :], imshape, P1, P2, R, t)


def gen_moving_avg(correspondences, clouds):
    # generates a "moving average"
    # beware that initial frames become increasingly "blurred" over a greater number of input frames
    # should only be used for small numbers of frames
    pass



def gen_binned_cloud(points):
    detail = 5  # bins per unit

    print("data shape", points.shape)

    xmin, ymin, zmin = np.floor(np.min(points, axis=1)).astype(int)
    xmax, ymax, zmax = np.ceil(np.max(points, axis=1)).astype(int)
    print("data shape", points.shape)
    print("data min", np.min(points, axis=1))
    print("data max", np.max(points, axis=1))

    # print("flattening to xy plane")
    # data = pointcloud.align_points_with_xy(data)

    xarr, yarr = np.arange(xmin, xmax+1, 1/detail), np.arange(ymin, ymax+1, 1/detail)
    X, Y = np.meshgrid(xarr, yarr)
    yshape, xshape = X.shape
    print("X shape", X.shape)
    print("Y shape", Y.shape)

    # index data
    indexdata = np.vstack([
        np.rint((points[0, :] - xmin) * detail),
        np.rint((points[1, :] - ymin) * detail),
        points[2, :]
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
    return avgcloud


def gen_world_avg_pairs_gc(vid, fnums):
    # generate frame number pairs
    fnumpairs = [(fnums[i], fnums[i+1]) for i in range(len(fnums)-1)]

    vel0 = generate_registrations.load_velocity_fields(vid, *fnumpairs[0])
    # generate pair velocity fields
    # vels = np.array(list(map(lambda fnm: generate_registrations.load_velocity_fields(*fnm), fnumpairs)))
    print(vel0.shape)
    # crop vels
    vel0 = vel0[:, 50:-50, 50:-50]
    print(vel0.shape)

    nregs = len(fnumpairs)
    imgshape = vel0[0].shape
    shapey, shapex = imgshape  # todo: move before cropping

    velshape = vel0[0].shape
    X, Y = np.meshgrid(np.arange(velshape[1], dtype=np.float32),
                       np.arange(velshape[0], dtype=np.float32))

    # print("cloudshape", clouds[0].imageshape)
    print("velshape", vel0[0, :, :].shape)

    # generate moving average of clouds
    avgperiod = 5
    avgpoints = np.ndarray((3, 0))
    # generate pair clouds
    clouds = []
    vels = []
    for fnumpair in fnumpairs[:avgperiod-1]:
        vel = generate_registrations.load_velocity_fields(vid, *fnumpair)[:, 50:-50, 50:-50]
        vels.append(vel)
        corr = create_pixel_correspondences(vel)
        # print(vel.shape)
        P1, P2, R, t = estimate_projections(corr)
        cloud = generate_cloud(corr, P1, P2, R, t)
        clouds.append(cloud)
    cloudshape = clouds[0].get_shaped().shape
    for i in range(nregs - (avgperiod - 1)):
        print("i, lenclouds", i, len(clouds))
        vel = generate_registrations.load_velocity_fields(
            vid,
            *fnumpairs[i+avgperiod-1]
        )[:, 50:-50, 50:-50]
        vels.append(vel)
        corr = create_pixel_correspondences(vel)
        # print(vel.shape)
        P1, P2, R, t = estimate_projections(corr)
        cloud = generate_cloud(corr, P1, P2, R, t)
        clouds.append(cloud)

        assert len(clouds) == avgperiod  # todo: remove check

        avg = np.zeros(cloudshape)
        # avg = np.zeros_like(clouds[i].get_shaped())
        # print("avg shape", avg.shape)
        for j in range(avgperiod):
            p = clouds[j].get_shaped()
            mapX = (X + vels[j][0, :, :] * shapex).astype('float32')
            mapY = (Y + vels[j][1, :, :] * shapey).astype('float32')
            avg = p + cv2.remap(avg, mapX, mapY,
                            interpolation=cv2.INTER_LINEAR,  # INTER_LANCZOS4,
                            borderMode=cv2.BORDER_TRANSPARENT)
        avg /= avgperiod

        crop = 50
        print("avg shape", avg.shape)
        avgcloudX = avg[400:-crop, crop:-crop, 0]
        avgcloudY = avg[400:-crop, crop:-crop, 1]
        avgcloudZ = avg[400:-crop, crop:-crop, 2]
        avgcloudXYZ = np.vstack((avgcloudX.flat, avgcloudY.flat, avgcloudZ.flat))

        # avgcloud.points = pointcloud.align_points_with_xy(avgcloud.points)
        # pointcloud.visualise_worlds_mplotlib(avgcloud)

        # shift into global coordinates, trusting R and t from the projection matrices to be correct
        avgpoints = clouds[-1].R.dot(avgpoints) + clouds[-1].t  # shift the previous values into current space
        avgpoints = np.hstack((avgpoints, avgcloudXYZ))  # then add current points

        # move clouds stack along
        # clouds = clouds[:-1]
        clouds = clouds[1:]
        vels = vels[1:]

    # del vels
    # del clouds
    # world = gen_binned_cloud(avgpoints)
    # del avgpoints
    # pointcloud.visualise_worlds_mplotlib(world)
    return avgpoints


def generate_world(vid, start, stop):
    avgpoints = gen_world_avg_pairs_gc(vid, list(range(start, stop, 3)))

    # bin, flatten, and render
    world = gen_binned_cloud(avgpoints)
    del avgpoints
    # pointcloud.visualise_worlds_mplotlib(world)  #, fname='test_save.png')
    pointcloud.visualise_heatmap(world, fname='{}_{}_singletrain_heatmap'.format(start, stop))


def generate_world3(vid, start, stop):
    f1 = list(range(start,   stop-2, 3))
    f2 = list(range(start+1, stop-1, 3))
    f3 = list(range(start+2, stop,   3))
    print(f1, f2, f3)

    points1 = gen_world_avg_pairs_gc(f1)
    points2 = gen_world_avg_pairs_gc(f2)
    points3 = gen_world_avg_pairs_gc(f3)

    ## transform points to last frame
    # interpolate between final pair
    finalpair = f3[-2:]
    print(finalpair)

    vel = generate_registrations.load_velocity_fields(vid, *finalpair)[:, 50:-50, 50:-50]
    vel1 = vel * 2/3
    vel2 = vel * 1/3

    imgshape = vel[0].shape
    shapey, shapex = imgshape  #todo: move before cropping

    # generate pair clouds
    # corr1 = create_pixel_correspondences(vel1)
    # corr2 = create_pixel_correspondences(vel2)
    # # print(vel.shape)
    # _, _, R1, t1 = estimate_projections(corr1)
    # _, _, R2, t2 = estimate_projections(corr2)
    #
    # print("R1, t1", R1, t1)
    # print("R2, t2", R2, t2)
    #
    # # avgpoints = points1
    # # avgpoints = R1.dot(avgpoints) + t1
    # # avgpoints = np.hstack((avgpoints, points2))
    # # avgpoints = R2.dot(avgpoints) + t2
    # # avgpoints = np.hstack((avgpoints, points3))
    #
    # avgpoints = np.hstack((
    #     # points1,
    #     # points2,
    #     R1.dot(points1) + t1,
    #     R2.dot(points2) + t2,
    #     points3
    # ))

    # Assume R = eye  # todo: cube root homogeneous matrix
    corr = create_pixel_correspondences(vel)
    _, _, R, T = estimate_projections(corr)
    avgpoints = np.hstack((
        points1 + T * 2/3,
        points2 + T * 1/3,
        points3
    ))
    print("R, T", R, T)


    # cloud1 = pointcloud.PointCloud(pointcloud.align_points_with_xy(avgcloud.points)
    # pointcloud.visualise_worlds_mplotlib(avgcloud)

    # bin, flatten, and render
    world = gen_binned_cloud(avgpoints)
    del avgpoints
    # pointcloud.visualise_worlds_mplotlib(world)
    pointcloud.visualise_heatmap(world, fname='{}_{}_tripletrain_heatmap'.format(start, stop))


if __name__ == "__main__":
    vid = video.Video(r"../../../../../YUNC0001.mp4")
    print(vid.fname)
    print(vid.shape)
    print(vid.fps)

    # generate_world3(vid, 9900, 9920)
    generate_world(vid, 9900, 9920)
    # generate_world3(13100, 13200)
    # generate_world3(14550, 14800)
    # generate_world3(15000, 15200)
