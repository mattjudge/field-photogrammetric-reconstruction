
import logging

import numpy as np
import cv2
import multiprocessing

import generate_registrations
import pointcloud
import video


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

    # print("recovering pose")
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

    # print("triangulating vertices")
    points = cv2.triangulatePoints(P1.astype(float), P2.astype(float), w1.astype(float), w2.astype(float))
    # world = cv2.triangulatePoints(P, P2, w1, w2)  # crashes, bug in opencv
    # print("triangulated.")
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


def gen_binned_points(points, detail=50, minpointcount=4):
    # detail = bins per unit
    # minpointcount = min number of points in a bin for that bin to be accepted

    # remove outliers
    def getoutliermask(data, m=3.):
        # ref: https://stackoverflow.com/a/16562028
        d = np.abs(data - np.median(data))
        mdev = np.median(d)
        s = d / mdev if mdev else 0.
        return s < m

    # outliermask = getoutliermask(points[0, :])
    # outliermask = np.logical_and(
    #     getoutliermask(points[0, :]),
    #     getoutliermask(points[1, :]),
    #     getoutliermask(points[2, :])
    # )
    # outliermasksum = np.sum(outliermask)
    # print("outliermask accepts {} out of {} ({}%)".format(
    #     outliermasksum,
    #     points[2, :].shape,
    #     int(outliermasksum / points[2, :].shape * 100)
    # ))
    # points = points[:, outliermask]

    xmin, ymin, zmin = np.floor(np.min(points, axis=1)).astype(int)
    xmax, ymax, zmax = np.ceil(np.max(points, axis=1)).astype(int)
    logging.info("Data shape: {}".format(points.shape))
    logging.info("Data min:   {}".format(np.min(points, axis=1)))
    logging.info("Data max:   {}".format(np.max(points, axis=1)))

    xarr, yarr = np.arange(xmin, xmax+1, 1/detail), \
                 np.arange(ymin, ymax+1, 1/detail)
    X, Y = np.meshgrid(xarr, yarr)
    logging.info("X,Y shape: {}, {}".format(X.shape, Y.shape))

    avgz, xedges, yedges = np.histogram2d(
        points[0, :], points[1, :],
        bins=(xarr, yarr),
        range=None, normed=False,
        weights=points[2, :]
    )
    pcount, _, _ = np.histogram2d(
        points[0, :], points[1, :],
        bins=(xarr, yarr),
        range=None, normed=False,
        weights=None
    )
    avgz = avgz.T  # np.histogram2d returns with x on first dim (unconventional)
    pcount = pcount.T
    logging.info("Binned shape: {}".format(avgz.shape))

    assert np.sum(pcount) == points.shape[1]
    print("Binned {} points".format(int(np.sum(pcount))))
    print("Mean bin size: ", np.mean(pcount))

    pvals = pcount >= minpointcount  # mask to filter bins by minimum number of child points
    avgz[pvals] /= pcount[pvals]
    print("Accepted {} points ({}%)".format(
        int(np.sum(pcount[pvals])),
        int(np.sum(pcount[pvals]) / np.sum(pcount) * 100)))
    print("Mean filtered bin size", np.mean(pcount[pvals]))

    binnedXY = np.vstack([
        X[:-1, :-1][pvals],  # remove last column and row as they are upper bounds
        Y[:-1, :-1][pvals]   # and filter by pvals mask
    ]) + 0.5 * 1/detail      # offset for bin center
    logging.info("binnedXY shape: {}".format(binnedXY.shape))

    print("Aligning points with XY plane")
    avgpoints = pointcloud.align_points_with_xy(np.vstack([binnedXY, avgz[pvals].flatten()]))
    return avgpoints


def gen_world_avg_pairs_gc(vid, fnums):
    # generate frame number pairs
    fnumpairs = [(fnums[i], fnums[i+1]) for i in range(len(fnums)-1)]

    vel0 = generate_registrations.load_velocity_fields(vid, *fnumpairs[0])

    # generate pair velocity fields
    # print("initial vel field shape", vel0.shape)
    # crop vels
    vel0 = vel0[:, 50:-50, 50:-50]
    # print("cropped vel field shape", vel0.shape)

    nregs = len(fnumpairs)
    imgshape = vel0[0].shape
    shapey, shapex = imgshape  # todo: move before cropping?

    velshape = vel0[0].shape
    del vel0
    X, Y = np.meshgrid(np.arange(velshape[1], dtype=np.float32),
                       np.arange(velshape[0], dtype=np.float32))

    # generate moving average of clouds
    avgperiod = 5
    avgpoints = np.ndarray((3, 0))
    # generate pair clouds
    clouds = []
    vels = []

    i = 0
    for fnumpair in fnumpairs[:avgperiod-1]:
        progress = int(i / nregs * 100)
        print("\rProcessing frames\t{}%".format(progress), end='')
        i += 1

        vel = generate_registrations.load_velocity_fields(
            vid, *fnumpair
        )[:, 50:-50, 50:-50]
        vels.append(vel)
        corr = create_pixel_correspondences(vel)

        P1, P2, R, t = estimate_projections(corr)
        cloud = generate_cloud(corr, P1, P2, R, t)
        clouds.append(cloud)

    cloudshape = clouds[0].get_shaped().shape

    for i in range(nregs - (avgperiod - 1)):
        progress = int((i + avgperiod) / nregs * 100)
        print("\rProcessing frames\t{}%".format(progress), end='')

        vel = generate_registrations.load_velocity_fields(
            vid, *fnumpairs[i+avgperiod-1]
        )[:, 50:-50, 50:-50]
        vels.append(vel)
        corr = create_pixel_correspondences(vel)

        P1, P2, R, t = estimate_projections(corr)
        cloud = generate_cloud(corr, P1, P2, R, t)
        clouds.append(cloud)

        assert len(clouds) == avgperiod  # todo: remove check

        avg = np.zeros(cloudshape)
        for j in range(avgperiod):
            p = clouds[j].get_shaped()
            mapX = (X + vels[j][0, :, :] * shapex).astype('float32')
            mapY = (Y + vels[j][1, :, :] * shapey).astype('float32')
            avg = p + cv2.remap(avg, mapX, mapY,
                            interpolation=cv2.INTER_LINEAR,  # INTER_LANCZOS4,
                            borderMode=cv2.BORDER_TRANSPARENT)
        avg /= avgperiod

        crop = 50
        avgcloudX = avg[400:-crop, crop:-crop, 0]
        avgcloudY = avg[400:-crop, crop:-crop, 1]
        avgcloudZ = avg[400:-crop, crop:-crop, 2]
        avgcloudXYZ = np.vstack((avgcloudX.flat, avgcloudY.flat, avgcloudZ.flat))

        # avgcloud.points = pointcloud.align_points_with_xy(avgcloud.points)
        # pointcloud.visualise_worlds_mplotlib(avgcloud)

        # shift into global coordinates, trusting R and t from the projection matrices to be correct
        avgpoints = clouds[-1].R.dot(avgpoints) + clouds[-1].t  # shift the previous values into current space
        avgpoints = np.hstack((avgpoints, avgcloudXYZ))  # then add current points
        # print("avgpoints bytes", avgpoints.nbytes)

        # move clouds stack along
        # clouds = clouds[1:]
        # vels = vels[1:]
        del clouds[0]
        del vels[0]

    print("\nProcessed frames")

    return avgpoints


def bin_and_render(avgpoints, fname=None):
    # bin, flatten, and render
    print("Binning points")
    world = gen_binned_points(avgpoints)
    del avgpoints

    # if fname is not None:
    #     print("exporting .mat file")
    #     savemat('{}.mat'.format(fname), {
    #         'X': X.flatten(),
    #         'Y': Y.flatten(),
    #         'Z': Z.flatten()
    #     })

    pointcloud.visualise_heatmap(world, fname=fname)


def generate_world(vid, start, stop):
    avgpoints = gen_world_avg_pairs_gc(vid, list(range(start, stop, 3)))
    bin_and_render(avgpoints, './output/{}_{}_singletrain_heatmap_neg'.format(start, stop))


def multiprocfunc(f):
    vidl = video.Video(vid.fname)  # todo: don't rely on vid from global
    return gen_world_avg_pairs_gc(vidl, f)


def generate_world3(vid, start, stop, multiproc=True):
    f1 = list(range(start,   stop-2, 3))
    f2 = list(range(start+1, stop-1, 3))
    f3 = list(range(start+2, stop,   3))
    logging.debug("Frame lists: \n{}\n{}\n{}".format(f1, f2, f3))

    if multiproc:
        print("Starting multiprocessing pool")
        with multiprocessing.Pool() as p:
            points1, points2, points3 = p.map(
                multiprocfunc,
                (f1, f2, f3)
            )
    else:
        points1 = gen_world_avg_pairs_gc(vid, f1)
        points2 = gen_world_avg_pairs_gc(vid, f2)
        points3 = gen_world_avg_pairs_gc(vid, f3)

    # # transform points to last frame
    # interpolate between final pair
    finalpair = f3[-2:]
    logging.debug("finalpair {}".format(finalpair))

    vel = generate_registrations.load_velocity_fields(vid, *finalpair)[:, 50:-50, 50:-50]
    vel1 = vel * 2/3
    vel2 = vel * 1/3

    imgshape = vel[0].shape
    shapey, shapex = imgshape  # todo: move before cropping

    # Assume R = eye  # todo: cube root homogeneous matrix
    corr = create_pixel_correspondences(vel)
    _, _, R, T = estimate_projections(corr)
    avgpoints = np.hstack((
        points1 + T * 2/3,
        points2 + T * 1/3,
        points3
    ))
    # print("R, T", R, T)

    bin_and_render(avgpoints, './output/{}_{}_tripletrain_heatmap_neg'.format(start, stop))


def gen_frame_pair(vid, f0, f1):
    vel = generate_registrations.load_velocity_fields(
        vid, f0, f1
    )[:, 50:-50, 50:-50]

    corr = create_pixel_correspondences(vel)
    P1, P2, R, t = estimate_projections(corr)
    cloud = generate_cloud(corr, P1, P2, R, t)
    cloud.points = pointcloud.align_points_with_xy(cloud.points)

    pointcloud.visualise_worlds_mplotlib(cloud)
    # pointcloud.visualise_heatmap(cloud.points, fname="./output/{}_{}_single_pair".format(f0, f1))


if __name__ == "__main__":
    vid = video.Video(r"../../../../../YUNC0001.mp4")
    print("Loaded video {fname}, shape: {shape}, fps: {fps}".format(fname=vid.fname, shape=vid.shape, fps=vid.fps))

    # gen_frame_pair(vid, 9900, 9903)

    # generate_world(vid, 9900, 9920)
    # generate_world3(vid, 20750, 20850)
    # generate_world3(vid, 20750, 20800)
    # generate_world3(vid, 9900, 10100)
    # generate_world3(vid, 9900, 9930)
    generate_world3(vid, 26400, 26460)
    # generate_world(vid, 26400, 26460)
    # generate_world3(vid, 10101, 10160)
    # generate_world3(vid, 26400, 26500)
    # generate_world(vid, 31302, 31600)
    # generate_world(vid, 31590, 31900)

    # generate_world3(13100, 13200)
    # generate_world3(vid, 14550, 14610)
    # generate_world3(15000, 15200)

    print("Done.")
