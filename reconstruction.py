
import logging

import numpy as np
import cv2
import multiprocessing

import dtcwt_registration
import pointcloud
import video
from caching import cache_numpy_result


_cam_mat_K = None


def set_camera_matrix(mat):
    global _cam_mat_K
    _cam_mat_K = mat


def get_fundamental(u1, v1, u2, v2):
    u1 = u1.reshape((-1, 1))
    v1 = v1.reshape((-1, 1))
    u2 = u2.reshape((-1, 1))
    v2 = v2.reshape((-1, 1))
    lhs = np.hstack([u2*u1, u2*v1, u2, v2*u1, v2*v1, v2, u1, v1, np.ones_like(v1)])
    U, s, VT = np.linalg.svd(lhs)
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


def estimate_projections(correspondences, K):
    # points should be pre-cropped to ensure good data points (otherwise E becomes unstable)
    corr1, corr2 = correspondences

    w1 = np.vstack((corr1[:, :, 0].flat, corr1[:, :, 1].flat))
    w2 = np.vstack((corr2[:, :, 0].flat, corr2[:, :, 1].flat))

    E, mask = cv2.findEssentialMat(w1.transpose(), w2.transpose(), K)

    # TODO: refine sampling method
    retval, R, t, mask = cv2.recoverPose(E, w1[:, ::100].transpose(), w2[:, ::100].transpose(), K, mask=None)
    P1, P2 = get_projections_from_rt(K, R, t)
    return P1, P2, R, t


@cache_numpy_result(True, hash_method='readable')
def triangulate_frames(vid, fnum1, fnum2):
    vel = dtcwt_registration.load_velocity_fields(vid, fnum1, fnum2)[:, 50:-50, 50:-50]
    corr1, corr2 = create_pixel_correspondences(vel)
    P1, P2, R, t = estimate_projections((corr1, corr2), _cam_mat_K)

    w1 = np.vstack((corr1[:, :, 0].flat, corr1[:, :, 1].flat))
    w2 = np.vstack((corr2[:, :, 0].flat, corr2[:, :, 1].flat))
    points = cv2.triangulatePoints(P1.astype(float), P2.astype(float), w1.astype(float), w2.astype(float))
    points = points[:-1, :] / points[-1, :]

    return points, vel, P1, P2, R, t


def generate_frame_pair_cloud(vid, fnumpair):
    points, vel, P1, P2, R, t = triangulate_frames(vid, *fnumpair)
    imshape = vel.shape[1:]
    return pointcloud.PointCloud(points, imshape, P1, P2, R, t), vel


def get_outlier_mask(points, percentile_discard):
    # filter outliers
    # print(np.median(points, axis=1))
    # print(np.percentile(points, [0., 10., 25., 50., 75., 90., 100.], axis=1))
    limits = np.percentile(points, [float(percentile_discard), 100.0 - percentile_discard], axis=1)
    lower_limit = limits[0, :]
    upper_limit = limits[-1, :]

    outlier_mask = (
        (points[0, :] >= lower_limit[0]) & (points[0, :] <= upper_limit[0]) &
        (points[1, :] >= lower_limit[1]) & (points[1, :] <= upper_limit[1]) &
        (points[2, :] >= lower_limit[2]) & (points[2, :] <= upper_limit[2])
    )
    return outlier_mask


def gen_binned_points(points, detail=50, minpointcount=4):
    logging.info("binning points shape: {}".format(points.shape))
    # detail = bins per unit
    # minpointcount = min number of points in a bin for that bin to be accepted

    orig_points_shape = points.shape
    points = points[:, get_outlier_mask(points, 1)]
    # points = points[:, get_outlier_mask(pointcloud.align_points_with_xy(points), 10)]
    print("Removed outliers, kept {0:.0%}".format(points.shape[1] / orig_points_shape[1]))

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


def generate_world_cloud(vid, fnums, avg_size=5):
    # generate frame number pairs
    fnumpairs = [(fnums[i], fnums[i+1]) for i in range(len(fnums)-1)]
    nregs = len(fnumpairs)
    if nregs < avg_size:
        msg = "Moving average size avg_size ({a_s}) exceeds number of frame pairs ({f_p})".format(
            a_s=avg_size, f_p=nregs)
        raise ValueError(msg)

    # get cloud dimensions
    cloud0, vel0 = generate_frame_pair_cloud(vid, fnumpairs[0])
    cloudshape = cloud0.get_shaped().shape
    del cloud0, vel0

    shapey, shapex, _ = cloudshape
    X, Y = np.meshgrid(np.arange(shapex, dtype=np.float32),
                       np.arange(shapey, dtype=np.float32))

    # generate moving average of clouds
    avgpoints = np.ndarray((3, 0))
    clouds = []
    vels = []

    for i, fnumpair in enumerate(fnumpairs):
        print("\rProcessing frames {0:>4.0%}".format(i/nregs), end='')

        cloud, vel = generate_frame_pair_cloud(vid, fnumpair)
        clouds.append(cloud)
        vels.append(vel)

        if i >= avg_size-1:
            assert len(clouds) == avg_size  # todo: remove check

            moving_avg = np.zeros(cloudshape)
            for j in range(avg_size):
                p = clouds[j].get_shaped()
                mapX = (X + vels[j][0, :, :] * shapex).astype('float32')
                mapY = (Y + vels[j][1, :, :] * shapey).astype('float32')
                moving_avg = p + cv2.remap(moving_avg, mapX, mapY, interpolation=cv2.INTER_LINEAR)
                # try INTER_LANCZOS4, borderMode=cv2.BORDER_TRANSPARENT)
            moving_avg /= avg_size

            crop = 50
            avgcloudX = moving_avg[400:-crop, crop:-crop, 0]
            avgcloudY = moving_avg[400:-crop, crop:-crop, 1]
            avgcloudZ = moving_avg[400:-crop, crop:-crop, 2]
            avgcloudXYZ = np.vstack((avgcloudX.flat, avgcloudY.flat, avgcloudZ.flat))
            # avgcloudXYZ = np.vstack((avg[:, :, 0].flat, avg[:, :, 1].flat, avg[:, :, 2].flat))

            # shift into global coordinates, trusting R and t from the projection matrices to be correct
            avgpoints = clouds[-1].R.dot(avgpoints) + clouds[-1].t  # shift the previous values into current space
            avgpoints = np.hstack((avgpoints, avgcloudXYZ))  # then add current points
            # print("avgpoints bytes", avgpoints.nbytes)

            # move clouds stack along
            del clouds[0], vels[0]

    print("\rProcessed frames")
    return avgpoints


def bin_and_render(avgpoints, fname=None):
    # bin, flatten, and render
    print("Binning points")
    world = gen_binned_points(avgpoints)
    del avgpoints
    pointcloud.visualise_heatmap(world, path=fname, mode='plain')


def _multiproc_process_frame_collection(vid_path, frames):
    vidl = video.Video(vid_path)
    return generate_world_cloud(vidl, frames)


def reconstruct_world(clip, frame_step, include_intermediates=False, multiproc=True):
    frame_collection = tuple(
        range(clip.start_frame + offset, clip.stop_frame - (frame_step-offset), frame_step)
        for offset in range(frame_step if include_intermediates else 1)
    )

    if multiproc:
        print("Starting multiprocessing pool")
        with multiprocessing.Pool() as p:
            points_collection = p.starmap(
                _multiproc_process_frame_collection,
                ((clip.video.path, frames) for frames in frame_collection)
            )
    else:
        points_collection = list(map(
            lambda frames: generate_world_cloud(clip.video, frames),
            frame_collection
        ))

    if include_intermediates:
        # transform points to last frame
        finalpair = frame_collection[-1][-2:]
        logging.debug("finalpair {}".format(finalpair))

        vel = dtcwt_registration.load_velocity_fields(clip.video, *finalpair)[:, 50:-50, 50:-50]

        # Assume R = eye  # todo: cube root homogeneous matrix?
        corr = create_pixel_correspondences(vel)
        _, _, R, T = estimate_projections(corr, _cam_mat_K)
        points = np.hstack((
            points_collection[i] + T * (frame_step-(i+1))/frame_step for i in range(frame_step)
        ))
    else:
        points = points_collection[0]
    del points_collection

    print("Binning points")
    return gen_binned_points(points)


def reconstruct_frame_pair(vid, f0, f1):
    cloud, vel = generate_frame_pair_cloud(vid, (f0, f1))
    cloud.points = pointcloud.align_points_with_xy(cloud.points)
    pointcloud.visualise_worlds_mplotlib(cloud)
    # pointcloud.visualise_heatmap(cloud.points, fname="./output/{}_{}_single_pair".format(f0, f1))


def render_reconstruct_world(clip, frame_step, path=None, include_intermediates=False, multiproc=True):
    world_points = reconstruct_world(clip=clip, frame_step=frame_step,
                                     include_intermediates=include_intermediates,
                                     multiproc=multiproc)

    save_path = '{base}_{start}_{stop}_{step}_{incl}'.format(
        base=path, start=clip.start_frame, stop=clip.stop_frame, step=frame_step,
        incl=('multitrain' if include_intermediates else 'singletrain')
    )

    return pointcloud.visualise_heatmap(world_points, path=save_path, mode='plain')


if __name__ == "__main__":
    # logging.root.setLevel(logging.DEBUG)
    vid = video.Video(r"../../../../../YUNC0001.mp4")
    clip = video.Clip(vid, 26400, 26460)
    # 9900, 9920
    # 20750, 20850
    # 20750, 20800
    # 9900, 10100
    # 9900, 9930
    # 26400, 26460
    # 10101, 10160
    # 26400, 26500
    # 31302, 31600
    # 31590, 31900
    # 13100, 13200
    # 14550, 14610
    # 15000, 15200

    print("Loaded video {fname}, shape: {shape}, fps: {fps}, start: {start}, stop: {stop}".format(
        fname=clip.video.path, shape=clip.video.shape, fps=clip.video.fps,
        start=clip.start_frame, stop=clip.stop_frame))

    # estimated from tractor treads in video
    set_camera_matrix(np.array(
        [[1883, 0, 910],
         [0, 1883, 490],
         [0, 0, 1]]
    ))

    # reconstruct_world(clip, 3, include_intermediates=True)
    render_reconstruct_world(clip, 3, path='./output/', include_intermediates=True, multiproc=True)

    print("Done.")