
"""
Author: Matt Judge 2017

This module provides:
    :func:`render_reconstruct_world` as a helper function to reconstruct and render a clip of video
    :func:`reconstruct_world` to create a 3D reconstruction of a clip of video
    :func:`reconstruct_frame_pair` to triangulate and reconstruct two frames
    :func:`generate_world_cloud` to generate a dense point cloud from a video, utilising a moving average
    :func:`gen_binned_points` to bin a dense point cloud and average over reliable bins
    :func:`get_outlier_mask` to determine outliers in a dense point cloud
    :func:`generate_frame_pair_cloud` to create an instance of :class:`pointcloud.Pointcloud` from two frames
    :func:`triangulate_frames` to generate a point cloud from two frames
    :func:`estimate_projections` to estimate projection matrices (P1, P2, R, T) from pixel
            correspondences and camera matrix
    :func:`create_pixel_correspondences` to create pixel correspondences from relative motion velocities
    :func:`get_projections_from_rt` to get projection matrices from R and T

    And legacy functions:
    :func:`get_fundamental` to get the Fundamental matrix from corresponding pixel positions
    :func:`get_rt` to get rotation R and translation T matrices from the essential matrix E
"""

import logging
import multiprocessing

import cv2
import numpy as np

from field_reconstruction import dtcwt_registration, pointcloud, video
from field_reconstruction.numpy_caching import np_cache


def get_fundamental(u1, v1, u2, v2):
    """Legacy function to get the Fundamental matrix from corresponding pixel positions"""
    u1 = u1.reshape((-1, 1))
    v1 = v1.reshape((-1, 1))
    u2 = u2.reshape((-1, 1))
    v2 = v2.reshape((-1, 1))
    lhs = np.hstack([u2*u1, u2*v1, u2, v2*u1, v2*v1, v2, u1, v1, np.ones_like(v1)])
    U, s, VT = np.linalg.svd(lhs)
    NS = VT[-1:, :].transpose()
    return NS.reshape((3, 3)) / NS[-1]


def get_rt(E):
    """Legacy function to get rotation R and translation T matrices from the essential matrix E"""
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
    """
    Get projection matrices from R and T
    :param K: [3,3] Camera calibration matrix
    :param R: [3,3] Rotation matrix
    :param t: [3,1] Translation matrix
    :return: P1, P2, projection matrices, both [3,4]
    """
    P1 = K.dot(
        np.hstack([np.eye(3), np.zeros((3, 1))])
    )
    P2 = K.dot(
        np.hstack([R, t])
    )
    return P1, P2


def create_pixel_correspondences(vel):
    """
    Create pixel correspondences from relative motion velocities
    :param vel: Motion velocities from :func:`dtcwt_registration.load_velocity_fields`, a [2,Y,X] array
    :return: tuple of two pixel correspondences, each a [Y,X] array corresponding to one frame
    """
    velx, vely = vel
    imshape = velx.shape
    shapey, shapex = imshape

    X, Y = np.meshgrid(np.arange(shapex), np.arange(shapey))

    # velocity vectors map pixels in f2 to their locations in f1
    u1shaped = X + velx
    v1shaped = Y + vely
    u2shaped, v2shaped = X, Y

    return np.dstack((u1shaped, v1shaped)), np.dstack((u2shaped, v2shaped))


def estimate_projections(correspondences, K):
    """
    Estimate the projection matrices given point correspondences and the camera calibration matrix K
    :param correspondences: Tuple of two frame correspondences (each [X,Y] matrices). Should be
        pre-cropped to ensure good data points (otherwise E becomes unstable)
    :param K: [3,3] Camera calibration matrix
    :return:P1, P2, R, t Camera projection matrices:
        P1: Projection to frame 1
        P2: Projection to frame 2
        R: Rotation from frame 1 to frame 2
        t: Translation from frame 1 to frame 2
    """
    corr1, corr2 = correspondences

    w1 = np.vstack((corr1[:, :, 0].flat, corr1[:, :, 1].flat))
    w2 = np.vstack((corr2[:, :, 0].flat, corr2[:, :, 1].flat))

    E, mask = cv2.findEssentialMat(w1.transpose(), w2.transpose(), K)

    # TODO: refine sampling method
    retval, R, t, mask = cv2.recoverPose(E, w1[:, ::100].transpose(), w2[:, ::100].transpose(), K, mask=None)
    P1, P2 = get_projections_from_rt(K, R, t)
    return P1, P2, R, t


@np_cache(True, hash_method='readable')
def triangulate_frames(vid, frame_pair, K):
    """
    Perform point triangulation from two frames of a video
    :param vid: :class:video.Video object from which to take the frames
    :param frame_pair: Tuple of two frame numbers (frame1, frame2)
    :param K: [3,3] Camera calibration matrix
    :returns: points, velocities, P1, P2, R, t
    WHERE
      - points are a [3, N] numpy array point cloud
      - velocities are the velocities returned by the dtcwt transform
        as a [2, Y, X] numpy array (see :func:`dtcwt_registration.load_velocity_fields`)
      - P1, P2, R, t are the projection matrix parameters
                    returned by :func:`estimate_projections`)
    """
    vel = dtcwt_registration.load_velocity_fields(vid, *frame_pair)[:, 50:-50, 50:-50]
    corr1, corr2 = create_pixel_correspondences(vel)
    P1, P2, R, t = estimate_projections((corr1, corr2), K)

    w1 = np.vstack((corr1[:, :, 0].flat, corr1[:, :, 1].flat))
    w2 = np.vstack((corr2[:, :, 0].flat, corr2[:, :, 1].flat))
    points = cv2.triangulatePoints(P1.astype(float), P2.astype(float), w1.astype(float), w2.astype(float))
    points = points[:-1, :] / points[-1, :]

    return points, vel, P1, P2, R, t


def generate_frame_pair_cloud(vid, frame_pair, K):
    """
    Generates a instance of :class:`pointcloud.Pointcloud` from a pair of frames of a :class:`video.Video`.
    :param vid: :class:video.Video object from which to take the frames
    :param frame_pair: Tuple of two frame numbers (frame1, frame2)
    :param K: [3,3] Camera calibration matrix
    :return: pointcloud, velocities
    WHERE
      - pointcloud is an instance of :class:`pointcloud.Pointcloud`
      - velocities are the velocities returned by the dtcwt transform
        as a [2, Y, X] numpy array (see :func:`dtcwt_registration.load_velocity_fields`)
    """
    points, vel, P1, P2, R, t = triangulate_frames(vid, frame_pair, K)
    imshape = vel.shape[1:]
    return pointcloud.PointCloud(points, imshape, P1, P2, R, t), vel


def get_outlier_mask(points, percentile_discard):
    """
    Generate a mask identifying outliers in a point cloud
    :param points: A [3, N] numpy array of points
    :param percentile_discard: The percentile to discard symmetrically (i.e. a :param:percentile_discard of 5
                discards points which fall into the first or last 1% of the data in the x, y, or z dimensions.
    :return: outlier_mask, a [N,] boolean numpy array, where True values correspond to an outlying point
    """
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
    """
    Bin points from a point cloud, ignoring outliers
    :param points: A [3, N] numpy array of points
    :param detail: The bins per point cloud unit
    :param minpointcount: Minimum number of points in a bin considered to generate a reliable mean
    :return: binned_points, a [3,N] numpy array of the binned points
    """
    logging.info("binning points shape: {}".format(points.shape))
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


def generate_world_cloud(vid, K, fnums, avg_size=5):
    """
    Generate a point cloud from multiple video frames
    :param vid: :class:video.Video object from which to take the frames
    :param K: [3,3] Camera calibration matrix
    :param fnums: An iterable of frame numbers from which to create the point cloud
    :param avg_size: The number of frame-pairs to combine in a moving average
    :return: points, a [3,N] numpy array of the triangulated points computed from the video
    """
    fnumpairs = [(fnums[i], fnums[i+1]) for i in range(len(fnums)-1)]
    nregs = len(fnumpairs)
    if nregs < avg_size:
        msg = "Moving average size avg_size ({a_s}) exceeds number of frame pairs ({f_p})".format(
            a_s=avg_size, f_p=nregs)
        raise ValueError(msg)

    # get cloud dimensions
    cloud0, vel0 = generate_frame_pair_cloud(vid, fnumpairs[0], K)
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

        cloud, vel = generate_frame_pair_cloud(vid, fnumpair, K)
        clouds.append(cloud)
        vels.append(vel)

        if i >= avg_size-1:
            assert len(clouds) == avg_size  # todo: remove check

            moving_avg = np.zeros(cloudshape)
            for j in range(avg_size):
                p = clouds[j].get_shaped()
                mapX = (X + vels[j][0, :, :]).astype('float32')
                mapY = (Y + vels[j][1, :, :]).astype('float32')
                moving_avg = p + cv2.remap(moving_avg, mapX, mapY, interpolation=cv2.INTER_LINEAR)
                # try INTER_LANCZOS4, borderMode=cv2.BORDER_TRANSPARENT)
                # dst(x,y) = src(map_x(x,y), map_y(x,y))

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


def reconstruct_frame_pair(vid, K, f0, f1):
    """
    Legacy function to compute a reconstruction of the world from just two frames
    :param vid: input video
    :param K: camera calibration matrix
    :param f0: First frame number
    :param f1: Second frame number
    """
    cloud, vel = generate_frame_pair_cloud(vid, K, (f0, f1))
    points = pointcloud.align_points_with_xy(cloud.points)
    pointcloud.visualise_heatmap(points, fname="./output/{}_{}_single_pair".format(f0, f1))


def _multiproc_process_frame_collection(vid_path, frames, K):
    vidl = video.Video(vid_path)
    return generate_world_cloud(vidl, K, frames)


def reconstruct_world(clip, K, frame_step, include_intermediates=False, multiproc=True):
    """
    Generate a filtered, averaged point cloud from a video.
    :param clip: An instance of :class:`video.Clip` as the input video
    :param K: [3,3] The camera calibration matrix
    :param frame_step: The fixed step between frame numbers from which to determine pairs of frames to triangulate
    :param include_intermediates: Boolean. When enabled, all frames are included in the reconstruction.
        To illustrate, with a frame step of 3:
            Disabled: result = function(f0->f3, f3->f6, f6->f9, etc)  # a single 'frame train'
            Enabled:  result = function(f0->f3, f1->f4, f2->f5, f3->f6, f4->f7, f5->f8, etc)  # a multi 'frame train'
    :param multiproc: Boolean. When enabled, reconstruction of each frame train is processed in parallel
    :return: points, a [3,N] numpy array of x,y,z point coordinates of the reconstructed world
    """
    frame_collection = tuple(
        range(clip.start_frame + offset, clip.stop_frame - (frame_step-offset), frame_step)
        for offset in range(frame_step if include_intermediates else 1)
    )

    if multiproc:
        print("Starting multiprocessing pool")
        with multiprocessing.Pool() as p:
            points_collection = p.starmap(
                _multiproc_process_frame_collection,
                ((clip.video.path, frames, K) for frames in frame_collection)
            )
    else:
        points_collection = list(map(
            lambda frames: generate_world_cloud(clip.video, K, frames),
            frame_collection
        ))

    if include_intermediates:
        # transform points to last frame
        finalpair = frame_collection[-1][-2:]
        logging.debug("finalpair {}".format(finalpair))

        vel = dtcwt_registration.load_velocity_fields(clip.video, *finalpair)[:, 50:-50, 50:-50]

        # Assume R = eye  # todo: cube root homogeneous matrix?
        corr = create_pixel_correspondences(vel)
        _, _, R, T = estimate_projections(corr, K)
        points = np.hstack((
            points_collection[i] + T * (frame_step-(i+1))/frame_step for i in range(frame_step)
        ))
    else:
        points = points_collection[0]
    del points_collection

    print("Binning points")
    return gen_binned_points(points)


def render_reconstruct_world(clip, K, frame_step, path=None, include_intermediates=False, multiproc=True,
                             render_mode='standard', render_scale=1, render_gsigma=0):
    """
    A helper wrapper function. **See documentation for :func:`reconstruction.reconstruct_world` and
        :func:`pointcloud.visualise_heatmap` for detailed parameter documentation.**
    :param clip: Input video clip
    :param K: Camera calibration matrix
    :param frame_step: Frame step
    :param path: Output path
    :param include_intermediates: Boolean to enable computation of mutiple frame trains
    :param multiproc: Boolean to enable parallel frame train processing
    :param render_mode: Mode to render with
    :param render_scale: Scale to render with
    :param render_gsigma: Level of Gaussian smoothing
    :return: matplotlib figure
    """
    world_points = reconstruct_world(clip=clip, K=K, frame_step=frame_step,
                                     include_intermediates=include_intermediates, multiproc=multiproc)

    save_path = '{base}_{start}_{stop}_{step}_{incl}'.format(
        base=path, start=clip.start_frame, stop=clip.stop_frame, step=frame_step,
        incl=('multitrain' if include_intermediates else 'singletrain')
    ) if path is not None else None

    return pointcloud.visualise_heatmap(
        world_points, path=save_path, mode=render_mode, scale=render_scale, gsigma=render_gsigma
    )


if __name__ == "__main__":
    pass
