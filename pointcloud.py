import logging

import numpy as np

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import matplotlib.pyplot as plt
from scipy import interpolate, ndimage
from scipy.io import savemat


class PointCloud:
    def __init__(self, points, imageshape, P1, P2, R, t):
        """
        Generate a point cloud
        :param points: A 3xN homogeneous array of [X, Y, Z]^T points in 3D space
        :param imageshape: The (y,x) image shape from which the points were obtained
        :param P1: The projection matrix of the first frame camera
        :param P2: The projection matrix of the second frame camera
        :param R: The rotation mapping the first camera to the second, followed by translation t
        :param t: The translation mapping the first camera to the second after rotation R
        """

        # check dimensions
        assert points.shape == (3, np.product(imageshape))

        self.points = points
        self.imageshape = imageshape
        self.P1 = P1
        self.P2 = P2
        self.R = R
        self.t = t

    def get_shaped(self):
        """
        :return: A XxYx3 array of the points reshaped into self.imageshape
        """
        return np.dstack([
            self.points[0, :].reshape(self.imageshape),
            self.points[1, :].reshape(self.imageshape),
            self.points[2, :].reshape(self.imageshape)
        ])


def align_points_with_xy(points):
    """
    Applies rotation and translation to align points with the xy plane
    Ref: http://math.stackexchange.com/questions/1167717/transform-a-plane-to-the-xy-plane
    :param points: Points to align
    :return: Aligned points
    """

    # regular grid covering the domain of the data
    X, Y = np.meshgrid(np.arange(-0.5, 0.5, 0.05), np.arange(-0.5, 0.5, 0.05))
    XX = X.flatten()
    YY = Y.flatten()
    notnan_points = points[:, ~np.isnan(points[-1,:])]
    data = notnan_points.transpose()
    # best-fit linear plane
    A = np.c_[data[:, 0], data[:, 1], np.ones(data.shape[0])]
    C, _, _, _ = np.linalg.lstsq(A, data[:, 2])  # coefficients

    # evaluate it on grid
    Z = C[0] * X + C[1] * Y + C[2]

    # or expressed using matrix/vector product
    # Z = np.dot(np.c_[XX, YY, np.ones(XX.shape)], C).reshape(X.shape)

    centroid = np.mean(notnan_points, axis=1, keepdims=True)
    logging.info("Centroid: {}".format(centroid))
    points -= centroid

    cos_t = 1 / np.sqrt(C[0] ** 2 + C[1] ** 2 + 1)
    sin_t = np.sin(np.arccos(cos_t))
    ux = cos_t * -C[1]
    uy = cos_t * C[0]
    n = np.sqrt(ux ** 2 + uy ** 2)
    ux /= n
    uy /= n

    R = np.array([
        [cos_t + ux ** 2 * (1 - cos_t), ux * uy * (1 - cos_t), uy * sin_t],
        [ux * uy * (1 - cos_t), cos_t + uy ** 2 * (1 - cos_t), -ux * sin_t],
        [-uy * sin_t, ux * sin_t, cos_t]
    ])

    return R.dot(points)


def set_axes_equal(ax):
    """
    Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc..  This is one possible solution to Matplotlib's
    ax.set_aspect('equal') and ax.axis('equal') not working for 3D.

    Ref: http://stackoverflow.com/a/31364297

    :param ax: a matplotlib axis, e.g., as output from plt.gca().
    :return: None
    """

    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    # The plot bounding box is a sphere in the sense of the infinity
    # norm, hence I call half the max range the plot radius.
    plot_radius = 0.5*max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])


def visualise_heatmap(points, fname=None, detail=30, gsigma=0, mode='cutthru'):
    # detail = bins per unit
    pts = points[:, ~np.isnan(points[-1, :])]

    xmin, ymin, zmin = np.floor(np.min(pts, axis=1)).astype(int)
    xmax, ymax, zmax = np.ceil(np.max(pts, axis=1)).astype(int)
    logging.info("data shape: {}".format(pts.shape))
    logging.info("data min  : {}".format(np.min(pts, axis=1)))
    logging.info("data max  : {}".format(np.max(pts, axis=1)))

    xarr, yarr = np.arange(xmin, xmax, 1 / detail), np.arange(ymin, ymax, 1 / detail)
    X, Y = np.meshgrid(xarr, yarr)
    logging.info("X shape: {}".format(X.shape))
    logging.info("Y shape: {}".format(Y.shape))
    print("Interpolating Z")

    Z = -interpolate.griddata(np.vstack([pts[0, :], pts[1, :]]).T, pts[2, :].T,
                              np.vstack([X.flatten(), Y.flatten()]).T, method='linear'
                              ).reshape(X.shape)
    logging.info("Z shape: {}".format(Z.shape))

    if gsigma > 0:
        Z = ndimage.gaussian_filter(Z, sigma=gsigma, order=0)

    finshape = Z.shape
    logging.info("Final Z shape: {}".format(Z.shape))

    print("Rendering")

    # scale XYZ
    scale = 3.3
    X /= scale
    Y /= scale
    Z /= scale

    if mode == 'cutthru':
        # create a 2 X 2 grid
        # gs = grd.GridSpec(3, 2, height_ratios=[6, 1, 1], width_ratios=[10, 1], wspace=0.2)
        fig, axes = plt.subplots(3, 2, sharex='col', subplot_kw=dict(),
                                 gridspec_kw=dict(height_ratios=[4, 1, 1], width_ratios=[10, 1], wspace=0.2))

        # image plot
        ax = axes[0, 0]
        p = ax.imshow(Z, cmap='gray',
                      extent=(np.min(X), np.max(X), np.max(Y), np.min(Y)),
                      interpolation='nearest', aspect='equal', origin='upper')  # set the aspect ratio to auto to fill the space.
        # ax.set_xlabel('x [m]')
        ax.set_ylabel('y [m]')

        rowA = -500
        rowB = -200
        ax.plot((np.min(X), np.max(X)), (Y[rowA, 0], Y[rowA, -1]), 'b-')
        ax.plot((np.min(X), np.max(X)), (Y[rowB, 0], Y[rowB, -1]), 'r-')

        # color bar in it's own axis
        colorAx = axes[0, 1]
        cb = plt.colorbar(p, cax=colorAx)
        cb.set_label('Crop height deviation (z) [m]')

        # line plot
        ax2 = axes[1, 0]
        ax2.spines['right'].set_visible(False)
        ax2.spines['top'].set_visible(False)
        ax2.xaxis.set_ticks_position('bottom')
        ax2.yaxis.set_ticks_position('left')

        # ax2.set_aspect('auto')
        # ax2.set_xlabel('x [m]')
        ax2.set_ylabel('z [m]')
        ax2.set_xlim((np.min(X), np.max(X)))
        ax2.plot(X[rowA, :], Z[rowA, :], "b-")

        # line plot
        ax3 = axes[2, 0]
        ax3.spines['right'].set_visible(False)
        ax3.spines['top'].set_visible(False)
        ax3.xaxis.set_ticks_position('bottom')
        ax3.yaxis.set_ticks_position('left')

        ax3.set_xlabel('x [m]')
        ax3.set_ylabel('z [m]')
        ax3.set_xlim((np.min(X), np.max(X)))
        ax3.plot(X[rowB, :], Z[rowB, :], "r-")

        # hide unwanted
        axes[1, 1].axis('off')
        axes[2, 1].axis('off')

    else:
        fig = plt.figure()
        ax = fig.gca()
        p = plt.imshow(Z, cmap='gray',  # cmap='hot',
                   extent=(np.min(X), np.max(X), np.max(Y), np.min(Y)),
                   interpolation='nearest', aspect='equal', origin='upper')  # set the aspect ratio to auto to fill the space.
        ax.set_xlabel('x [m]')
        ax.set_ylabel('y [m]')
        cb = fig.colorbar(p)
        cb.set_label('Crop height deviation (z) [m]')

    if fname is not None:
        fname = '{}_gsigma{}'.format(fname, gsigma)
        fig.savefig('{}_mode{}.pdf'.format(fname, mode), dpi=1000)
        savemat('{}.mat'.format(fname), {
            'X': X,
            'Y': Y,
            'Z': Z
        })

    plt.show()
    return fig


def visualise_worlds_mplotlib(*worlds, method="surf", fname=None):
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.set_aspect('equal')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')

    if method == "surf":
        if len(worlds) == 1:
            shaped = worlds[0].get_shaped()
            X, Y, Z = shaped[:, :, 0], shaped[:, :, 1], shaped[:, :, 2]

            logging.info("Z range: {}, {}".format(np.nanmin(Z), np.nanmax(Z)))
            surf = ax.plot_surface(X, Y, Z, cmap=cm.hot, linewidth=0, antialiased=False,
                                   vmin=np.nanmin(Z), vmax=np.nanmax(Z))  # these limits seem to make it less
                                                                        # sharp, but are required to deal with NaNs
            surf.cmap.set_under('black')
            fig.colorbar(surf, extend='both')
        else:
            for i, world in enumerate(worlds):
                shaped = world.get_shaped()
                X, Y, Z = shaped[:, :, 0], shaped[:, :, 1], shaped[:, :, 2]
                surf = ax.plot_surface(X, Y, Z, linewidth=0, antialiased=False, rcount=10, ccount=10)#, color=('r','g','b','y')[i])
    else:
        # method == "scatter"
        # requires heavy graphics
        for world in worlds:
            X, Y, Z = world.points
            ax.scatter(X, Y, Z, linewidth=0, antialiased=False, marker="o")

    set_axes_equal(ax)
    if fname is not None:
        plt.savefig(fname)
    plt.show()
    return plt


def visualise_world_visvis(X, Y, Z, format="surf"):
    import visvis as vv

    # m2 = vv.surf(worldx[::detail], worldy[::detail], worldz[::detail])

    app = vv.use()
    # prepare axes
    a = vv.gca()
    a.cameraType = '3d'
    a.daspectAuto = False
    # print("view", a.camera.GetViewParams())
    # a.SetView(loc=(-1000,0,0))
    # a.camera.SetView(None, loc=(-1000,0,0))

    if format == "surf":
        l = vv.surf(X, Y, Z)
        a.SetLimits(rangeX=(-0.2, 0.2), rangeY=(-0.5, 0.5), rangeZ=(-0.5, 0), margin=0.02)
    else:
        # draw points
        pp = vv.Pointset(np.concatenate([X.flatten(), Y.flatten(), Z.flatten()], axis=0).reshape((-1, 3)))
        l = vv.plot(pp, ms='.', mc='r', mw='5', ls='', mew=0)
        l.alpha = 0.2
    app.Run()
