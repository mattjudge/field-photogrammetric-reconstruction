import numpy as np

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import matplotlib.pyplot as plt


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
        self.points = points
        self.imageshape = imageshape
        self.P1 = P1
        self.P2 = P2
        self.R = R
        self.t = t

    def get_shaped(self):
        """
        :return: A 3xYxX array of the points reshaped into self.imageshape
        """
        return np.stack([
            self.points[0, :].reshape(self.imageshape),
            self.points[1, :].reshape(self.imageshape),
            self.points[2, :].reshape(self.imageshape)
        ], axis=0)


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
    data = points.transpose()
    # best-fit linear plane
    A = np.c_[data[:, 0], data[:, 1], np.ones(data.shape[0])]
    C, _, _, _ = np.linalg.lstsq(A, data[:, 2])  # coefficients

    # evaluate it on grid
    Z = C[0] * X + C[1] * Y + C[2]

    # or expressed using matrix/vector product
    # Z = np.dot(np.c_[XX, YY, np.ones(XX.shape)], C).reshape(X.shape)

    centroid = np.mean(points, axis=1, keepdims=True)
    print("centroid", centroid)
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
    '''Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc..  This is one possible solution to Matplotlib's
    ax.set_aspect('equal') and ax.axis('equal') not working for 3D.

    Ref: http://stackoverflow.com/a/31364297

    Input
      ax: a matplotlib axis, e.g., as output from plt.gca().
    '''

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


def visualise_worlds_mplotlib(*worlds, method="surf"):
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.set_aspect('equal')

    if method == "surf":
        if len(worlds) == 1:
            X, Y, Z = worlds[0].get_shaped()
            surf = ax.plot_surface(X, Y, Z, cmap=cm.hot,
                                   linewidth=0, antialiased=False)
            fig.colorbar(surf)
        else:
            for i, world in enumerate(worlds):
                X, Y, Z = world.get_shaped()
                surf = ax.plot_surface(X, Y, Z, linewidth=0, antialiased=False) #, rcount=10, ccount=10, color=('r','g','b','y')[i])
    else:
        # method == "scatter"
        # requires heavy graphics
        for world in worlds:
            X, Y, Z = world.points
            ax.scatter(X, Y, Z, linewidth=0, antialiased=False, marker="o")

    set_axes_equal(ax)
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
