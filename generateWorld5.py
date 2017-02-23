
import numpy as np
import cv2

import generate_registrations


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

    # d = 100
    # Fu1, Fv1, Fu2, Fv2 = u1[::d], v1[::d], u2[::d], v2[::d]

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
    # fku = 1700

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
    # 4 possible:  [R_1, t], [R_1, -t], [R_2, t], [R_2, -t]
    # R2b, R1b, tb = cv2.decomposeEssentialMat(E2)

    R1b, R2b, tb = cv2.decomposeEssentialMat(E2)
    # P1, P2 = getProjections(K, R1b, tb)
    # P1, P2 = getProjections(K, R1b, -tb)
    P1, P2 = getProjections(K, R2b, tb)  # appears correct
    # P1, P2 = getProjections(K, R2b, -tb)

    # P1 = K.dot(np.hstack([R1b, np.zeros((3, 1))]))
    # P2 = K.dot(np.hstack([R2b, t], axis=1))
    # print("P1b", P1b)
    # print("P2b", P2b)

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

    
def align_point_cloud_with_xy(cloud):
    # regular grid covering the domain of the data
    X, Y = np.meshgrid(np.arange(-0.5, 0.5, 0.05), np.arange(-0.5, 0.5, 0.05))
    XX = X.flatten()
    YY = Y.flatten()
    data = cloud.transpose()
    # best-fit linear plane
    A = np.c_[data[:, 0], data[:, 1], np.ones(data.shape[0])]
    C, _, _, _ = np.linalg.lstsq(A, data[:, 2])  # coefficients

    # evaluate it on grid
    Z = C[0] * X + C[1] * Y + C[2]

    # or expressed using matrix/vector product
    # Z = np.dot(np.c_[XX, YY, np.ones(XX.shape)], C).reshape(X.shape)

    centroid = np.mean(cloud, axis=1, keepdims=True)
    print("centroid", centroid)
    cloud -= centroid

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

    return R.dot(cloud)


def visualise_world_mplotlib(X, Y, Z):
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.pyplot as plt
    from matplotlib import cm
    from matplotlib.ticker import LinearLocator, FormatStrFormatter

    fig = plt.figure()
    ax = fig.gca(projection='3d')

    # Plot the surface.
    surf = ax.plot_surface(X, Y, Z, cmap=cm.hot,
                           linewidth=0, antialiased=False)

    ax.set_aspect('equal')
    # Customize the z axis.
    # ax.set_zlim(-1.01, 1.01)
    # ax.zaxis.set_major_locator(LinearLocator(10))
    # ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

    # Add a color bar which maps values to colors.
    # fig.colorbar(surf, shrink=0.5, aspect=5)
    fig.colorbar(surf)

    max_range = np.array([X.max() - X.min(), Y.max() - Y.min(), Z.max() - Z.min()]).max() / 2.0

    mid_x = (X.max() + X.min()) * 0.5
    mid_y = (Y.max() + Y.min()) * 0.5
    mid_z = (Z.max() + Z.min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    # ax.set_zlim(mid_z - max_range/50, mid_z + max_range/50)

    plt.show()


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


def generate_world(fname1, fname2, visual=True, crop=32):
    vel = generate_registrations.load_velocity_fields(fname1, fname2)
    world, shape = genWorld(vel)

    world = align_point_cloud_with_xy(world)

    X = world[0, :].reshape(shape)[crop:-crop:, crop:-crop:]
    Y = world[1, :].reshape(shape)[crop:-crop:, crop:-crop:]
    Z = world[2, :].reshape(shape)[crop:-crop:, crop:-crop:]

    if visual:
        # visualise_world_visvis(X, Y, Z)
        visualise_world_mplotlib(X, Y, Z)

    return world

    
    
if __name__ == "__main__":
    generate_world('frame9903', 'frame9906')
