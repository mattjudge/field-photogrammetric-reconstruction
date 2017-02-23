
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
    imagesize = vel.shape
    shapex = imagesize[2]
    shapey = imagesize[1]

    X, Y = np.meshgrid(np.arange(shapex), np.arange(shapey))

    # velocity vectors map pixels in f2 to their locations in f1
    u1shaped = X + vel[0]*shapex
    v1shaped = Y + vel[1]*shapey
    u2shaped, v2shaped = X, Y

    u1 = u1shaped.flatten()
    v1 = v1shaped.flatten()
    u2 = u2shaped.flatten()
    v2 = v2shaped.flatten()

    w1 = np.vstack([u1, v1])
    w2 = np.vstack([u2, v2])
    print("w1 as int", w1.astype(int))
    print("w2 as int", w2.astype(int))


    fku = 1883  # estimated
    # fku = 1700

    K = np.array([[fku,     0, shapex//2],
                  [0,     fku, shapey//2],
                  [0,       0,         1]])

    # E, mask = cv2.findEssentialMat(w1.transpose(), w2.transpose(), 2000, (shapey//2, shapex//2))
    E, mask = cv2.findEssentialMat(w1.transpose(), w2.transpose(), K)
    print("E", E)

    # R, t = getRandT(E)
    # P1, P2 = getProjections(K, R, t)
    # print("P1", P1)
    # print("P2", P2)

    # 4 possible:  [R_1, t], [R_1, -t], [R_2, t], [R_2, -t]
    # R2b, R1b, tb = cv2.decomposeEssentialMat(E2)

    R1, R2, t = cv2.decomposeEssentialMat(E)
    print("R1", R1)
    print("R2", R2)
    print("t", t)

    ## an attempt to choose the correct rotation matrix
    ## TODO: work out the maths behind it
    # we want R2 to be close to the identity matrix?
    if np.linalg.norm(np.eye(3) - np.abs(R1)) < np.linalg.norm(np.eye(3) - np.abs(R2)):
        R1, R2 = R2, R1

    # P1, P2 = getProjections(K, R1, t)
    # P1, P2 = getProjections(K, R1, -t)
    P1, P2 = getProjections(K, R2, t)  # appears correct?, light
    # P1, P2 = getProjections(K, R2, -t) # appears correct, dark


    print("triangulating vertices")
    finalshape = (shapey, shapex)
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


def generate_world_average(fnames):
    vels = np.array(list(map(lambda fnm: generate_registrations.load_velocity_fields(*fnm),
                        [(fnames[i], fnames[i+1]) for i in range(len(fnames)-1)])))

    print(vels.shape)

    nregs = vels.shape[0]
    imagesize = vels[0][0].shape
    shapex = imagesize[1]
    shapey = imagesize[0]

    X, Y = np.meshgrid(np.arange(shapex), np.arange(shapey))

    worldavg = np.zeros((3, shapey, shapex))
    cumreg = np.array([X, Y]).astype('float32')

    for vel in vels:
        # print(vel.shape)
        world, shape = genWorld(vel)

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


    crop = 50
    X = worldavg[0, :].reshape(shape)[crop:-crop:, crop:-crop:]
    Y = worldavg[1, :].reshape(shape)[crop:-crop:, crop:-crop:]
    Z = worldavg[2, :].reshape(shape)[crop:-crop:, crop:-crop:]
    croppedshape = X.shape

    flattenedworld = np.vstack([X.flatten(), Y.flatten(), Z.flatten()])
    worldavg = align_point_cloud_with_xy(flattenedworld)

    crop = None
    X = worldavg[0, :].reshape(croppedshape)#[crop:-crop:, crop:-crop:]
    Y = worldavg[1, :].reshape(croppedshape)#[crop:-crop:, crop:-crop:]
    Z = worldavg[2, :].reshape(croppedshape)#[crop:-crop:, crop:-crop:]

    visualise_world_mplotlib(X, Y, Z)

    return world


    
    
if __name__ == "__main__":
    # generate_world('frame9900', 'frame9903')
    # generate_world('frame9903', 'frame9906')
    # generate_world('frame9906', 'frame9909')
    # generate_world('frame9909', 'frame9912')
    # generate_world('frame9912', 'frame9915')
    # generate_world('frame9915', 'frame9918')
    # generate_world('frame9918', 'frame9921')
    # generate_world('frame9921', 'frame9924')
    # generate_world('frame9924', 'frame9927') # ok
    # generate_world('frame9927', 'frame9930')

    # generate_world_average(('frame9900', 'frame9903', 'frame9906', 'frame9909'))
    generate_world_average(('frame9900', 'frame9903', 'frame9906', 'frame9909', 'frame9912', 'frame9915'))
    # generate_world_average(('frame9900', 'frame9903', 'frame9906', 'frame9909', 'frame9912', 'frame9915', 'frame9918'))
