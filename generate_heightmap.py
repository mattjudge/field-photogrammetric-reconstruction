
import logging

import numpy as np
import matplotlib
# matplotlib.use('Agg')
import pylab
import matplotlib.pyplot as plt
import matplotlib.animation
from mpl_toolkits.mplot3d import Axes3D
import cv2
import visvis

import dtcwt.registration
import planevel
import generate_registrations


logging.basicConfig(level=logging.INFO)


def averagemany_predicted_to_actual_diff(fnames, planeargs):
    savenm = '_averagemany_%s_%s.png' % (fnames[0], fnames[-1])

    # tvxs1, tvys1
    realregs = np.array(list(map(lambda fnm: generate_registrations.load_velocity_fields(*fnm),
                        [(fnames[i], fnames[i+1]) for i in range(len(fnames)-1)])))

    shape = realregs[0][0].shape

    sx = np.linspace(-1, 1, shape[1])
    sy = np.linspace(-1, 1, shape[0])
    x, y = np.meshgrid(sx, sy)

    # vxs2, vys2
    planeregs = np.array(list(map(lambda pa: planevel.get_plane_flow(x, y, *pa), planeargs)))
    X, Y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))

    diffs = planeregs - realregs
    diffs /= np.abs(planeregs)
    realregs[:, 0, :, :] *= shape[1]
    realregs[:, 1, :, :] *= shape[0]

    cumreg = np.array([X, Y]).astype('float32')  #np.zeros_like(realregs[0])
    # print(cumreg.shape, realregs[-1].shape)
    avg = diffs[-1]
    for i in range(-1, -len(realregs), -1):  # for each vel
        cumreg += realregs[i]
        dfx = cv2.remap(diffs[i-1][0], cumreg[0].astype('float32'), cumreg[1].astype('float32'), cv2.INTER_LINEAR)
        dfy = cv2.remap(diffs[i-1][1], cumreg[0].astype('float32'), cumreg[1].astype('float32'), cv2.INTER_LINEAR)
        avg += np.array([dfx, dfy])
        print(np.linalg.norm(avg), np.linalg.norm(cumreg))

    diffx = avg[0] * shape[1] / len(realregs)
    diffy = avg[1] * shape[0] / len(realregs)


    # diffx = np.divide(diffx, np.abs(planeregs[0][0]))
    # diffy = np.divide(diffy, np.abs(planeregs[0][1]))
    print('yo')
    print(np.linalg.norm(diffx))
    print(np.linalg.norm(planeregs[0][0]))
    # diffx /= np.abs(planeregs[0][0]) * shape[1]
    # diffy /= np.abs(planeregs[0][1]) * shape[0]
    print(np.linalg.norm(diffx))
    fname1 = fnames[0]


    img = cv2.imread('./data/%s.png' % fname1)
    frame = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ###
    # Tx, Ty, Tz, mx, my, z0, f
    # planevel.show_plane(x, y, *planeargs12[3:])
    planevel.show_quiver(X, Y, planeregs[0][0], planeregs[0][1], 100, 5, '_out_plane.png')
    planevel.show_quiver(X, Y, realregs[0][0]/shape[1], realregs[0][1]/shape[0], 100, 5, '_out_real.png')
    # planevel.show_quiver(X, Y, diffx[0], realregs[0][1], 100, 20, '_out_real.png')

    pylab.figure()
    pylab.imshow(frame[50:-50, 50:-50])#, interpolation='none', cmap=pylab.cm.hot, vmin=-4, vmax=4)
    pylab.colorbar()
    pylab.title('frame %s' % fname1)
    pylab.savefig('frame_%s' % savenm)


    pylab.figure()
    pylab.imshow(diffx[50:-50, 50:-50], interpolation='none', cmap=pylab.cm.hot)#, vmin=-4, vmax=4)
    pylab.colorbar()
    pylab.title('x velocity errors / px')
    pylab.savefig('veldiff_x_%s' % savenm)

    pylab.figure()
    pylab.imshow(diffy[50:-50, 50:-50], interpolation='none', cmap=pylab.cm.hot)#, vmin=-4, vmax=4)
    pylab.colorbar()
    pylab.title('y velocity errors / px')
    pylab.savefig('veldiff_y_%s' % savenm)

    print(np.min(diffx))
    # diffx += np.min(diffx)
    # diffx += 150
    # diffy += np.min(diffy)
    # diffy = np.sqrt(diffx*diffx + diffy*diffy)

    # map onto world coords
    # imx, imy = np.meshgrid(np.arange(-1, 1), np.arange(-1, 1))
    Tx, Ty, Tz, mx, my, z0, f = planeargs[-1]
    wx = X*f*z0 / (f - -x*my - -y*my)# - 100
    wy = Y*f*z0 / (f - -x*my - -y*my)# - 100
    diffy = cv2.remap(diffy, wx.astype('float32'), wy.astype('float32'), cv2.INTER_LINEAR)

    pylab.figure()
    pylab.imshow(diffy[50:-50, 50:-50], interpolation='none', cmap=pylab.cm.hot)#, vmin=-4, vmax=4)
    pylab.colorbar()
    pylab.title('velocity errors / px')
    pylab.savefig('veldiff_m_%s' % savenm)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    # ax.plot(frame[50:-50, 50:-50], interpolation='none', cmap=pylab.cm.hot)
    # ax.plot(diffy[50:-50, 50:-50], interpolation='none', cmap=pylab.cm.hot, alpha=0.6)
    im = plt.imshow(diffy[50:-50, 50:-50], interpolation='none', cmap=pylab.cm.hot)
    def init_animation():
        # ax.title('y velocity errors / px')
        # ax.savefig('veldiff_y.png')
        return im,

    def animate(i):
        print('frame %i' % i)
        if i == 1:
            cax = ax.imshow(frame[50:-50, 50:-50])#, interpolation='none', cmap=pylab.cm.hot)
        else:
            cax = ax.imshow(diffy[50:-50, 50:-50], interpolation='none', cmap=pylab.cm.hot)#, alpha=0.6)
            # fig.colorbar(cax)
        return cax,

    # ani = matplotlib.animation.FuncAnimation(fig, animate, frames=2, interval=50, save_count=2)#init_func=init_animation,
    # plt.show()

    # pylab.show()
    # pylab.draw(fig)

    # Writer = matplotlib.animation.writers['ffmpeg']
    # Writer = matplotlib.animation.writers['ImageMagick']
    # Writer = matplotlib.animation.ImageMagickWriter
    # writer = Writer(fps=2, metadata=dict(artist='Me'), bitrate=1800)
    # ani.save('./animation.gif', writer=writer)
    # ani.save('./animation.mp4', writer='imagemagick')
    # ani.save('./animation.gif')


    # fig = plt.figure()
    # ax = fig.add_subplot(1, 1, 1, projection='3d')
    # ax.plot_surface(X, Y, diffy)
    # ax.plot_trisurf(X.flatten(), Y.flatten(), diffy.flatten())
    print('rendering')
    # xi = np.linspace(0, shape[1])
    # yi = np.linspace(0, shape[0])
    # X, Y = np.meshgrid(xi, yi)
    Z = diffy/80

    # Tx, Ty, Tz, mx, my, z0, f = planeargs[0]
    # wx = X*f*z0 / (f - X*my - Y*my)
    # wy = Y*f*z0 / (f - X*my - Y*my)
    # Z = cv2.remap(Z, wx.astype('float32'), wy.astype('float32'), cv2.INTER_LINEAR)


    # ax.plot_surface(X, Y, Z, rstride=10, cstride=10, cmap=matplotlib.cm.jet, linewidth=1)#, antialiased=True)
    # plt.show()
    import visvis
    f = visvis.gca()
    # m = visvis.grid(xi, yi, Z)
    f.daspect = 1, -1, 30  # z x 30
    # draped colors
    print(X.shape, Y.shape, Z.shape)
    # m = visvis.surf(X[50:-50, 50:-50], Y[50:-50, 50:-50], Z[50:-50, 50:-50], cv2.cvtColor(img, cv2.COLOR_BGR2RGB)[50:-50, 50:-50, :])
    m2 = visvis.surf(X[50:-50, 50:-50], Y[50:-50, 50:-50], Z[50:-50, 50:-50])
    m2.colormap = visvis.CM_HOT
    plt.waitforbuttonpress()

    # pylab.figure()
    # mags = np.sqrt(diffx ** 2 + diffy ** 2)
    # mags = mags[50:-50, 50:-50]
    # pylab.imshow(mags, interpolation='none', cmap=pylab.cm.hot)
    # pylab.colorbar()
    # pylab.title('Magnitude of velocity errors / px')
    # pylab.savefig('veldiff_mag.png')
    #
    # pylab.show()


def average_predicted_to_actual_diff(fname1, fname2, fname3, planeargs12, planeargs23):
    savenm = '_average_%s_%s.png' % (fname1, fname2)
    tvxs1, tvys1 = generate_registrations.load_velocity_fields(fname1, fname2)
    tvxs2, tvys2 = generate_registrations.load_velocity_fields(fname2, fname3)
    shape = tvxs1.shape

    sx = np.linspace(-1, 1, shape[1])
    sy = np.linspace(-1, 1, shape[0])
    x, y = np.meshgrid(sx, sy)

    vxs1, vys1 = planevel.get_plane_flow(x, y, *planeargs12)
    vxs2, vys2 = planevel.get_plane_flow(x, y, *planeargs23)
    X, Y = np.meshgrid(np.arange(tvxs1.shape[1]), np.arange(tvxs1.shape[0]))
    diffx1 = vxs1 - tvxs1
    diffy1 = vys1 - tvys1
    diffx2 = vxs2 - tvxs2
    diffy2 = vys2 - tvys2

    # map frame 1 velocities onto frame 2
    diffx1mapper = (np.multiply(tvxs1, shape[1]) + X).astype('float32')
    diffy1mapper = (np.multiply(tvys1, shape[0]) + Y).astype('float32')

    diffx1mapped = cv2.remap(diffx1, diffx1mapper, diffy1mapper, cv2.INTER_LINEAR)
    diffy1mapped = cv2.remap(diffy1, diffx1mapper, diffy1mapper, cv2.INTER_LINEAR)

    diffx = np.multiply(0.5, diffx1mapped + diffx2)
    diffy = np.multiply(0.5, diffy1mapped + diffy2)

    diffx *= shape[1]
    diffy *= shape[0]


    img = cv2.imread('./data/%s.png' % fname1)
    frame = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    pylab.figure()
    pylab.imshow(frame[50:-50, 50:-50])#, interpolation='none', cmap=pylab.cm.hot, vmin=-4, vmax=4)
    pylab.colorbar()
    pylab.title('frame %s' % fname1)
    pylab.savefig('frame_%s' % savenm)


    pylab.figure()
    pylab.imshow(diffx[50:-50, 50:-50], interpolation='none', cmap=pylab.cm.hot, vmin=-4, vmax=4)
    pylab.colorbar()
    pylab.title('x velocity errors / px')
    pylab.savefig('veldiff_x_%s' % savenm)

    pylab.figure()
    pylab.imshow(diffy[50:-50, 50:-50], interpolation='none', cmap=pylab.cm.hot, vmin=-4, vmax=4)
    pylab.colorbar()
    pylab.title('y velocity errors / px')
    pylab.savefig('veldiff_y_%s' % savenm)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    # ax.plot(frame[50:-50, 50:-50], interpolation='none', cmap=pylab.cm.hot)
    # ax.plot(diffy[50:-50, 50:-50], interpolation='none', cmap=pylab.cm.hot, alpha=0.6)
    im = plt.imshow(diffy[50:-50, 50:-50], interpolation='none', cmap=pylab.cm.hot)
    def init_animation():
        # ax.title('y velocity errors / px')
        # ax.savefig('veldiff_y.png')
        return im,

    def animate(i):
        print('frame %i' % i)
        if i == 1:
            cax = ax.imshow(frame[50:-50, 50:-50])#, interpolation='none', cmap=pylab.cm.hot)
        else:
            cax = ax.imshow(diffy[50:-50, 50:-50], interpolation='none', cmap=pylab.cm.hot)#, alpha=0.6)
            # fig.colorbar(cax)
        return cax,

    # ani = matplotlib.animation.FuncAnimation(fig, animate, frames=2, interval=50, save_count=2)#init_func=init_animation,
    # plt.show()

    # pylab.show()
    # pylab.draw(fig)

    Writer = matplotlib.animation.writers['ffmpeg']
    # Writer = matplotlib.animation.writers['ImageMagick']
    # Writer = matplotlib.animation.ImageMagickWriter
    writer = Writer(fps=2, metadata=dict(artist='Me'), bitrate=1800)
    # ani.save('./animation.gif', writer=writer)
    # ani.save('./animation.mp4', writer='imagemagick')
    # ani.save('./animation.gif')


    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    # ax.plot_surface(X, Y, diffy)
    # ax.plot_trisurf(X.flatten(), Y.flatten(), diffy.flatten())
    print('rendering')
    # xi = np.linspace(0, shape[1])
    # yi = np.linspace(0, shape[0])
    # X, Y = np.meshgrid(xi, yi)
    Z = diffy

    # ax.plot_surface(X, Y, Z, rstride=10, cstride=10, cmap=matplotlib.cm.jet, linewidth=1)#, antialiased=True)
    # plt.show()
    import visvis
    f = visvis.gca()
    # m = visvis.grid(xi, yi, Z)
    f.daspect = 1, -1, 30  # z x 10
    # draped colors
    print(X.shape, Y.shape, Z.shape)
    # m = visvis.surf(X[50:-50, 50:-50], Y[50:-50, 50:-50], Z[50:-50, 50:-50], cv2.cvtColor(img, cv2.COLOR_BGR2RGB)[50:-50, 50:-50, :])
    m2 = visvis.surf(X[50:-50, 50:-50], Y[50:-50, 50:-50], Z[50:-50, 50:-50])
    m2.colormap = visvis.CM_HOT
    plt.waitforbuttonpress()

    # pylab.figure()
    # mags = np.sqrt(diffx ** 2 + diffy ** 2)
    # mags = mags[50:-50, 50:-50]
    # pylab.imshow(mags, interpolation='none', cmap=pylab.cm.hot)
    # pylab.colorbar()
    # pylab.title('Magnitude of velocity errors / px')
    # pylab.savefig('veldiff_mag.png')
    #
    # pylab.show()

def predicted_to_actual_diff(fname1, fname2, planeargs=None):
    """Generate predicted velocities from the modelled plane, and find
    the difference with the actual velocity field between the two frames"""
    if planeargs is None:
        # Tx, Ty, Tz, mx, my, z0, f
        # args = np.array([ -4.71771865e-04,   7.49684144e-03,  -3.44752634e-02,
        #     -1.98693869e-02,  -4.98004919e-01,   1.43187857e+00,
        #      1.15023796e+01])
        initplaneargs = np.array([-1.67849264e-03, 2.25666383e-02, -9.56481633e-02,
                         -3.83649454e-02, -4.53280430e-01, 1.39258114e+00,
                         9.90727195e+00])
        planeargs = planevel.find_optimum(fname1, fname2, initplaneargs)
        print('plane args', planeargs)


    savenm = '%s_%s.png' % (fname1, fname2)
    tvxs, tvys = generate_registrations.load_velocity_fields(fname1, fname2)
    shape = tvxs.shape

    sx = np.linspace(-1, 1, shape[1])
    sy = np.linspace(-1, 1, shape[0])
    x, y = np.meshgrid(sx, sy)

    vxs, vys = planevel.get_plane_flow(x, y, *planeargs)
    X, Y = np.meshgrid(np.arange(tvxs.shape[1]), np.arange(tvxs.shape[0]))
    planevel.show_quiver(X, Y, vxs, vys, 50, 1, 'vel_predicted_plane_%s' % savenm)
    planevel.show_quiver(X, Y, tvxs, tvys, 50, 1, 'vel_actual_%s' % savenm)

    diffx = vxs - tvxs
    diffy = vys - tvys

    planevel.show_quiver(X, Y, diffx, diffy, 50, 1, 'veldiff_field_%s' % savenm)

    diffx *= shape[1]
    diffy *= shape[0]


    img = cv2.imread('./data/%s.png' % fname1)
    frame = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    pylab.figure()
    pylab.imshow(frame[50:-50, 50:-50])#, interpolation='none', cmap=pylab.cm.hot, vmin=-4, vmax=4)
    pylab.colorbar()
    pylab.title('frame %s' % fname1)
    pylab.savefig('frame_%s' % savenm)


    pylab.figure()
    pylab.imshow(diffx[50:-50, 50:-50], interpolation='none', cmap=pylab.cm.hot, vmin=-4, vmax=4)
    pylab.colorbar()
    pylab.title('x velocity errors / px')
    pylab.savefig('veldiff_x_%s' % savenm)

    pylab.figure()
    pylab.imshow(diffy[50:-50, 50:-50], interpolation='none', cmap=pylab.cm.hot, vmin=-4, vmax=4)
    pylab.colorbar()
    pylab.title('y velocity errors / px')
    pylab.savefig('veldiff_y_%s' % savenm)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    # ax.plot(frame[50:-50, 50:-50], interpolation='none', cmap=pylab.cm.hot)
    # ax.plot(diffy[50:-50, 50:-50], interpolation='none', cmap=pylab.cm.hot, alpha=0.6)
    im = plt.imshow(diffy[50:-50, 50:-50], interpolation='none', cmap=pylab.cm.hot)
    def init_animation():
        # ax.title('y velocity errors / px')
        # ax.savefig('veldiff_y.png')
        return im,

    def animate(i):
        print('frame %i' % i)
        if i == 1:
            cax = ax.imshow(frame[50:-50, 50:-50])#, interpolation='none', cmap=pylab.cm.hot)
        else:
            cax = ax.imshow(diffy[50:-50, 50:-50], interpolation='none', cmap=pylab.cm.hot)#, alpha=0.6)
            # fig.colorbar(cax)
        return cax,

    # ani = matplotlib.animation.FuncAnimation(fig, animate, frames=2, interval=50, save_count=2)#init_func=init_animation,
    # plt.show()

    # pylab.show()
    # pylab.draw(fig)

    Writer = matplotlib.animation.writers['ffmpeg']
    # Writer = matplotlib.animation.writers['ImageMagick']
    # Writer = matplotlib.animation.ImageMagickWriter
    writer = Writer(fps=2, metadata=dict(artist='Me'), bitrate=1800)
    # ani.save('./animation.gif', writer=writer)
    # ani.save('./animation.mp4', writer='imagemagick')
    # ani.save('./animation.gif')


    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    # ax.plot_surface(X, Y, diffy)
    # ax.plot_trisurf(X.flatten(), Y.flatten(), diffy.flatten())
    print('rendering')
    # xi = np.linspace(0, shape[1])
    # yi = np.linspace(0, shape[0])
    # X, Y = np.meshgrid(xi, yi)
    Z = diffy

    # ax.plot_surface(X, Y, Z, rstride=10, cstride=10, cmap=matplotlib.cm.jet, linewidth=1)#, antialiased=True)
    # plt.show()
    import visvis
    f = visvis.gca()
    # m = visvis.grid(xi, yi, Z)
    f.daspect = 1, -1, 30  # z x 10
    # draped colors
    print(X.shape, Y.shape, Z.shape)
    # m = visvis.surf(X[50:-50, 50:-50], Y[50:-50, 50:-50], Z[50:-50, 50:-50], cv2.cvtColor(img, cv2.COLOR_BGR2RGB)[50:-50, 50:-50, :])
    m2 = visvis.surf(X[50:-50, 50:-50], Y[50:-50, 50:-50], Z[50:-50, 50:-50])
    m2.colormap = visvis.CM_HOT
    plt.waitforbuttonpress()

    # pylab.figure()
    # mags = np.sqrt(diffx ** 2 + diffy ** 2)
    # mags = mags[50:-50, 50:-50]
    # pylab.imshow(mags, interpolation='none', cmap=pylab.cm.hot)
    # pylab.colorbar()
    # pylab.title('Magnitude of velocity errors / px')
    # pylab.savefig('veldiff_mag.png')
    #
    # pylab.show()


def predicted_to_actual(fname1, fname2):
    """Generate a predicted frame using predicted plane velocities and find the velocities
    between the predicted frame and the actual frame"""
    f1 = cv2.cvtColor(cv2.imread('./data/%s.png' % fname1), cv2.COLOR_BGR2GRAY)
    shape = f1.shape[:2]

    sx = np.linspace(-1, 1, shape[1])
    sy = np.linspace(-1, 1, shape[0])
    x, y = np.meshgrid(sx, sy)

    #Tx, Ty, Tz, mx, my, z0, f
    args = np.array([ -4.71771865e-04,   7.49684144e-03,  -3.44752634e-02,
        -1.98693869e-02,  -4.98004919e-01,   1.43187857e+00,
         1.15023796e+01])
    vpredx, vpredy = planevel.get_plane_flow(x, y, *args)

    X, Y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))

    # vpredx = np.multiply(np.multiply(vpredx, 3), X) + X
    # vpredy = np.multiply(np.multiply(vpredy, 3), Y) + Y
    vpredx = np.multiply(vpredx, X) + X
    vpredy = np.multiply(vpredy, Y) + Y
    vpredx = vpredx.astype('float32')
    vpredy = vpredy.astype('float32')

    fpredict = cv2.remap(f1, vpredx, vpredy, cv2.INTER_LINEAR)
    cv2.imwrite("new_predicted_frame.png", fpredict)

    t1 = generate_registrations.take_transform(fpredict)
    t2 = generate_registrations.load_transform_frame(fname2)
    avecs = dtcwt.registration.estimatereg(t1, t2)
    vx, vy = dtcwt.registration.velocityfield(avecs, shape, method='nearest')

    # vx -= np.median(vx.flat)
    # vy -= np.median(vy.flat)

    planevel.show_quiver(X, Y, vx, vy, 100, 20, 'vel_predictedframediff_field.png')

    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    mags = np.sqrt((shape[1] * vx) ** 2 + (shape[0] * vy) ** 2)
    mags = mags[50:-50, 50:-50]
    # ax.plot_surface(X, Y, mags)
    pylab.imshow(mags, interpolation='none', cmap=pylab.cm.hot)
    pylab.colorbar()
    pylab.title('Magnitude of velocity errors / px')
    pylab.savefig('vel_predictedframediff_mag.png')

    pylab.show()


def gen_actual(fname1, fname2):
    tvxs, tvys = generate_registrations.load_velocity_fields(fname1, fname2)
    shape = tvxs.shape
    mags = np.sqrt((shape[1] * tvxs) ** 2 + (shape[0] * tvys) ** 2)
    # ax.plot_surface(X, Y, mags)
    pylab.imshow(mags, interpolation='none', cmap=pylab.cm.hot)
    pylab.colorbar()
    pylab.title('Magnitude of velocity / px')
    pylab.savefig('cropped_actual.png')


def new_algo(fname1, fname2):

    savenm = '%s_%s_newalgo.png' % (fname1, fname2)
    tvxs, tvys = generate_registrations.load_velocity_fields(fname1, fname2)
    shape = tvxs.shape

    sx = np.linspace(-1, 1, shape[1])
    sy = np.linspace(-1, 1, shape[0])
    x, y = np.meshgrid(sx, sy)
    X, Y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))

    vxs = tvxs * shape[1]
    vys = tvys * shape[0]

    m = 0.2  # distance drone has moved (scaling factor)
    # theta1 = np.deg2rad(40)
    # theta2 = np.deg2rad(82.7)
    theta1 = np.deg2rad(40)
    theta2 = np.deg2rad(72)
    aov = theta2 - theta1
    # theta1 = 0  # offset from vertical to innermost viewing angle
    # aov = 2.71  # angle of view
    radppx = aov / shape[0]  # radians per pixel in vertical plane
    #angles = np.linspace(aov,0,shape[0]).reshape(-1,1).repeat(shape[1], axis=1)  # angle of each row
    # angles = Y[::-1,::] * radppx
    r = 0.5 * shape[0] / np.tan(0.5 * aov)
    print(Y[:shape[0]//2 + 1])
    # ang = np.arctan(Y[::2]/(2*r))
    # angles = np.concatenate((aov/2 + ang[::-1], aov/2 - ang[:]))
    # angles += theta1
    angles = theta1 + aov/2 + np.arctan((2*Y[::-1] - shape[0])/(2*r))
    print(angles)
    print("angles shape", angles.shape)
    print(angles)

    shiftedangles = cv2.remap(angles, X.astype('float32'), (Y-vys).astype('float32'), cv2.INTER_LINEAR)
    # shiftedangles = (vys + Y) * radppx
    # shiftedangles = angles + (vys * radppx)
    # shiftedangles = angles[(Y-vys).astype('int'), X.astype('int')]

    e1 =  angles
    e2 = shiftedangles
    # e1, e2 = e2, e1
    d = -m * np.cos(e1) * np.cos(e2) / (np.cos(e1)*np.sin(e2) - np.sin(e1)*np.cos(e2))
    d = -d  # calc'd dist is from drone (downwards direction)
    p = -m * np.sin(e1) * np.cos(e2) / (np.cos(e1)*np.sin(e2) - np.sin(e1)*np.cos(e2))
    # d = angles
    # print(Y)
    # print(angles)
    # print(newangles)
    print(np.sum(angles))
    print(np.sum(shiftedangles))
    print(np.sum(vys))
    print(d.shape)
    print(np.linalg.norm(angles))
    print(np.linalg.norm(shiftedangles))
    print(np.linalg.norm(vys))

    img = cv2.imread('./data/%s.png' % fname1)
    frame = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    pylab.figure()
    pylab.imshow(frame[50:-50, 50:-50])#, interpolation='none', cmap=pylab.cm.hot, vmin=-4, vmax=4)
    pylab.colorbar()
    pylab.title('frame %s' % fname1)
    pylab.savefig('frame_%s' % savenm)


    pylab.figure()
    pylab.imshow(d[50:-50, 50:-50], interpolation='none')#, cmap=pylab.cm.hot, vmin=-4, vmax=4)
    pylab.colorbar()
    pylab.title('y velocity errors / px')
    pylab.savefig('veldiff_y_%s' % savenm)


    pylab.figure()
    pylab.imshow(p[50:-50, 50:-50], interpolation='none')#, cmap=pylab.cm.hot, vmin=-4, vmax=4)
    pylab.colorbar()
    pylab.title('y position / px')
    pylab.savefig('p_%s' % savenm)

    Z = d
    f = visvis.gca()
    # m = visvis.grid(xi, yi, Z)
    f.daspect = 1, -1, 30  # z x 10
    # draped colors
    print(X.shape, Y.shape, Z.shape)
    # m = visvis.surf(X[50:-50, 50:-50], Y[50:-50, 50:-50], Z[50:-50, 50:-50], cv2.cvtColor(img, cv2.COLOR_BGR2RGB)[50:-50, 50:-50, :])
    m2 = visvis.surf(X[50:-50, 50:-50], Y[50:-50, 50:-50], Z[50:-50, 50:-50])
    # m2 = visvis.surf(X[300:-100, 100:-100], Y[300:-100, 100:-100], Z[300:-100, 100:-100])
    m2.colormap = visvis.CM_HOT
    plt.waitforbuttonpress()




if __name__ == '__main__':
    # predicted_to_actual_diff('frame9900', 'frame9903', np.array([-1.67849264e-03, 2.25666383e-02, -9.56481633e-02,
    #                      -3.83649454e-02, -4.53280430e-01, 1.39258114e+00, 9.90727195e+00]))
    # predicted_to_actual_diff('frame9903', 'frame9906', np.array([ -1.48934135e-03,   2.22083172e-02,  -9.56233987e-02,
    #                         -3.12542802e-02, -4.51714788e-01,   1.38886661e+00,   9.90725322e+00]))
    #   predicted_to_actual_diff('frame9906', 'frame9909', np.array([ -1.27780976e-03,   2.28267637e-02,  -9.80714738e-02,  -2.98015928e-02,
    # -4.56325230e-01,   1.42068716e+00,   9.90484345e+00]))
    # averagemany_predicted_to_actual_diff(('frame9900', 'frame9903', 'frame9906', 'frame9909'),
    #                                      (np.array([-1.67849264e-03, 2.25666383e-02, -9.56481633e-02,
    #                                                 -3.83649454e-02, -4.53280430e-01, 1.39258114e+00, 9.90727195e+00]),
    #                                       np.array([-1.48934135e-03, 2.22083172e-02, -9.56233987e-02,
    #                                                 -3.12542802e-02, -4.51714788e-01,   1.38886661e+00,   9.90725322e+00]),
    #                                       np.array([-1.27780976e-03, 2.28267637e-02, -9.80714738e-02, -2.98015928e-02,
    #                                                 -4.56325230e-01, 1.42068716e+00, 9.90484345e+00])))
    #   averagemany_predicted_to_actual_diff(('frame23750', 'frame23752', 'frame23754', 'frame23756'),
    #                                        (np.array([  1.15666718e-03,   2.14093518e-02,  -8.24336339e-02,   1.31248279e-02,
    # -4.63319026e-01,   1.60261989e+00,   9.87242643e+00]),
    #                                         np.array([  1.33177705e-03,   2.27207625e-02,  -8.84550958e-02,   2.49825491e-02,
    # -4.52572001e-01,   1.73230194e+00,   9.90191487e+00]),
    #                                         np.array([  1.46361732e-03,   2.11674410e-02,  -8.20001910e-02,   8.16957726e-03,
    # -4.77802249e-01,   1.61163646e+00,   9.87439840e+00])))
    new_algo('frame9900', 'frame9903')