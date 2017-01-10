#!/usr/bin/env python

import numpy as np
# from matplotlib import pyplot
import pylab
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import cv2
from scipy.optimize import minimize

import generate_registrations

#based on http://www.cns.nyu.edu/~david/handouts/motion.pdf


def get_plane_flow(x, y, Tx, Ty, Tz, mx, my, z0, f):
    # tilt = np.pi / 3
    # v = 5
    # % T = [0;
    # v * cos(tilt);
    # v * sin(tilt)];
    # T = np.array([[0], [10], [50]])
    Txy = np.array([[Tx], [Ty]])

    coeff = (1 - mx * x - my * y) / (f * z0)

    Ax = np.multiply(coeff, (np.dot(np.array([-f, 0]), Txy) + x * Tz))
    Ay = np.multiply(coeff, (np.dot(np.array([0, -f]), Txy) + y * Tz))

    # vzip = zip(x, y)
    #
    # print(list(vzip))

    return Ax, Ay


def get_err(args, tvxs, tvys, x, y):
    vxs, vys = get_plane_flow(x, y, *args)
    diff = np.linalg.norm(tvxs-vxs) + np.linalg.norm(tvys-vys)
    print(diff)
    return diff


def show_quiver(x, y, vxs, vys, step, sc, fname):
    pylab.figure()
    lengthy = x.shape[0]
    lengthx = x.shape[1]
    pylab.quiver(x[::step, ::step], y[::step, ::step],
           -sc * vxs[::step, ::step] * lengthx, -sc * vys[::step, ::step] * lengthy,
           color='b', angles='xy', scale_units='xy', scale=1)
    # pylab.quiver(x, y, vxs, vys)
    pylab.gca().invert_yaxis()
    pylab.savefig(fname)
    # pylab.show()


def show_plane(x, y, mx, my, z0, f):
    Z = f * z0 / (f - x*mx - y*my)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(x, y, Z, rstride=4, cstride=4, color='b')
    plt.savefig('test.png')
    plt.show()


def find_optimum(fname1, fname2, init=None):
    if init is None:
        #Tx, Ty, Tz, mx, my, z0, f
        # init = np.array([0, -10, -50, 0, 0.8, 1, 10])
        init = np.array([-1.67849264e-03, 2.25666383e-02, -9.56481633e-02,
                                               -3.83649454e-02, -4.53280430e-01, 1.39258114e+00, 9.90727195e+00])

    tvxs, tvys = generate_registrations.load_velocity_fields(fname1, fname2)

    sx = np.linspace(-1, 1, tvxs.shape[1])
    sy = np.linspace(-1, 1, tvxs.shape[0])
    x, y = np.meshgrid(sx, sy)

    res = minimize(get_err, init, (tvxs, tvys, x, y), tol=0.1)
    return res.x



if __name__ == '__main__':
    # get_plane_flow(1920, 990)
    print(find_optimum('frame23754', 'frame23756'))

    # tvxs, tvys = generate_registrations.load_velocity_fields('frame10700', 'frame10701')
    #
    # sx = np.linspace(-1, 1, tvxs.shape[1])
    # sy = np.linspace(-1, 1, tvxs.shape[0])
    # x, y = np.meshgrid(sx, sy)
    #
    # #Tx, Ty, Tz, mx, my, z0, f
    # args = np.array([ -4.71771865e-04,   7.49684144e-03,  -3.44752634e-02,
    #     -1.98693869e-02,  -4.98004919e-01,   1.43187857e+00,
    #      1.15023796e+01])
    #
    # vxs, vys = get_plane_flow(x, y, *args)
    #
    # X, Y = np.meshgrid(np.arange(tvxs.shape[1]), np.arange(tvxs.shape[0]))
    # show_quiver(X, Y, vxs, vys, 100, -20, 'velplane.png')
    # show_quiver(X, Y, tvxs, tvys, 100, -20, 'velreal.png')