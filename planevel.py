#!/usr/bin/env python

import numpy as np
# from matplotlib import pyplot
import pylab
import cv2
from scipy.optimize import minimize

import generate_registrations



def get_plane_flow(x, y, Tx, Ty, Tz, mx, my, z0, f):
    # tilt = np.pi / 3
    # v = 5
    # % T = [0;
    # v * cos(tilt);
    # v * sin(tilt)];
    # T = np.array([[0], [10], [50]])
    Txy = np.array([[Tx], [Ty]])

    coeff = (1 - mx * x - my * y) / (f * z0)

    vxs = np.multiply(coeff, (np.dot(np.array([-f, 0]), Txy) + x * Tz))
    vys = np.multiply(coeff, (np.dot(np.array([0, -f]), Txy) + y * Tz))

    return vxs, vys


def get_err(args, tvxs, tvys, x, y):
    vxs, vys = get_plane_flow(x, y, *args)
    diff = np.linalg.norm(tvxs-vxs) + np.linalg.norm(tvys-vys)
    print(diff)
    return diff


def show_quiver(x, y, vxs, vys, step, sc, fname):
    pylab.figure()
    lengthx = x.shape[1]
    lengthy = x.shape[0]
    pylab.quiver(x[::step, ::step], y[::step, ::step],
           -sc * vxs[::step, ::step] * lengthy, -sc * vys[::step, ::step] * lengthx,
           color='b', angles='xy', scale_units='xy', scale=1)
    # pylab.quiver(x, y, vxs, vys)
    pylab.show()
    pylab.savefig(fname)


def find_optimum():
    tvxs, tvys = generate_registrations.load_velocity_fields('frame10700', 'frame10701')

    sx = np.linspace(-1, 1, tvxs.shape[1])
    sy = np.linspace(-1, 1, tvxs.shape[0])
    x, y = np.meshgrid(sx, sy)

    #Tx, Ty, Tz, mx, my, z0, f
    init = np.array([0, -10, -50, 0, 0.8, 1, 10])

    res = minimize(get_err, init, (tvxs, tvys, x, y), tol=0.1)
    print(res)



if __name__ == '__main__':
    # get_plane_flow(1920, 990)
    # find_optimum()

    tvxs, tvys = generate_registrations.load_velocity_fields('frame10700', 'frame10701')

    sx = np.linspace(-1, 1, tvxs.shape[1])
    sy = np.linspace(-1, 1, tvxs.shape[0])
    x, y = np.meshgrid(sx, sy)

    #Tx, Ty, Tz, mx, my, z0, f
    args = np.array([ -4.71771865e-04,   7.49684144e-03,  -3.44752634e-02,
        -1.98693869e-02,  -4.98004919e-01,   1.43187857e+00,
         1.15023796e+01])

    vxs, vys = get_plane_flow(x, y, *args)

    X, Y = np.meshgrid(np.arange(tvxs.shape[1]), np.arange(tvxs.shape[0]))
    show_quiver(X, Y, vxs, vys, 100, -20, 'velplane.png')
    show_quiver(X, Y, tvxs, tvys, 100, -20, 'velreal.png')