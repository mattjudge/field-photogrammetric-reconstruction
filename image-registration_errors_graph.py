#!/usr/bin/env python
"""
An example of image registration via the DTCWT.

This script demonstrates some methods for image registration using the DTCWT.

"""

from __future__ import division, print_function

import itertools
import logging
import os

from matplotlib.pyplot import *
import numpy as np
import cv2
from scipy.ndimage.filters import gaussian_filter


import dtcwt
from dtcwt.numpy import Transform2d
import dtcwt.sampling
from dtcwt.registration import *

logging.basicConfig(level=logging.INFO)


def geterror(f1, pixx, pixy):
    dimy, dimx = f1.shape[:2]
    shiftx, shifty = np.ones(f1.shape), np.ones(f1.shape)
    shiftx *= -pixx
    shifty *= -pixy
    posx = np.array(range(dimx)).reshape(1,dimx).repeat(dimy,0)
    posy = np.array(range(dimy)).reshape(dimy,1).repeat(dimx, 1)
    #print(posx.shape, posy.shape)
    shiftxtot = (shiftx + posx).astype('float32')
    shiftytot = (shifty + posy).astype('float32')
    f2 = cv2.remap(f1, shiftxtot, shiftytot, cv2.INTER_LINEAR)
    #cv2.imshow("window", f2)
    #cv2.waitKey()
    logging.info('Taking DTCWT')
    nlevels = 7
    trans = Transform2d()
    t1 = trans.forward(f1, nlevels=nlevels)
    t2 = trans.forward(f2, nlevels=nlevels)

    # Solve for transform
    logging.info('Finding flow')
    avecs = estimatereg(t1, t2)
    print(avecs.shape)
    # print(avecs)

    logging.info('Computing velocity field')
    X, Y = np.meshgrid(np.arange(f1.shape[1]), np.arange(f1.shape[0]))
    vxs, vys = velocityfield(avecs, f1.shape, method='nearest')
    print(f1.shape)

    figure(figsize=(16, 9))
    subplot(223)
    sc = 1
    step = 100
    imshow(np.dstack((f1, f2, np.zeros_like(f2))))
    quiver(X[::step, ::step], Y[::step, ::step],
           -sc * vxs[::step, ::step] * f1.shape[1], -sc * vys[::step, ::step] * f1.shape[0],
           color='b', angles='xy', scale_units='xy', scale=1)
    title('Computed velocity field, x{0}'.format(sc))

    subplot(224)
    # imshow(np.sqrt(vxs*vxs + vys*vys), interpolation='none', cmap=cm.hot)
    # vel = np.sqrt(vxs * vxs + vys * vys)
    vel = np.sqrt((f1.shape[1] * vxs - shiftx) ** 2 + (f1.shape[0] * vys - shifty) ** 2)
    vel = vel[100:-100,100:-100]
    imshow(vel, interpolation='none', cmap=cm.hot)
    colorbar()
    title('Magnitude of velocity errors / px')

    savefig('__registration.png')


def register_images(fname1):
    im1 = cv2.imread(fname1)
    f1 = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
    geterror(f1, 0, 45)

register_images("vlcsnap-2016-10-20-13h36m38s764cropped.png")
