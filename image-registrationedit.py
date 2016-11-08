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

def register_frames(f1, f2):
    # Take the DTCWT of both frames.

    cenx = 1166
    ceny = -597
    gxs, gys = np.zeros(f1.shape), np.zeros(f1.shape)
    predict_v_x, predict_v_y = np.zeros(f1.shape), np.zeros(f1.shape)
    for posx in range(f1.shape[1]):
        for posy in range(f1.shape[0]):
            # gxs[posy, posx] = -consx * (posx - cenx)
            # gys[posy, posx] = -consy * (posy - ceny)
            predict_v_x[posy, posx] = -2e-06 * (posx - cenx) + 0.0002
            predict_v_y[posy, posx] = -9e-06*posy - 0.0006
            gxs[posy, posx] = predict_v_x[posy, posx]*f1.shape[1] + posx
            gys[posy, posx] = predict_v_y[posy, posx]*f1.shape[0] + posy
    #cgxs, cgys = cv2.convertMaps(gxs, gys, cv2.CV_32FC1)
    gxs = gxs.astype('float32')
    gys = gys.astype('float32')
    impredict = cv2.remap(f1, gxs, gys, cv2.INTER_LINEAR)
    #cv2.imshow("window", )
    cv2.imwrite("predicted_frame.png", impredict)
    f1 = impredict
    #exit()
    #cv2.waitKey()
    # vxs -= gxs
    # vys -= gys
    #
    # vxs -= np.median(vxs.flat)
    # vys -= np.median(vys.flat)



    logging.info('Taking DTCWT')
    nlevels = 7
    trans = Transform2d()
    t1 = trans.forward(f1, nlevels=nlevels)
    t2 = trans.forward(f2, nlevels=nlevels)

    # Solve for transform
    logging.info('Finding flow')
    avecs = estimatereg(t1, t2)
    print(avecs.shape)
    #print(avecs)

    logging.info('Computing warped image')
    warped_f1 = warp(f1, avecs, method='bilinear')

    logging.info('Computing velocity field')
    X, Y = np.meshgrid(np.arange(f1.shape[1]), np.arange(f1.shape[0]))
    vxs, vys = velocityfield(avecs, f1.shape, method='nearest')
    print(f1.shape)

    # vxs -= gaussian_filter(vxs, sigma=50)
    # vys -= gaussian_filter(vys, sigma=50)

    # vxs -= np.median(vxs.flat)
    # vys -= np.median(vys.flat)

    # vxs -= np.mean(vxs, axis=0)
    # vys -= np.mean(vys, axis=0)
    # for h in range(0, f1.shape[0], 40):
    #     print(h - ceny, h, vys[h, cenx])
    # for h in range(0, f1.shape[1], 40):
    #     print(h - cenx, h, vxs[500, h])

    figure(figsize=(16,9))

    subplot(221)
    imshow(np.dstack((f1, f2, np.zeros_like(f1))))
    title('Overlaid frames')

    subplot(222)
    imshow(np.dstack((warped_f1, f2, np.zeros_like(f2))))
    title('Frame 1 warped to Frame 2 (image domain)')

    subplot(223)
    sc = 20
    step = 100
    imshow(np.dstack((f1, f2, np.zeros_like(f2))))
    quiver(X[::step,::step], Y[::step,::step],
           -sc*vxs[::step,::step]*f1.shape[1], -sc*vys[::step,::step]*f1.shape[0],
           color='b', angles='xy', scale_units='xy', scale=1)
    title('Computed velocity field, x{0}'.format(sc))

    subplot(224)
    #imshow(np.sqrt(vxs*vxs + vys*vys), interpolation='none', cmap=cm.hot)
    #vel = np.sqrt(vxs * vxs + vys * vys)
    vel = np.sqrt((f1.shape[1]*(vxs-predict_v_x))**2 + (f1.shape[0]*(vys-predict_v_y))**2)

    imshow(vel, interpolation='none', cmap=cm.hot)
    colorbar()
    title('Magnitude of velocity errors / px')

    show()
    savefig('__registration.png')


def register_images(fname1, fname2):
    im1 = cv2.imread(fname1)
    im2 = cv2.imread(fname2)
    f1 = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
    f2 = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)
    register_frames(f1, f2)


#register_images("tsukuba_L.jpg", "tsukuba_R.jpg")
#register_images("vlcsnap-2016-10-20-13h36m38s764.png", "vlcsnap-2016-10-20-13h38m27s793.png")
register_images("vlcsnap-2016-10-20-13h36m38s764cropped.png", "vlcsnap-2016-10-20-13h38m27s793cropped.png")
#register_images("warpf1crop.png", "warpf2crop.png")
