#!/usr/bin/env python

from __future__ import division, print_function

import logging

from matplotlib.pyplot import *
import numpy as np
import cv2

import pickle

from dtcwt.numpy import Transform2d
from dtcwt.registration import *

logging.basicConfig(level=logging.INFO)


def take_transform(frame):
    trans = Transform2d()
    return trans.forward(frame, nlevels=7)


def load_transform_frame(fname):
    logging.info('Taking transform')
    path = './data/transform_%s.pickle' % fname
    try:
        # tform = np.load(path)
        with open(path, 'rb') as fileob:
            tform = pickle.load(fileob)
    except IOError:
        img = cv2.imread('./data/%s.png' % fname)
        frame = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        tform = take_transform(frame)

        with open(path, 'wb') as fileob:
            pickle.dump(tform, fileob)
        # np.save(path, tform)
    return tform


def load_flow(fname1, fname2):
    logging.info('Finding flow')
    path = './data/flow_%s_%s.npy' % (fname1, fname2)
    try:
        flow = np.load(path)
    except IOError:
        flow = estimatereg(load_transform_frame(fname1), load_transform_frame(fname2))
        np.save(path, flow)
    return flow


def load_velocity_fields(fname1, fname2, shape=None):
    logging.info('Computing velocities')
    if shape is None:
        shape = cv2.imread('./data/%s.png' % fname1).shape[:2]
    path = './data/velocities_%s_%s.npy' % (fname1, fname2)
    try:
        vels = np.load(path)
    except IOError:
        vels = velocityfield(load_flow(fname1, fname2), shape, method='nearest')
        np.save(path, vels)
    return vels


if __name__ == '__main__':
    # estimatereg(load_transform_frame('frame10701'), load_transform_frame('frame10701'))
    # print(load_transform_frame('frame10700'))
    load_velocity_fields('frame10700', 'frame10701')
    load_velocity_fields('frame10700', 'frame10702')
    load_velocity_fields('frame10700', 'frame10703')