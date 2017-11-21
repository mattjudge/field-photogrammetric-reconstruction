#!/usr/bin/env python

import logging

import numpy as np
import cv2
import pickle

from dtcwt.numpy import Transform2d
import dtcwt.registration as reg


def take_transform(frame):
    trans = Transform2d()
    return trans.forward(frame, nlevels=7)


def load_frame(vid, fnum):
    path = './data/{}.png'.format(fnum)
    img = cv2.imread(path)
    if img is None:
        # generate frame from video
        img = vid.get_frame_number(fnum)
        # used to cache frame here
    return img


def load_transform_frame(vid, fnum, cache=False):
    logging.debug('Taking transform of frame {}'.format(fnum))
    path = './data/transform_{}.pickle'.format(fnum)
    try:
        # tform = np.load(path)
        with open(path, 'rb') as fileob:
            tform = pickle.load(fileob)
    except (IOError, FileNotFoundError) as e:
        frame = cv2.cvtColor(load_frame(vid, fnum), cv2.COLOR_BGR2GRAY)
        tform = take_transform(frame)
        if cache:
            with open(path, 'wb') as fileob:
                pickle.dump(tform, fileob)
                # np.save(path, tform)
    return tform


def load_flow(vid, fnum1, fnum2, cache=True):
    logging.debug('Finding flow of frames {} & {}'.format(fnum1, fnum2))
    path = './data/flow_{}_{}.npy'.format(fnum1, fnum2)
    try:
        flow = np.load(path)
    except (IOError, FileNotFoundError) as e:
        flow = reg.estimatereg(load_transform_frame(vid, fnum1), load_transform_frame(vid, fnum2))
        if cache:
            np.save(path, flow)
    return flow


def load_velocity_fields(vid, fnum1, fnum2, cache=False):
    # cache is off by default, because of large file sizes, and low computation expense
    logging.debug('Computing velocities of frames {} & {}'.format(fnum1, fnum2))
    path = './data/velocities_{}_{}.npy'.format(fnum1, fnum2)
    try:
        vels = np.load(path)
    except (IOError, FileNotFoundError) as e:
        vels = np.array(reg.velocityfield(load_flow(vid, fnum1, fnum2), vid.shape, method='nearest'))
        if cache:
            np.save(path, vels)
    return vels


if __name__ == '__main__':
    pass
