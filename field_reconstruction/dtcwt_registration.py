
"""
Author: Matt Judge 2017

This module provides:
    :func:`take_transform` to take the DTCWT transform of a frame from a video
    :func:`load_flow` to load the registration which maps frame 1 to frame 2 of a video
    :func:`load_velocity_fields` to load the velocity fields mapping frame 1 to frame 2 of a video
"""

import logging

import cv2
import dtcwt.registration as reg
import numpy as np
from dtcwt.numpy import Transform2d

from field_reconstruction.numpy_caching import np_cache

_transform2d = Transform2d()


# @np_cache(True, hash_method='readable')
def take_transform(vid, fnum):
    """
    Takes the DTCWT transform of a frame from a video
    :param vid: The video from which to take the frame, an instance of :class:`video.Video`
    :param fnum: The frame number to transform
    :return: transform, the transformed frame (A :class:`dtcwt.Pyramid` compatible object
        representing the transform-domain signal)
    """
    logging.debug('Taking transform of frame {}'.format(fnum))
    frame = cv2.cvtColor(vid.get_frame_number(fnum), cv2.COLOR_BGR2GRAY)
    return _transform2d.forward(frame, nlevels=7)


@np_cache(False)
def load_flow(vid, fnum1, fnum2):
    """
    Load the registration which maps frame 1 to frame 2 of a video.
    :param vid: The video from which to take the frames, an instance of :class:`video.Video`
    :param fnum1: First frame number
    :param fnum2: Second frame number
    :return: The DTCWT affine distortion parameters, a [N,M,6] array
    """
    logging.debug('Finding flow of frames {} & {}'.format(fnum1, fnum2))
    return reg.estimatereg(take_transform(vid, fnum1), take_transform(vid, fnum2))


@np_cache(True, hash_method='readable')
def load_velocity_fields(vid, fnum1, fnum2):
    """
    Load the velocity fields mapping frame 1 to frame 2 of a video
    :param vid: The video from which to take the frames, an instance of :class:`video.Video`
    :param fnum1: First frame number
    :param fnum2: Second frame number
    :return: The velocity field, a [2,Y,X] array
    """
    # todo: don't upscale the flow to the full image size, since the velocity appears in blocks
    logging.debug('Computing velocities of frames {} & {}'.format(fnum1, fnum2))
    return np.array(reg.velocityfield(load_flow(vid, fnum1, fnum2), vid.shape, method='nearest'))


if __name__ == '__main__':
    pass
