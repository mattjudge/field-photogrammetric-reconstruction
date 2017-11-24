import logging

import numpy as np
import cv2

from dtcwt.numpy import Transform2d
import dtcwt.registration as reg

from caching import cache_numpy_result

_transform2d = Transform2d()


def take_transform(vid, fnum):
    logging.debug('Taking transform of frame {}'.format(fnum))
    frame = cv2.cvtColor(vid.get_frame_number(fnum), cv2.COLOR_BGR2GRAY)
    return _transform2d.forward(frame, nlevels=7)


@cache_numpy_result(False)
def load_flow(vid, fnum1, fnum2):
    logging.debug('Finding flow of frames {} & {}'.format(fnum1, fnum2))
    return reg.estimatereg(take_transform(vid, fnum1), take_transform(vid, fnum2))


@cache_numpy_result(True, hash_method='readable')
def load_velocity_fields(vid, fnum1, fnum2):
    logging.debug('Computing velocities of frames {} & {}'.format(fnum1, fnum2))
    return np.array(reg.velocityfield(load_flow(vid, fnum1, fnum2), vid.shape, method='nearest'))


if __name__ == '__main__':
    pass
