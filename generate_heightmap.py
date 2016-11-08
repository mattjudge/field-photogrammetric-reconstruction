

import logging

import numpy as np
import pylab
import cv2

import dtcwt.registration
import planevel
import generate_registrations


logging.basicConfig(level=logging.INFO)


def predicted_to_actual_diff(fname1, fname2):
    """Generate predicted velocities from the modelled plane, and find
    the difference with the actual velocity field between the two frames"""
    tvxs, tvys = generate_registrations.load_velocity_fields('frame10700', 'frame10701')
    shape = tvxs.shape

    sx = np.linspace(-1, 1, shape[1])
    sy = np.linspace(-1, 1, shape[0])
    x, y = np.meshgrid(sx, sy)

    #Tx, Ty, Tz, mx, my, z0, f
    args = np.array([ -4.71771865e-04,   7.49684144e-03,  -3.44752634e-02,
        -1.98693869e-02,  -4.98004919e-01,   1.43187857e+00,
         1.15023796e+01])
    vxs, vys = planevel.get_plane_flow(x, y, *args)

    diffx = tvxs - vxs
    diffy = tvys - vys

    X, Y = np.meshgrid(np.arange(tvxs.shape[1]), np.arange(tvxs.shape[0]))
    planevel.show_quiver(X, Y, diffx, diffy, 100, -20, 'veldiff_field.png')

    pylab.figure()
    mags = np.sqrt((shape[1] * diffx) ** 2 + (shape[0] * diffy) ** 2)
    pylab.imshow(mags, interpolation='none', cmap=pylab.cm.hot)
    pylab.colorbar()
    pylab.title('Magnitude of velocity errors / px')
    pylab.savefig('veldiff_mag.png')

    pylab.show()


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

    vx -= np.median(vx.flat)
    vy -= np.median(vy.flat)

    planevel.show_quiver(X, Y, vx, vy, 100, -20, 'vel_predictedframediff_field.png')

    pylab.figure()
    mags = np.sqrt((shape[1] * vx) ** 2 + (shape[0] * vy) ** 2)
    pylab.imshow(mags, interpolation='none', cmap=pylab.cm.hot)
    pylab.colorbar()
    pylab.title('Magnitude of velocity errors / px')
    pylab.savefig('vel_predictedframediff_mag.png')

    pylab.show()

if __name__ == '__main__':
    # predicted_to_actual('frame10700', 'frame10701')
    predicted_to_actual("vlcsnap-2016-10-20-13h36m38s764cropped", "vlcsnap-2016-10-20-13h38m27s793cropped")
