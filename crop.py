import cv2


def crop(fname, tl, shape):
    #shape is x, y
    # tl is x, y coord of top left corner
    img = cv2.imread('./data/%s.png' % fname)
    imgcrop = img[tl[1]:tl[1]+shape[1], tl[0]:tl[0]+shape[0],:]
    cv2.imwrite('./data/%s_cropped.png' % fname, imgcrop)


if __name__ == '__main__':
    tl = (800,400)
    shape = (512, 256)
    crop('frame10700', tl, shape)
    crop('frame10701', tl, shape)
    crop('frame10702', tl, shape)
    crop('frame10703', tl, shape)
