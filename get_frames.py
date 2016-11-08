import cv2

# constants found in
# https://github.com/opencv/opencv/blob/e3ae36dcb3c1d523802f8642e5c3984db43637c4/modules/python/src2/defs

vidcap = cv2.VideoCapture(r"C:\Users\mattc\OneDrive\Documents\YUNC0001.mp4")


def save_frame_number(fnum):
    vidcap.set(1, fnum)

    success, frame = vidcap.read()
    cv2.imwrite("./data/frame%d.png" % fnum, frame)
    print("written frame%d" % fnum)


def get_frame_number_at_seconds(t):
    fps = vidcap.get(5)
    return t*fps


if __name__ == "__main__":
    save_frame_number(10700)
    save_frame_number(10701)
    save_frame_number(10702)
    save_frame_number(10703)
    #get_frame_number_at_seconds(214)
