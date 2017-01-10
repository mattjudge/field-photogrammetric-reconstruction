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


def save_frame_at_seconds(t):
    save_frame_number(get_frame_number_at_seconds(t))


if __name__ == "__main__":
    # save_frame_number(9909)
    # save_frame_at_seconds(7*60+55)
    save_frame_number(23752)
    save_frame_number(23754)
    save_frame_number(23756)