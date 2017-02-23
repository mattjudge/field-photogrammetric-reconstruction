import cv2

# constants found in
# https://github.com/opencv/opencv/blob/e3ae36dcb3c1d523802f8642e5c3984db43637c4/modules/python/src2/defs

vidcap = cv2.VideoCapture(r"../../../../../YUNC0001.mp4")


def get_frame_number(fnum):
    vidcap.set(1, fnum)
    success, frame = vidcap.read()
    if not success:
        raise LookupError("Error reading frame from video file")
    return frame



def save_frame_number(fnum):
    frame = get_frame_number(fnum)
    cv2.imwrite("./data/frame%d.png" % fnum, frame)
    print("written frame%d" % fnum)
    return frame


def get_frame_number_at_seconds(t):
    fps = vidcap.get(5)
    return t*fps


def save_frame_at_seconds(t):
    save_frame_number(get_frame_number_at_seconds(t))


if __name__ == "__main__":
    save_frame_number(9909)
    # save_frame_at_seconds(60*7+32)
    # save_frame_number(23752)
