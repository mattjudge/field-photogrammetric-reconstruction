import cv2


class Video:
    def __init__(self, fname=None):
        self.fname = fname
        self.vidcap = None
        self.fps = None
        self.shape = None
        if self.fname is not None:
            self.open(self.fname)

    def open(self, fname):
        self.fname = fname
        self.vidcap = cv2.VideoCapture(self.fname)
        self.fps = self.vidcap.get(cv2.CAP_PROP_FPS)
        self.shape = (
            int(self.vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            int(self.vidcap.get(cv2.CAP_PROP_FRAME_WIDTH))
        )

    def get_frame_number(self, fnum):
        self.vidcap.set(cv2.CAP_PROP_POS_FRAMES, fnum)
        success, frame = self.vidcap.read()
        if not success:
            raise LookupError("Error reading frame from video file")
        return frame

    def save_frame_number(self, fnum):
        frame = self.get_frame_number(fnum)
        path = "./data/frame{}.png".format(fnum, frame)
        cv2.imwrite(path, frame)
        print("written", path)
        return frame

    def get_frame_number_at_seconds(self, t):
        return t * self.fps

    def save_frame_at_seconds(self, t):
        self.save_frame_number(self.get_frame_number_at_seconds(t))


if __name__ == "__main__":
    vid = Video(r"../../../../../YUNC0001.mp4")
    print(vid.fname)
    print(vid.shape)
    print(vid.fps)
    # vid.save_frame_number(9909)
    # save_frame_at_seconds(60*7+32)
