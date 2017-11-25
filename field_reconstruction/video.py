import logging
import os.path
import cv2


class Video:
    def __init__(self, path):
        self.path = path
        self.name = os.path.splitext(os.path.basename(self.path))[0]
        self.vidcap = cv2.VideoCapture()
        if not self.vidcap.open(self.path):
            raise IOError(None, "Error opening video", self.path)
        self.fps = self.vidcap.get(cv2.CAP_PROP_FPS)
        self.shape = (
            int(self.vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            int(self.vidcap.get(cv2.CAP_PROP_FRAME_WIDTH))
        )
        self.frame_count = self.vidcap.get(cv2.CAP_PROP_FRAME_COUNT)

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.path

    def get_frame_number(self, fnum):
        self.vidcap.set(cv2.CAP_PROP_POS_FRAMES, fnum)
        success, frame = self.vidcap.read()
        if not success:
            raise LookupError("Error reading frame from video file")
        return frame

    def save_frame_number(self, fnum, path):
        frame = self.get_frame_number(fnum)
        fname = os.path.join(path, "{}_frame_{}.png".format(self.name, fnum))
        cv2.imwrite(fname, frame)
        logging.debug("written", fname)
        return frame

    def get_frame_number_at_seconds(self, t):
        return t * self.fps

    def save_frame_at_seconds(self, t):
        return self.save_frame_number(self.get_frame_number_at_seconds(t))


class Clip:
    def __init__(self, video, startframe=0, stopframe=None):
        self.video = video
        self.start_frame = startframe
        if stopframe is None:
            self.stop_frame = self.video.frame_count
        elif stopframe > self.video.frame_count:
            raise IndexError("Clip must be within video length (stopframe value too large)")
        self.stop_frame = stopframe  # stopframe is NOT included in the clip
        self.frame_count = self.stop_frame - self.start_frame

    def get_frame_number(self, frame_number):
        if not -self.stop_frame <= frame_number < self.stop_frame:
            raise IndexError("Frame number must be within the clip length")
        elif frame_number < 0:
            frame_number += self.stop_frame
        else:
            frame_number += self.start_frame
        return self.video.get_frame_number(frame_number)


if __name__ == "__main__":
    pass
