#!/usr/bin/env python

import numpy as np
from field_reconstruction import video, reconstruction


def clip_heatmap():
    # logging.root.setLevel(logging.DEBUG)
    # vid = video.Video(r"video1.mp4")
    vid = video.Video(r"video2.mp4")
    clip = video.Clip(vid, startframe=0, stopframe=298)

    print("Loaded video {fname}, shape: {shape}, fps: {fps}, start: {start}, stop: {stop}".format(
        fname=clip.video.path, shape=clip.video.shape, fps=clip.video.fps,
        start=clip.start_frame, stop=clip.stop_frame))

    # estimated from tractor treads in video
    camera_proj_mat = np.array(
        [[1883, 0, 910],
         [0, 1883, 490],
         [0, 0, 1]]
    )

    reconstruction.render_reconstruct_world(
        clip, camera_proj_mat, frame_step=3, path='./output/', include_intermediates=False, multiproc=False,
        render_mode='standard', render_scale=3.3, render_gsigma=1
    )

    print("Done.")


if __name__ == '__main__':
    clip_heatmap()
