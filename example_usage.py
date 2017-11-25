#!/usr/bin/env python

import numpy as np
from field_reconstruction import video, reconstruction


def clip_heatmap():
    # logging.root.setLevel(logging.DEBUG)
    vid = video.Video(r"../../../../../YUNC0001.mp4")
    clip = video.Clip(vid, 26400, 26460)
    # 9900, 9920
    # 20750, 20850
    # 20750, 20800
    # 9900, 10100
    # 9900, 9930
    # 26400, 26460
    # 10101, 10160
    # 26400, 26500
    # 31302, 31600
    # 31590, 31900
    # 13100, 13200
    # 14550, 14610
    # 15000, 15200

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
        clip, camera_proj_mat, frame_step=3, path='./output/', include_intermediates=True, multiproc=True,
        render_mode='standard', render_scale=3.3, render_gsigma=1
    )

    print("Done.")


if __name__ == '__main__':
    clip_heatmap()
