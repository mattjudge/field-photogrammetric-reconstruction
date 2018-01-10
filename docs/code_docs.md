## field_reconstruction


## field_reconstruction.video


Author: Matt Judge 2017

This module provides:
- class `Video`: as a wrapper over a :class:`cv2.VideoCapture` object
- class `Clip`: as a wrapper over a :class:`Video` object, with start and stop frame numbers

### field_reconstruction.video.Video

```python
Video(self, path)
```

Class to wrap over a :class:`cv2.VideoCapture` object

#### field_reconstruction.video.Video.get_frame_number

```python
Video.get_frame_number(self, fnum)
```

Get a frame from the video
- param `fnum`: The frame number
- return: frame: A BGR image in the form of a [X,Y,3] numpy array

#### field_reconstruction.video.Video.save_frame_number

```python
Video.save_frame_number(self, fnum, path)
```

Save a frame from the video to disk
- param `fnum`: The frame number
- param `path`: The directory in which to save the frame
- return: frame: The saved frame as a [X,Y,3] numpy array

#### field_reconstruction.video.Video.get_frame_number_at_seconds

```python
Video.get_frame_number_at_seconds(self, t)
```

Get the frame number at a time in the video
- param `t`: Time in seconds
- return: Frame number

#### field_reconstruction.video.Video.save_frame_at_seconds

```python
Video.save_frame_at_seconds(self, t, path)
```

Save a frame from a time in the video
- param `t`: Time in seconds
- param `path`: Directory in which to save the frame
- return: frame: The saved frame as a [X,Y,3] numpy array

### field_reconstruction.video.Clip

```python
Clip(self, video, startframe=0, stopframe=None)
```

A wrapper around a :class:`Video` object, with start and stop frame numbers

#### field_reconstruction.video.Clip.get_frame_number

```python
Clip.get_frame_number(self, frame_number)
```

Gets a frame from the clip, relative to the clip's starting frame
- param `frame_number`: The frame number in the clip to obtain
- return: frame: An BGR image in the form of a [X,Y,3] numpy array

## field_reconstruction.reconstruction


Author: Matt Judge 2017

This module provides:
- func `render_reconstruct_world`: as a helper function to reconstruct and render a clip of video
- func `reconstruct_world`: to create a 3D reconstruction of a clip of video
- func `reconstruct_frame_pair`: to triangulate and reconstruct two frames
- func `generate_world_cloud`: to generate a dense point cloud from a video, utilising a moving average
- func `gen_binned_points`: to bin a dense point cloud and average over reliable bins
- func `get_outlier_mask`: to determine outliers in a dense point cloud
- func `generate_frame_pair_cloud`: to create an instance of :class:`pointcloud.Pointcloud` from two frames
- func `triangulate_frames`: to generate a point cloud from two frames
- func `estimate_projections`: to estimate projection matrices (P1, P2, R, T) from pixel correspondences and camera matrix
- func `create_pixel_correspondences`: to create pixel correspondences from relative motion velocities
- func `get_projections_from_rt`: to get projection matrices from R and T

    And legacy functions:
- func `get_fundamental`: to get the Fundamental matrix from corresponding pixel positions
- func `get_rt`: to get rotation R and translation T matrices from the essential matrix E

### field_reconstruction.reconstruction.get_fundamental

```python
get_fundamental(u1, v1, u2, v2)
```
Legacy function to get the Fundamental matrix from corresponding pixel positions
### field_reconstruction.reconstruction.get_rt

```python
get_rt(E)
```
Legacy function to get rotation R and translation T matrices from the essential matrix E
### field_reconstruction.reconstruction.get_projections_from_rt

```python
get_projections_from_rt(K, R, t)
```

Get projection matrices from R and T
- param `K`: [3,3] Camera calibration matrix
- param `R`: [3,3] Rotation matrix
- param `t`: [3,1] Translation matrix
- return: P1, P2, projection matrices, both [3,4]

### field_reconstruction.reconstruction.create_pixel_correspondences

```python
create_pixel_correspondences(vel)
```

Create pixel correspondences from relative motion velocities
- param `vel`: Motion velocities from :func:`dtcwt_registration.load_velocity_fields`, a [2,Y,X] array
- return: tuple of two pixel correspondences, each a [Y,X] array corresponding to one frame

### field_reconstruction.reconstruction.estimate_projections

```python
estimate_projections(correspondences, K)
```

Estimate the projection matrices given point correspondences and the camera calibration matrix K
- param `correspondences`: Tuple of two frame correspondences (each [X,Y] matrices). Should be
    pre-cropped to ensure good data points (otherwise E becomes unstable)
- param `K`: [3,3] Camera calibration matrix
- return:P1, P2, R, t Camera projection matrices:
    P1: Projection to frame 1
    P2: Projection to frame 2
    R: Rotation from frame 1 to frame 2
    t: Translation from frame 1 to frame 2

### field_reconstruction.reconstruction.triangulate_frames

```python
triangulate_frames(vid, frame_pair, K)
```

Perform point triangulation from two frames of a video
- param `vid`: :class:video.Video object from which to take the frames
- param `frame_pair`: Tuple of two frame numbers (frame1, frame2)
- param `K`: [3,3] Camera calibration matrix
  :returns: points, velocities, P1, P2, R, t
  WHERE
  - points are a [3, N] numpy array point cloud
  - velocities are the velocities returned by the dtcwt transform
    as a [2, Y, X] numpy array (see :func:`dtcwt_registration.load_velocity_fields`)
  - P1, P2, R, t are the projection matrix parameters returned by :func:`estimate_projections`)

### field_reconstruction.reconstruction.generate_frame_pair_cloud

```python
generate_frame_pair_cloud(vid, frame_pair, K)
```

Generates an instance of :class:`pointcloud.Pointcloud` from a pair of frames of a :class:`video.Video`.
- param `vid`: :class:video.Video object from which to take the frames
- param `frame_pair`: Tuple of two frame numbers (frame1, frame2)
- param `K`: [3,3] Camera calibration matrix
- return: pointcloud, velocities
  WHERE
  - pointcloud is an instance of :class:`pointcloud.Pointcloud`
  - velocities are the velocities returned by the dtcwt transform
    as a [2, Y, X] numpy array (see :func:`dtcwt_registration.load_velocity_fields`)

### field_reconstruction.reconstruction.get_outlier_mask

```python
get_outlier_mask(points, percentile_discard)
```

Generate a mask identifying outliers in a point cloud
- param `points`: A [3, N] numpy array of points
- param `percentile_discard`: The percentile to discard symmetrically (i.e. a :param:percentile_discard of 5 discards points which fall into the first or last 1% of the data in the x, y, or z dimensions.
- return: outlier_mask, a [N,] boolean numpy array, where True values correspond to an outlying point

### field_reconstruction.reconstruction.gen_binned_points

```python
gen_binned_points(points, detail=50, minpointcount=4)
```

Bin points from a point cloud, ignoring outliers
- param `points`: A [3, N] numpy array of points
- param `detail`: The bins per point cloud unit
- param `minpointcount`: Minimum number of points in a bin considered to generate a reliable mean
- return: binned_points, a [3,N] numpy array of the binned points

### field_reconstruction.reconstruction.generate_world_cloud

```python
generate_world_cloud(vid, K, fnums, avg_size=5)
```

Generate a point cloud from multiple video frames
- param `vid`: :class:video.Video object from which to take the frames
- param `K`: [3,3] Camera calibration matrix
- param `fnums`: An iterable of frame numbers from which to create the point cloud
- param `avg_size`: The number of frame-pairs to combine in a moving average
- return: points, a [3,N] numpy array of the triangulated points computed from the video

### field_reconstruction.reconstruction.reconstruct_frame_pair

```python
reconstruct_frame_pair(vid, K, f0, f1)
```

Legacy function to compute a reconstruction of the world from just two frames
- param `vid`: input video
- param `K`: camera calibration matrix
- param `f0`: First frame number
- param `f1`: Second frame number

### field_reconstruction.reconstruction.reconstruct_world

```python
reconstruct_world(clip, K, frame_step, include_intermediates=False, multiproc=True)
```

Generate a filtered, averaged point cloud from a video.
- param `clip`: An instance of :class:`video.Clip` as the input video
- param `K`: [3,3] The camera calibration matrix
- param `frame_step`: The fixed step between frame numbers from which to determine pairs of frames to triangulate
- param `include_intermediates`: Boolean. When enabled, all frames are included in the reconstruction.
    To illustrate, with a frame step of 3:
        Disabled: result = function(f0->f3, f3->f6, f6->f9, etc)  # a single 'frame train'
        Enabled:  result = function(f0->f3, f1->f4, f2->f5, f3->f6, f4->f7, f5->f8, etc)  # a multi 'frame train'
- param `multiproc`: Boolean. When enabled, reconstruction of each frame train is processed in parallel
- return: points, a [3,N] numpy array of x,y,z point coordinates of the reconstructed world

### field_reconstruction.reconstruction.render_reconstruct_world

```python
render_reconstruct_world(clip, K, frame_step, path=None, include_intermediates=False, multiproc=True, render_mode='standard', render_scale=1, render_gsigma=0)
```

A helper wrapper function. **See documentation for :func:`reconstruction.reconstruct_world` and :func:`pointcloud.visualise_heatmap` for detailed parameter documentation.**
- param `clip`: Input video clip
- param `K`: Camera calibration matrix
- param `frame_step`: Frame step
- param `path`: Output path
- param `include_intermediates`: Boolean to enable computation of mutiple frame trains
- param `multiproc`: Boolean to enable parallel frame train processing
- param `render_mode`: Mode to render with
- param `render_scale`: Scale to render with
- param `render_gsigma`: Level of Gaussian smoothing
- return: matplotlib figure

## field_reconstruction.dtcwt_registration


Author: Matt Judge 2017

This module provides:
- func `take_transform`: to take the DTCWT transform of a frame from a video
- func `load_flow`: to load the registration which maps frame 1 to frame 2 of a video
- func `load_velocity_fields`: to load the velocity fields mapping frame 1 to frame 2 of a video

### field_reconstruction.dtcwt_registration.take_transform

```python
take_transform(vid, fnum)
```

Takes the DTCWT transform of a frame from a video
- param `vid`: The video from which to take the frame, an instance of :class:`video.Video`
- param `fnum`: The frame number to transform
- return: transform, the transformed frame (A :class:`dtcwt.Pyramid` compatible object
    representing the transform-domain signal)

### field_reconstruction.dtcwt_registration.load_flow

```python
load_flow(vid, fnum1, fnum2)
```

Load the registration which maps frame 1 to frame 2 of a video.
- param `vid`: The video from which to take the frames, an instance of :class:`video.Video`
- param `fnum1`: First frame number
- param `fnum2`: Second frame number
- return: The DTCWT affine distortion parameters, a [N,M,6] array

### field_reconstruction.dtcwt_registration.load_velocity_fields

```python
load_velocity_fields(vid, fnum1, fnum2)
```

Load the velocity fields mapping frame 1 to frame 2 of a video
- param `vid`: The video from which to take the frames, an instance of :class:`video.Video`
- param `fnum1`: First frame number
- param `fnum2`: Second frame number
- return: The velocity field, a [2,Y,X] array

## field_reconstruction.pointcloud


Author: Matt Judge 2017, except `set_axes_equal`

This module provides:
- class `Pointcloud`: as a container for point clouds and associated projection matrices
- func `align_points_with_xy`: to align point clouds on the XY plane
- func `visualise_heatmap`: to interpolate and render pre-binned point clouds

### field_reconstruction.pointcloud.align_points_with_xy

```python
align_points_with_xy(points)
```

Applies rotation and translation to align point cloud with the xy plane
Maths Ref: http://math.stackexchange.com/questions/1167717/transform-a-plane-to-the-xy-plane
- param `points`: [3,N] numpy array of points to align
- return: [3,N] numpy array of aligned points

### field_reconstruction.pointcloud.set_axes_equal

```python
set_axes_equal(ax)
```

Make sure axes of 3D plot have equal scale so that spheres appear as spheres,
cubes as cubes, etc..  This is one possible solution to Matplotlib's
ax.set_aspect('equal') and ax.axis('equal') not working for 3D.

Ref: http://stackoverflow.com/a/31364297

- param `ax`: a matplotlib axis, e.g., as output from plt.gca().
- return: None

### field_reconstruction.pointcloud.visualise_heatmap

```python
visualise_heatmap(points, path=None, detail=30, gsigma=0, scale=1, mode='standard')
```

Interpolates a point cloud into a regular grid, rendering a heatmap and optionally saving as .png and .mat files.
The .mat file can be further processed by external tools into a surface plot.
- param `points`: A [3,N] numpy array of points
- param `path`: The path in which to save output files. No files will be saved if set to None.
- param `detail`: The detail with which to interpolate the point cloud
- param `gsigma`: The level of gaussian smoothing to apply (default to 0, no smoothing)
- param `scale`: Scale to apply to the 3 axis, defaults to 1
- param `mode`: Either 'standard' or 'cutthru'.
  - 'standard': Render a standard heatmap
  - 'cutthru': Include two cross sectional lines
- return:

