# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Utilities for reading from the DeepView dataset and computing a PSV."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import math
import os

import numpy as np
from PIL import Image
import tensorflow as tf
import transformations


class Camera(object):
  """Represents a Camera with intrinsics and world from/to camera transforms.

  Attributes:
    w_f_c: The world from camera 4x4 matrix.
    c_f_w: The camera from world 4x4 matrix.
    intrinsics: The camera intrinsics as a 3x3 matrix.
    inv_intrinsics: The inverse of camera intrinsics matrix.
  """

  def __init__(self, intrinsics, w_f_c):
    """Constructor.

    Args:
      intrinsics: A numpy 3x3 array representing intrinsics.
      w_f_c: A numpy 4x4 array representing wFc.
    """
    self.intrinsics = intrinsics
    self.inv_intrinsics = intrinsics.getI()
    self.w_f_c = w_f_c
    self.c_f_w = w_f_c.getI()


class View(object):
  """Represents an image and associated camera geometry.

  Attributes:
    camera: The camera for this view.
    image: The np array containing the image data.
    image_path: The file path to the image.
    shape: The 2D shape of the image.
  """

  def __init__(self, image_path, shape, camera):
    self.image_path = image_path
    self.shape = shape
    self.camera = camera
    self.image = None


def _WorldFromCameraFromViewDict(view_json):
  """Fills the world from camera transform from the view_json.

  Args:
    view_json: A dictionary of view parameters.

  Returns:
     A 4x4 transform matrix representing the world from camera transform.
  """

  # The camera model transforms the 3d point X into a ray u in the local
  # coordinate system:
  #
  #  u = R * (X[0:2] - X[3] * c)
  #
  # Meaning the world from camera transform is [inv(R), c]

  transform = np.identity(4)
  position = view_json['position']
  transform[0:3, 3] = (position[0], position[1], position[2])
  orientation = view_json['orientation']
  angle_axis = np.array([orientation[0], orientation[1], orientation[2]])
  angle = np.linalg.norm(angle_axis)
  epsilon = 1e-7
  if abs(angle) < epsilon:
    # No rotation
    return np.matrix(transform)

  axis = angle_axis / angle
  rot_mat = transformations.quaternion_matrix(
      transformations.quaternion_about_axis(-angle, axis))
  transform[0:3, 0:3] = rot_mat[0:3, 0:3]
  return np.matrix(transform)


def _IntrinsicsFromViewDict(view_params):
  """Fills the intrinsics matrix from view_params.

  Args:
    view_params: Dict view parameters.

  Returns:
     A 3x3 matrix representing the camera intrinsics.
  """
  intrinsics = np.matrix(np.identity(3))
  intrinsics[0, 0] = view_params['focal_length']
  intrinsics[1, 1] = (
      view_params['focal_length'] * view_params['pixel_aspect_ratio'])
  intrinsics[0, 2] = view_params['principal_point'][0]
  intrinsics[1, 2] = view_params['principal_point'][1]
  return intrinsics


def ReadView(base_dir, view_json):
  return View(
      image_path=os.path.join(base_dir, view_json['relative_path']),
      shape=(int(view_json['height']), int(view_json['width'])),
      camera=Camera(
          _IntrinsicsFromViewDict(view_json),
          _WorldFromCameraFromViewDict(view_json)))


def ReadScene(base_dir):
  """Reads a scene from the directory base_dir."""
  with open(os.path.join(base_dir, 'models.json')) as f:
    model_json = json.load(f)

  all_views = []
  for views in model_json:
    all_views.append([ReadView(base_dir, view_json) for view_json in views])
  return all_views


def InterpolateDepths(near_depth, far_depth, num_depths):
  """Returns num_depths from (far_depth, near_depth), interpolated in inv depth.


  Args:
    near_depth: The first depth.
    far_depth: The last depth.
    num_depths: The total number of depths to create, include near_depth and
      far_depth are always included and other depths are interpolated between
      them, in inverse depth space.

  Returns:
    The depths sorted in descending order (so furthest first). This order is
    useful for back to front compositing.
  """

  inv_near_depth = 1.0 / near_depth
  inv_far_depth = 1.0 / far_depth
  depths = []
  for i in range(0, num_depths):
    fraction = float(i) / float(num_depths - 1)
    inv_depth = inv_far_depth + (inv_near_depth - inv_far_depth) * fraction
    depths.append(1.0 / inv_depth)
  return depths


def CreateDepthPlaneCameras(w_f_c, fov_degrees, shape, depths):
  """Creates depth plane Cameras for each of depths.


  Note that a depth plane is paramaterized by the extrinsic 3D transform and a
  2D mapping from the plane's coordinate system to pixels in the planes texture
  map. We slightly abuse the notion of a camera and use a Camera object as a
  container for these two transformations for depth planes.


  Creates depth plane cameras for the passed depth "centered" on the camera with
  transform w_f_c.
  A separate camera will be created for each depth in depths and each
  depth camera will have spatial size and intrinsics such that its
  "field of view" from the w_f_c origin will be fov_degrees.

  Args:
    w_f_c: The world from camera transform that these planes are created at.
    fov_degrees: Tuple of [vertical, horizontal] field of view for depth planes.
    shape: The shape of the depth planes (height, width, num_channels).
    depths: The depths along which to compute the planes.

  Returns:
    Returns a list of depth planes.
  """
  tan_v = math.tan(math.radians(fov_degrees[0]) * 0.5)
  tan_h = math.tan(math.radians(fov_degrees[1]) * 0.5)
  c_f_p = np.matrix(np.identity(4))
  cameras = []
  for depth in depths:
    x_size = tan_h * depth
    y_size = tan_v * depth
    c_f_p[0, 3] = -x_size
    c_f_p[1, 3] = -y_size
    c_f_p[2, 3] = depth
    intrinsics = np.matrix(np.identity(3))
    intrinsics[0, 0] = shape[1] / (x_size * 2.0)
    intrinsics[1, 1] = shape[0] / (y_size * 2.0)

    cameras.append(Camera(intrinsics, w_f_c * c_f_p))
  return cameras


def WarpCoordinatesWithHomography(homography, rect):
  """Computes the warped coordinates from rect through homography.

  Computes the corresponding coordinates on the image for each pixel of rect.
  Note that the returned coordinates are in x, y order.
  The returned image can be used to warp from the image to the
  pixels of the depth_plane within rect.
  warp_coordinates = ApplyHomographyToCoords(....)
  warped_from_image(x, y) = image(warp_coordinates(x, y)[0],
                                  warp_coordinates(x, y)[1])

  Args:
    homography: A 3x3 tensor representing the transform applied to the
      coordinates inside rect.
   rect: An integer tensor [start_y, start_x, end_y, end_x] representing a rect.

  Returns:
    Returns a rect.height * rect.width * 2 tensor filled with image
    coordinates.
  """
  ys = tf.cast(tf.range(rect[0], rect[2]), tf.float32)
  xs = tf.cast(tf.range(rect[1], rect[3]), tf.float32)

  # Adds 0.5, as pixel centers are assumed to be at half integer coordinates.
  image_coords_t = tf.stack(tf.meshgrid(xs, ys), axis=-1) + 0.5
  hom_image_coords_t = tf.concat(
      (image_coords_t, tf.ones([rect[2] - rect[0], rect[3] - rect[1], 1])),
      axis=-1)

  hom_warped_coords = tf.einsum('ijk,lk->ijl', hom_image_coords_t, homography)
  return hom_warped_coords[:, :, :-1] / hom_warped_coords[:, :, 2:3]


def ImageFromPlane(camera, w_f_p):
  """Computes the homography from the plane's space to the camera's image.

  Points on the plane in the plane's space have coordinates (x, y, 0). The
  homography computed maps from x, y to image pixel coordinates in the camera.
  Note that this maps from the plane's geometric coordinate system (i.e. *not*
  the image associated with the plane) to the image's pixels.

  Args:
    camera: A camera instance.
    w_f_p: The transform from the plane to the world, see top of file.

  Returns:
    Returns a numpy 3x3 matrix representing the homography.
  """
  c_f_p = camera.c_f_w * w_f_p
  # Make the homograpy, it can be shown that it's the 3x4 sub-matrix with the
  # third column removed.
  # See http://szeliski.org/Book/drafts/SzeliskiBook_20100903_draft.pdf
  # pages 55-56.
  hom = np.matrix(np.identity(3))
  hom[:, 0] = c_f_p[0:3, 0]
  hom[:, 1] = c_f_p[0:3, 1]
  hom[:, 2] = c_f_p[0:3, 3]
  return camera.intrinsics * hom


def ComputeImageFromPlaneHomographies(depth_cameras, image_camera):
  """Compute the homographies from the depth planes to the image.

  The returned homography will map a pixel on a depth plance to a (floating
  point) pixel in the image camera.

  Args:
    depth_cameras: A list of "depth" cameras instances.
    image_camera: Homographies are computed from this camera's coordinate system
      to each of the depth cameras.

  Returns:
    The list of length len(depth_cameras), containing the 3x3 float32 np.array
    representing the homographies.
  """
  image_from_plane_mats = []

  for depth_camera in depth_cameras:
    image_from_plane_mats.append(
        np.asarray(
            ImageFromPlane(image_camera, depth_camera.w_f_c) *
            depth_camera.inv_intrinsics).astype(np.float32))
  return np.stack(image_from_plane_mats)


def ReadViewImages(views):
  """Reads the images for the passed views."""
  for view in views:
    # Keep images unnormalized as uint8 to save RAM and transmission time to
    # and from the GPU.
    view.image = np.array(Image.open(view.image_path))


def WriteNpToImage(np_image, path):
  """Writes an image as a numpy array to the passed path.

     If the input has more than four channels only the first four will be
     written. If the input has a single channel it will be duplicated and
     written as a three channel image.
  Args:
    np_image: A numpy array.
    path: The path to write to.

  Raises:
    IOError: if the image format isn't recognized.
  """

  min_value = np.amin(np_image)
  max_value = np.amax(np_image)
  if min_value < 0.0 or max_value > 255.1:
    print('Warning: Outside image bounds, min: %f, max:%f, clipping.',
          min_value, max_value)
    np.clip(np_image, 0.0, 255.0)
  if np_image.shape[2] == 1:
    np_image = np.concatenate((np_image, np_image, np_image), axis=2)

  if np_image.shape[2] == 3:
    image = Image.fromarray(np_image.astype(np.uint8))
  elif np_image.shape[2] == 4:
    image = Image.fromarray(np_image.astype(np.uint8), 'RGBA')

  _, ext = os.path.splitext(path)
  ext = ext[1:]
  if ext.lower() == 'png':
    image.save(path, format='PNG')
  elif ext.lower() in ('jpg', 'jpeg'):
    image.save(path, format='JPEG')
  else:
    raise IOError('Unrecognized format for %s' % path)
