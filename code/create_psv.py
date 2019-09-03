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

"""Simple program that reads a DeepView scene and output its plane sweep volume.

Example usage:

python create_psv.py --input_dir $PATH_TO_DEEPVIEW_DATA/800/scene_063/
    --output_dir /tmp/

Writes the average of the plane sweep volumes (PSVs) of the first rig position
to /tmp/.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import tensorflow as tf
tf.enable_eager_execution()
import utils

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('input_dir', '', 'Input scene directory.')

flags.DEFINE_integer('num_planes', 20, 'Number of planes to use in the PSV.')

flags.DEFINE_float('near_depth', 1.0,
                   'The depth of the first depth plane, in meters.')

flags.DEFINE_float('far_depth', 100.0,
                   'The depth of the last depth plane, in meters.')

flags.DEFINE_string('output_dir', '/tmp/',
                    'The average plane sweep volume will be written here.')


def main(_):
  # Reads the first rig position for the input scene.
  # Could also include views from other rig positions.
  views = utils.ReadScene(FLAGS.input_dir)[0]
  # Read the images for the views in this scene.
  utils.ReadViewImages(views)

  # Sets up the "cameras" for the plane sweep volume.
  # Create the plane sweep volume at view 7.
  psv_view = views[7]

  depth_plane_shape = psv_view.image.shape[:2]

  # These depth of fields approximately enclose the scene images.
  vert_fov_degrees, horz_fov_degrees = 100.0, 130.0
  # Computes the depths, distributed in 1 / depth.
  depths = utils.InterpolateDepths(FLAGS.near_depth, FLAGS.far_depth,
                                   FLAGS.num_planes)
  depth_plane_cameras = utils.CreateDepthPlaneCameras(
      psv_view.camera.w_f_c, [vert_fov_degrees, horz_fov_degrees],
      depth_plane_shape, depths)

  # Warps each image to the PSV and accumulates it.
  accumulated_warped_images = 0.0
  for view_index, view in enumerate(views):
    print('Warping view %d to plane sweep volume...' % view_index)
    # Compute the homographies from the depth plane space to the image space.
    image_from_plane_mats = tf.constant(
        utils.ComputeImageFromPlaneHomographies(depth_plane_cameras,
                                                view.camera))

    # Compute warped coordinates, that maps each pixel on a depth plane to a
    # position on the image.
    warped_coordinates = tf.stack([
        utils.WarpCoordinatesWithHomography(
            m, [0, 0, depth_plane_shape[0], depth_plane_shape[1]])
        for m in image_from_plane_mats
    ])
    # Append an alpha channel to the input image, this will be used to record
    # the number of image contribution to the pixels on the depth planes.
    image_with_alpha = tf.concat([
        tf.cast(view.image, tf.float32) / 255.0,
        tf.ones([view.shape[0], view.shape[1], 1])
    ],
                                 axis=-1)

    # Warp the image using the resampler.
    warped_images = tf.contrib.resampler.resampler(
        tf.broadcast_to(image_with_alpha,
                        [FLAGS.num_planes, view.shape[0], view.shape[1], 4]),
        warped_coordinates)
    accumulated_warped_images = warped_images + accumulated_warped_images

  # Divide out the rgb by the alpha to get the average RGB at each depth plane
  # pixel.
  average_rgb = accumulated_warped_images[..., :3] / (
      accumulated_warped_images[..., 3:] + 1e-5)
  for d in range(average_rgb.shape[0]):
    utils.WriteNpToImage(average_rgb[d].numpy() * 255.0,
                         os.path.join(FLAGS.output_dir, 'psv_%03d.png' % d))
  print(
      'Wrote plane sweep volume to',
      os.path.join(FLAGS.output_dir,
                   'psv_000...%03d.png' % (FLAGS.num_planes - 1)))


if __name__ == '__main__':
  tf.compat.v1.app.run()
