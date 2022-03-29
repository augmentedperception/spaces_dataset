"""Computes SSIM for experiments from rendered and ground truth images.

# To compute metrics on DeepView data.
  compute_metrics ../data/eval/ deep_view_synth_%02d.png /tmp/deep_view.txt

# To compute metrics on Soft3d data.
  compute_metrics ../data/eval/ soft3d_synth_%02d.png /tmp/soft3d.txt

See figure 4 from the paper for details on the experiments. For all experiments,
the first rig position was used for both input and target views.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import numpy as np
from PIL import Image
from skimage.measure import compare_ssim


def compute_ssim(edge_crop, image_0_path, image_1_path):
  image_0 = np.array(Image.open(image_0_path)).astype(np.float32) / 255.0
  image_1 = np.array(Image.open(image_1_path)).astype(np.float32) / 255.0
  b, l, t, r = edge_crop

  image_0 = image_0[b:-t, l:-r]
  image_1 = image_1[b:-t, l:-r]
  score = 0.0
  for c in range(3):
    score += compare_ssim(
        image_0[..., c],
        image_1[..., c],
        gaussian_weights=True,
        sigma=1.5,
        use_sample_covariance=False,
        data_range=1.0)
  return score / 3


def main(argv):
  """Computes metrics on previously computed model outputs."""
  assert len(argv) >= 2 and len(argv) < 4, argv
  base_dir = argv[0]
  # Should be either deep_view_synth_%02d.png or soft3d_synth_%02d.png
  rendered_image_name_pattern = argv[1]
  if len(argv) == 3:
    output_path = argv[2]
    f = open(output_path, "w")
  else:
    f = None

  def _output(*args):
    print(*args)
    if f:
      print(*args, file=f)

  gt_image_name_pattern = "ground_truth_%02d.png"
  experiment_names = ["small_quad", "medium_quad", "large_quad", "dense"]
  scene_names = [
      "scene_000", "scene_009", "scene_010", "scene_023", "scene_024",
      "scene_052", "scene_056", "scene_062", "scene_063", "scene_073"
  ]
  view_indexes_dict = {
      "small_quad": [5, 6, 7],
      "medium_quad": [2, 4, 5, 6, 7, 11],
      "large_quad": [1, 2, 4, 5, 6, 7, 8, 10, 11],
      "dense": [5, 7, 10, 11]
  }

  # Note following Kalantari et al, we always crop in from the edges slightly as
  # they tend to be unreliable an unrepresentative.
  # For certain viewpoints we crop in more, for example a view that is at the
  # top of a quad is unreliable near the top edge, so we crop in further there,
  # similarly for images on the bottom, left and right edges.
  small_crop = 16
  big_crop = 40
  std_crop = (small_crop, small_crop, small_crop, small_crop)
  left_crop = (small_crop, big_crop, small_crop, small_crop)
  top_crop = (big_crop, small_crop, small_crop, small_crop)
  right_crop = (small_crop, small_crop, small_crop, big_crop)
  bottom_crop = (small_crop, small_crop, big_crop, small_crop)

  scene_to_crop_dict = {
      "small_quad": {
          5: left_crop,
          7: right_crop
      },
      "medium_quad": {
          2: top_crop,
          4: left_crop,
          7: right_crop,
          11: bottom_crop
      },
      "large_quad": {
          1: top_crop,
          2: top_crop,
          4: left_crop,
          8: right_crop,
          10: bottom_crop,
          11: bottom_crop
      },
      "dense": {}
  }

  per_experiment_scores = []
  for experiment_name in experiment_names:
    view_indexes = view_indexes_dict[experiment_name]
    _output(experiment_name)
    image_to_crop_dict = scene_to_crop_dict[experiment_name]
    average_over_scenes_score = 0.0
    for scene_name in scene_names:
      _output("\t", scene_name)
      average_per_scene_score = 0.0
      for view_index in view_indexes:
        rendered_image_base_name = rendered_image_name_pattern % view_index
        rendered_image_path = os.path.join(base_dir, experiment_name,
                                           scene_name, rendered_image_base_name)
        gt_image_base_name = gt_image_name_pattern % view_index
        gt_image_path = os.path.join(base_dir, experiment_name, scene_name,
                                     gt_image_base_name)
        crop = image_to_crop_dict.get(view_index, std_crop)
        ssim_score = compute_ssim(crop, rendered_image_path, gt_image_path)
        average_per_scene_score += ssim_score
        _output("\t\t", rendered_image_base_name, gt_image_base_name,
                ssim_score)
      average_per_scene_score /= len(view_indexes)
      _output("\t\t", "Scene average:", average_per_scene_score)
      average_over_scenes_score += average_per_scene_score
    average_over_scenes_score /= len(scene_names)
    per_experiment_scores.append(average_over_scenes_score)
  for experiment_name, score in zip(experiment_names, per_experiment_scores):
    _output(experiment_name, "average:", score)


if __name__ == "__main__":
  main(sys.argv[1:])
