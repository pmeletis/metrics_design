import os.path
import argparse
import numpy as np
import json
import yaml
import glob
import multiprocessing
import functools
import traceback
from PIL import Image
import os.path as op
import sys
sys.path.append('panoptic_parts')

from panoptic_parts.utils.experimental_evaluation_PQPart import evaluate_PQPart_multicore
from merge_eval_spec import PPSEvalSpec


def filepaths_pairs_fn(dst_name, filepath_pattern_gt_pan_part, basepath_pred):
  # returns a list of tuples with paths
  filepaths_gt_pan_part = glob.glob(filepath_pattern_gt_pan_part)
  print(f"Found {len(filepaths_gt_pan_part)} ground truth labels.")
  pairs = list()
  for fp_gt_pan_part in filepaths_gt_pan_part:
    if dst_name == 'Cityscapes Panoptic Parts':
      image_id = op.basename(fp_gt_pan_part)[:-23]
      fp_pred = op.join(basepath_pred, image_id + "_gtFine_leftImg8bit.png")
    elif dst_name == 'PASCAL Panoptic Parts':
      image_id = op.basename(fp_gt_pan_part)[:-4]
      fp_pred = op.join(basepath_pred, image_id + ".png")
    assert op.isfile(fp_pred), fp_pred
    pairs.append((fp_gt_pan_part, fp_pred))
  return pairs


def pred_reader_fn(fp_pred):
  # This function assumes that predictions are saved in the 3-channel format
  part_pred_sample = np.array(Image.open(fp_pred), dtype=np.int32)
  pan_classes = part_pred_sample[..., 0]
  pan_inst_ids = part_pred_sample[..., 1]
  parts_output = part_pred_sample[..., 2]
  return pan_classes, pan_inst_ids, parts_output


def evaluate(eval_spec_path, basepath_gt, basepath_pred):
  spec = PPSEvalSpec(eval_spec_path)
  dst_name = spec._dspec.dataset_name
  if dst_name == 'Cityscapes Panoptic Parts':
    filepath_pattern_gt_pan_part = op.join(basepath_gt, '*', '*.tif')
  elif dst_name == 'PASCAL Panoptic Parts':
    filepath_pattern_gt_pan_part = op.join(basepath_gt, '*.tif')

  filepaths_pairs = filepaths_pairs_fn(dst_name, filepath_pattern_gt_pan_part, basepath_pred)

  results = evaluate_PQPart_multicore(spec, filepaths_pairs, pred_reader_fn)

  print(*map(lambda d: ', '.join(map(lambda t: f'{t[0]}: {t[1]:.3f}', d.items())),
             results[0]),
        sep='\n')
  print(*map(lambda t: f'{t[0]:15} ' + ', '.join(map(lambda t: f'{t[0]}: {t[1]:.3f}', t[1].items())),
             zip(spec.eval_sid2scene_label.values(), results[1].values())),
        sep='\n')


if __name__ == '__main__':
  # python -m eval_PQPart "[WIP]cpp_official_evalspec_grouped.yaml" \
  #                       "/home/cylu/Documents/fpsnetv2/datasets/Cityscapes/validation/part_gt" \
  #                       "/home/cylu/Documents/fpsnetv2/datasets/Cityscapes/predictions/20200917_fpsnet_parts_0/part_panoptic_cs"

  # python -m eval_PQPart "[WIP]cpp_official_evalspec_grouped.yaml" \
  #                       "/media/panos/data/datasets/cityscapes_parts/labelling/releases/20201021/cityscapes_panoptic_parts_v1.1/gtFine/val" \
  #                       "/media/panos/data/logdir/part-aware-panoptic-segmentation/cpp_merged_part_aware_panseg"

  # python -m eval_PQPart "[WIP]ppp_official_evalspec.yaml" \
  #                       "/media/panos/data/datasets/pascal_panoptic_parts/releases/20210503/pascal_panoptic_parts_v2/validation" \
  #                       "/media/panos/data/logdir/part-aware-panoptic-segmentation/ppp_merged_part_aware_panseg"

  parser = argparse.ArgumentParser()
  parser.add_argument('eval_spec_path')
  parser.add_argument('basepath_gt')
  parser.add_argument('basepath_pred')
  args = parser.parse_args()

  evaluate(args.eval_spec_path, args.basepath_gt, args.basepath_pred)
