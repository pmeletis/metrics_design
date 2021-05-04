import os.path

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
from eval_spec_PQPart import PQPartEvalSpec



FILEPATH_EVALUATION_DEF = 'cpp_PQPart_23_evalspec.yaml'
FILEPATH_PATTERN_GT_PAN_PART_CPP = op.join('/home/cylu/Documents/fpsnetv2/datasets/Cityscapes/validation/part_gt',
                                           '*', '*.tif')
FILEPATH_GT_PAN_CPP = op.join('/home/cylu/Documents/fpsnetv2/datasets/Cityscapes/validation/panoptic')
BASEPATH_PRED = "/home/cylu/Documents/fpsnetv2/datasets/Cityscapes/predictions/20200917_fpsnet_parts_0/part_panoptic_cs"


def filepaths_pairs_fn(filepath_pattern_gt_pan_part, filepath_pattern_gt_pan, basepath_pred):
  # return a list of tuples with paths
  filepaths_gt_pan_part = glob.glob(filepath_pattern_gt_pan_part)
  print(f"Found {len(filepaths_gt_pan_part)} ground truth labels.")
  pairs = list()
  for fp_gt_pan_part in filepaths_gt_pan_part:
    image_id = op.basename(fp_gt_pan_part)[:-23]
    fp_gt_pan = op.join(filepath_pattern_gt_pan, image_id + '_gtFine_instanceIds.png')
    assert op.isfile(fp_gt_pan)
    ########################
    # Adapt to your system #
    # here we use the ground truth paths for predictions
    fp_pred = op.join(basepath_pred, image_id + "_gtFine_leftImg8bit.png")
    assert op.isfile(fp_pred)
    # assert False, 'delete this when adapted to your needs'
    ########################
    pairs.append((image_id, fp_gt_pan_part, fp_gt_pan, fp_pred))

  return pairs

######################
# Adapt to your case #
# here we assume that predictions are encoded as ground truth
def pred_reader_fn(fp_pred):
  part_pred_sample = np.array(Image.open(fp_pred)).astype(np.int32)
  pan_classes = part_pred_sample[..., 0]
  pan_inst_ids = part_pred_sample[..., 1]
  parts_output = part_pred_sample[..., 2]
  return pan_classes, pan_inst_ids, parts_output
######################


spec = PQPartEvalSpec(FILEPATH_EVALUATION_DEF)
filepaths_pairs = filepaths_pairs_fn(FILEPATH_PATTERN_GT_PAN_PART_CPP, FILEPATH_GT_PAN_CPP, BASEPATH_PRED)

results = evaluate_PQPart_multicore(spec, filepaths_pairs, pred_reader_fn)
print(results)
