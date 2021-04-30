import numpy as np
import os
import json
from tqdm import tqdm
from PIL import Image

from panopticapi import combine_semantic_and_instance_predictions
from pycocotools import mask

from dataset_spec import DatasetSpec
from merge_eval_spec import PPSEvalSpec

from tmp_utils import get_filenames_in_dir, find_filename_in_list

def _prepare_mappings(sid_pid2part_seg_label, void):
  # TODO(daan): check whether it matters that this is too large
  # TODO(daan): see if it is okay that 100 is hardcoded
  # TODO(daan): clean up
  # Get the maximum amount of part_seg labels
  num_part_seg_labels = np.max(
    list(sid_pid2part_seg_label.values()))

  sids2part_seg_ids = dict()
  for class_key in sid_pid2part_seg_label.keys():
    class_id = class_key // 100
    if class_id in sids2part_seg_ids.keys():
      if sid_pid2part_seg_label[class_key] not in sids2part_seg_ids[class_id]:
        sids2part_seg_ids[class_id].append(sid_pid2part_seg_label[class_key])
      else:
        raise ValueError(
          'A part_id can only be shared between different semantic classes, not within a single semantic class.')
    else:
      sids2part_seg_ids[class_id] = [sid_pid2part_seg_label[class_key]]

  sids2pids_eval = dict()
  for class_key in sid_pid2part_seg_label.keys():
    class_id = class_key // 100
    if class_id in sids2pids_eval.keys():
      if class_key % 100 not in sids2pids_eval[class_id]:
        sids2pids_eval[class_id].append(class_key % 100)
    else:
      sids2pids_eval[class_id] = [class_key % 100]

  part_seg_ids2eval_pids_per_sid = dict()
  for class_key in sids2part_seg_ids.keys():
    arr = np.ones(num_part_seg_labels + 1, np.uint8) * void
    arr[sids2part_seg_ids[class_key]] = sids2pids_eval[class_key]
    part_seg_ids2eval_pids_per_sid[class_key] = arr

  return sids2part_seg_ids, part_seg_ids2eval_pids_per_sid

def _create_categories_list(eval_spec):
  category_list = list()

  for eval_id in eval_spec.eval_sid2scene_label.keys():
    category_dict = dict()
    category_dict['id'] = eval_id
    category_dict['name'] = eval_spec.eval_sid2scene_label[eval_id]
    # TODO(daan): get function in eval_spec to get (color_from_eval_id) functionality
    category_dict['color'] = eval_spec.dataset_spec.scene_class2color[category_dict['name']]
    if eval_id in eval_spec.eval_sid_things:
      category_dict['isthing'] = 1
    else:
      category_dict['isthing'] = 0

    category_list.append(category_dict)

  return category_list

def merge(eval_spec_path,
          panoptic_pred_dir,
          panoptic_pred_json,
          part_pred_path,
          output_dir):
  """

  Args:
    eval_spec_path:
    panoptic_pred_dir:
    panoptic_pred_json:
    part_pred_path:
    output_dir:

  Returns:

  """
  eval_spec = PPSEvalSpec(eval_spec_path)

  # Get category information from EvalSpec
  categories_list = _create_categories_list(eval_spec)
  categories_json = os.path.join(output_dir, 'categories.json')
  with open(categories_json, 'w') as fp:
    json.dump(categories_list, fp)

  # Get the sid_pid -> part_seg mapping from the EvalSpec
  sid_pid2part_seg_label = eval_spec.eval_sid_pid2eval_pid_flat

  # Get the void label from the EvalSpec definition
  void = eval_spec.ignore_label

  # Load list of images with their properties
  with open(panoptic_pred_json, 'r') as fp:
    panoptic_dict = json.load(fp)

  # If the output directory does not exist, create it
  if not os.path.exists(output_dir):
    print("Creating output directory at {}".format(output_dir))
    os.mkdir(output_dir)

  # Prepare the mappings from predictions to evaluation and vice versa
  sids2part_seg_ids, part_seg_ids2eval_pids_per_sid = _prepare_mappings(sid_pid2part_seg_label, void)

  # Load panoptic annotations
  annotations = panoptic_dict['annotations']

  # Get filenames in directory with part segmentation predictions
  fn_partseg = get_filenames_in_dir(part_pred_path)

  annotation_count = 0
  print("Merging panoptic and part predictions to PPS, and saving...")
  # TODO(daan): implement a check to see whether the image set correponds with image set from images.json (to be created)
  for annotation in tqdm(annotations):
    annotation_count += 1
    file_name = annotation['file_name']
    image_id = annotation['image_id']

    f_partseg = find_filename_in_list(image_id, fn_partseg, 'part seg')

    # Load and decode panoptic predictions
    pred_pan = np.array(Image.open(os.path.join(panoptic_pred_dir, file_name)))
    pred_pan_flat = pred_pan[..., 0] + pred_pan[..., 1] * 256 + pred_pan[..., 2] * 256 ** 2
    h, w = pred_pan.shape[0], pred_pan.shape[1]

    # Load part predictions
    pred_part = np.array(Image.open(f_partseg))

    class_canvas = np.ones((h, w), dtype=np.int32) * void
    inst_canvas = np.zeros((h, w), dtype=np.int32)
    # TODO(daan): check whether we can also set part_canvas init to 255
    part_canvas = np.zeros((h, w), dtype=np.int32)

    segment_count = 0
    for segment in annotation['segments_info']:
      segment_count += 1
      # TODO(daan): keep track of # instances per sid, and only relevant for things
      if segment_count > 255:
        raise ValueError('More than 255 instances. This is currently not supported.')
      segment_id = segment['id']
      cat_id = segment['category_id']

      mask = pred_pan_flat == segment_id

      # TODO(daan): is this the best way to see if a category has parts, or should we get it from l_parts?
      # Loop over all scene-level categories
      if cat_id in sids2part_seg_ids.keys():
        # If category has parts
        # Check what pids are possible for the sid
        plausible_parts = sids2part_seg_ids[cat_id]
        plausible_parts_mask = np.isin(pred_part, plausible_parts)

        # TODO(daan): see if this can also already be done for the entire part prediction, instead of per class
        # Get the mapping from part_seg ids to evaluation pids, given the sid
        part_seg_ids2eval_pids = part_seg_ids2eval_pids_per_sid[cat_id]
        part_canvas[mask] = void

        # Convert the part seg ids to the desired evaluation pids, and store them in the tensor with part labels
        part_canvas[np.logical_and(mask, plausible_parts_mask)] = part_seg_ids2eval_pids[
          pred_part[np.logical_and(mask, plausible_parts_mask)]]

        # Store the category id and instance id in the respective tensors
        class_canvas[mask] = cat_id
        inst_canvas[mask] = segment_count

      else:
        # If category does not have parts
        mask = pred_pan_flat == segment_id

        # Store the category id and instance id in the respective tensors
        class_canvas[mask] = cat_id
        inst_canvas[mask] = segment_count
        # Store a dummy part id
        part_canvas[mask] = 1

    pred_pan_part = np.stack([class_canvas, inst_canvas, part_canvas], axis=2)
    img_pan_part = Image.fromarray(pred_pan_part.astype(np.uint8))
    img_pan_part.save(os.path.join(output_dir, file_name))

  print("Merging finished.")


if __name__ == '__main__':
  # TODO(daan): inlcude args to run from command line
  eval_spec_path = "/home/ddegeus/nvme/projects/metrics_design/[WIP]cpp_official_evalspec.yaml"

  panoptic_pred_dir = "/home/ddegeus/hdnew/output_dir/panoptic_parts/merge_to_panoptic/test/panoptic"
  panoptic_pred_json = "/home/ddegeus/hdnew/output_dir/panoptic_parts/merge_to_panoptic/test/panoptic.json"
  part_pred_path = '/home/ddegeus/nvme/projects/part_panoptic/experiments/baselines/cityscapes/partseg/bsanet/parts_ungrouped/pred_val'

  output_dir = "/home/ddegeus/hdnew/output_dir/panoptic_parts/merge_to_pps/test"

  merge(eval_spec_path,
        panoptic_pred_dir,
        panoptic_pred_json,
        part_pred_path,
        output_dir)