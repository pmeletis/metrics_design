import numpy as np
import os
import json
from PIL import Image
from tqdm import tqdm

from panopticapi import combine_semantic_and_instance_predictions
from pycocotools import mask

from dataset_spec import DatasetSpec
from merge_eval_spec import PPSEvalSpec

from tmp_utils import get_filenames_in_dir, find_filename_in_list


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


def _stuff_segmentation_to_coco(sem_pred_dir, images_list, stuff_labels):
  segmentation_coco_format = list()

  fn_semseg = get_filenames_in_dir(sem_pred_dir)

  # TODO(daan): multi-cpu processing
  for image in tqdm(images_list):
    image_id = image['id']

    f_semseg = find_filename_in_list(image_id, fn_semseg, 'instseg')
    semseg_map = np.array(Image.open(f_semseg)).astype(np.uint8)

    for label_id in stuff_labels:
      # Create mask and encode it
      label_mask = semseg_map == label_id

      # If the category is not in the prediction, go to the next category
      if not np.any(label_mask):
        continue
      label_mask = np.expand_dims(label_mask, axis=2)
      label_mask = np.asfortranarray(label_mask)
      RLE = mask.encode(label_mask)
      assert len(RLE) == 1
      RLE = RLE[0]

      # When using Python3, we convert the encoding to ascii format, to be serializable as json
      RLE['counts'] = RLE['counts'].decode('ascii')

      # Add encoded data to the list
      segmentation_coco_format.append({'segmentation': RLE,
                                       'image_id': image_id,
                                       'category_id': int(label_id)})

  return segmentation_coco_format


def _instance_cs_to_coco_format(inst_pred_dir, images_list):
  instseg_coco_format = list()

  fn_instseg = get_filenames_in_dir(inst_pred_dir)

  # TODO(daan): multi-cpu processing
  for image in tqdm(images_list):
    image_id = image['id']
    h, w = image['height'], image['width']

    # Find the txt file for the image in the directory with predictions
    f_instseg = find_filename_in_list(image_id, fn_instseg, 'instseg', ext='.txt')

    instseg_masks = list()
    instseg_classes = list()
    instseg_scores = list()
    with open(f_instseg, 'r') as txtfile:
      # For each line in the txt file, load the corresponding image and store the mask, class and score
      for txtline in txtfile:
        inst_mask_file = os.path.join(inst_pred_dir, txtline.split(' ')[0])
        instseg_masks.append(np.array(Image.open(inst_mask_file)) / 255)
        instseg_classes.append(txtline.split(' ')[1])
        instseg_scores.append(txtline.split(' ')[2])
    instseg_masks = np.reshape(np.array(instseg_masks).astype(np.uint8),
                               (-1, h, w))

    # Encode mask as RLE as expected by the COCO format
    RLE = mask.encode(np.asfortranarray(np.transpose(instseg_masks, (1, 2, 0))))
    for i, _ in enumerate(instseg_masks):
      # Store all in a list, as expected by the COCO format

      # When using Python3, we convert the encoding to ascii format, to be serializable as json
      RLE[i]['counts'] = RLE[i]['counts'].decode('ascii')

      instseg_coco_format.append({'segmentation': RLE[i],
                                  'score': float(instseg_scores[i]),
                                  'image_id': image_id,
                                  'category_id': int(instseg_classes[i])})

  return instseg_coco_format


def merge(eval_spec_path,
          inst_pred_path,
          sem_pred_path,
          output_dir,
          images_json,
          instseg_format='COCO'):
  """

  Args:
    eval_spec_path:
    inst_pred_path:
    sem_pred_path:
    output_dir:
    images_json:
    instseg_format:

  Returns:

  """

  eval_spec = PPSEvalSpec(eval_spec_path)

  assert instseg_format in ['Cityscapes', 'COCO'], \
    "instseg_format should be \'Cityscapes\' or \'COCO\'"

  # If the output directory does not exist, create it
  if not os.path.exists(output_dir):
    print("Creating output directory at {}".format(output_dir))
    os.mkdir(output_dir)

  # Load list of images with their properties
  with open(images_json, 'r') as fp:
    images_dict = json.load(fp)

  # Get category information from EvalSpec
  categories_list = _create_categories_list(eval_spec)
  categories_json = os.path.join(output_dir, 'categories.json')
  with open(categories_json, 'w') as fp:
    json.dump(categories_list, fp)

  # Get the list of all images in the dataset (split)
  images_list = images_dict['images']

  print("Loading instance segmentation predictions")
  # Load instance segmentation predictions
  if instseg_format == 'Cityscapes':
    print("Converting inst seg predictions from CS to COCO format")
    # If in Cityscapes format, convert to expected COCO format
    inst_pred_list = _instance_cs_to_coco_format(inst_pred_path, images_list)
  elif instseg_format == 'COCO':
    # If in COCO format, load the json file into a list
    with open(inst_pred_path, 'r') as fp:
      inst_pred_list = json.load(fp)

  # Load semantic segmentation predictions, filter out stuff classes and convert to COCO format
  print("Loading semantic segmentation predictions, and converting to COCO format")
  stuff_labels = eval_spec.eval_sid_stuff
  sem_pred_list = _stuff_segmentation_to_coco(sem_pred_path, images_list, stuff_labels=stuff_labels)

  instseg_json_file = os.path.join(output_dir, 'inst_pred.json')
  with open(instseg_json_file, 'w') as fp:
    json.dump(inst_pred_list, fp)

  semseg_json_file = os.path.join(output_dir, 'sem_pred.json')
  with open(semseg_json_file, 'w') as fp:
    json.dump(sem_pred_list, fp)

  output_json = os.path.join(output_dir, 'panoptic.json')
  output_dir_files = os.path.join(output_dir, 'panoptic')
  combine_semantic_and_instance_predictions.combine_predictions(semseg_json_file,
                                                                instseg_json_file,
                                                                images_json,
                                                                categories_json,
                                                                output_dir_files,
                                                                output_json,
                                                                confidence_thr=0.5,
                                                                overlap_thr=0.5,
                                                                stuff_area_limit=1024)

  print("Merging finished.")


if __name__ == '__main__':
  # TODO(daan): inlcude args to run from command line
  eval_spec_path = "/home/ddegeus/nvme/projects/metrics_design/[WIP]cpp_official_evalspec.yaml"
  inst_pred_dir = "/home/ddegeus/nvme/projects/part_panoptic/experiments/baselines/cityscapes/instseg/mask r-cnn/pred_val/"
  sem_pred_dir = "/home/ddegeus/nvme/projects/part_panoptic/experiments/baselines/cityscapes/semseg/deeplabv3_plus/pred_val"
  images_json = "/home/ddegeus/hdnew/dataset/CityscapesPanParts/gtFinePanopticParts_trainval/gtFinePanopticParts/val/images.json"

  output_dir = "/home/ddegeus/hdnew/output_dir/panoptic_parts/merge_to_panoptic/test/"
  instseg_format = 'Cityscapes'

  merge(eval_spec_path,
        inst_pred_dir,
        sem_pred_dir,
        output_dir,
        images_json,
        instseg_format=instseg_format)