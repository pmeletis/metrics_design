import os
import json
import functools
import traceback
import multiprocessing
import numpy as np
from PIL import Image
from collections import defaultdict
from sklearn.metrics import confusion_matrix


class PQStatCat():
  def __init__(self):
    self.iou = 0.0
    self.tp = 0
    self.fp = 0
    self.fn = 0

  def __iadd__(self, pq_stat_cat):
    self.iou += pq_stat_cat.iou
    self.tp += pq_stat_cat.tp
    self.fp += pq_stat_cat.fp
    self.fn += pq_stat_cat.fn
    return self


class PQStat():
  def __init__(self):
    self.pq_per_cat = defaultdict(PQStatCat)

  def __getitem__(self, i):
    return self.pq_per_cat[i]

  def __iadd__(self, pq_stat):
    for label, pq_stat_cat in pq_stat.pq_per_cat.items():
      self.pq_per_cat[label] += pq_stat_cat
    return self

  def pq_average(self, cat_definition):
    cats_w_parts = []
    cats_no_parts = []
    for i, part_cats in enumerate(cat_definition['cat_def']):
      if len(part_cats['parts_cls']) > 1:
        cats_w_parts.append(i)
      else:
        cats_no_parts.append(i)

    pq, sq, rq, n = 0, 0, 0, 0
    pq_p, sq_p, rq_p, n_p = 0, 0, 0, 0
    pq_np, sq_np, rq_np, n_np = 0, 0, 0, 0
    per_class_results = {}
    for label in range(cat_definition['num_cats']):
      iou = self.pq_per_cat[label].iou
      tp = self.pq_per_cat[label].tp
      fp = self.pq_per_cat[label].fp
      fn = self.pq_per_cat[label].fn
      if tp + fp + fn == 0:
        per_class_results[label] = {'pq': 0.0, 'sq': 0.0, 'rq': 0.0}
        continue
      n += 1
      pq_class = iou / (tp + 0.5 * fp + 0.5 * fn)
      sq_class = iou / tp if tp != 0 else 0
      rq_class = tp / (tp + 0.5 * fp + 0.5 * fn)
      per_class_results[label] = {'pq': pq_class, 'sq': sq_class, 'rq': rq_class}
      pq += pq_class
      sq += sq_class
      rq += rq_class

      if label in cats_w_parts:
        n_p += 1
        pq_p += pq_class
        sq_p += sq_class
        rq_p += rq_class
      elif label in cats_no_parts:
        n_np += 1
        pq_np += pq_class
        sq_np += sq_class
        rq_np += rq_class

    return [{'pq': pq / n, 'sq': sq / n, 'rq': rq / n, 'n': n},
            {'pq_p': pq_p / n_p, 'sq_p': sq_p / n_p, 'rq_p': rq_p / n_p, 'n_p': n_p},
            {'pq_np': pq_np / n_np, 'sq_np': sq_np / n_np, 'rq_np': rq_np / n_np, 'n_np': n_np}], per_class_results


def prediction_parsing(sem_map, inst_map, part_map, cat_definition, thresh=0):
  '''
  parse the predictions (macro-semantic map, instance map, part-semantic map) into the dict format for evaluation.
  [optional]: drop the instances with pixels less than a threshold, which is important if loads of tiny instances exists.

  Args:   sem_map, inst_map, part_map: 2D numpy arrays, with the same size (H,W)
          cat_definition: e.g.
                          {
                              "num_cats": 2,
                              "cat_def":  [{
                                              "sem_cls": [24, 25],
                                              "parts_cls": [1, 2, 3, 4]
                                          },
                                          {
                                              "sem_cls": [26, 27, 28],
                                              "parts_cls": [1, 2, 3, 4, 5]
                                          }]
                          }
          thresh: drop the instances with pixles less than a threshhold
  Returns: a dict:
          {
              cat #0: {   "num_instances": int,
                          "binary_masks": numpy array with size num_instances*h*w
                          "parts_annotation": numpy array with size num*instances*h*w
                          (for each instance, one layer for mask (binary), one layer for parts segmentation with
                          annotation from 1 to 4 or 5)
                      }
              ,
              cat #1: {
                          ...
                      }
              ,
              ...
          }
  '''

  # shape check
  assert sem_map.shape == inst_map.shape == part_map.shape
  h, w = sem_map.shape

  # map's dtype cannot be uint, since -1 flag is used later
  sem_map = sem_map.astype(np.int32)
  inst_map = inst_map.astype(np.int32)
  part_map = part_map.astype(np.int32)


  meta_dict = {}

  # cat_id is 0, 1, 2, ...,
  for cat_id in range(cat_definition['num_cats']):
    sem_cls = cat_definition['cat_def'][cat_id]['sem_cls']
    parts_cls = cat_definition['cat_def'][cat_id]['parts_cls']

    # empty list for multiple semantic classes for a single category
    binary_masks_list = []
    parts_annotations_list = []

    for sem_idx in sem_cls:
      selected = sem_map == sem_idx
      selected_ins = inst_map.copy()
      selected_ins[np.invert(selected)] = -1
      if -1 in selected_ins:
        idxs, counts = np.unique(selected_ins, return_counts=True)
        # get rid of -1 label stats
        idxs = idxs[1:]
        counts = counts[1:]
      else:
        # only if all the pixels belong to the same semantic classes, then there will be no -1 label
        idxs, counts = np.unique(selected_ins, return_counts=True)

      # drop the instances that are too small if they are with part annotations
      if len(parts_cls) > 1:
        idxs = idxs[counts > thresh]
        counts = counts[counts > thresh]

      binary_masks = np.zeros((idxs.shape[0], h, w)).astype(np.int32)
      parts_annotations = np.zeros((idxs.shape[0], h, w)).astype(np.int32)

      # save the masks and part-level annotations
      for i in range(len(idxs)):
        binary_masks[i, :, :] = selected_ins == idxs[i]
        if len(parts_cls) > 1:
          temp_parts = np.zeros((h, w))
          temp_parts[selected_ins == idxs[i]] = part_map[selected_ins == idxs[i]]
          parts_annotations[i, :, :] = temp_parts

      binary_masks_list.append(binary_masks)
      parts_annotations_list.append(parts_annotations)

    binary_masks_per_cat = np.concatenate(binary_masks_list)
    parts_annotations_per_cat = np.concatenate(parts_annotations_list)
    num_instances_per_cat = binary_masks_per_cat.shape[0]

    meta_dict[cat_id] = {'num_instances': num_instances_per_cat,
                         'binary_masks': binary_masks_per_cat,
                         'parts_annotation': parts_annotations_per_cat
                         }

  return meta_dict


def annotation_parsing(sample, cat_definition, thresh=0):
  '''
  parse the numpy encoding defined by dataset definition.
  [optional]: drop the instances with pixels less than a threshold, which is important if loads of tiny instances exists.

  Args:   sample: a numpy array with ground truth annotation
          cat_definition: e.g.
                          {
                              "num_cats": 2,
                              "cat_def":  [{
                                              "sem_cls": [24, 25],
                                              "parts_cls": [1, 2, 3, 4]
                                          },
                                          {
                                              "sem_cls": [26, 27, 28],
                                              "parts_cls": [1, 2, 3, 4, 5]
                                          }]
                          }
          thresh: drop the instances with pixels less than a threshold
  Returns: a dict:
          {
              cat #0: {   "num_instances": int,
                          "binary_masks": numpy array with size num_instances*h*w
                          "parts_annotation": numpy array with size num*instances*h*w
                          (for each instance, one layer for mask (binary), one layer for parts segmentation with
                          annotation from 1 to 4 or 5)
                      }
              ,
              cat #1: {
                          ...
                      }
              ,
              ...
          }
  '''

  h, w = sample.shape

  sem_map = np.where(sample <= 99,
               sample,
               np.where(sample <= 99_999,
                        sample // 10 ** 3,
                        sample // 10 ** 5))
  inst_map = np.where(sample <= 99,
                      np.zeros_like(sample),
                      np.where(sample <= 99_999,
                               sample % 10 ** 3,
                               (sample % 10 ** 5) // 10 ** 2))
  part_map = np.where(sample <= 99_999,
                      np.zeros_like(sample),
                      (sample % 10 ** 5) % 10 ** 2)

  meta_dict = {}

  # cat_id is 0, 1, 2, ...,
  for cat_id in range(cat_definition['num_cats']):
    sem_cls = cat_definition['cat_def'][cat_id]['sem_cls']
    parts_cls = cat_definition['cat_def'][cat_id]['parts_cls']

    # empty list for multiple semantic classes for a single category
    binary_masks_list = []
    parts_annotations_list = []

    for sem_idx in sem_cls:
      selected = sem_map == sem_idx
      # # delete some instances that should have parts, but actually not annotated, happening in Cityscapes
      if len(parts_cls) > 1:
        # if there are instances with parts not annotated as instances with parts
        selected = np.logical_and(selected, sample // 10 ** 5 > 0)
      selected_ins = inst_map.copy()
      selected_ins[np.invert(selected)] = -1
      if -1 in selected_ins:
        idxs, counts = np.unique(selected_ins, return_counts=True)
        # get rid of -1 label stat
        idxs = idxs[1:]
        counts = counts[1:]
      else:
        # only used if all the pixels belong to the same semantic classes, then there will be no -1 label
        idxs, counts = np.unique(selected_ins, return_counts=True)

      # drop the instances that are too small if they are with part annotations
      if len(parts_cls) > 1:
        idxs = idxs[counts > thresh]
        counts = counts[counts > thresh]

      binary_masks = np.zeros((idxs.shape[0], h, w)).astype(np.int32)
      parts_annotations = np.zeros((idxs.shape[0], h, w)).astype(np.int32)

      # write the masks and part-level annotations
      for i in range(len(idxs)):
        binary_masks[i, :, :] = selected_ins == idxs[i]
        if len(parts_cls) > 1:
          temp_parts = np.zeros((h, w)).astype(np.int32)
          temp_parts[selected_ins == idxs[i]] = part_map[selected_ins == idxs[i]]
          parts_annotations[i, :, :] = temp_parts

      # delete some instances that claimed to have parts, but actually not annotated (all zeros pixels), happening in pascal
      if len(parts_cls) > 1:
        delete_idx = []
        for i in range(idxs.shape[0]):
          temp_binary_msk = binary_masks[i, :, :]
          temp_parts_anno = parts_annotations[i, :, :]
          part_elements = np.unique(temp_parts_anno[temp_binary_msk > 0.5])
          if part_elements.size == 1 and 0 in part_elements:
            print('found in valid segment in annotation, deleted')
            delete_idx.append(i)
        binary_masks = np.delete(binary_masks, delete_idx, 0)
        parts_annotations = np.delete(parts_annotations, delete_idx, 0)

      binary_masks_list.append(binary_masks)
      parts_annotations_list.append(parts_annotations)

    binary_masks_per_cat = np.concatenate(binary_masks_list)
    parts_annotations_per_cat = np.concatenate(parts_annotations_list)
    num_instances_per_cat = binary_masks_per_cat.shape[0]

    meta_dict[cat_id] = {'num_instances': num_instances_per_cat,
                         'binary_masks': binary_masks_per_cat,
                         'parts_annotation': parts_annotations_per_cat
                         }

  return meta_dict


def generate_ignore_info(panoptic_dict, panoptic_ann_img, image_id, void=0):
  # Create empty ignore_img and ignore_dict
  ignore_img = np.zeros_like(panoptic_ann_img).astype(np.uint8)
  ignore_dict = dict()

  # Get panoptic segmentation in the correct format
  pan_ann_format = panoptic_ann_img[..., 0] + panoptic_ann_img[..., 1] * 256 + panoptic_ann_img[..., 2] * 256 * 256

  # Store overall void info in ignore_img and ignore_dict
  overall_void = pan_ann_format == void
  ignore_img[overall_void] = 255
  ignore_dict['255'] = 255

  # Retrieve annotation corresponding to image_id
  annotation_dict = dict()
  for annotation in panoptic_dict['annotations']:
    if annotation['image_id'] == image_id:
      annotation_dict = annotation

  if len(annotation_dict) == 0:
    raise KeyError('ImageID is not present in the panoptic annotation dict.')

  # Find crowd annotations and add them to ignore_img and ignore_dict
  for inst_annotation in annotation_dict['segments_info']:
    if inst_annotation['iscrowd'] == 1:
      crowd_instance_id = inst_annotation['id']
      category_id = inst_annotation['category_id']
      crowd_mask = pan_ann_format == crowd_instance_id
      ignore_img[crowd_mask] = category_id
      ignore_dict[str(category_id)] = category_id

  return ignore_img[:, :, 0], ignore_dict


def ignore_img_parsing(sample, cat_definition):
  '''
  parse the ignore_img, which contains crowd (with semantics id) and void region (255)

  Args:   sample: a numpy array
          cat_definition: e.g.
                          {
                              "num_cats": 2,
                              "cat_def":  [{
                                              "sem_cls": [24, 25],
                                              "parts_cls": [1, 2, 3, 4]
                                          },
                                          {
                                              "sem_cls": [26, 27, 28],
                                              "parts_cls": [1, 2, 3, 4, 5]
                                          }]
                          }
  Returns: a dict:
          {
              cat #0: {
                          "binary_masks": numpy array with size num_instances*h*w
                         }
              ,
              cat #1: {
                          ...
                      }
              ,
              ...
          }
  '''
  h, w = sample.shape

  meta_dict = {}

  # cat_id is 0, 1, 2, ...
  for cat_id in range(cat_definition['num_cats']):
    sem_cls = cat_definition['cat_def'][cat_id]['sem_cls']

    binary_masks_per_cat_void = np.zeros((h, w), dtype=np.uint8)
    binary_masks_per_cat_crowd = np.zeros((h, w), dtype=np.uint8)

    for sem_idx in sem_cls:
      binary_masks_per_cat_crowd[sample == sem_idx] = 1

    binary_masks_per_cat_void[sample == 255] = 1

    meta_dict[cat_id] = {'void_masks': binary_masks_per_cat_void,
                         'crowd_masks': binary_masks_per_cat_crowd}

  return meta_dict


def pq_part(pred_meta, gt_meta, crowd_meta, cat_definition):
  '''

  Args: three meta_dict of the prediction and ground truth, and crowd_instances with definition:
          {
              cat #0: {   "num_instances": int,
                          "binary_masks": numpy array with size num_instances*h*w
                          "parts_annotation": numpy array with size num*instances*h*w
                          (for each instance, one layer for mask (binary), one layer for parts segmentation with
                          annotation from 1 to 4 or 5)
                      }
              ,
              cat #1: {
                          ...
                      }
              ,
              ...
          }
      , cat_definition: e.g.
                          {
                              "num_cats": 2,
                              "cat_def":  [{
                                              "sem_cls": [24, 25],
                                              "parts_cls": [1, 2, 3, 4]
                                          },
                                          {
                                              "sem_cls": [26, 27, 28],
                                              "parts_cls": [1, 2, 3, 4, 5]
                                          }]
                          }

  Returns: an instance PQStat
  '''

  pq_stat = PQStat()

  for cat_id in range(cat_definition['num_cats']):
    pred_ins_dict = pred_meta[cat_id]
    gt_ins_dict = gt_meta[cat_id]
    crowd_ins_dict = crowd_meta[cat_id]

    num_ins_pred = pred_ins_dict['num_instances']
    masks_pred = pred_ins_dict['binary_masks'].astype(np.int32)
    parts_pred = pred_ins_dict['parts_annotation'].astype(np.int32)

    num_ins_gt = gt_ins_dict['num_instances']
    masks_gt = gt_ins_dict['binary_masks'].astype(np.int32)
    parts_gt = gt_ins_dict['parts_annotation'].astype(np.int32)

    masks_crowd = crowd_ins_dict['crowd_masks'].astype(np.int32)
    masks_void = crowd_ins_dict['void_masks'].astype(np.int32)
    masks_void_and_crowd = np.logical_or(masks_void, masks_crowd)

    for i in range(num_ins_gt):
      temp_gt_mask = np.logical_and(masks_gt[i, :, :], np.logical_not(masks_crowd))
      if np.sum(temp_gt_mask) == 0:
        num_ins_gt -= 1
        masks_gt = np.delete(masks_gt, i, 0)
        parts_gt = np.delete(parts_gt, i, 0)
        break

    unmatched_pred = list(range(num_ins_pred))
    for i in range(num_ins_gt):
      temp_gt_mask = np.logical_and(masks_gt[i, :, :], np.logical_not(masks_crowd))
      temp_gt_parts = parts_gt[i, :, :]

      for j in range(num_ins_pred):
        if j not in unmatched_pred: continue
        temp_pred_mask = masks_pred[j, :, :]
        temp_pred_parts = parts_pred[j, :, :]

        mask_inter_sum = np.sum(np.logical_and(temp_gt_mask, temp_pred_mask))
        mask_union_sum = np.sum(np.logical_or(temp_gt_mask, temp_pred_mask)) - np.sum(
          np.logical_and(masks_void, temp_pred_mask))
        mask_iou = mask_inter_sum / mask_union_sum

        if mask_iou > 0.5:
          unmatched_pred.remove(j)
          if len(cat_definition['cat_def'][cat_id]['parts_cls']) > 1:
            # if defnied with multiple part labels
            msk_not_defined_in_gt = np.logical_and(temp_gt_parts == 0, temp_gt_mask)
            msk_ignore = np.logical_or(masks_void_and_crowd, msk_not_defined_in_gt)
            msk_evaluated = np.logical_not(msk_ignore)
            cm = confusion_matrix(temp_gt_parts[msk_evaluated].reshape(-1), temp_pred_parts[msk_evaluated].reshape(-1))
            void_in_pred = 255 in np.unique(temp_pred_parts[msk_evaluated])
            if cm.size != 0:
              inter = np.diagonal(cm)
              union = np.sum(cm, 0) + np.sum(cm, 1) - np.diagonal(cm)
              ious = inter / np.where(union > 0, union, np.ones_like(union))
              if void_in_pred:
                ious = ious[:-1]
              mean_iou = np.mean(ious)
            else:
              raise Exception('empty CM')
            pq_stat[cat_id].tp += 1
            pq_stat[cat_id].iou += mean_iou
          else:
            # if defnied without part label, i.e. part label always being 1
            pq_stat[cat_id].tp += 1
            pq_stat[cat_id].iou += mask_iou
          break

    for j in range(num_ins_pred):
      if j not in unmatched_pred: continue
      temp_pred_mask = masks_pred[j, :, :]
      mask_inter_sum = np.sum(np.logical_and(masks_void_and_crowd, temp_pred_mask))
      mask_pred_sum = np.sum(temp_pred_mask)
      if mask_inter_sum / mask_pred_sum <= 0.5:
        pq_stat[cat_id].fp += 1

    pq_stat[cat_id].fn = num_ins_gt - pq_stat[cat_id].tp

  return pq_stat


# The decorator is used to prints an error thrown inside process
def get_traceback(f):
  @functools.wraps(f)
  def wrapper(*args, **kwargs):
    try:
      return f(*args, **kwargs)
    except Exception as e:
      print('Caught exception in worker thread:')
      traceback.print_exc()
      raise e

  return wrapper


@get_traceback
def evaluate_single_core(proc_id, fn_pairs, pred_reader_fn, gt_panoptic_dict, cat_definition):
  # Initialize PartPQ statistics
  pq_stats_split = PQStat()


  counter = 0
  # Loop over all predictions
  for fn_pair in fn_pairs:
    counter += 1
    print(counter)
    image_id = fn_pair[0]
    gt_pan_part_file = fn_pair[1]
    gt_pan_file = fn_pair[2]
    pred_file = fn_pair[3]


    # partPQ eval starts here
    # Load GT annotation file for this image

    part_gt_sample = np.array(Image.open(gt_pan_part_file)).astype(np.int32)
    part_gt_dict = annotation_parsing(part_gt_sample, cat_definition, thresh=0)

    # Load prediction for this image

    pan_classes, pan_inst_ids, parts_output = pred_reader_fn(pred_file)

    # Feed to PQ_part evaluator
    part_pred_dict = prediction_parsing(pan_classes, pan_inst_ids,
                                        parts_output, cat_definition, thresh=0)

    # Generate ignore_data
    panoptic_ann_img = np.array(Image.open(gt_pan_file)).astype(np.int32)

    ignore_img, ignore_dict = generate_ignore_info(gt_panoptic_dict, panoptic_ann_img, image_id)
    crowd_dict = ignore_img_parsing(ignore_img, cat_definition)

    temp_pq_part = pq_part(part_pred_dict, part_gt_dict, crowd_dict, cat_definition)
    # print(temp_pq_part.pq_average(cat_definition))

    pq_stats_split += temp_pq_part

  return pq_stats_split


def evaluate_PQPart_multicore(spec, filepaths_pairs, pred_reader_fn, cpu_num=8):

  gt_panoptic_dict = spec.gt_panoptic_dict
  cat_definition = spec.cat_definition


  fn_splits = np.array_split(filepaths_pairs, cpu_num)
  print("Number of cores: {}, images per core: {}".format(cpu_num, len(fn_splits[0])))
  workers = multiprocessing.Pool(processes=cpu_num)
  processes = []
  for proc_id, fn_split in enumerate(fn_splits):
      p = workers.apply_async(evaluate_single_core,
                              (proc_id, fn_split, pred_reader_fn, gt_panoptic_dict, cat_definition))
      processes.append(p)

  pq_stats_global = PQStat()
  # split_pq_stats = evaluate_single_core(fn_total, fn_anns, fn_preds, gt_folder_pan, gt_panoptic_dict)
  for p in processes:
    pq_stats_global += p.get()

  results = pq_stats_global.pq_average(cat_definition)
  return results
