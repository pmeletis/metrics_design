"""
This module contains tools for handling evaluation specifications.
"""
import warnings
import sys
sys.path.append('panoptic_parts')
import yaml
import platform
from operator import itemgetter

from panoptic_parts.utils.experimental_evaluation import parse__sid_pid2eid__v2
from panoptic_parts.utils.utils import (
    _sparse_ids_mapping_to_dense_ids_mapping as dict_to_numpy)
from specs.dataset_specs.dataset_spec import DatasetSpec


class PPSEvalSpec(object):
  """
  This class creates an evaluation specification from a YAML specification file and provides
  convenient attributes from the specification and useful functions. Moreover, it provides
  defaults and specification checking.

  Accessible specification attributes:
    - 

  """
  def __init__(self, spec_path):
    """
    Args:
      spec_path: a YAML evaluation specification
    """
    with open(spec_path) as fd:
      espec = yaml.load(fd, Loader=yaml.Loader)

    self._spec_version = espec['version']
    # self.sid_pid2eid__template = espec['sid_pid2eid__template']
    # self.eval_id2scene_part_class = espec['eval_id2scene_part_class']
    self._dspec = DatasetSpec(espec['dataset_spec_path'])

    self.ignore_label = espec['ignore_label']

    # Dataset ids -> evaluation ids
    self.dataset_sid_pid2eval_sid_pid = espec['dataset_sid_pid2eval_sid_pid']
    self.dataset_sid2eval_sid = espec['dataset_sid2eval_sid']

    # Evaluation scene+part ids -> Evaluation flat part ids (for 'flat' part segmentation)
    self.eval_sid_pid2eval_pid_flat = espec['eval_sid_pid2eval_pid_flat']

    # Evaluation ids -> Labels
    self.eval_sid2scene_label = espec['eval_sid2scene_label']
    self.eval_pid_flat2scene_part_label = espec['eval_pid_flat2scene_part_label']

    # Get all valid evaluation sid and sid_pids
    eval_sid_total = set(self.dataset_sid2eval_sid.values())
    eval_sid_total.remove('IGNORED')
    self.eval_sid_total = list(eval_sid_total)
    eval_sid_pid_total = set(self.dataset_sid_pid2eval_sid_pid.values())
    eval_sid_pid_total.remove('IGNORED')
    self.eval_sid_pid_total = list(eval_sid_pid_total)

    # NEW:
    self.eval_sid_things = espec['eval_sid_things']
    self.eval_sid_stuff = espec['eval_sid_stuff']
    self.eval_sid_parts = espec['eval_sid_parts']
    self.eval_sid_no_parts = espec['eval_sid_no_parts']

    eval_sid_total_th_st = list(set(self.eval_sid_things + self.eval_sid_stuff))
    eval_sid_total_p_np = list(set(self.eval_sid_parts + self.eval_sid_no_parts))
    if not set(eval_sid_total_p_np) == set(eval_sid_total):
      raise ValueError('The defined set of scene classes with and without parts'
                       'is not equal to the total set of scene categories.')
    if not set(eval_sid_total_th_st) == set(eval_sid_total):
      raise ValueError('The defined set of things and stuff scene classes '
                       'is not equal to the total set of scene categories.')

    self._extract_useful_attributes()

  def _extract_useful_attributes(self):
    self.dataset_spec = self._dspec

    sids_eval2pids_eval = dict()
    for class_key in self.eval_sid_pid_total:
      class_id = class_key // 100
      if class_id in sids_eval2pids_eval.keys():
        if class_key % 100 not in sids_eval2pids_eval[class_id]:
          sids_eval2pids_eval[class_id].append(class_key % 100)
      else:
        sids_eval2pids_eval[class_id] = [class_key % 100]

    cat_definition = dict()
    cat_definition['num_cats'] = len(self.eval_sid_total)
    cat_definition['cat_def'] = list()
    for sid in self.eval_sid_total:
      cat_def = dict()
      cat_def['sem_cls'] = [sid]
      if sid in self.eval_sid_parts:
        if sid in sids_eval2pids_eval.keys():
          if len(sids_eval2pids_eval[sid]) > 1:
            cat_def['parts_cls'] = sids_eval2pids_eval[sid]
          else:
            # TODO(daan): make sure this is the behavior we want
            raise ValueError("Semantic category {} only has 1 part id defined in the EvalSpec: {}, "
                             "so in our format it is not treated as a class with parts. "
                             "Remove it as a class with parts, in the EvalSpec.".format(sid, sids_eval2pids_eval[sid]))
        else:
          raise ValueError("Semantic category {} has no part ids defined in the EvalSpec, "
                           "so it cannot be treated as a class with parts. "
                           "Remove it as a class with parts, in the EvalSpec.".format(sid))
      else:
        cat_def['parts_cls'] = [1]
        if sid in sids_eval2pids_eval.keys():
          if len(sids_eval2pids_eval[sid]) > 1:
            warnings.warn("Note: Semantic category {} will be treated as a class without parts according to EvalSpec, "
                          "even though there are {} parts defined for it.".format(sid, len(sids_eval2pids_eval[sid])),
                          Warning)
      cat_definition['cat_def'].append(cat_def)

    self.cat_definition = cat_definition

    # self.sid_pid2eval_id = parse__sid_pid2eid__v2(self.sid_pid2eid__template)
    # TODO(panos): here we assume that IGNORE eval_id exists and is the max eval_id
    # self.eid_ignore = max(self.sid_pid2eval_id.values())
    # self.sp2e_np = dict_to_numpy(self.sid_pid2eval_id, self.eid_ignore)
    # self.scene_part_classes = list(
    #     map(itemgetter(1), sorted(self.eval_id2scene_part_class.items())))
    # self.Nclasses = len(self.scene_part_classes)

if __name__ == '__main__':
  spec = PPSEvalSpec('cpp_iouparts_24_evalspec.yaml')
  breakpoint()
