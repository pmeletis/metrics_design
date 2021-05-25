"""
This module contains tools for handling evaluation specifications.
"""
import yaml
import json
import platform
from operator import itemgetter

from panoptic_parts.utils.experimental_evaluation_IOU import parse__sid_pid2eid__v2
from panoptic_parts.utils.utils import (
    _sparse_ids_mapping_to_dense_ids_mapping as dict_to_numpy)
from panoptic_parts.specs.dataset_spec import DatasetSpec

class SegmentationPartsEvalSpec(object):
  """
  This class creates an evaluation specification from a YAML specification file and provides
  convenient attributes from the specification and useful functions. Moreover, it provides
  defaults and specification checking.

  Accessible specification attributes:
    - dataset_spec: the associated dataset specification
    - Nclasses: the number of evaluated classes (including ignored and background)
    - scene_part_classes: list of str, the names of the scene-part classes for evaluation,
        ordered by the eval id
    - eid_ignore: the eval_id to be ignored in evaluation
    - sid_pid2eval_id: dict, maps all sid_pid (0-99_99) to an eval_id,
        according to the template in specification yaml
    - sp2e_np: np.ndarray, shape: (10000,), sid_pid2eval_id as an array for dense gathering,
        position i has the sid_pid2eval_id[i] value
  
  Member functions:
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
    self.sid_pid2eid__template = espec['sid_pid2eid__template']
    self.eval_id2scene_part_class = espec['eval_id2scene_part_class']
    self._dspec = DatasetSpec(espec['dataset_spec_path'])

    self._extract_useful_attributes()

  def _extract_useful_attributes(self):
    self.dataset_spec = self._dspec
    self.sid_pid2eval_id = parse__sid_pid2eid__v2(self.sid_pid2eid__template)
    # TODO(panos): here we assume that IGNORE eval_id exists and is the max eval_id
    self.eid_ignore = max(self.sid_pid2eval_id.values())
    self.sp2e_np = dict_to_numpy(self.sid_pid2eval_id, self.eid_ignore)
    self.scene_part_classes = list(
        map(itemgetter(1), sorted(self.eval_id2scene_part_class.items())))
    self.Nclasses = len(self.scene_part_classes)


if __name__ == '__main__':
  spec = SegmentationPartsEvalSpec('cpp_iouparts_24_evalspec.yaml')
  breakpoint()