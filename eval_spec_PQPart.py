"""
This module contains tools for handling evaluation specifications.
"""
import sys
sys.path.append('panoptic_parts')
import yaml
import json
import platform
from operator import itemgetter

from dataset_spec import DatasetSpec

class PQPartEvalSpec(object):
  """
  This class creates an evaluation specification from a YAML specification file and provides
  convenient attributes from the specification and useful functions. Moreover, it provides
  defaults and specification checking.

  Accessible specification attributes:
    - dataset_spec: the associated dataset specification
    - cat_definition: the category/class definition for the evaluation of PQ_part

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
    self.cat_definition = espec['cat_definition']
    with open(espec['panoptic_dict_path'], 'r') as fp:
      self.gt_panoptic_dict = json.load(fp)
    # self._dspec = DatasetSpec(espec['dataset_spec_path'])


if __name__ == '__main__':
  spec = SegmentationPartsEvalSpec('cpp_iouparts_24_evalspec.yaml')
  breakpoint()
