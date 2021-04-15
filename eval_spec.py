"""
This module contains tools for handling evaluation specifications.
"""
import yaml
import platform

from dataset_spec import DatasetSpec

class IOUEvalSpec(object):
  """
  This class creates an evaluation specification from a YAML specification file and provides convenient attributes from the specification and useful functions. Moreover, it provides defaults and specification checking.

  Accessible specification attributes:
   - dataset_spec: the associated dataset specification
  
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
    self._dspec = DatasetSpec(espec['dataset_spec_path'])
    self.dataset_spec = self._dspec
    self._scene_class_new2scene_classes_old = espec.get('scene_class2part_classes')
    if self._scene_class_new2scene_classes_old is None:
      print('TODO')
      # generate default mapping from _dspec
    self._part_groupings = espec.get('part_groupings')

    self._extract_useful_attributes()

  def _extract_useful_attributes(self):
    pass


if __name__ == '__main__':
  spec = IOUEvalSpec('ppp_20_58_iou_evalspec.yaml')
  breakpoint()
