"""
This module contains tools for handling evaluation specifications.
"""
import yaml
import platform

from dataset_spec import DatasetSpec

class IOUPartsEvalSpec(object):
  """
  This class creates an evaluation specification from a YAML specification file and provides
  convenient attributes from the specification and useful functions. Moreover, it provides
  defaults and specification checking.

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
    self.sid_pid2eid__template = espec['sid_pid2eid__template']
    self.eval_id2scene_part_class = espec['eval_id2scene_part_class']

    self._extract_useful_attributes()

  def _extract_useful_attributes(self):
    pass


if __name__ == '__main__':
  spec = IOUPartsEvalSpec('cpp_iouparts_24_evalspec.yaml')
  breakpoint()
