"""
This module contains tools for handling dataset specifications.
"""
import copy
from functools import partial
import platform
version = platform.python_version()
if float(version[:3]) <= 3.6:
  raise EnvironmentError('At least Python 3.7 is needed for ordered dict functionality.')

import yaml


class DatasetSpec(object):
  """
  This class creates a dataset specification from a YAML specification file, so properties
  in the specification are easily accessed. Moreover, it provides defaults and specification checking.

  Accessible specification attributes:
    - scene_class2part_classes: 
    - l: list of str, the names of the scene-level semantic classes
    - l_things: list of str, the names of the scene-level things classes
    - l_stuff: list of str, the names of the scene-level stuff classes
    - l_parts: list of str, the names of the scene-level classes with parts
    - l_noparts: list of str, the names of the scene-level classes without parts
    - sid2scene_class: dict, mapping from sid to scene-level semantic class name
    - sid2scene_color: dict, mapping from sid to scene-level semantic class color
    - scene_class_from_sid(sid)
    - sid_from_scene_class(name)
    - part_classes_from_sid(sid)
    - scene_color_from_scene_class(name)
    - scene_color_from_sid(sid)

  A special 'UNLABELED' semantic class is defined for the scene-level and part-level abstractions.
  This class must have sid/pid = 0 and is added by befault to the attributes of this class if
  it does not exist in yaml specification.

  It holds that:
    - the special 'UNLABELED' class ∈ l, l_stuff, l_noparts
    - l = l_things ∪ l_stuff
    - l = l_parts ∪ l_noparts
  """
  def __init__(self, spec_path):
    """
    Args:
      spec_path: a YAML panoptic parts dataset specification
    """
    with open(spec_path) as fd:
      spec = yaml.load(fd, Loader=yaml.Loader)

    self._spec_version = spec['version']
    # describes the semantic information layer
    self._scene_class2part_classes = spec['scene_class2part_classes']
    # describes the instance information layer
    self._scene_classes_with_instances = spec['scene_classes_with_instances']
    self._scene_class2color = spec.get('scene_class2color')
    if self._scene_class2color is None:
      raise ValueError(
          '"scene_class2color" in dataset_spec must be provided for now. '
          'In the future random color assignment will be implemented.')
    self._countable_parts_grouping = spec.get('countable_parts_grouping')

    self._extract_useful_attributes()

  def _extract_useful_attributes(self):

    def _check_and_append_unlabeled(seq, unlabeled_dct=None):
      seq = copy.copy(seq)
      if 'UNLABELED' not in seq:
        if isinstance(seq, dict):
          seq_new = unlabeled_dct
          seq_new.update(seq)
        elif isinstance(seq, list):
          seq_new = ['UNLABELED'] + seq
      if list(seq_new)[0] != 'UNLABELED':
        raise ValueError(
            f'"UNLABELED" class exists in seq but not at position 0. seq: {seq}')
      return seq_new

    # check and append (if doesn't exist) the special UNLABELED key to
    # scene_class2part_classes and scene_class2color attributes
    self.scene_class2part_classes = _check_and_append_unlabeled(self._scene_class2part_classes,
                                                                {'UNLABELED': []})
    self.scene_class2part_classes = dict(
        zip(self.scene_class2part_classes.keys(),
            map(_check_and_append_unlabeled,
                self.scene_class2part_classes.values())))
    self.scene_class2color = _check_and_append_unlabeled(self._scene_class2color,
                                                         {'UNLABELED': [0, 0, 0]})

    self.l = list(self.scene_class2part_classes)
    self.l_things = self._scene_classes_with_instances
    self.l_stuff = list(set(self.l) - set(self.l_things))
    self.l_parts = list(filter(lambda k: len(self.scene_class2part_classes[k]) >= 2,
                               self.scene_class2part_classes))
    self.l_noparts = list(set(self.l) - set(self.l_parts))
    self.sid2scene_class = dict(enumerate(self.l))
    self.sid2scene_color = {sid: self.scene_class2color[name] for sid, name in self.sid2scene_class.items()}
    self.sid2part_classes = {sid: part_classes
                             for sid, part_classes in enumerate(self.scene_class2part_classes.values())}

  def sid_from_scene_class(self, name):
    return self.l.index(name)

  def scene_class_from_sid(self, sid):
    return self.l[sid]

  def scene_color_from_scene_class(self, name):
    return self._scene_class2color[name]

  def scene_color_from_sid(self, sid):
    return self.sid2scene_color[sid]
  
  def part_classes_from_sid(self, sid):
    return self.sid2part_classes[sid]


if __name__ == '__main__':
  # spec = DatasetSpec('ppp_datasetspec.yaml')
  spec = DatasetSpec('cpp_datasetspec.yaml')
  breakpoint()
