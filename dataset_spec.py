"""
This module contains tools for handling dataset specifications.
"""
import yaml
import platform

class DatasetSpec(object):
  """
  This class creates a dataset specification from a YAML specification file, where are properties
  in the specification are easily accessed. Moreover, it provides defaults and specification checking.

  Accessible specification attributes:
   - l: list of str, the names of the scene-level semantic classes
   - l_things: list of str, the names of the scene-level things classes
   - l_stuff: list of str, the names of the scene-level stuff classes
   - l_parts: list of str, the names of the scene-level classes with parts
   - l_noparts: list of str, the names of the scene-level classes without parts
   - sid2scene_class: dict, mapping from sid to scene-level semantic class name
   - sid2scene_color: dict, mapping from sid to scene-level semantic class color
  
  Member functions:
   - scene_class_from_sid(sid)
   - sid_from_scene_class(name)
   - scene_color_from_scene_class(name)
   - scene_color_from_sid(sid)
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
    version = platform.python_version()
    if float(version[:3]) <= 3.6:
      raise EnvironmentError('At least Python 3.7 is needed for ordered dict functionality.')
    self.l = list(self._scene_class2part_classes.keys())
    self.l_things = self._scene_classes_with_instances
    self.l_stuff = list(set(self.l) - set(self.l_things))
    self.l_parts = list(filter(lambda k: bool(self._scene_class2part_classes[k]),
                               self._scene_class2part_classes.keys()))
    self.l_noparts = list(set(self.l) - set(self.l_parts))
    if 'unlabeled' not in self.l:
      self.l.insert(0, 'unlabeled')
    else:
      if self.l[0] != 'unlabeled':
        raise ValueError(
            '"unlabeled" scene-level class exists in scene_class2part_classes.keys() but not in position 0.')
    self.sid2scene_class = dict(enumerate(self.l))
    self.sid2scene_color = {sid: self._scene_class2color[name] for sid, name in self.sid2scene_class.items()}

  def sid_from_scene_class(self, name):
    return self.l.index(name)

  def scene_class_from_sid(self, sid):
    return self.l[sid]

  def scene_color_from_scene_class(self, name):
    return self._scene_class2color[name]

  def scene_color_from_sid(self, sid):
    return self.sid2scene_color[sid]


if __name__ == '__main__':
  # spec = DatasetSpec('ppp_datasetspec.yaml')
  spec = DatasetSpec('cpp_datasetspec.yaml')
  breakpoint()
