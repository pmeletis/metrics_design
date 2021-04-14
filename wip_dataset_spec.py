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
   - l: list, the names of the scene-level semantic class
   - l_things: list, the names of the scene-level things semantic class
   - l_stuff: list, the names of the scene-level stuff semantic class
   - l_parts:
   - l_noparts:
   - sid2scene_class: dict, mapping from sid to scene-level semantic class name
  
  Member functions:
   - scene_class_from_sid(sid)
   - sid_from_scene_class(name)
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
    if self.l[0] != 'unlabeled':
      self.l.insert(0, 'unlabeled')
    self.sid2scene_class = dict(enumerate(self.l))

  def sid_from_scene_class(self, name):
    return self.l.index(name)

  def scene_class_from_sid(self, sid):
    return self.l[sid]


if __name__ == '__main__':
  spec = DatasetSpec('/home/panos/git/github/pmeletis/metrics_design/ppp_datasetspec.yaml')
  breakpoint()
