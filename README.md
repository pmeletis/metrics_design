# Evaluation code design

## Updates
 - UPDATE 15/05: To run the refactored code: `cd metrics_design/panoptic_parts`
 - UPDATE 29/04: The decode_uids function provides now mapping of the part-level instance information layer (a.k.a instance-wise parts per things class) by providing the dataset_spec as:
   ```python
   sids, iids, pids = decode_uids(uids, experimental_dataset_spec=DatasetSpec(yaml_path))
   ```
   With this change original pids in the PPP annotations for e.g. aeroplane engine pids: {6, 7, ..., 14, 15} will all become 6, which is the pid defined by the `scene_class2part_classes` field in yaml. FYI: the sid_pid can be retrieved by `dataset_spec.sid_pid_from_scene_class_part_class('aeroplane', 'engine')` → 1_06.
 - UPDATE 26/04: Extended the documentation for DatasetSpec class with examples.
 - UPDATE 15/04: It seems that it's handier to include the dataset spec path inside the eval_spec, so the evaluation function can have the following signature (using Python typing package):

   ```python
   evaluate_metric(eval_spec_path: str,
                   gt_preds_paths: List[Tuple[str, str]]
   ) -> Dict[str, Union[float, list]]
   ```

## Metrics for Panoptic Parts datasets
Two families of metrics are provided for the evaluation of performance: IoU and PQ.

### Part Segmentation: <img src="https://render.githubusercontent.com/render/math?math=IoU^{\text{parts}}">

Run in terminal:
```bash
python -m eval_segmentation_parts path_evalspec basepath_gt basepath_pred
```

### Part-aware Panoptic Segmentation: PQ, PQ_parts <img src="https://render.githubusercontent.com/render/math?math=PQ^{\text{parts}}">
```bash
python -m eval_PQPart path_evalspec basepath_gt basepath_pred
```

## Dataset release history

Dataset releases and Github repository versions association:
 - v2: `v2` or `master` branch
 - v1: `v1` branch

### Cityscapes Panoptic Parts

 - v2: Rename GT files for complying with official Cityscapes naming scheme
 - v1: Initial release

### PASCAL Panoptic Parts

 - v2: Re-assign pids for the `tvmonitor` scene-level class, with pids: `screen`: 1, `tvframe`: 2. This change makes PPP part labels a pure superset of PASCAL-Parts-58. This is a a breaking change. The sid_pid are as following:
   - v2: 20_00: (`tvmonitor`, `UNLABELED`), 20_01: (`tvmonitor`, `screen`), 20_02: (`tvmonitor`, `frame`).
   - v1: 20_00: (`tvmonitor`, `UNLABELED`), 20_01: (`tvmonitor`, `screen`).

   Moreover, in this release the `UNLABELED` pid (00) is removed from the uid as it serves no purpose and the uids do not change, since XX_XXX_00 = XX_XXX.
 - v1: Initial release
