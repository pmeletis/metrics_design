# Evaluation code design

## Updates
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

## Almost finalized
 - Dataset specifications
   - ppp_datasetspec.yaml
   - cpp_datasetspec.yaml
 - Evaluation specifications
   - cpp_iouparts_{9, 10, 23, 24}_evalspec.yaml
   - ppp_iouparts_{7, 58}_evalspec.yaml
   - cpp_partPQ_24_evalspec.yaml
 - Specification parsing
   - dataset_spec.py
   - eval_spec.py
 - Evaluation scripts
   - eval_segmentation_parts.py
   - eval_PQPart.py

## ToDo's
 - add eval spec and test PPQ metric for PPP
 - remove the generate_ignore_info and requirements (gt_panoptic_dict, panoptic_ann_img, image_id) as this info can be extracted from tiff + dataset_spec
 - welcome page and CVPR READMEs
 - refactor eval_* scripts

## Timeline
 - 30/04: Finalize ToDo's
 - 02/05: Complete internal testing by three of us
 - 04/05: Publish repo
 - 05/05: Upload CVPR to arxiv
 - 18/05: Update Tech report on arxiv
