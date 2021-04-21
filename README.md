# Evaluation code design

## Updates
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
   - dataset_spec.py
 - Evaluation specifications
   - cpp_iouparts_9_evalspec.yaml
   - cpp_iouparts_10_evalspec.yaml
   - cpp_iouparts_23_evalspec.yaml
   - cpp_iouparts_24_evalspec.yaml
   - ppp_iouparts_7_evalspec.yaml
   - ppp_iouparts_58_evalspec.yaml
   - eval_spec.py
 - Evaluation scripts
   - eval_segmentation_parts.py
