# Generate part-aware panoptic segmentation results

Here, we provide a guide for generating Part-aware Panoptic Segmentation (PPS) results as in in our CVPR paper. 

### To do

**To check**:
 - mIOU evaluator (check that results are the same that we get in the paper)
 - Check that sofa and other things classes with only 1 part are not evaluated at part-level, and also not merged at part-level, because that makes no sense.

**TODO**:
- Check all functionality also for Pascal!
- Check that sofa and other things classes with only 1 part are not evaluated at part-level, and also not merged at part-level, because that makes no sense.
- Create official_evalspec for Pascal

## Prepare EvalSpec and dataset information
Before generating the Part-aware Panoptic Segmentation (PPS) results, you have to specify the dataset you wish to do this for. This consists of two parts:
1. Defining what category definition you wish to use, by using the EvalSpec.
2. Defining which images your dataset contains, and what their properties are.

### EvalSpec
In the EvalSpec, we define the following properties
* The classes that are to be evaluated, both on scene-level and part-level
* The split between _things_ and _stuff_ categories, and _parts_ and _no-parts_ categories
* The category definition and numbering that we expect for the predictions.

For the datasets that we define and use in our paper, we provide the `EvalSpec` that we use:
* `cpp_PartPQ_evalspec_default.yaml`: Cityscapes Panoptic Parts default (parts not grouped)
* `cpp_PartPQ_evalspec_grouped.yaml`: Cityscapes Panoptic Parts default (similar parts grouped)
* `ppp_PartPQ_evalspec_default.yaml`: Panoptic Panoptic Parts default

**TMP. TODO: rename these to proper names.** For now, while WIP:
* `[WIP]cpp_official_evalspec.yaml`: Cityscapes Panoptic Parts default (parts not grouped)
* `[WIP]cpp_official_evalspec_grouped.yaml`: Cityscapes Panoptic Parts default (similar parts grouped)
* `[WIP]ppp_official_evalspec.yaml`: Panoptic Panoptic Parts default


Using these `EvalSpec` definitions, we map the label definition for the raw ground-truth to the definition that we use for evaluation.
**NOTE**: This `EvalSpec` also determines how our merging and evaluation code expects the predictions. 

Examples for CPP default:
* In `eval_sid2_scene_label`, we list the evaluation ids for the scene-level classes and their labels.
  * Following this, the prediction label for `road` is `7`, `car` is `26`, etc.
* In `eval_pid_flat2scene_part_class`, we list the flat evaluation ids for part-level classes as we expect it in a part segmentation output:
  * Each part has a unique id (unless part grouping is used)
  * Following this, the prediction label for `person-head` is `2`, `rider-head` is `6`, etc.
* If you do not use the merging code, we expect you to deliver the predictions in the 3-channel format, as explained HERE (**TODO: introduce link to evaluation readme**). In this case, the expected `part_id` can be determined using `eval_sid_pid2eval_pid_flat` and the other data in the `EvalSpec`.
  * The combined `sid_pid` prediction label for `person-head` is `24_02`, so the part id is `2` (and the scene id is `24`). 
  
You can adjust the EvalSpec according to your needs, so you can adjust the mappings and the label definition you use for evaluation.

### Dataset information
To run the merging scripts, we need to know what images are in a given split of a dataset. 
Therefore, for each split (e.g., Cityscapes Panoptic Parts val), we create a json file called `ìmages.json`.

This `ìmages.json` follows the format also used in the [panopticapi](https://github.com/cocodataset/panopticapi), and contains of:
* A dictionary with the key `'images'`, for which the value is:
  * A list of dictionaries with image information. For each image, the dictionary contains:
    * `file_name`: the file name of the RGB image (NOT the ground-truth file).
    * `image_id`: a unique identifier for each image.
    * `height` and `width`: the pixel dimensions of the RGB image (and ground-truth file).

NOTE: the `image_id` defined here, should be unique, and should be used in the names of all prediction files, as explained later.

To generate the `images.json` file for Cityscapes, run the following script:

```
create_image_list(dataset_dir, 
                  output_dir, 
                  dataset):

# dataset_dir: path to the PPS ground-truths file for the data split (e.g. 'DATASET_DIR/gtFinePanopticParts_trainval/gtFinePanopticParts/val')
# óutput_dir: directory where the images.json file will be stored
# dataset: 'Cityscapes' or 'Pascal'
```

## Get results for subtasks
To generate Part-aware Panoptic Segmentation (PPS) predictions, we need to merge panoptic segmentation and part segmentation predictions. Here, we explain how to retrieve and format the predictions on these subtasks, before merging to PPS.

### Panoptic segmentation
There are two options to get panoptic segmentation results:
1. Merge semantic segmentation and instance segmentation predictions. See below how to format and merge these predictions.
2. Do predictions with network that outputs panoptic segmentation results directly.

In the case of option 2, the output needs to be stored in the format as defined for the [COCO dataset](https://cocodataset.org/#format-results):
1. A folder with PNG files storing the ids for all predicted segments.
2. A single .json file storing the semantic information for all images.

For more details on the format, check [here](https://cocodataset.org/#format-results).

**Example Cityscapes Panoptic Parts**: for a baseline in our paper, we generate results for Cityscapes using the provided ResNet-50 model from the [UPSNet repository](https://github.com/uber-research/UPSNet).

### Semantic segmentation
For semantic segmentation:
* For each image, the semantic segmentation prediction should be stored as a single PNG
* Shape: the shape of the corresponding image, i.e., `2048 x 1024` for Cityscapes.
* Each pixel has one value: the scene-level `category_id`, as defined in the `EvalSpec`.
* Name of the files: should include the unique `image_id` as defined in `images.json`.


**Example Cityscapes Panoptic Parts**: for a baseline in our paper, we generate results for Cityscapes using the [provided Xception-65 model](https://github.com/tensorflow/models/blob/master/research/deeplab/g3doc/model_zoo.md) from the official [DeepLabv3+ repository](https://github.com/tensorflow/models/blob/master/research/deeplab/g3doc/cityscapes.md).


### Instance segmentation
For instance segmentation, we accept two formats:
1. COCO format (as defined [here](https://cocodataset.org/#format-data).)
2. Cityscapes format (as defined in the comments for [Instance Level Semantic Labeling here](https://github.com/mcordts/cityscapesScripts#evaluation).)
  

For the **COCO format**, we expect:
* A single .json file per image 
* Each json file named as `ìmage_id.json`, with the `image_id` as defined in `images.json`.
* The category id in the json file should be the scene-level id as defined in the `EvalSpec`.

For the **Cityscapes format**, we expect:
* A single .txt file per image, containing per-instance info on each line:\
```relPathPrediction1 labelIDPrediction1 confidencePrediction1 ```

* The category id, (`labelIDPrediction` in the example), should be the scene-level id as defined in the `EvalSpec`.
* The name of each .txt file contains the `image_id` as defined in `images.json`.
* A singe .png containing with a mask prediction for each individual detected instance.
* See the [official Cityscapes repository](https://github.com/mcordts/cityscapesScripts/blob/master/cityscapesscripts/evaluation/evalInstanceLevelSemanticLabeling.py) for more details.


When merging with semantic segmentation to panoptic segmentation, indicate which instance segmentation format ('COCO' or 'Cityscapes') is used.

**Example Cityscapes Panoptic Parts**: for a baseline in our paper, we generate results for Cityscapes using the official [provided ResNet-50-FPN Mask R-CNN](https://github.com/facebookresearch/detectron2/blob/master/MODEL_ZOO.md) model from the [Detectron2 repository](https://github.com/facebookresearch/detectron2).


### Part segmentation
For part segmentation, we expect predictions in the same format as semantic segmentation:
* For each image, the part segmentation prediction should be stored as a single PNG
* Shape: the shape of the corresponding image, i.e., `2048 x 1024` for Cityscapes.
* Each pixel has one value: the _flat_ part-level `category_id`, as defined in the `EvalSpec`.
* Name of the files: should include the unique `image_id` as defined in `images.json`.


**Example Cityscapes Panoptic Parts**: for a baseline in our paper, we have trained a [BSANet](http://cvteam.net/projects/2019/multiclass-part.html) model with ResNet-101 backbone on our part annotations for the Cityscapes dataset. [These can be downloaded here](INSERT LINK!). **TODO: INSERT LINK**


## Merge instance and semantic segmentation to panoptic segmentation
To use the merging script, you need [pycocotools](https://github.com/cocodataset/cocoapi) and [panopticapi](https://github.com/cocodataset/panopticapi).

These can be installed through pip:
```
pip install pycocotools
pip install git+https://github.com/cocodataset/panopticapi.git
```

To merge to panoptic, run the following script

```
merge_to_panoptic.merge(eval_spec_path,
                        inst_pred_path,
                        sem_pred_path,
                        output_dir,
                        images_json,
                        instseg_format=instseg_format)

# eval_spec_path: path to the EvalSpec
# inst_pred_dir: path where the instance segmentation predictions are stored (a directory when instseg_format='Cityscapes', a JSON file when instseg_format='COCO')
# sem_pred_path: path where the semantic segmentation predictions are stored
# output_dir: directory where you wish to store the panoptic segmentation predictions
# images_json: the json file with a list of images and corresponding images ids (TODO: generate)
# instseg_format: instance segmentation encoding format (either 'COCO' or 'Cityscapes')
```

## Merge panoptic segmentation and part segmentation to PPS
To merge panoptic segmentation and part segmentation to the Part-aware Panoptic Segmentation (PPS) format, run the code below. 
It stores the PPS predictions as a 3-channel PNG in shape `[height x width x 3]`, where the 3 channels encode the `[scene_category_id, scene_instance_id, part_category_id]`.

```
merge_to_pps.merge(eval_spec_path,
                   panoptic_pred_dir,
                   panoptic_pred_json,
                   part_pred_path,
                   output_dir)

# eval_spec_path: path to the EvalSpec
# panoptic_pred_dir: directory where the panoptic segmentation predictions (png files) are stored
# panoptic_pred_json: path to the .json file with the panoptic segmentation predictions
# part_pred_path: directory where the part predictions are stored
# output_dir: directory where you wish to store the panoptic segmentation predictions
```


## Evaluate results
TODO by Chenyang

```python
evaluate_metric(eval_spec_path: str, gt_preds_paths: List(Tuple(str, str))) -> Dict(str, Union(float, list))
```


## Visualize results
TODO by Panos


## References and useful links
- Cityscapes datset
- Cityscapes scripts
- COCO dataset
- COCO API
- Pascal VOC 2010