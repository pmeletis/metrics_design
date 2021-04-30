import json
import glob
import os
from PIL import Image

def create_image_list(dataset_dir, output_dir, dataset=None):
  images_list = list()

  # Get all filenames in the GT directory
  filenames = [file for file in glob.glob(dataset_dir + "/*")]
  if dataset == 'Cityscapes':
    filenames.extend([file for file in glob.glob(dataset_dir + "/*/*")])

  for filename in filenames:
    if filename.endswith(str('.tif')):
      image_dict = dict()
      file_name_gt = os.path.basename(filename)

      # Set names for file_name and image_id
      if dataset == 'Cityscapes':
        file_name = file_name_gt.replace('_gtFinePanopticParts.tif', '_gtFine_leftImg8bit.png')
        image_id = file_name_gt.replace('_gtFinePanopticParts.tif', '')
      else:
        file_name = file_name_gt.replace('.tif', '.png')
        image_id = file_name_gt.replace('.tif', '')
      image_dict['file_name'] = file_name
      image_dict['id'] = image_id

      # Open gt image and store image dimensions
      img = Image.open(filename)
      image_dict['width'], image_dict['height'] = img.size[0:2]

      images_list.append(image_dict)

  images_dict = {'images': images_list}

  # Save images.json file
  output_path = os.path.join(output_dir, 'images.json')
  with open(output_path, 'w') as fp:
    json.dump(images_dict, fp)

  print("Created images list and stored at {}.".format(output_path))

if __name__ == '__main__':
  # TODO(daan): inlcude args to run from command line
  dataset_dir = "/home/ddegeus/hdnew/dataset/CityscapesPanParts/gtFinePanopticParts_trainval/gtFinePanopticParts/val"
  output_dir = "/home/ddegeus/hdnew/dataset/CityscapesPanParts/gtFinePanopticParts_trainval/gtFinePanopticParts/val"

  create_image_list(dataset_dir, output_dir, dataset='Cityscapes')