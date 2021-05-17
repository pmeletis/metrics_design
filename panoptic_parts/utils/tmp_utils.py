import glob


def get_filenames_in_dir(directory):
  filenames = [file for file in glob.glob(directory + "/*")]
  filenames.extend([file for file in glob.glob(directory + "/*/*")])
  return filenames


def find_filename_in_list(filename, filename_list, subject='', ext=None):
  f_found = None
  for fs in filename_list:
    if ext is not None:
      if filename in fs and fs.endswith(str(ext)):
        f_found = fs
    else:
      if filename in fs:
        f_found = fs

  if f_found is None:
    raise FileNotFoundError('There is no corresponding ' + str(subject) + ' prediction file for ' + filename)

  return f_found